#include "apps/gui/viewer/viewport_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>

#include <QColor>
#include <QFile>
#include <QImage>
#include <QMatrix4x4>
#include <QSize>

// The RHI headers are Qt's private-module ones (Qt6::GuiPrivate); there is no public
// <QShader> header, only <rhi/qshader.h>.
#include <rhi/qrhi.h>
#include <rhi/qshader.h>

#include "apps/gui/viewer/camera_controller.hpp"

namespace alphashape::gui {

namespace {

/// One uniform block shared by every shader in shaders/: 2 mat4 + 2 vec4 = 160 bytes.
constexpr quint32 kUniformSize = 2U * 64U + 2U * 16U;
/// Screen-facing point sprites; this is their edge length in device pixels.
constexpr float kPointSize = 3.0F;

constexpr quint32 kPointStride = 6U * 4U;  // position(3f) + colour(3f)
constexpr quint32 kMeshStride = 11U * 4U;  // position(3f) colour(3f) normal(3f) uv(2f)
constexpr quint32 kLineStride = 6U * 4U;   // position(3f) + colour(3f)
constexpr quint32 kQuadStride = 2U * 4U;   // corner(2f)

/// Two triangles as a strip, corners in [-1, 1]^2.
constexpr float kQuadCorners[4][2] = {{-1.0F, -1.0F}, {1.0F, -1.0F}, {-1.0F, 1.0F}, {1.0F, 1.0F}};

// The strides above are the single source of truth for the GPU layout: if a Scene struct's
// layout ever changes, these fail here instead of producing silently wrong vertices.
static_assert(sizeof(ScenePointVertex) == kPointStride, "ScenePointVertex must stay packed");
static_assert(sizeof(SceneMeshVertex) == kMeshStride, "SceneMeshVertex must stay packed");
static_assert(sizeof(SceneLineVertex) == kLineStride, "SceneLineVertex must stay packed");
static_assert(alignof(SceneMeshVertex) <= 4, "the vertex arrays must need no alignment padding");
static_assert(offsetof(SceneMeshVertex, normal) == 12, "mesh normal must follow the position");
static_assert(offsetof(SceneMeshVertex, colour) == 24, "mesh colour must follow the normal");
static_assert(offsetof(SceneMeshVertex, uv) == 36, "mesh uv must follow the colour");

QShader load_shader(const char* name) {
    QFile file(QStringLiteral(":/alphashape/shaders/") + QLatin1String(name));
    if (!file.open(QIODevice::ReadOnly)) {
        return QShader();
    }
    return QShader::fromSerialized(file.readAll());
}

QMatrix4x4 to_qt(const Eigen::Matrix4d& matrix) {
    QMatrix4x4 out;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            out(row, column) = static_cast<float>(matrix(row, column));
        }
    }
    return out;
}

Eigen::Matrix4d to_eigen(const QMatrix4x4& matrix) {
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            out(row, column) = static_cast<double>(matrix(row, column));
        }
    }
    return out;
}

/// The GPU vertex layouts, matching the shaders' attribute locations.
QRhiVertexInputLayout point_layout() {
    QRhiVertexInputLayout layout;
    layout.setBindings({
        QRhiVertexInputBinding(kQuadStride),
        QRhiVertexInputBinding(kPointStride, QRhiVertexInputBinding::PerInstance),
    });
    layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0),
        QRhiVertexInputAttribute(1, 1, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(1, 2, QRhiVertexInputAttribute::Float3, 12),
    });
    return layout;
}

QRhiVertexInputLayout mesh_layout() {
    QRhiVertexInputLayout layout;
    layout.setBindings({QRhiVertexInputBinding(kMeshStride)});
    // SceneMeshVertex: position(3f) normal(3f) colour(3f) uv(2f) -> locations 0..3.
    layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 12),
        QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 24),
        QRhiVertexInputAttribute(0, 3, QRhiVertexInputAttribute::Float2, 36),
    });
    return layout;
}

QRhiVertexInputLayout line_layout() {
    QRhiVertexInputLayout layout;
    layout.setBindings({QRhiVertexInputBinding(kLineStride)});
    layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 12),
    });
    return layout;
}

/// The pixel size of the page the scene wants: its texture, or 1x1 for the white default.
QSize page_size_of(const Scene& scene) {
    if (scene.texture.empty()) {
        return QSize(1, 1);
    }
    return QSize(scene.texture.width, scene.texture.height);
}

}  // namespace

ViewportRenderer::ViewportRenderer() = default;

ViewportRenderer::~ViewportRenderer() { release(); }

void ViewportRenderer::set_scene(Scene scene) {
    scene_ = std::move(scene);
    upload_pending_ = true;
    // Size the per-scene buffers now: this is called from the GUI thread between frames, which
    // is the same point at which QRhiWidget re-creates its own colour texture on a resize.
    // Creating a resource while a frame is being recorded is what has to be avoided, not here.
    if (rhi_ != nullptr) {
        size_scene_buffers();
    }
}

bool ViewportRenderer::size_scene_buffers() {
    if (rhi_ == nullptr) {
        return false;
    }
    const auto ensure = [&](QRhiBuffer*& buffer, QRhiBuffer::UsageFlags usage, quint32 size) {
        const quint32 wanted = std::max(size, 1U);
        if (buffer != nullptr && buffer->size() == wanted) {
            return true;
        }
        if (buffer != nullptr) {
            // A buffer the last frame drew with may still be in flight: deleteLater() is QRhi's
            // own answer to that (it defers until the frame that used it has been submitted).
            buffer->deleteLater();
            buffer = nullptr;
        }
        buffer = rhi_->newBuffer(QRhiBuffer::Static, usage, wanted);
        return buffer != nullptr && buffer->create();
    };
    const bool ok =
        ensure(mesh_vertices_, QRhiBuffer::VertexBuffer,
               static_cast<quint32>(scene_.mesh_vertices.size() * kMeshStride)) &&
        ensure(mesh_indices_, QRhiBuffer::IndexBuffer,
               static_cast<quint32>(scene_.mesh_indices.size() * sizeof(std::uint32_t))) &&
        ensure(tie_points_, QRhiBuffer::VertexBuffer,
               static_cast<quint32>(scene_.tie_points.size() * kPointStride)) &&
        ensure(dense_points_, QRhiBuffer::VertexBuffer,
               static_cast<quint32>(scene_.dense_points.size() * kPointStride)) &&
        ensure(lines_, QRhiBuffer::VertexBuffer,
               static_cast<quint32>(scene_.lines.size() * kLineStride));
    if (!ok) {
        error_ = QStringLiteral("could not create the scene buffers");
    }
    return ok;
}

void ViewportRenderer::release() {
    // Resources must be destroyed while their QRhi is still alive; the widget calls this from
    // releaseResources(), i.e. exactly before the old QRhi can go away.
    destroy_resources();
    rhi_ = nullptr;
    render_pass_ = nullptr;
    sample_count_ = 1;
    upload_pending_ = true;
    upload_quad_ = true;
    stats_ = Stats{};
}

void ViewportRenderer::destroy_resources() {
    delete pipeline_mesh_;
    pipeline_mesh_ = nullptr;
    delete pipeline_points_;
    pipeline_points_ = nullptr;
    delete pipeline_lines_;
    pipeline_lines_ = nullptr;
    delete bindings_;
    bindings_ = nullptr;
    delete uniforms_;
    uniforms_ = nullptr;
    delete sampler_;
    sampler_ = nullptr;
    delete page_;
    page_ = nullptr;
    delete point_quad_;
    point_quad_ = nullptr;
    upload_quad_ = true;
    delete mesh_vertices_;
    mesh_vertices_ = nullptr;
    delete mesh_indices_;
    mesh_indices_ = nullptr;
    delete tie_points_;
    tie_points_ = nullptr;
    delete dense_points_;
    dense_points_ = nullptr;
    delete lines_;
    lines_ = nullptr;
}

QString ViewportRenderer::backend_name() const {
    return rhi_ != nullptr ? QString::fromUtf8(rhi_->backendName()) : QStringLiteral("none");
}

bool ViewportRenderer::ensure(QRhi* rhi, QRhiRenderPassDescriptor* render_pass, int sample_count) {
    if (rhi == nullptr || render_pass == nullptr) {
        release();
        error_ = QStringLiteral("no render target");
        return false;
    }
    if (rhi != rhi_) {
        // A different device: nothing created against the old one may survive.
        release();
    }
    // A pipeline is compiled for one render pass descriptor and its sample count, so a change in
    // either is a full rebuild. Otherwise an intact resource set is left alone.
    const bool rebuild =
        render_pass != render_pass_ || sample_count != sample_count_ || !ready();
    if (!rebuild) {
        return true;
    }
    destroy_resources();
    if (!create_resources(rhi, render_pass, sample_count)) {
        destroy_resources();
        rhi_ = nullptr;
        render_pass_ = nullptr;
        return false;
    }
    rhi_ = rhi;
    render_pass_ = render_pass;
    sample_count_ = sample_count;
    upload_pending_ = true;
    // Now that there is a device, the per-scene buffers can be created at the right size.
    size_scene_buffers();
    return true;
}

bool ViewportRenderer::create_resources(QRhi* rhi, QRhiRenderPassDescriptor* render_pass,
                                        int sample_count) {
    const QShader mesh_vert = load_shader("mesh.vert.qsb");
    const QShader mesh_frag = load_shader("mesh.frag.qsb");
    const QShader point_vert = load_shader("points.vert.qsb");
    const QShader point_frag = load_shader("points.frag.qsb");
    const QShader line_vert = load_shader("lines.vert.qsb");
    const QShader line_frag = load_shader("lines.frag.qsb");
    if (!mesh_vert.isValid() || !mesh_frag.isValid() || !point_vert.isValid() ||
        !point_frag.isValid() || !line_vert.isValid() || !line_frag.isValid()) {
        error_ = QStringLiteral("the compiled shaders (:alphashape/shaders/*.qsb) are missing");
        return false;
    }

    uniforms_ = rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, kUniformSize);
    if (!uniforms_->create()) {
        error_ = QStringLiteral("could not create the uniform buffer");
        return false;
    }

    sampler_ = rhi->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::None,
                               QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge);
    if (!sampler_->create()) {
        error_ = QStringLiteral("could not create the sampler");
        return false;
    }

    // The page texture, at the size the current scene wants. Its content is uploaded in the
    // first frame (upload_scene).
    page_ = rhi->newTexture(QRhiTexture::RGBA8, page_size_of(scene_));
    if (!page_->create()) {
        error_ = QStringLiteral("could not create the texture");
        return false;
    }

    point_quad_ = rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer,
                                 static_cast<quint32>(sizeof(kQuadCorners)));
    if (!point_quad_->create()) {
        error_ = QStringLiteral("could not create the point-sprite quad");
        return false;
    }

    // Scene buffers are created by size_scene_buffers() (set_scene / upload_scene); the resource
    // set below is what the pipelines and bindings point at.

    bindings_ = rhi->newShaderResourceBindings();
    bindings_->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage,
                                                 uniforms_),
        QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage,
                                                  page_, sampler_),
    });
    if (!bindings_->create()) {
        error_ = QStringLiteral("could not create the shader resource bindings");
        return false;
    }

    const auto make_pipeline = [&](const QShader& vert, const QShader& frag,
                                   QRhiGraphicsPipeline::Topology topology,
                                   const QRhiVertexInputLayout& layout,
                                   QRhiGraphicsPipeline** into) {
        QRhiGraphicsPipeline* pipeline = rhi->newGraphicsPipeline();
        pipeline->setShaderStages(
            {{QRhiShaderStage::Vertex, vert}, {QRhiShaderStage::Fragment, frag}});
        pipeline->setTopology(topology);
        pipeline->setCullMode(QRhiGraphicsPipeline::None);
        pipeline->setDepthTest(true);
        pipeline->setDepthWrite(true);
        pipeline->setDepthOp(QRhiGraphicsPipeline::Less);
        pipeline->setSampleCount(sample_count);
        pipeline->setVertexInputLayout(layout);
        pipeline->setShaderResourceBindings(bindings_);
        pipeline->setRenderPassDescriptor(render_pass);
        if (!pipeline->create()) {
            delete pipeline;
            return false;
        }
        *into = pipeline;
        return true;
    };

    if (!make_pipeline(mesh_vert, mesh_frag, QRhiGraphicsPipeline::Triangles, mesh_layout(),
                       &pipeline_mesh_)) {
        error_ = QStringLiteral("could not create the mesh pipeline");
        return false;
    }
    if (!make_pipeline(point_vert, point_frag, QRhiGraphicsPipeline::TriangleStrip,
                       point_layout(), &pipeline_points_)) {
        error_ = QStringLiteral("could not create the point pipeline");
        return false;
    }
    if (!make_pipeline(line_vert, line_frag, QRhiGraphicsPipeline::Lines, line_layout(),
                       &pipeline_lines_)) {
        error_ = QStringLiteral("could not create the line pipeline");
        return false;
    }
    error_.clear();
    return true;
}

void ViewportRenderer::upload_scene(QRhiResourceUpdateBatch* batch) {
    size_scene_buffers();
    const auto upload = [&](QRhiBuffer* buffer, quint32 size, const void* data) {
        if (buffer == nullptr || size == 0U) {
            return;
        }
        batch->uploadStaticBuffer(buffer, 0, size, data);
    };

    upload(mesh_vertices_, static_cast<quint32>(scene_.mesh_vertices.size() * kMeshStride),
           scene_.mesh_vertices.data());
    upload(mesh_indices_,
           static_cast<quint32>(scene_.mesh_indices.size() * sizeof(std::uint32_t)),
           scene_.mesh_indices.data());
    upload(tie_points_, static_cast<quint32>(scene_.tie_points.size() * kPointStride),
           scene_.tie_points.data());
    upload(dense_points_, static_cast<quint32>(scene_.dense_points.size() * kPointStride),
           scene_.dense_points.data());
    upload(lines_, static_cast<quint32>(scene_.lines.size() * kLineStride), scene_.lines.data());

    if (point_quad_ != nullptr && upload_quad_) {
        batch->uploadStaticBuffer(point_quad_, kQuadCorners);
        upload_quad_ = false;
    }

    if (page_ != nullptr) {
        // The page texture is resized in place, exactly as QRhiWidget resizes its own colour
        // texture: the bindings and the pipelines keep pointing at the same object.
        const QSize wanted = page_size_of(scene_);
        if (page_->pixelSize() != wanted) {
            page_->setPixelSize(wanted);
            if (!page_->create()) {
                error_ = QStringLiteral("could not resize the texture page");
                upload_pending_ = false;
                return;
            }
        }
        if (scene_.texture.empty()) {
            const std::uint8_t white[4] = {255, 255, 255, 255};
            batch->uploadTexture(page_, QImage(white, 1, 1, QImage::Format_RGBA8888));
        } else {
            const QImage image(scene_.texture.rgba.data(), scene_.texture.width,
                               scene_.texture.height, QImage::Format_RGBA8888);
            batch->uploadTexture(page_, image);
        }
    }
    upload_pending_ = false;
}

void ViewportRenderer::render(QRhiCommandBuffer* cb, QRhiRenderTarget* target,
                              const CameraController& camera, const QSize& pixel_size) {
    stats_ = Stats{};
    if (rhi_ == nullptr || cb == nullptr || target == nullptr || !ready()) {
        return;
    }
    const int width = std::max(pixel_size.width(), 1);
    const int height = std::max(pixel_size.height(), 1);
    const double aspect = static_cast<double>(width) / static_cast<double>(height);

    QRhiResourceUpdateBatch* batch = rhi_->nextResourceUpdateBatch();
    if (upload_pending_) {
        upload_scene(batch);
    }

    // Clip <- world in one matrix: the backend's clip-space correction on the left of the camera
    // matrices, which are OpenGL-convention (see camera_controller.hpp).
    const QMatrix4x4 mvp =
        to_qt(to_eigen(rhi_->clipSpaceCorrMatrix()) * camera.view_projection(aspect));

    float values[kUniformSize / sizeof(float)] = {};
    std::memcpy(values, mvp.constData(), 16U * sizeof(float));
    const QMatrix4x4 identity;
    std::memcpy(values + 16, identity.constData(), 16U * sizeof(float));
    // One world-space key light, up and over the viewer's left shoulder.
    const double light[3] = {0.35, 0.45, 0.82};
    const double light_length =
        std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
    values[32] = static_cast<float>(light[0] / light_length);
    values[33] = static_cast<float>(light[1] / light_length);
    values[34] = static_cast<float>(light[2] / light_length);
    values[35] = 0.0F;
    values[36] = static_cast<float>(width);
    values[37] = static_cast<float>(height);
    values[38] = kPointSize;
    values[39] = 0.0F;
    batch->updateDynamicBuffer(uniforms_, 0, kUniformSize, values);

    const auto channel = [](float value) {
        return static_cast<int>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
    };
    const QColor clear(channel(background_.x()), channel(background_.y()),
                       channel(background_.z()));

    cb->beginPass(target, clear, {1.0F, 0}, batch);
    cb->setViewport(QRhiViewport(0.0F, 0.0F, static_cast<float>(width),
                                 static_cast<float>(height)));
    if (pipeline_mesh_ != nullptr && mesh_vertices_ != nullptr && mesh_indices_ != nullptr &&
        !scene_.mesh_indices.empty()) {
        cb->setGraphicsPipeline(pipeline_mesh_);
        cb->setShaderResources(bindings_);
        const QRhiCommandBuffer::VertexInput input = {mesh_vertices_, 0};
        cb->setVertexInput(0, 1, &input, mesh_indices_, 0, QRhiCommandBuffer::IndexUInt32);
        const quint32 indices = static_cast<quint32>(scene_.mesh_indices.size());
        cb->drawIndexed(indices);
        stats_.mesh_triangles = indices / 3U;
        stats_.textured = !scene_.texture.empty();
    }

    const auto draw_points = [&](QRhiBuffer* buffer, quint32 count) {
        if (pipeline_points_ == nullptr || buffer == nullptr || point_quad_ == nullptr ||
            count == 0U) {
            return;
        }
        cb->setGraphicsPipeline(pipeline_points_);
        cb->setShaderResources(bindings_);
        const QRhiCommandBuffer::VertexInput inputs[2] = {{point_quad_, 0}, {buffer, 0}};
        cb->setVertexInput(0, 2, inputs);
        cb->draw(4, count);
    };
    draw_points(dense_points_, static_cast<quint32>(scene_.dense_points.size()));
    stats_.dense_points = static_cast<quint32>(scene_.dense_points.size());
    draw_points(tie_points_, static_cast<quint32>(scene_.tie_points.size()));
    stats_.tie_points = static_cast<quint32>(scene_.tie_points.size());

    if (pipeline_lines_ != nullptr && lines_ != nullptr && !scene_.lines.empty()) {
        cb->setGraphicsPipeline(pipeline_lines_);
        cb->setShaderResources(bindings_);
        const QRhiCommandBuffer::VertexInput input = {lines_, 0};
        cb->setVertexInput(0, 1, &input);
        const quint32 vertices = static_cast<quint32>(scene_.lines.size());
        cb->draw(vertices);
        stats_.line_segments = vertices / 2U;
    }
    cb->endPass();
    stats_.draws = 1;
}

}  // namespace alphashape::gui
