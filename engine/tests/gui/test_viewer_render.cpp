// ViewerRender.*: the viewport's QRhi renderer, driven without a window.
//
// These create their own QRhi and render into an offscreen texture render target, so they need
// neither a QRhiWidget nor a platform integration that offers RHI-based rendering (the offscreen
// QPA plugin CI uses does not). What that buys per platform:
//
//   macOS   Metal (the runner has a paravirtual GPU)  -> real pixels are asserted
//   Windows D3D11 (WARP when there is no adapter)     -> real pixels are asserted
//   Linux   no Vulkan/GL platform integration under the offscreen plugin, so the suite falls
//           back to QRhi's Null backend: the pipeline/buffer/upload/draw recording is exercised
//           for real, the pixels are not (Null rasterises nothing)
//
// None of these ever fails the suite for "no GPU": a missing backend skips one test with a
// message that says which step could not run.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include <QColor>
#include <QFile>
#include <QImage>
#include <QOffscreenSurface>
#include <QSize>
#include <QString>
#include <QSurfaceFormat>
#include <QtGlobal>

// The Vulkan attempt is only compiled where this project would use it: the Linux lane, whose
// xcb platform can create a QVulkanInstance and whose lavapipe does the rasterising. Windows and
// macOS have their own native backend and no QVulkanInstance in the Qt build there.
#if defined(Q_OS_LINUX)
#define ALPHASHAPE_VIEWER_TRY_VULKAN 1
#include <QVulkanInstance>
#else
#define ALPHASHAPE_VIEWER_TRY_VULKAN 0
#endif

#include <rhi/qrhi.h>
#include <rhi/qshader.h>

#include "apps/gui/viewer/camera_controller.hpp"
#include "apps/gui/viewer/scene.hpp"
#include "apps/gui/viewer/viewport_renderer.hpp"
#include "common/core/chunk.hpp"
#include "model/mesh/mesh.hpp"

using namespace alphashape;
using namespace alphashape::gui;

namespace {

/// A QRhi of the platform's native backend, plus the reason when there is none. The objects the
/// backend borrows (the Vulkan instance, the GL fallback surface) live here too, so their
/// lifetime covers the QRhi's.
struct Device {
    // Declaration order is destruction order reversed: the borrowed objects (the Vulkan instance,
    // the GL fallback surface) are declared BEFORE the QRhi, so the QRhi is destroyed first and
    // never outlives what it borrows.
#if ALPHASHAPE_VIEWER_TRY_VULKAN
    std::unique_ptr<QVulkanInstance> vulkan_instance;
#endif
    std::unique_ptr<QOffscreenSurface> gl_surface;
    std::unique_ptr<QRhi> rhi;
    QString reason;
    bool software = false;

    bool usable() const { return rhi != nullptr; }
};

Device make_device() {
    Device device;
    const QRhi::Flags flags;

#if defined(Q_OS_MACOS)
    {
        QRhiMetalInitParams params;
        if (QRhi* rhi = QRhi::create(QRhi::Metal, &params, flags)) {
            device.rhi.reset(rhi);
            return device;
        }
        device.reason = QStringLiteral("Metal could not be initialised");
    }
#elif defined(Q_OS_WIN)
    {
        QRhiD3D11InitParams params;
        QRhi* rhi = QRhi::create(QRhi::D3D11, &params, flags);
        if (rhi == nullptr) {
            // A headless runner with no adapter still has WARP.
            rhi = QRhi::create(QRhi::D3D11, &params, QRhi::PreferSoftwareRenderer);
            device.software = rhi != nullptr;
        }
        if (rhi != nullptr) {
            device.rhi.reset(rhi);
            return device;
        }
        device.reason = QStringLiteral("D3D11 could not be initialised");
    }
#else
#if ALPHASHAPE_VIEWER_TRY_VULKAN
    {
        // Vulkan first: the Linux CI lane runs this suite under Xvfb so the xcb platform can
        // create an instance, and lavapipe does the rasterising.
        auto instance = std::make_unique<QVulkanInstance>();
        instance->setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
        // Without an explicit version QVulkanInstance asks for API version 0, which the loader's
        // validation layer rejects outright (the instance is then created invalid and the first
        // device call segfaults).
        instance->setApiVersion(QVersionNumber(1, 0));
        if (instance->create()) {
            QRhiVulkanInitParams params;
            params.inst = instance.get();
            if (QRhi* rhi = QRhi::create(QRhi::Vulkan, &params, flags)) {
                device.vulkan_instance = std::move(instance);
                device.rhi.reset(rhi);
                return device;
            }
        }
    }
#endif
    {
        // OpenGL next: it needs an offscreen surface, which needs a platform that reports
        // OpenGL support (xcb does, the plain offscreen plugin does not).
        QSurfaceFormat format;
        format.setDepthBufferSize(24);
        format.setVersion(3, 3);
        auto surface = std::unique_ptr<QOffscreenSurface>(
            QRhiGles2InitParams::newFallbackSurface(format));
        if (surface != nullptr && surface->isValid()) {
            QRhiGles2InitParams params;
            params.fallbackSurface = surface.get();
            params.format = format;
            if (QRhi* rhi = QRhi::create(QRhi::OpenGLES2, &params, flags)) {
                device.gl_surface = std::move(surface);
                device.rhi.reset(rhi);
                return device;
            }
        }
        device.reason = QStringLiteral("no Vulkan or OpenGL platform integration under this "
                                       "platform plugin");
    }
#endif

    // Nothing accelerated is available. This still drives the whole recording path for real
    // (pipelines, buffers, uploads, draws); it just rasterises nothing.
    QRhiNullInitParams null_params;
    if (QRhi* rhi = QRhi::create(QRhi::Null, &null_params)) {
        device.rhi.reset(rhi);
        device.software = true;
        if (device.reason.isEmpty()) {
            device.reason = QStringLiteral("only the Null backend is available");
        }
        return device;
    }
    if (device.reason.isEmpty()) {
        device.reason = QStringLiteral("no QRhi backend could be created");
    }
    return device;
}

/// A scene that is exactly one white-ish quad at the origin: no cameras, no points, no grid, so
/// "fit" frames the quad and the centre pixel is guaranteed to be covered by it.
Chunk make_quad_chunk() {
    Chunk chunk("Quad");
    Mesh& mesh = chunk.model();
    mesh.vertices = {
        Eigen::Vector3d(-1.0, -1.0, 0.0),
        Eigen::Vector3d(1.0, -1.0, 0.0),
        Eigen::Vector3d(1.0, 1.0, 0.0),
        Eigen::Vector3d(-1.0, 1.0, 0.0),
    };
    mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 2, 3)};
    mesh.colours.assign(mesh.vertices.size() * 3U, 200U);
    return chunk;
}

/// A render target plus the readback plumbing every render assertion needs.
class OffscreenTarget {
public:
    OffscreenTarget(QRhi* rhi, const QSize& size) : size_(size) {
        color_.reset(rhi->newTexture(QRhiTexture::RGBA8, size_, 1,
                                     QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
        ok_ = color_->create();
        if (!ok_) {
            return;
        }
        depth_.reset(rhi->newRenderBuffer(QRhiRenderBuffer::DepthStencil, size_, 1));
        ok_ = depth_->create();
        if (!ok_) {
            return;
        }
        QRhiTextureRenderTargetDescription description(QRhiColorAttachment(color_.get()),
                                                       depth_.get());
        target_.reset(rhi->newTextureRenderTarget(description,
                                                  // Keep the colour contents across passes: the
                                                  // readback below runs in its own frame.
                                                  QRhiTextureRenderTarget::PreserveColorContents));
        pass_.reset(target_->newCompatibleRenderPassDescriptor());
        target_->setRenderPassDescriptor(pass_.get());
        ok_ = target_->create();
    }

    bool ok() const { return ok_; }
    QRhiRenderTarget* target() { return target_.get(); }
    QRhiRenderPassDescriptor* pass() { return pass_.get(); }
    QSize size() const { return size_; }

    /// Read the rendered image back. Caller must have ended the frame.
    QImage read(QRhi* rhi) {
        QRhiReadbackResult result;
        bool completed = false;
        result.completed = [&completed]() { completed = true; };
        QRhiReadbackDescription description(color_.get());
        QRhiResourceUpdateBatch* batch = rhi->nextResourceUpdateBatch();
        batch->readBackTexture(description, &result);

        QRhiCommandBuffer* cb = nullptr;
        if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) {
            return QImage();
        }
        cb->resourceUpdate(batch);
        rhi->endOffscreenFrame();
        rhi->finish();
        if (!completed || result.data.isEmpty()) {
            return QImage();
        }
        // The colour texture is RGBA8; QRhi hands back tightly packed rows.
        QImage image(reinterpret_cast<const uchar*>(result.data.constData()),
                     result.pixelSize.width(), result.pixelSize.height(),
                     QImage::Format_RGBA8888);
        if (rhi->isYUpInFramebuffer()) {
            image = image.mirrored();
        }
        return image.copy();
    }

private:
    QSize size_;
    bool ok_ = false;
    std::unique_ptr<QRhiTexture> color_;
    std::unique_ptr<QRhiRenderBuffer> depth_;
    std::unique_ptr<QRhiTextureRenderTarget> target_;
    std::unique_ptr<QRhiRenderPassDescriptor> pass_;
};

/// Render `scene` with `camera` into a fresh target and return the image plus the stats.
struct RenderResult {
    bool ok = false;
    QImage image;
    ViewportRenderer::Stats stats;
    QString error;
};

RenderResult render_once(QRhi* rhi, Scene scene, const CameraController& camera, const QSize& size,
                         const QColor& background) {
    RenderResult result;
    OffscreenTarget target(rhi, size);
    if (!target.ok()) {
        result.error = QStringLiteral("could not create the offscreen render target");
        return result;
    }
    ViewportRenderer renderer;
    if (!renderer.ensure(rhi, target.pass(), 1)) {
        result.error = renderer.error();
        return result;
    }
    renderer.set_background(Eigen::Vector3f(static_cast<float>(background.redF()),
                                            static_cast<float>(background.greenF()),
                                            static_cast<float>(background.blueF())));
    renderer.set_scene(std::move(scene));

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) {
        result.error = QStringLiteral("could not begin an offscreen frame");
        return result;
    }
    renderer.render(cb, target.target(), camera, size);
    rhi->endOffscreenFrame();
    rhi->finish();

    result.stats = renderer.stats();
    result.image = target.read(rhi);
    result.ok = true;
    return result;
}

/// Distance between two colours, as the largest per-channel difference.
int channel_distance(const QColor& a, const QColor& b) {
    return std::max({std::abs(a.red() - b.red()), std::abs(a.green() - b.green()),
                     std::abs(a.blue() - b.blue())});
}

class ViewerRender : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = make_device();
        if (!device_.usable()) {
            GTEST_SKIP() << "no QRhi backend on this host: " << device_.reason.toStdString();
        }
    }

    Device device_;
};

}  // namespace

// A separate suite name from the fixture-based tests below: gtest forbids mixing TEST and TEST_F
// under one suite name. This one needs no device at all.
TEST(ViewerShaders, CompiledShadersCarryEveryBackendVariant) {
    // Backend-independent: this catches a broken qsb step (or a missing qt_add_shaders call)
    // even on a host where nothing can render.
    for (const char* name : {"mesh.vert.qsb", "mesh.frag.qsb", "points.vert.qsb",
                             "points.frag.qsb", "lines.vert.qsb", "lines.frag.qsb"}) {
        QFile file(QStringLiteral(":/alphashape/shaders/") + QLatin1String(name));
        ASSERT_TRUE(file.open(QIODevice::ReadOnly))
            << name << " is not in the resource bundle (:alphashape/shaders/)";
        const QByteArray bytes = file.readAll();
        ASSERT_FALSE(bytes.isEmpty()) << name << " is empty";
        const QShader shader = QShader::fromSerialized(bytes);
        ASSERT_TRUE(shader.isValid()) << name << " is not a serialized QShader";

        // Every backend pair qsb was asked for: SPIR-V (Vulkan), MSL (Metal), HLSL (D3D) and
        // GLSL (OpenGL). A missing one only fails on the backend that needs it at run time, so
        // it is checked here instead.
        const QShader::Stage expected_stage = QString::fromLatin1(name).endsWith(QLatin1String(".vert"))
                                                  ? QShader::VertexStage
                                                  : QShader::FragmentStage;
        if (shader.stage() != expected_stage) {
            continue;
        }
        bool saw_spirv = false;
        bool saw_msl = false;
        bool saw_hlsl = false;
        bool saw_glsl = false;
        for (const QShaderKey& key : shader.availableShaders()) {
            saw_spirv = saw_spirv || key.source() == QShader::SpirvShader;
            saw_msl = saw_msl || key.source() == QShader::MslShader;
            saw_hlsl = saw_hlsl || key.source() == QShader::HlslShader;
            saw_glsl = saw_glsl || key.source() == QShader::GlslShader;
        }
        EXPECT_TRUE(saw_spirv) << name << " has no SPIR-V variant";
        EXPECT_TRUE(saw_msl) << name << " has no Metal variant";
        EXPECT_TRUE(saw_hlsl) << name << " has no HLSL variant";
        EXPECT_TRUE(saw_glsl) << name << " has no GLSL variant";
    }
}

TEST_F(ViewerRender, PipelineAndBuffersBuildOnTheAvailableBackend) {
    OffscreenTarget target(device_.rhi.get(), QSize(96, 64));
    ASSERT_TRUE(target.ok());
    ViewportRenderer renderer;
    ASSERT_TRUE(renderer.ensure(device_.rhi.get(), target.pass(), 1))
        << renderer.error().toStdString();
    EXPECT_TRUE(renderer.ready());
    EXPECT_EQ(renderer.backend_name().toStdString(), device_.rhi->backendName());
    EXPECT_TRUE(renderer.error().isEmpty());

    // A second ensure() with the same device and pass is a no-op, not a rebuild.
    EXPECT_TRUE(renderer.ensure(device_.rhi.get(), target.pass(), 1));
}

TEST_F(ViewerRender, QuadSceneIsDrawnAndCoversTheCentrePixel) {
    CameraController camera;
    Scene scene = build_scene(make_quad_chunk(), scene_style_for(ThemeMode::Dark));
    ASSERT_TRUE(scene.has_bounds);
    camera.set_scene_bounds(scene.bounds_min, scene.bounds_max);
    camera.fit(1.5);

    const QColor background(10, 30, 50);
    const RenderResult result =
        render_once(device_.rhi.get(), std::move(scene), camera, QSize(120, 80), background);
    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.stats.mesh_triangles, 2U);
    EXPECT_EQ(result.stats.draws, 1);

    ASSERT_FALSE(result.image.isNull());
    if (device_.software) {
        GTEST_SKIP() << "the Null backend rasterises nothing (" << device_.reason.toStdString()
                     << "); only the recording path was exercised";
    }
    const QColor centre = result.image.pixelColor(result.image.width() / 2,
                                                  result.image.height() / 2);
    // The quad is mid-grey and lit, the background is dark blue: they cannot be confused.
    EXPECT_GT(channel_distance(centre, background), 40)
        << "centre pixel " << centre.name().toStdString() << " equals the clear colour";
}

TEST_F(ViewerRender, PointCloudCoversTheCentrePixel) {
    Chunk chunk;
    PointCloud& cloud = chunk.point_cloud();
    // A dense blob at the origin, with colours: the sprite pass, not the mesh pass, is what has
    // to put pixels there.
    for (int i = 0; i < 200; ++i) {
        cloud.points.push_back(Eigen::Vector3d::Zero());
    }
    cloud.colours.assign(cloud.points.size() * 3U, 240U);

    Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    CameraController camera;
    camera.set_scene_bounds(scene.bounds_min, scene.bounds_max);
    camera.fit(1.5);

    const QColor background(10, 30, 50);
    const RenderResult result =
        render_once(device_.rhi.get(), std::move(scene), camera, QSize(120, 80), background);
    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.stats.dense_points, 200U);
    if (device_.software) {
        GTEST_SKIP() << "the Null backend rasterises nothing";
    }
    const QColor centre = result.image.pixelColor(result.image.width() / 2,
                                                  result.image.height() / 2);
    EXPECT_GT(channel_distance(centre, background), 40)
        << "centre pixel " << centre.name().toStdString()
        << " equals the clear colour: the point sprite drew nothing";
}

TEST_F(ViewerRender, EmptySceneClearsToTheBackground) {
    CameraController camera;
    camera.set_scene_bounds(Eigen::Vector3d::Zero(), Eigen::Vector3d::Constant(1.0));
    camera.fit(1.5);
    const QColor background(10, 30, 50);
    const RenderResult result =
        render_once(device_.rhi.get(), Scene{}, camera, QSize(32, 32), background);
    ASSERT_TRUE(result.ok) << result.error.toStdString();
    EXPECT_EQ(result.stats.mesh_triangles, 0U);
    EXPECT_EQ(result.stats.line_segments, 0U);
    if (device_.software) {
        GTEST_SKIP() << "the Null backend rasterises nothing";
    }
    // Nothing was drawn: every pixel is the clear colour.
    EXPECT_LT(channel_distance(result.image.pixelColor(16, 16), background), 8);
}

TEST_F(ViewerRender, ResetSceneBetweenFramesIsSafe) {
    // The upload path is rebuilt between frames; this is the sequence a job that produced a
    // mesh (or removed one) causes.
    OffscreenTarget target(device_.rhi.get(), QSize(64, 48));
    ASSERT_TRUE(target.ok());
    ViewportRenderer renderer;
    ASSERT_TRUE(renderer.ensure(device_.rhi.get(), target.pass(), 1)) << renderer.error().toStdString();

    CameraController camera;
    camera.set_scene_bounds(Eigen::Vector3d::Zero(), Eigen::Vector3d::Constant(2.0));
    camera.fit(1.3);

    for (int frame = 0; frame < 3; ++frame) {
        Scene scene;
        if (frame != 1) {  // frame 1 draws nothing at all
            scene = build_scene(make_quad_chunk(), scene_style_for(ThemeMode::Dark));
        }
        renderer.set_scene(std::move(scene));
        QRhiCommandBuffer* cb = nullptr;
        ASSERT_EQ(device_.rhi->beginOffscreenFrame(&cb), QRhi::FrameOpSuccess);
        renderer.render(cb, target.target(), camera, QSize(64, 48));
        device_.rhi->endOffscreenFrame();
        EXPECT_EQ(renderer.stats().mesh_triangles, frame == 1 ? 0U : 2U) << "frame " << frame;
    }
    device_.rhi->finish();
}

TEST_F(ViewerRender, RendererReleaseIsSafeToRepeat) {
    OffscreenTarget target(device_.rhi.get(), QSize(32, 32));
    ASSERT_TRUE(target.ok());
    ViewportRenderer renderer;
    ASSERT_TRUE(renderer.ensure(device_.rhi.get(), target.pass(), 1)) << renderer.error().toStdString();
    renderer.release();
    EXPECT_FALSE(renderer.ready());
    renderer.release();
    // The renderer is reusable after release.
    EXPECT_TRUE(renderer.ensure(device_.rhi.get(), target.pass(), 1)) << renderer.error().toStdString();
}
