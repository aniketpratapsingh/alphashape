#pragma once

// ViewportRenderer: turns a Scene into QRhi draw calls.
//
// Split out of the widget so the render test can drive it with its own QRhi and render target
// (no QRhiWidget, no window, no platform integration). The widget owns one instance and calls
// ensure() from QRhiWidget::initialize() and render() from QRhiWidget::render().
//
// LIFECYCLE. QRhi resources belong to the QRhi that created them, pipelines are tied to a
// render pass descriptor and its sample count, and a resource the previous frame used must not
// be destroyed underneath it. The renderer therefore:
//   * creates every long-lived object (pipelines, bindings, sampler, the page texture, the unit
//     quad) in ensure(), i.e. from initialize(), and re-creates the whole set only when the QRhi
//     or the render pass actually changed;
//   * resizes the page texture and the vertex buffers in place (setPixelSize/setSize + create(),
//     exactly what QRhiWidget does for its own colour texture) when a new scene needs different
//     sizes, so no object a pipeline or the bindings point at is ever deleted mid-flight.
//
// One pipeline samples a texture unconditionally, so an untextured mesh binds a 1x1 white page
// rather than needing a second pipeline and a second shader for the untextured case.

#include <cstddef>
#include <cstdint>
#include <optional>

#include <QImage>
#include <QSize>
#include <QString>

#include <Eigen/Core>

#include "apps/gui/viewer/scene.hpp"

QT_BEGIN_NAMESPACE
class QRhi;
class QRhiBuffer;
class QRhiCommandBuffer;
class QRhiGraphicsPipeline;
class QRhiRenderPassDescriptor;
class QRhiRenderTarget;
class QRhiResourceUpdateBatch;
class QRhiSampler;
class QRhiShaderResourceBindings;
class QRhiTexture;
QT_END_NAMESPACE

namespace alphashape::gui {

class CameraController;

class ViewportRenderer {
public:
    /// What the last render() drew. The tests assert on it; the widget shows it in its tooltip.
    struct Stats {
        quint32 tie_points = 0;
        quint32 dense_points = 0;
        quint32 mesh_triangles = 0;
        quint32 line_segments = 0;
        int draws = 0;
        bool textured = false;
    };

    ViewportRenderer();
    ~ViewportRenderer();
    ViewportRenderer(const ViewportRenderer&) = delete;
    ViewportRenderer& operator=(const ViewportRenderer&) = delete;

    /// Replace the CPU scene. Buffers are (re)sized and uploaded on the next render().
    /// Takes ownership through a move, so rebuilding from the widget costs no copy: the caller
    /// hands its built Scene straight over.
    void set_scene(Scene&& scene);
    const Scene& scene() const { return scene_; }
    bool upload_pending() const { return upload_pending_; }

    /// Recolour the current scene in place and re-upload only the vertex streams `scope`
    /// names. The geometry (positions, normals, uvs), the index buffer and the texture page stay
    /// on the GPU: a selection change rewrites the frustum lines alone, a theme change the
    /// streams whose colours came from the theme. Nothing is rebuilt and nothing is copied.
    enum class RestyleScope : unsigned {
        None = 0U,
        Lines = 1U << 0U,        ///< grid, axes, bounding box, camera frustums
        Points = 1U << 1U,       ///< tie points and the dense cloud
        Mesh = 1U << 2U,         ///< the mesh's vertex colours
        All = Lines | Points | Mesh,
    };
    void restyle(const SceneStyle& style, std::optional<std::int32_t> highlight_camera,
                 RestyleScope scope);

    /// Create the pipelines and the fixed resources for `rhi` and the given render pass.
    /// Idempotent; re-creates the set when the QRhi, the render pass or the sample count
    /// changed. Returns false when something could not be created (error() says what), in which
    /// case render() draws nothing.
    bool ensure(QRhi* rhi, QRhiRenderPassDescriptor* render_pass, int sample_count);
    /// Release every QRhi resource. The widget calls this from releaseResources().
    void release();

    /// Record this frame: the clear plus one pass drawing the mesh, the point clouds and the
    /// line geometry. beginPass()/endPass() are done here, so call it with a frame recorded.
    void render(QRhiCommandBuffer* cb, QRhiRenderTarget* target, const CameraController& camera,
                const QSize& pixel_size);

    /// Clear colour of the next frame (the viewport background token of the theme).
    void set_background(const Eigen::Vector3f& colour) { background_ = colour; }
    const Eigen::Vector3f& background() const { return background_; }

    const Stats& stats() const { return stats_; }
    bool ready() const { return pipeline_mesh_ != nullptr && bindings_ != nullptr; }
    /// "Metal" / "Vulkan" / "D3D11" / "OpenGL", or "none" before ensure().
    QString backend_name() const;
    const QString& error() const { return error_; }

private:
    void destroy_resources();
    bool create_resources(QRhi* rhi, QRhiRenderPassDescriptor* render_pass, int sample_count);
    /// Create/resize the per-scene vertex/index buffers to hold `scene_`. Called between frames
    /// (set_scene) and again at upload time; a no-op when every size already matches.
    bool size_scene_buffers();
    /// Upload the current scene into buffers of the right size, plus the texture page.
    void upload_scene(QRhiResourceUpdateBatch* batch);
    /// Upload only the colours of the current scene (after restyle()).
    void upload_colours(QRhiResourceUpdateBatch* batch);

    QRhi* rhi_ = nullptr;
    QRhiRenderPassDescriptor* render_pass_ = nullptr;
    int sample_count_ = 1;

    // Fixed resources, created in ensure().
    QRhiBuffer* uniforms_ = nullptr;
    /// The aligned allocation size of uniforms_ (what ubufAligned(kUniformSize) returned).
    quint32 uniform_buffer_size_ = 0;
    QRhiSampler* sampler_ = nullptr;
    QRhiTexture* page_ = nullptr;
    QRhiBuffer* point_quad_ = nullptr;
    QRhiShaderResourceBindings* bindings_ = nullptr;
    QRhiGraphicsPipeline* pipeline_mesh_ = nullptr;
    QRhiGraphicsPipeline* pipeline_points_ = nullptr;
    QRhiGraphicsPipeline* pipeline_lines_ = nullptr;

    // Per-scene buffers, resized + uploaded in render().
    QRhiBuffer* mesh_vertices_ = nullptr;
    QRhiBuffer* mesh_indices_ = nullptr;
    QRhiBuffer* tie_points_ = nullptr;
    QRhiBuffer* dense_points_ = nullptr;
    QRhiBuffer* lines_ = nullptr;

    Scene scene_;
    /// The last uploaded page as an owned image. uploadTexture() reads it when the batch is
    /// submitted, after upload_scene() returned, so it must outlive that call and must not alias
    /// the scene's pixels (which a rebuild replaces).
    QImage page_image_;
    Eigen::Vector3f background_ = Eigen::Vector3f(0.11F, 0.14F, 0.16F);
    /// Upload the scene into the buffers on the next frame.
    bool upload_pending_ = true;
    /// Re-upload only these vertex colours on the next frame (a restyle).
    RestyleScope colours_pending_ = RestyleScope::None;
    /// The static point-sprite quad still has to be uploaded into point_quad_.
    bool upload_quad_ = true;
    Stats stats_;
    QString error_;
};

}  // namespace alphashape::gui
