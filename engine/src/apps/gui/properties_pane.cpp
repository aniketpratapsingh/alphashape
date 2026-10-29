#include "apps/gui/properties_pane.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include <QAbstractItemView>
#include <QHeaderView>
#include <QLabel>
#include <QStringList>
#include <QTableWidget>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"

namespace alphashape::gui {

namespace {

using Rows = QList<QPair<QString, QString>>;

QString qs(const std::string& text) { return QString::fromStdString(text); }
QString num(std::size_t n) { return QString::number(static_cast<qulonglong>(n)); }
QString num(double v) { return QString::number(v, 'g', 8); }
QString yes_no(bool v) { return v ? QObject::tr("yes") : QObject::tr("no"); }

QString vec3(const Eigen::Vector3d& v) {
    return QStringLiteral("(%1, %2, %3)").arg(v.x(), 0, 'g', 6).arg(v.y(), 0, 'g', 6).arg(v.z(), 0, 'g', 6);
}

QString size_text(int width, int height) {
    return QStringLiteral("%1 x %2").arg(width).arg(height);
}

void add(Rows& rows, const QString& name, const QString& value) {
    rows.append({name, value});
}

bool is_aligned(const Camera& camera) { return camera.pose != Transform::Identity(); }

QString crs_text(const CRS& crs) {
    if (crs.epsg != 0) {
        return QStringLiteral("EPSG:%1").arg(crs.epsg);
    }
    if (!crs.wkt.empty()) {
        return QObject::tr("Custom (WKT)");
    }
    return QObject::tr("Local coordinates");
}

QString camera_type_text(CameraType type) {
    return type == CameraType::Fisheye ? QObject::tr("Fisheye") : QObject::tr("Frame");
}

const Frame* frame_of(const Chunk& chunk, const Camera& camera) {
    for (const auto& frame : chunk.frames()) {
        if (frame && frame->camera_id() == camera.id) {
            return frame.get();
        }
    }
    return nullptr;
}

void describe_chunk(const Document& document, const Chunk& chunk, PropertySheet& sheet) {
    sheet.title = qs(chunk.label());
    Rows& r = sheet.rows;
    std::size_t aligned = 0;
    for (const Camera& camera : chunk.cameras()) {
        aligned += is_aligned(camera) ? 1U : 0U;
    }
    add(r, QObject::tr("Name"), qs(chunk.label()));
    add(r, QObject::tr("Active"), yes_no(chunk.id() == document.active_chunk_id()));
    add(r, QObject::tr("Cameras"), QObject::tr("%1 (%2 aligned)").arg(chunk.cameras().size()).arg(aligned));
    add(r, QObject::tr("Sensors"), num(chunk.sensors().size()));
    add(r, QObject::tr("Tie points"), num(chunk.tie_points().tracks.size()));
    add(r, QObject::tr("Point clouds"), num(chunk.point_clouds().size()));
    add(r, QObject::tr("Depth map sets"), num(chunk.depth_map_sets().size()));
    add(r, QObject::tr("Models"), num(chunk.models().size()));
    add(r, QObject::tr("Tiled models"), num(chunk.tiled_models().size()));
    add(r, QObject::tr("DEMs"), num(chunk.elevations().size()));
    add(r, QObject::tr("Orthomosaics"), num(chunk.orthomosaics().size()));
    add(r, QObject::tr("Markers"), num(chunk.markers().size()));
    add(r, QObject::tr("Scale bars"), num(chunk.scalebars().size()));
    add(r, QObject::tr("Shapes"), num(chunk.shapes().features.size()));
    add(r, QObject::tr("Coordinate system"), crs_text(chunk.crs()));
    add(r, QObject::tr("Region centre"), vec3(chunk.region().centre));
    add(r, QObject::tr("Region size"), vec3(chunk.region().size));
}

void describe_camera(const Chunk& chunk, const Camera& camera, PropertySheet& sheet) {
    sheet.title = qs(camera.label);
    Rows& r = sheet.rows;
    add(r, QObject::tr("Label"), qs(camera.label));
    const auto sensor_index = static_cast<std::size_t>(std::max(camera.sensor_id, 0));
    const bool has_sensor = camera.sensor_id >= 0 && sensor_index < chunk.sensors().size();
    add(r, QObject::tr("Sensor"), has_sensor ? qs(chunk.sensors()[sensor_index].label) : QObject::tr("(none)"));
    add(r, QObject::tr("Enabled"), yes_no(camera.enabled));
    add(r, QObject::tr("Aligned"), yes_no(is_aligned(camera)));
    const Frame* frame = frame_of(chunk, camera);
    add(r, QObject::tr("Image path"), frame != nullptr ? qs(frame->path()) : QObject::tr("(none)"));
    if (frame != nullptr && frame->width() > 0) {
        add(r, QObject::tr("Size"), size_text(frame->width(), frame->height()));
    } else if (has_sensor) {
        add(r, QObject::tr("Size"), size_text(chunk.sensors()[sensor_index].width, chunk.sensors()[sensor_index].height));
    }
    if (camera.reference.location) {
        add(r, QObject::tr("Reference (lon, lat, alt)"), vec3(*camera.reference.location));
    }
}

void describe_sensor(const Sensor& sensor, std::size_t index, PropertySheet& sheet) {
    sheet.title = sensor.label.empty() ? QObject::tr("Sensor %1").arg(index + 1) : qs(sensor.label);
    Rows& r = sheet.rows;
    add(r, QObject::tr("Label"), qs(sensor.label));
    add(r, QObject::tr("Type"), camera_type_text(sensor.type));
    add(r, QObject::tr("Resolution"), size_text(sensor.width, sensor.height));
    add(r, QObject::tr("Focal length (mm)"), sensor.focal_length_mm > 0 ? num(sensor.focal_length_mm) : QObject::tr("unknown"));
    add(r, QObject::tr("Pixel size (um)"), sensor.pixel_size_um > 0 ? num(sensor.pixel_size_um) : QObject::tr("unknown"));
    if (!sensor.make.empty() || !sensor.model.empty()) {
        add(r, QObject::tr("Make / model"), qs(sensor.make) + QStringLiteral(" / ") + qs(sensor.model));
    }
    if (!sensor.serial.empty()) {
        add(r, QObject::tr("Serial"), qs(sensor.serial));
    }
    const Calibration& c = sensor.calibration;
    const std::pair<const char*, double> params[] = {
        {"f (px)", c.f},   {"cx (px)", c.cx}, {"cy (px)", c.cy}, {"b1", c.b1}, {"b2", c.b2},
        {"k1", c.k1},      {"k2", c.k2},      {"k3", c.k3},      {"k4", c.k4}, {"p1", c.p1},
        {"p2", c.p2},
    };
    for (const auto& [name, value] : params) {
        add(r, QStringLiteral("Calibration %1").arg(QLatin1String(name)), num(value));
    }
    QStringList fixed;
    for (const std::string& name : sensor.fixed_params) {
        fixed << qs(name);
    }
    add(r, QObject::tr("Fixed parameters"), fixed.isEmpty() ? QObject::tr("(none)") : fixed.join(QStringLiteral(", ")));
}

template <class Slot>
void add_slot_common(Rows& r, const Slot& slot, std::int32_t active_id) {
    add(r, QObject::tr("Label"), qs(slot.label));
    add(r, QObject::tr("Active"), yes_no(slot.id == active_id));
    add(r, QObject::tr("Enabled"), yes_no(slot.enabled));
}

void describe_raster(Rows& r, const RasterGrid& grid, int epsg) {
    add(r, QObject::tr("Resolution"), size_text(grid.width, grid.height));
    add(r, QObject::tr("Pixel size"), QStringLiteral("%1 x %2").arg(grid.pixel_size_x, 0, 'g', 6).arg(grid.pixel_size_y, 0, 'g', 6));
    add(r, QObject::tr("Coordinate system"), epsg != 0 ? QStringLiteral("EPSG:%1").arg(epsg) : QObject::tr("Local coordinates"));
}

/// Describe a single focus item; false when it no longer exists.
bool describe_one(const Document& document, const NodeRef& ref, PropertySheet& sheet) {
    if (ref.kind == NodeKind::Document) {
        sheet.title = QObject::tr("Project");
        Rows& r = sheet.rows;
        add(r, QObject::tr("Path"), document.path().empty() ? QObject::tr("(unsaved)") : qs(document.path()));
        std::size_t chunks = 0;
        for (const auto& workspace : document.workspaces()) {
            chunks += workspace->chunks().size();
        }
        add(r, QObject::tr("Chunks"), num(chunks));
        add(r, QObject::tr("Active chunk"), num(static_cast<std::size_t>(std::max(document.active_chunk_id(), 0))));
        add(r, QObject::tr("Coordinate system"), crs_text(document.global_crs()));
        return true;
    }
    const Chunk* chunk = nullptr;
    for (const auto& workspace : document.workspaces()) {
        for (const auto& candidate : workspace->chunks()) {
            if (candidate->id() == ref.chunk_id) {
                chunk = candidate.get();
            }
        }
    }
    if (chunk == nullptr) {
        return false;
    }
    Rows& r = sheet.rows;
    const std::int32_t id = ref.item_id;
    switch (ref.kind) {
        case NodeKind::Chunk:
            describe_chunk(document, *chunk, sheet);
            return true;
        case NodeKind::Camera:
            for (const Camera& camera : chunk->cameras()) {
                if (camera.id == id) {
                    describe_camera(*chunk, camera, sheet);
                    return true;
                }
            }
            return false;
        case NodeKind::CameraGroup:
            for (const CameraGroup& group : chunk->camera_groups()) {
                if (group.id == id) {
                    sheet.title = qs(group.label);
                    add(r, QObject::tr("Label"), qs(group.label));
                    add(r, QObject::tr("Cameras"), num(group.members.size()));
                    if (group.type == CameraGroupType::Folder) {
                        add(r, QObject::tr("Folder"), qs(group.folder));
                    }
                    return true;
                }
            }
            return false;
        case NodeKind::Sensor:
            if (id >= 0 && static_cast<std::size_t>(id) < chunk->sensors().size()) {
                describe_sensor(chunk->sensors()[static_cast<std::size_t>(id)],
                                static_cast<std::size_t>(id), sheet);
                return true;
            }
            return false;
        case NodeKind::PointCloud:
            if (const auto* slot = find_asset(chunk->point_clouds(), id)) {
                sheet.title = qs(slot->label);
                add_slot_common(r, *slot, chunk->active_point_cloud_id());
                if (slot->data) {
                    const PointCloud& cloud = *slot->data;
                    add(r, QObject::tr("Points"), num(cloud.points.size()));
                    add(r, QObject::tr("Colours"), yes_no(!cloud.colours.empty()));
                    add(r, QObject::tr("Normals"), yes_no(!cloud.normals.empty()));
                    add(r, QObject::tr("Confidence"), yes_no(!cloud.confidence.empty()));
                } else {
                    add(r, QObject::tr("Points"), QObject::tr("(not loaded)"));
                }
                return true;
            }
            return false;
        case NodeKind::Model:
            if (const auto* slot = find_asset(chunk->models(), id)) {
                sheet.title = qs(slot->label);
                add_slot_common(r, *slot, chunk->active_model_id());
                if (slot->data) {
                    const Mesh& mesh = *slot->data;
                    add(r, QObject::tr("Vertices"), num(mesh.vertices.size()));
                    add(r, QObject::tr("Faces"), num(mesh.faces.size()));
                } else {
                    add(r, QObject::tr("Faces"), QObject::tr("(not loaded)"));
                }
                add(r, QObject::tr("Textured"), yes_no(!slot->textures.empty()));
                if (!slot->textures.empty()) {
                    add(r, QObject::tr("Texture pages"), num(slot->textures.size()));
                }
                add(r, QObject::tr("Vertex colours"),
                    slot->data ? yes_no(!slot->data->colours.empty()) : QObject::tr("(not loaded)"));
                return true;
            }
            return false;
        case NodeKind::DepthMapSet:
            if (const auto* slot = find_asset(chunk->depth_map_sets(), id)) {
                sheet.title = qs(slot->label);
                add_slot_common(r, *slot, chunk->active_depth_maps_id());
                if (slot->data) {
                    add(r, QObject::tr("Depth maps"), num(slot->data->size()));
                    if (!slot->data->empty()) {
                        const DepthMap& first = slot->data->front();
                        add(r, QObject::tr("Resolution"), size_text(first.width, first.height));
                        add(r, QObject::tr("Downscale"), QString::number(first.downscale));
                    }
                } else {
                    add(r, QObject::tr("Depth maps"), QObject::tr("(not loaded)"));
                }
                return true;
            }
            return false;
        case NodeKind::TiledModel:
            if (const auto* slot = find_asset(chunk->tiled_models(), id)) {
                sheet.title = qs(slot->label);
                add_slot_common(r, *slot, chunk->active_tiled_model_id());
                add(r, QObject::tr("Size (bytes)"),
                    slot->data ? num(slot->data->payload.size()) : QObject::tr("(not loaded)"));
                return true;
            }
            return false;
        case NodeKind::Elevation:
            if (const auto* slot = find_asset(chunk->elevations(), id)) {
                sheet.title = qs(slot->label);
                add_slot_common(r, *slot, chunk->active_elevation_id());
                if (slot->data) {
                    describe_raster(r, slot->data->raster.grid, slot->data->epsg());
                    add(r, QObject::tr("Source"), qs(slot->data->source));
                    add(r, QObject::tr("No-data value"), num(static_cast<double>(slot->data->nodata)));
                } else {
                    add(r, QObject::tr("Resolution"), QObject::tr("(not loaded)"));
                }
                return true;
            }
            return false;
        case NodeKind::Orthomosaic:
            if (const auto* slot = find_asset(chunk->orthomosaics(), id)) {
                sheet.title = qs(slot->label);
                add_slot_common(r, *slot, chunk->active_orthomosaic_id());
                if (slot->data) {
                    describe_raster(r, slot->data->raster.grid, slot->data->epsg());
                    add(r, QObject::tr("Surface"), qs(slot->data->surface));
                } else {
                    add(r, QObject::tr("Resolution"), QObject::tr("(not loaded)"));
                }
                return true;
            }
            return false;
        case NodeKind::Marker:
            for (const Marker& marker : chunk->markers()) {
                if (marker.id == id) {
                    sheet.title = qs(marker.label);
                    add(r, QObject::tr("Label"), qs(marker.label));
                    add(r, QObject::tr("Projections"), num(marker.projections.size()));
                    add(r, QObject::tr("Reference (lon, lat, alt)"),
                        marker.reference ? vec3(*marker.reference) : QObject::tr("(none)"));
                    add(r, QObject::tr("Position"),
                        marker.position ? vec3(*marker.position) : QObject::tr("(not estimated)"));
                    return true;
                }
            }
            return false;
        case NodeKind::ScaleBar:
            for (const Scalebar& bar : chunk->scalebars()) {
                if (bar.id == id) {
                    sheet.title = qs(bar.label);
                    add(r, QObject::tr("Label"), qs(bar.label));
                    add(r, QObject::tr("Reference length (m)"), num(bar.reference_length_m));
                    add(r, QObject::tr("Accuracy (m)"), num(bar.reference_accuracy_m));
                    add(r, QObject::tr("Estimated length (m)"), num(bar.estimated_length_m));
                    return true;
                }
            }
            return false;
        case NodeKind::TiePoints: {
            const TiePoints& tie = chunk->tie_points();
            sheet.title = QObject::tr("Tie points");
            add(r, QObject::tr("Tracks"), num(tie.tracks.size()));
            add(r, QObject::tr("Keypoints"), num(tie.total_keypoints()));
            add(r, QObject::tr("Matched pairs"), num(tie.matches.size()));
            return true;
        }
        case NodeKind::Cameras: {
            std::size_t aligned = 0;
            for (const Camera& camera : chunk->cameras()) {
                aligned += is_aligned(camera) ? 1U : 0U;
            }
            sheet.title = QObject::tr("Cameras");
            add(r, QObject::tr("Total"), num(chunk->cameras().size()));
            add(r, QObject::tr("Aligned"), num(aligned));
            add(r, QObject::tr("Groups"), num(chunk->camera_groups().size()));
            return true;
        }
        case NodeKind::Sensors:
            sheet.title = QObject::tr("Sensors");
            add(r, QObject::tr("Total"), num(chunk->sensors().size()));
            return true;
        case NodeKind::Markers:
            sheet.title = QObject::tr("Markers");
            add(r, QObject::tr("Total"), num(chunk->markers().size()));
            return true;
        case NodeKind::ScaleBars:
            sheet.title = QObject::tr("Scale bars");
            add(r, QObject::tr("Total"), num(chunk->scalebars().size()));
            return true;
        case NodeKind::Shapes:
            sheet.title = QObject::tr("Shapes");
            add(r, QObject::tr("Features"), num(chunk->shapes().features.size()));
            add(r, QObject::tr("Groups"), num(chunk->shapes().groups.size()));
            return true;
        case NodeKind::DepthMaps:
            sheet.title = QObject::tr("Depth maps");
            add(r, QObject::tr("Sets"), num(chunk->depth_map_sets().size()));
            return true;
        case NodeKind::PointClouds:
            sheet.title = QObject::tr("Point cloud");
            add(r, QObject::tr("Instances"), num(chunk->point_clouds().size()));
            return true;
        case NodeKind::Models:
            sheet.title = QObject::tr("Model");
            add(r, QObject::tr("Instances"), num(chunk->models().size()));
            return true;
        case NodeKind::TiledModels:
            sheet.title = QObject::tr("Tiled model");
            add(r, QObject::tr("Instances"), num(chunk->tiled_models().size()));
            return true;
        case NodeKind::Elevations:
            sheet.title = QObject::tr("DEM");
            add(r, QObject::tr("Instances"), num(chunk->elevations().size()));
            return true;
        case NodeKind::Orthomosaics:
            sheet.title = QObject::tr("Orthomosaic");
            add(r, QObject::tr("Instances"), num(chunk->orthomosaics().size()));
            return true;
        case NodeKind::Document:
            return false;
    }
    return false;
}

}  // namespace

PropertySheet describe(const Document& document, const QList<NodeRef>& items) {
    PropertySheet sheet;
    if (items.isEmpty()) {
        return sheet;
    }
    if (items.size() > 1) {
        // Several items: the sheet says what is selected and describes nothing else.
        sheet.title = QObject::tr("%1 items selected").arg(items.size());
        bool same = true;
        for (const NodeRef& item : items) {
            same = same && item.kind == items.front().kind;
        }
        if (same) {
            add(sheet.rows, QObject::tr("Selected"), num(static_cast<std::size_t>(items.size())));
        }
        return sheet;
    }
    PropertySheet one;
    if (describe_one(document, items.front(), one)) {
        return one;
    }
    return sheet;
}

PropertiesPane::PropertiesPane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent)
    : QWidget(parent), app_(app), selection_(selection) {
    setObjectName(QStringLiteral("pane.properties"));
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    title_ = new QLabel(this);
    title_->setObjectName(QStringLiteral("properties.title"));
    title_->setContentsMargins(8, 6, 8, 6);
    layout->addWidget(title_);

    hint_ = new QLabel(tr("Select an item in the Workspace to see its properties."), this);
    hint_->setObjectName(QStringLiteral("properties.hint"));
    hint_->setAlignment(Qt::AlignCenter);
    hint_->setWordWrap(true);
    layout->addWidget(hint_, 1);

    table_ = new QTableWidget(0, 2, this);
    table_->setObjectName(QStringLiteral("properties.table"));
    table_->setHorizontalHeaderLabels({tr("Property"), tr("Value")});
    table_->verticalHeader()->hide();
    table_->horizontalHeader()->setStretchLastSection(true);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setAlternatingRowColors(true);
    table_->setShowGrid(false);
    layout->addWidget(table_, 1);

    connect(&selection_, &WorkspaceSelection::changed, this, &PropertiesPane::rebuild);
    connect(&app_, &GuiApp::undo_changed, this, &PropertiesPane::rebuild);
    connect(&app_, &GuiApp::document_replaced, this, &PropertiesPane::rebuild);
    // Save As changes the Document "Path" row without touching the document's content.
    connect(&app_, &GuiApp::path_changed, this, &PropertiesPane::rebuild);
    // Tasks change the document without going through commands, so every finished job re-reads it.
    connect(&app_.jobs(), &JobManager::job_finished, this,
            [this](quint64, JobManager::State, const QString&) { rebuild(); });
    rebuild();
}

QString PropertiesPane::value(const QString& name) const {
    for (const auto& row : sheet_.rows) {
        if (row.first == name) {
            return row.second;
        }
    }
    return {};
}

// Concurrency rule (see GuiApp): never read the document while a job is active. A trigger that
// arrives meanwhile only sets stale_; the sheet keeps its last content until job_finished.
void PropertiesPane::rebuild() {
    if (app_.jobs().busy()) {
        stale_ = true;
        return;
    }
    stale_ = false;
    sheet_ = app_.read_document(
        [this](const Document& document) { return describe(document, selection_.items()); });
    const bool empty = sheet_.title.isEmpty() && sheet_.rows.isEmpty();
    title_->setText(sheet_.title);
    title_->setVisible(!empty);
    hint_->setVisible(empty);
    table_->setVisible(!empty);
    table_->setRowCount(static_cast<int>(sheet_.rows.size()));
    for (int i = 0; i < sheet_.rows.size(); ++i) {
        table_->setItem(i, 0, new QTableWidgetItem(sheet_.rows[i].first));
        table_->setItem(i, 1, new QTableWidgetItem(sheet_.rows[i].second));
    }
}

}  // namespace alphashape::gui
