#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "common/core/chunk.hpp"

namespace alphashape {

namespace {

/// Append a message naming a container and the offending id.
void report_id(std::vector<std::string>& out, const char* container,
               std::int32_t id, const char* problem) {
    out.push_back(std::string(container) + ": id " + std::to_string(id) + " " +
                  problem);
}

/// Check that `items` is sorted ascending by id with no duplicates and every
/// id is below `next_id`. `name` is used in messages.
template <class T>
void check_sorted_unique(const std::vector<T>& items, std::int32_t next_id,
                         const char* name, std::vector<std::string>& out) {
    std::set<std::int32_t> seen;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const std::int32_t id = items[i].id;
        if (id >= next_id) {
            report_id(out, name, id, "is not below next_id");
        }
        if (i > 0 && items[i - 1].id >= id) {
            report_id(out, name, id, "is out of order or duplicated");
        }
        if (!seen.insert(id).second) {
            report_id(out, name, id, "appears more than once");
        }
    }
}

/// Check that the active id of an asset vector names an existing slot, and is
/// -1 exactly when the vector is empty.
template <class Slot>
void check_active(const std::vector<Slot>& items, std::int32_t active_id,
                  const char* name, std::vector<std::string>& out) {
    if (items.empty()) {
        if (active_id != -1) {
            out.push_back(std::string("active ") + name + " id is " +
                          std::to_string(active_id) + " but there are no " + name +
                          " assets");
        }
        return;
    }
    for (const auto& slot : items) {
        if (slot.id == active_id) {
            return;
        }
    }
    out.push_back(std::string("active ") + name + " id " + std::to_string(active_id) +
                  " does not name an existing " + name + " asset");
}

/// Check group members name existing objects of the group's kind.
void check_group_members(const std::vector<std::int32_t>& members,
                         const std::set<std::int32_t>& known, const char* name,
                         std::int32_t group_id, std::vector<std::string>& out) {
    for (const std::int32_t member : members) {
        if (known.find(member) == known.end()) {
            out.push_back(std::string(name) + " id " + std::to_string(group_id) +
                          " references missing member " + std::to_string(member));
        }
    }
}

}  // namespace

std::vector<std::string> Chunk::validate() const {
    std::vector<std::string> out;

    if (next_id_ < 0) {
        out.push_back("next_id is negative");
    }

    // -- ids: sorted, unique, below next_id --------------------------------
    check_sorted_unique(markers_, next_id_, "markers", out);
    check_sorted_unique(marker_groups_, next_id_, "marker_groups", out);
    check_sorted_unique(scalebars_, next_id_, "scalebars", out);
    check_sorted_unique(camera_groups_, next_id_, "camera_groups", out);
    check_sorted_unique(shapes_.features, next_id_, "shapes", out);
    check_sorted_unique(shapes_.groups, next_id_, "shape_groups", out);
    check_sorted_unique(trajectories_, next_id_, "trajectories", out);
    check_sorted_unique(tiled_models_, next_id_, "tiled_models", out);
    if (panorama_.has_value()) {
        check_sorted_unique(std::vector<PanoramaAsset>{*panorama_}, next_id_,
                            "panorama", out);
    }
    check_sorted_unique(point_clouds_, next_id_, "point_clouds", out);
    check_sorted_unique(models_, next_id_, "models", out);
    check_sorted_unique(depth_map_sets_, next_id_, "depth_maps_sets", out);
    check_sorted_unique(elevations_, next_id_, "elevations", out);
    check_sorted_unique(orthomosaics_, next_id_, "orthomosaics", out);

    // -- active ids name existing assets (or -1 iff the vector is empty) ----
    check_active(point_clouds_, active_point_cloud_id_, "point cloud", out);
    check_active(models_, active_model_id_, "model", out);
    check_active(depth_map_sets_, active_depth_maps_id_, "depth-map set", out);
    check_active(elevations_, active_elevation_id_, "elevation", out);
    check_active(orthomosaics_, active_orthomosaic_id_, "orthomosaic", out);

    // -- cameras and markers are the reference sets ------------------------
    std::set<std::int32_t> camera_ids;
    for (const auto& camera : cameras_) {
        camera_ids.insert(camera.id);
    }
    std::set<std::int32_t> marker_ids;
    for (const auto& marker : markers_) {
        marker_ids.insert(marker.id);
    }

    // -- masks: valid, keyed by an existing camera, unique per camera ------
    std::set<int> masked_cameras;
    for (const auto& mask : masks_) {
        if (!mask.is_valid()) {
            out.push_back("mask for camera " + std::to_string(mask.camera_id) +
                          " has inconsistent dimensions");
        }
        if (camera_ids.find(mask.camera_id) == camera_ids.end()) {
            out.push_back("mask references missing camera " +
                          std::to_string(mask.camera_id));
        }
        if (!masked_cameras.insert(mask.camera_id).second) {
            out.push_back("duplicate mask for camera " +
                          std::to_string(mask.camera_id));
        }
    }
    for (std::size_t i = 1; i < masks_.size(); ++i) {
        if (masks_[i - 1].camera_id >= masks_[i].camera_id) {
            out.push_back("masks are not sorted by camera id or are duplicated");
            break;
        }
    }

    // -- marker projections ------------------------------------------------
    for (const auto& marker : markers_) {
        for (std::size_t i = 0; i < marker.projections.size(); ++i) {
            const auto& projection = marker.projections[i];
            if (camera_ids.find(projection.camera_id) == camera_ids.end()) {
                out.push_back("marker " + std::to_string(marker.id) +
                              " projects into missing camera " +
                              std::to_string(projection.camera_id));
            }
            if (i > 0 && marker.projections[i - 1].camera_id >= projection.camera_id) {
                out.push_back("marker " + std::to_string(marker.id) +
                              " projections are out of order or duplicated");
            }
        }
    }

    // -- group members -----------------------------------------------------
    std::set<std::int32_t> shape_ids;
    for (const auto& shape : shapes_.features) {
        shape_ids.insert(shape.id);
    }
    for (const auto& group : marker_groups_) {
        check_group_members(group.members, marker_ids, "marker_group", group.id, out);
    }
    for (const auto& group : shapes_.groups) {
        check_group_members(group.members, shape_ids, "shape_group", group.id, out);
    }
    for (const auto& group : camera_groups_) {
        check_group_members(group.members, camera_ids, "camera_group", group.id, out);
    }

    // Marker group back-references must also resolve.
    std::set<std::int32_t> marker_group_ids;
    for (const auto& group : marker_groups_) {
        marker_group_ids.insert(group.id);
    }
    for (const auto& marker : markers_) {
        if (marker.group_id >= 0 && marker_group_ids.find(marker.group_id) ==
                                        marker_group_ids.end()) {
            out.push_back("marker " + std::to_string(marker.id) +
                          " references missing group " +
                          std::to_string(marker.group_id));
        }
    }

    // -- scalebars ---------------------------------------------------------
    for (const auto& bar : scalebars_) {
        for (const ScalebarEndpoint* const endpoint : {&bar.from, &bar.to}) {
            const std::set<std::int32_t>& known =
                endpoint->kind == ScalebarEndpoint::Kind::Marker ? marker_ids
                                                                : camera_ids;
            if (known.find(endpoint->id) == known.end()) {
                out.push_back("scalebar " + std::to_string(bar.id) +
                              " references missing " +
                              (endpoint->kind == ScalebarEndpoint::Kind::Marker
                                   ? "marker "
                                   : "camera ") +
                              std::to_string(endpoint->id));
            }
        }
        if (!std::isfinite(bar.reference_length_m) || bar.reference_length_m < 0.0) {
            out.push_back("scalebar " + std::to_string(bar.id) +
                          " has invalid reference length");
        }
    }

    // -- assets ------------------------------------------------------------
    if (panorama_.has_value() && !panorama_->data) {
        out.push_back("panorama has a null data pointer");
    }
    for (const auto& asset : tiled_models_) {
        if (!asset.data) {
            report_id(out, "tiled_model", asset.id, "has a null data pointer");
        }
    }
    for (const auto& asset : point_clouds_) {
        if (!asset.data) {
            report_id(out, "point_cloud", asset.id, "has a null data pointer");
        }
    }
    for (const auto& asset : models_) {
        if (!asset.data) {
            report_id(out, "model", asset.id, "has a null data pointer");
        }
    }
    for (const auto& asset : depth_map_sets_) {
        if (!asset.data) {
            report_id(out, "depth_maps_set", asset.id, "has a null data pointer");
        }
    }
    for (const auto& asset : elevations_) {
        if (!asset.data) {
            report_id(out, "elevation", asset.id, "has a null data pointer");
        }
    }
    for (const auto& asset : orthomosaics_) {
        if (!asset.data) {
            report_id(out, "orthomosaic", asset.id, "has a null data pointer");
        }
    }

    // -- active tiled model ------------------------------------------------
    if (tiled_models_.empty()) {
        if (active_tiled_model_id_ != -1) {
            out.push_back("active tiled model id is " +
                          std::to_string(active_tiled_model_id_) +
                          " but there are no tiled models");
        }
    } else {
        bool found = false;
        for (const auto& asset : tiled_models_) {
            if (asset.id == active_tiled_model_id_) {
                found = true;
                break;
            }
        }
        if (!found) {
            out.push_back("active tiled model id " +
                          std::to_string(active_tiled_model_id_) +
                          " does not name an existing tiled model");
        }
    }

    return out;
}

}  // namespace alphashape
