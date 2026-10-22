#include "common/core/chunk_ops.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "common/core/frame.hpp"

namespace alphashape {
namespace {

/// id translation for one append. Camera ids, frame ids and the shared
/// chunk-local counter are three independent namespaces.
struct IdMaps {
    std::unordered_map<std::int32_t, std::int32_t> cameras;
    std::unordered_map<std::int32_t, std::int32_t> frames;
    std::unordered_map<std::int32_t, std::int32_t> local;
};

std::int32_t translate(const std::unordered_map<std::int32_t, std::int32_t>& map,
                       std::int32_t id, const char* what) {
    if (id < 0) {
        return id;  // "unset" survives a copy unchanged
    }
    const auto it = map.find(id);
    if (it == map.end()) {
        throw ValueError(std::string("chunk operations: ") + what + " " +
                         std::to_string(id) + " is not in the source chunk");
    }
    return it->second;
}

/// True when `id` names an object in `map` (used for soft references such as a
/// model's source asset, which may legitimately point outside a partial copy).
bool has_mapping(const std::unordered_map<std::int32_t, std::int32_t>& map,
                 std::int32_t id) {
    return id >= 0 && map.find(id) != map.end();
}

std::int32_t next_unused(const std::vector<Camera>& cameras) {
    std::int32_t next = 0;
    for (const Camera& camera : cameras) {
        next = std::max(next, camera.id + 1);
    }
    return next;
}

std::int32_t next_unused(const std::vector<std::shared_ptr<Frame>>& frames) {
    std::int32_t next = 0;
    for (const auto& frame : frames) {
        next = std::max(next, frame->id() + 1);
    }
    return next;
}

/// Deep copy of one asset slot: the payload is cloned, never shared.
template <class Slot>
Slot clone_slot(const Slot& slot) {
    Slot copy = slot;
    using Payload = typename Slot::value_type;
    copy.data = slot.data ? std::make_shared<Payload>(*slot.data)
                          : std::make_shared<Payload>();
    return copy;
}

/// Append `source` slots to `slots` with fresh ids. `active` is the target's
/// active id; when the target has no assets yet it takes the remapped source
/// active (so a copy keeps the same asset active as its source), or the first
/// appended slot when the source has no active asset. Returns the active id.
template <class Slot, class Allocate>
std::int32_t append_assets(std::vector<Slot>& slots, const std::vector<Slot>& source,
                           Allocate allocate, std::int32_t active,
                           std::int32_t source_active) {
    const bool first_kind = slots.empty() && !source.empty();
    std::int32_t remapped_source_active = -1;
    for (const Slot& slot : source) {
        Slot copy = clone_slot(slot);
        copy.id = allocate(slot.id);
        slots.push_back(std::move(copy));
        if (slot.id == source_active) {
            remapped_source_active = slots.back().id;
        }
    }
    if (!first_kind) {
        return active;
    }
    if (remapped_source_active >= 0) {
        return remapped_source_active;
    }
    return active != -1 ? active : slots.front().id;
}

/// Sensor identity for dedup: exact type, geometry, EXIF identity and
/// calibration. Label and fixed-parameter lists are presentation, not
/// identity, and are deliberately excluded (WP-11.6 risks: never merge two
/// distinct sensors).
bool same_sensor_identity(const Sensor& a, const Sensor& b) {
    return a.type == b.type && a.width == b.width && a.height == b.height &&
           a.pixel_size_um == b.pixel_size_um &&
           a.focal_length_mm == b.focal_length_mm && a.make == b.make &&
           a.model == b.model && a.serial == b.serial &&
           a.calibration == b.calibration;
}

using AllocateId = std::function<std::int32_t(std::int32_t)>;

/// Append every container of `source` to `target` (which may be empty),
/// rewiring references through `maps`. Region, transform, CRS and metadata are
/// the caller's business: an append into a populated chunk must not touch them.
void append_containers(Chunk& target, const Chunk& source, const ChunkCopyOptions& options,
                       IdMaps& maps) {
    if (&target == &source) {
        throw ValueError("chunk operations: cannot append a chunk to itself");
    }

    // -- sensors (deduplicated on exact identity) --------------------------
    std::vector<std::int32_t> sensor_map(source.sensors().size(), -1);
    for (std::size_t i = 0; i < source.sensors().size(); ++i) {
        const Sensor& sensor = source.sensors()[i];
        bool found = false;
        for (std::size_t j = 0; j < target.sensors().size(); ++j) {
            if (same_sensor_identity(target.sensors()[j], sensor)) {
                sensor_map[i] = static_cast<std::int32_t>(j);
                found = true;
                break;
            }
        }
        if (!found) {
            target.sensors().push_back(sensor);
            sensor_map[i] = static_cast<std::int32_t>(target.sensors().size() - 1);
        }
    }

    // -- cameras and frames ------------------------------------------------
    std::int32_t next_camera = next_unused(target.cameras());
    for (const Camera& camera : source.cameras()) {
        // A chunk with no sensors at all (an import that never resolved EXIF,
        // or a bare test fixture) has nothing to translate; sensor_id is
        // carried verbatim, exactly as a Chunk copy would. With sensors
        // present a dangling reference is a real defect and fails loudly.
        if (!source.sensors().empty() &&
            (camera.sensor_id < 0 ||
             static_cast<std::size_t>(camera.sensor_id) >= source.sensors().size())) {
            throw ValueError("chunk operations: camera " + std::to_string(camera.id) +
                             " references sensor " + std::to_string(camera.sensor_id) +
                             ", which the source chunk does not have");
        }
        if (!maps.cameras.emplace(camera.id, next_camera).second) {
            throw ValueError("chunk operations: duplicate camera id " +
                             std::to_string(camera.id) + " in the source chunk");
        }
        ++next_camera;
    }
    std::int32_t next_frame = next_unused(target.frames());
    for (const auto& frame : source.frames()) {
        if (!maps.frames.emplace(frame->id(), next_frame).second) {
            throw ValueError("chunk operations: duplicate frame id " +
                             std::to_string(frame->id()) + " in the source chunk");
        }
        ++next_frame;
    }
    for (const Camera& source_camera : source.cameras()) {
        Camera camera = source_camera;
        camera.id = translate(maps.cameras, camera.id, "camera");
        if (!source.sensors().empty()) {
            camera.sensor_id =
                sensor_map[static_cast<std::size_t>(source_camera.sensor_id)];
        }
        target.cameras().push_back(std::move(camera));
    }
    for (const auto& source_frame : source.frames()) {
        auto frame = std::make_shared<Frame>(*source_frame);
        frame->set_id(translate(maps.frames, frame->id(), "frame"));
        frame->set_camera_id(
            translate(maps.cameras, frame->camera_id(), "frame camera"));
        target.frames().push_back(std::move(frame));
    }

    // -- the shared chunk-local namespace ----------------------------------
    const AllocateId allocate = [&target, &maps](std::int32_t source_id) {
        const std::int32_t id = target.allocate_id();
        if (!maps.local.emplace(source_id, id).second) {
            throw ValueError("chunk operations: duplicate object id " +
                             std::to_string(source_id) + " in the source chunk");
        }
        return id;
    };

    std::vector<Marker> markers = source.markers();
    for (Marker& marker : markers) {
        marker.id = allocate(marker.id);
    }
    std::vector<MarkerGroup> marker_groups = source.marker_groups();
    for (MarkerGroup& group : marker_groups) {
        group.id = allocate(group.id);
        for (std::int32_t& member : group.members) {
            member = translate(maps.local, member, "marker group member");
        }
        std::sort(group.members.begin(), group.members.end());
    }
    for (Marker& marker : markers) {
        marker.group_id = translate(maps.local, marker.group_id, "marker group");
        for (MarkerProjection& projection : marker.projections) {
            projection.camera_id =
                translate(maps.cameras, projection.camera_id, "marker projection camera");
        }
        std::sort(marker.projections.begin(), marker.projections.end(),
                  [](const MarkerProjection& a, const MarkerProjection& b) {
                      return a.camera_id < b.camera_id;
                  });
    }

    std::vector<Scalebar> scalebars = source.scalebars();
    for (Scalebar& bar : scalebars) {
        bar.id = allocate(bar.id);
    }
    for (Scalebar& bar : scalebars) {
        for (ScalebarEndpoint* const endpoint : {&bar.from, &bar.to}) {
            endpoint->id =
                endpoint->kind == ScalebarEndpoint::Kind::Marker
                    ? translate(maps.local, endpoint->id, "scalebar marker")
                    : translate(maps.cameras, endpoint->id, "scalebar camera");
        }
    }

    std::vector<CameraGroup> camera_groups = source.camera_groups();
    for (CameraGroup& group : camera_groups) {
        group.id = allocate(group.id);
        for (std::int32_t& member : group.members) {
            member = translate(maps.cameras, member, "camera group member");
        }
    }

    Shapes shapes = source.shapes();
    for (Shape& shape : shapes.features) {
        shape.id = allocate(shape.id);
    }
    for (ShapeGroup& group : shapes.groups) {
        group.id = allocate(group.id);
        for (std::int32_t& member : group.members) {
            member = translate(maps.local, member, "shape group member");
        }
    }

    std::vector<Trajectory> trajectories = source.trajectories();
    for (Trajectory& trajectory : trajectories) {
        trajectory.id = allocate(trajectory.id);
    }

    std::vector<Mask> masks = source.masks();
    for (Mask& mask : masks) {
        mask.camera_id = translate(maps.cameras, mask.camera_id, "mask camera");
    }
    std::sort(masks.begin(), masks.end(), [](const Mask& a, const Mask& b) {
        return a.camera_id < b.camera_id;
    });

    // Ids only ever grow, and every namespace continues after the target's
    // maximum, so appending to the end keeps each container sorted.
    const auto append_ids = [](auto& destination, auto& added) {
        destination.insert(destination.end(),
                           std::make_move_iterator(added.begin()),
                           std::make_move_iterator(added.end()));
    };
    append_ids(target.markers(), markers);
    append_ids(target.marker_groups(), marker_groups);
    append_ids(target.scalebars(), scalebars);
    append_ids(target.camera_groups(), camera_groups);
    append_ids(target.trajectories(), trajectories);
    append_ids(target.shapes().features, shapes.features);
    append_ids(target.shapes().groups, shapes.groups);
    append_ids(target.masks(), masks);

    // -- assets ------------------------------------------------------------
    // Asset slots take their ids from the shared counter, so a model's
    // source_id can only be rewired after every slot id is known: the loop
    // below records the new ids first and rewires at the very end.
    std::vector<std::int64_t> model_sources;
    if (options.copy_assets) {
        target.set_active_point_cloud_id(
            append_assets(target.point_clouds(), source.point_clouds(), allocate,
                          target.active_point_cloud_id(),
                          source.active_point_cloud_id()));
        target.set_active_model_id(
            append_assets(target.models(), source.models(), allocate,
                          target.active_model_id(), source.active_model_id()));
        target.set_active_elevation_id(
            append_assets(target.elevations(), source.elevations(), allocate,
                          target.active_elevation_id(),
                          source.active_elevation_id()));
        target.set_active_orthomosaic_id(
            append_assets(target.orthomosaics(), source.orthomosaics(), allocate,
                          target.active_orthomosaic_id(),
                          source.active_orthomosaic_id()));
        target.set_active_tiled_model_id(
            append_assets(target.tiled_models(), source.tiled_models(), allocate,
                          target.active_tiled_model_id(),
                          source.active_tiled_model_id()));

        const std::size_t first_new_model =
            target.models().size() - source.models().size();
        for (std::size_t i = 0; i < source.models().size(); ++i) {
            model_sources.push_back(target.models()[first_new_model + i].source_id);
        }

        // One panorama slot exists; an append into a chunk that already has one
        // keeps the existing image rather than silently replacing it.
        if (source.panorama().has_value() && !target.panorama().has_value()) {
            PanoramaAsset panorama = clone_slot(*source.panorama());
            panorama.id = allocate(source.panorama()->id);
            target.panorama() = std::move(panorama);
        }
    }

    if (options.copy_depth_maps && !source.depth_map_sets().empty()) {
        const bool first_kind = target.depth_map_sets().empty();
        std::int32_t active = target.active_depth_maps_id();
        for (const DepthMapsAsset& source_set : source.depth_map_sets()) {
            DepthMapsAsset set = clone_slot(source_set);
            set.id = allocate(source_set.id);
            if (set.data != nullptr) {
                for (DepthMap& map : *set.data) {
                    map.camera_id =
                        translate(maps.cameras, map.camera_id, "depth map camera");
                }
            }
            if (first_kind && source_set.id == source.active_depth_maps_id()) {
                active = set.id;
            }
            target.depth_map_sets().push_back(std::move(set));
        }
        if (first_kind && active == -1) {
            active = target.depth_map_sets().front().id;
        }
        target.set_active_depth_maps_id(active);
    }

    // A model's source_id links it to the point cloud or depth-map set it was
    // built from. Rewire it now that every asset id in every kind is known; a
    // partial copy (assets or depth maps switched off) drops the link.
    if (!model_sources.empty()) {
        const std::size_t first_new_model =
            target.models().size() - model_sources.size();
        for (std::size_t i = 0; i < model_sources.size(); ++i) {
            const std::int64_t old_source = model_sources[i];
            ModelAsset& model = target.models()[first_new_model + i];
            if (old_source < 0 ||
                !has_mapping(maps.local, static_cast<std::int32_t>(old_source))) {
                model.source_id = -1;
            } else {
                model.source_id = static_cast<std::int64_t>(
                    maps.local.at(static_cast<std::int32_t>(old_source)));
            }
        }
    }

    // -- tie points ---------------------------------------------------------
    if (options.copy_tie_points) {
        TiePoints tie_points = source.tie_points();
        for (KeypointSet& set : tie_points.keypoint_sets) {
            set.camera_id =
                translate(maps.cameras, set.camera_id, "keypoint set camera");
        }
        for (PairMatches& pair : tie_points.matches) {
            pair.camera_a = translate(maps.cameras, pair.camera_a, "match camera");
            pair.camera_b = translate(maps.cameras, pair.camera_b, "match camera");
        }
        std::sort(tie_points.matches.begin(), tie_points.matches.end(),
                  [](const PairMatches& a, const PairMatches& b) {
                      return a.camera_a != b.camera_a ? a.camera_a < b.camera_a
                                                      : a.camera_b < b.camera_b;
                  });
        for (Track& track : tie_points.tracks) {
            for (Observation& observation : track.observations) {
                observation.camera_id =
                    translate(maps.cameras, observation.camera_id, "track camera");
            }
            std::sort(track.observations.begin(), track.observations.end(),
                      [](const Observation& a, const Observation& b) {
                          return a.camera_id < b.camera_id;
                      });
        }
        append_ids(target.tie_points().keypoint_sets, tie_points.keypoint_sets);
        append_ids(target.tie_points().matches, tie_points.matches);
        append_ids(target.tie_points().tracks, tie_points.tracks);
    }
}

}  // namespace

Chunk duplicate_chunk(const Chunk& source, std::int32_t chunk_id, std::string label,
                      const ChunkCopyOptions& options) {
    Chunk copy(std::move(label));
    copy.set_id(chunk_id);
    IdMaps maps;
    append_containers(copy, source, options, maps);
    copy.region() = source.region();
    copy.transform() = source.transform();
    copy.meta() = source.meta();
    copy.crs() = source.crs();
    return copy;
}

void append_chunk(Chunk& target, const Chunk& source, const ChunkCopyOptions& options) {
    IdMaps maps;
    append_containers(target, source, options, maps);
}

bool crs_compatible(const Chunk& a, const Chunk& b) {
    // "Shares a CRS, or one of them is local" (plan §4.8).
    if (!a.crs().is_set() || !b.crs().is_set()) {
        return true;
    }
    return a.crs() == b.crs();
}

Chunk merge_chunks(const std::vector<const Chunk*>& sources, std::int32_t chunk_id,
                   std::string label) {
    if (sources.empty()) {
        throw ValueError("merge_chunks: need at least one chunk");
    }
    for (const Chunk* source : sources) {
        if (source == nullptr) {
            throw ValueError("merge_chunks: a source chunk is null");
        }
    }
    for (std::size_t i = 1; i < sources.size(); ++i) {
        if (!crs_compatible(*sources[0], *sources[i])) {
            throw ValueError(
                "merge_chunks: chunk " + std::to_string(sources[i]->id()) +
                " has CRS epsg " + std::to_string(sources[i]->crs().epsg) +
                ", which does not match chunk " + std::to_string(sources[0]->id()) +
                " (epsg " + std::to_string(sources[0]->crs().epsg) + ")");
        }
    }

    Chunk merged(std::move(label));
    merged.set_id(chunk_id);
    merged.crs() = sources[0]->crs();
    merged.transform() = sources[0]->transform();
    merged.meta() = sources[0]->meta();

    // Bounding box of every source region, so the merged chunk's region covers
    // all of its content.
    Eigen::Vector3d low = sources[0]->region().centre - 0.5 * sources[0]->region().size;
    Eigen::Vector3d high = sources[0]->region().centre + 0.5 * sources[0]->region().size;
    for (std::size_t i = 1; i < sources.size(); ++i) {
        const Region& region = sources[i]->region();
        low = low.cwiseMin(region.centre - 0.5 * region.size);
        high = high.cwiseMax(region.centre + 0.5 * region.size);
    }
    merged.region().centre = 0.5 * (low + high);
    merged.region().size = high - low;

    for (const Chunk* source : sources) {
        IdMaps maps;
        append_containers(merged, *source, ChunkCopyOptions{}, maps);
    }
    return merged;
}

}  // namespace alphashape
