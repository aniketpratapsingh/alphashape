#include "common/core/chunk.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/core/frame.hpp"

namespace alphashape {

namespace {

/// splitmix64's finaliser: a bijection on 64 bits with good avalanche, used to
/// fold one more value into a running fingerprint.
std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// Fold one double in by its exact IEEE-754 bit pattern. The project writer
/// emits %.17g, which round-trips every finite double, so a pose read back from
/// a project hashes identically to the one it was written from — and any real
/// change of value changes the bits.
std::uint64_t fold_double(std::uint64_t h, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return mix64(h ^ bits);
}

std::uint64_t fold_int(std::uint64_t h, std::int64_t v) {
    return mix64(h ^ static_cast<std::uint64_t>(v));
}

/// "<Kind> <n>" with the smallest n >= 1 not already used as an exact label in
/// `existing`. Keeps labels stable when assets are added and removed.
template <class Slot>
std::string default_asset_label(const std::vector<Slot>& existing, const char* kind) {
    for (int n = 1;; ++n) {
        const std::string candidate = std::string(kind) + " " + std::to_string(n);
        bool taken = false;
        for (const auto& slot : existing) {
            if (slot.label == candidate) {
                taken = true;
                break;
            }
        }
        if (!taken) {
            return candidate;
        }
    }
}

/// Copy a slot vector with deep-cloned payloads. AssetSlot::data is a
/// shared_ptr, so the implicit copy would alias the payload; every slot gets
/// its own payload object here.
template <class Slot>
std::vector<Slot> clone_slots(const std::vector<Slot>& source) {
    std::vector<Slot> out = source;
    for (auto& slot : out) {
        if (slot.data) {
            slot.data = std::make_shared<typename Slot::value_type>(*slot.data);
        } else {
            slot.data = std::make_shared<typename Slot::value_type>();
        }
    }
    return out;
}

template <class Slot>
Slot* find_slot(std::vector<Slot>& slots, std::int32_t id) {
    for (auto& slot : slots) {
        if (slot.id == id) {
            return &slot;
        }
    }
    return nullptr;
}

template <class Slot>
const Slot* find_slot(const std::vector<Slot>& slots, std::int32_t id) {
    for (const auto& slot : slots) {
        if (slot.id == id) {
            return &slot;
        }
    }
    return nullptr;
}

/// Append `slot` keeping the vector sorted by id (ids are allocated
/// monotonically, so this is almost always a push_back).
template <class Slot>
Slot& append_slot(std::vector<Slot>& slots, Slot slot) {
    auto it = slots.begin();
    while (it != slots.end() && it->id < slot.id) {
        ++it;
    }
    return *slots.insert(it, std::move(slot));
}

}  // namespace

Chunk::Chunk(std::string label) : label_(std::move(label)) {}

Chunk::Chunk(const Chunk& other) { copy_from(other); }

Chunk& Chunk::operator=(const Chunk& other) {
    if (this != &other) {
        copy_from(other);
    }
    return *this;
}

void Chunk::copy_from(const Chunk& other) {
    id_ = other.id_;
    label_ = other.label_;
    sensors_ = other.sensors_;
    cameras_ = other.cameras_;
    frames_ = other.frames_;
    tie_points_ = other.tie_points_;

    point_clouds_ = clone_slots(other.point_clouds_);
    active_point_cloud_id_ = other.active_point_cloud_id_;
    models_ = clone_slots(other.models_);
    active_model_id_ = other.active_model_id_;
    depth_map_sets_ = clone_slots(other.depth_map_sets_);
    active_depth_maps_id_ = other.active_depth_maps_id_;
    elevations_ = clone_slots(other.elevations_);
    active_elevation_id_ = other.active_elevation_id_;
    orthomosaics_ = clone_slots(other.orthomosaics_);
    active_orthomosaic_id_ = other.active_orthomosaic_id_;

    next_id_ = other.next_id_;
    markers_ = other.markers_;
    marker_groups_ = other.marker_groups_;
    scalebars_ = other.scalebars_;
    camera_groups_ = other.camera_groups_;
    masks_ = other.masks_;
    shapes_ = other.shapes_;
    trajectories_ = other.trajectories_;
    panorama_ = other.panorama_;
    if (panorama_.has_value()) {
        panorama_->data = panorama_->data
                              ? std::make_shared<Image>(*other.panorama_->data)
                              : std::make_shared<Image>();
    }
    tiled_models_ = clone_slots(other.tiled_models_);
    active_tiled_model_id_ = other.active_tiled_model_id_;

    region_ = other.region_;
    transform_ = other.transform_;
    meta_ = other.meta_;
    crs_ = other.crs_;
}

// -- asset addition --------------------------------------------------------

PointCloudAsset& Chunk::add_point_cloud(std::string label) {
    PointCloudAsset slot;
    slot.id = allocate_id();
    slot.label = label.empty() ? default_asset_label(point_clouds_, "Point Cloud")
                               : std::move(label);
    active_point_cloud_id_ = slot.id;
    return append_slot(point_clouds_, std::move(slot));
}

ModelAsset& Chunk::add_model(std::string label) {
    ModelAsset slot;
    slot.id = allocate_id();
    slot.label = label.empty() ? default_asset_label(models_, "Model") : std::move(label);
    active_model_id_ = slot.id;
    return append_slot(models_, std::move(slot));
}

DepthMapsAsset& Chunk::add_depth_maps(std::string label) {
    DepthMapsAsset slot;
    slot.id = allocate_id();
    slot.label = label.empty() ? default_asset_label(depth_map_sets_, "Depth Maps")
                               : std::move(label);
    active_depth_maps_id_ = slot.id;
    return append_slot(depth_map_sets_, std::move(slot));
}

ElevationAsset& Chunk::add_elevation(std::string label) {
    ElevationAsset slot;
    slot.id = allocate_id();
    slot.label =
        label.empty() ? default_asset_label(elevations_, "Elevation") : std::move(label);
    active_elevation_id_ = slot.id;
    return append_slot(elevations_, std::move(slot));
}

OrthomosaicAsset& Chunk::add_orthomosaic(std::string label) {
    OrthomosaicAsset slot;
    slot.id = allocate_id();
    slot.label = label.empty() ? default_asset_label(orthomosaics_, "Orthomosaic")
                               : std::move(label);
    active_orthomosaic_id_ = slot.id;
    return append_slot(orthomosaics_, std::move(slot));
}

// -- legacy single-slot accessors -----------------------------------------
// Const overloads fall back to shared empty singletons when the vector is
// empty; non-const overloads lazily create (and activate) the first asset.

namespace {

const PointCloud& empty_point_cloud() {
    static const PointCloud empty;
    return empty;
}
const Mesh& empty_model() {
    static const Mesh empty;
    return empty;
}
const std::vector<DepthMap>& empty_depth_maps() {
    static const std::vector<DepthMap> empty;
    return empty;
}
const std::vector<Image>& empty_textures() {
    static const std::vector<Image> empty;
    return empty;
}
const Elevation& empty_elevation() {
    static const Elevation empty;
    return empty;
}
const Orthomosaic& empty_orthomosaic() {
    static const Orthomosaic empty;
    return empty;
}

/// The payload of `slot`, created on demand. The non-const accessors hand out a mutable
/// reference, so they must never dereference a cleared payload; a slot whose `data` was reset
/// (validate() reports it) gets a fresh, empty one the same way `Chunk::point_cloud() = ...`
/// creates its slot.
template <class Slot>
typename Slot::value_type& payload_of(Slot& slot) {
    if (!slot.data) {
        slot.data = std::make_shared<typename Slot::value_type>();
    }
    return *slot.data;
}

}  // namespace

std::uint64_t depth_map_fingerprint_seed() { return 0x243F6A8885A308D3ULL; }

std::uint64_t depth_map_fingerprint_fold_double(std::uint64_t h, double value) {
    return fold_double(h, value);
}

std::uint64_t depth_map_fingerprint_fold_int(std::uint64_t h, std::int64_t value) {
    return fold_int(h, value);
}

std::uint64_t depth_map_fingerprint_fold_string(std::uint64_t h, const std::string& text) {
    // Length first, then the bytes: without the length "ab"+"c" and "a"+"bc"
    // would collide, which for a path pair is a real (if unlikely) confusion.
    h = fold_int(h, static_cast<std::int64_t>(text.size()));
    for (const char c : text) {
        h = mix64(h ^ static_cast<std::uint64_t>(static_cast<unsigned char>(c)));
    }
    return h;
}

namespace {

void fold_calibration(std::uint64_t& h, const Calibration& cal) {
    h = fold_int(h, static_cast<std::int64_t>(cal.type));
    h = fold_int(h, cal.width);
    h = fold_int(h, cal.height);
    for (const double value : {cal.f, cal.cx, cal.cy, cal.b1, cal.b2, cal.k1, cal.k2, cal.k3,
                               cal.k4, cal.p1, cal.p2}) {
        h = fold_double(h, value);
    }
}

}  // namespace

std::uint64_t depth_map_fingerprint(const Camera& camera, const Sensor& sensor, int downscale,
                                    const DepthMapFingerprintContext* context) {
    std::uint64_t h = depth_map_fingerprint_seed();  // pi's digits: an arbitrary non-zero seed
    // The pose, row-major. A re-alignment changes it, so a map built before the
    // re-alignment stops matching.
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            h = fold_double(h, camera.pose(r, c));
        }
    }
    // The calibration the map was derived from, and the reduction factor: the
    // same pose with a different lens, or at a different working resolution, is
    // a different map.
    fold_calibration(h, sensor.calibration);
    h = fold_int(h, downscale);

    if (context != nullptr) {
        // The NEIGHBOURHOOD: a depth map is estimated against the neighbours, so
        // moving any one of them changes this map as much as moving the
        // reference does. Ids and poses are folded in sorted-by-id order so the
        // hash depends on the SET, not on select_views' tie ordering.
        std::vector<std::size_t> order(context->neighbour_ids.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return context->neighbour_ids[a] < context->neighbour_ids[b];
        });
        h = fold_int(h, context->max_neighbors);
        h = fold_int(h, static_cast<std::int64_t>(context->neighbour_ids.size()));
        for (const std::size_t i : order) {
            h = fold_int(h, context->neighbour_ids[i]);
            if (i < context->neighbour_poses.size()) {
                const Transform& pose = context->neighbour_poses[i];
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        h = fold_double(h, pose(r, c));
                    }
                }
            }
            if (i < context->neighbour_calibrations.size()) {
                fold_calibration(h, context->neighbour_calibrations[i]);
            }
        }
        h = fold_double(h, context->depth_min);
        h = fold_double(h, context->depth_max);
        // The pixels themselves: the resolved path, plus size/mtime when they
        // were stat-able (0 when not, which hashes consistently either way).
        h = depth_map_fingerprint_fold_string(h, context->image_path);
        h = fold_int(h, static_cast<std::int64_t>(context->image_size));
        h = fold_int(h, context->image_mtime);
        // The search configuration that produces the result.
        h = fold_int(h, context->window_radius);
        h = fold_int(h, context->window_step);
        h = fold_int(h, context->iterations);
        // Coarse-to-fine: only a pyramid run is a different map from a flat one.
        // A map built before the pyramid existed has no record here (0), and a
        // run that resolves to one level is the very same search, so neither
        // may invalidate a stored map.
        if (context->pyramid_levels > 1) {
            h = fold_int(h, context->pyramid_levels);
        }
        h = fold_int(h, static_cast<std::int64_t>(context->seed));
    }

    // mix64 is a bijection on 64 bits and the seed is non-zero, but a single
    // final mix keeps the guarantee local: fold_int() already mixed, so only the
    // impossible all-zero state needs excluding. Remap it rather than return it,
    // because 0 is the "unknown" sentinel.
    return h == 0 ? 1 : h;
}

std::vector<DepthMap>& Chunk::depth_maps() {
    if (depth_map_sets_.empty()) {
        add_depth_maps();
    }
    DepthMapsAsset* slot = find_slot(depth_map_sets_, active_depth_maps_id_);
    if (slot == nullptr) {
        slot = &depth_map_sets_.front();
    }
    return payload_of(*slot);
}

const std::vector<DepthMap>& Chunk::depth_maps() const {
    const DepthMapsAsset* slot = find_slot(depth_map_sets_, active_depth_maps_id_);
    // A slot whose payload was unloaded (data == nullptr) reads as empty, like a missing one.
    return slot == nullptr || !slot->data ? empty_depth_maps() : *slot->data;
}

PointCloud& Chunk::point_cloud() {
    if (point_clouds_.empty()) {
        add_point_cloud();
    }
    PointCloudAsset* slot = find_slot(point_clouds_, active_point_cloud_id_);
    if (slot == nullptr) {
        slot = &point_clouds_.front();
    }
    return payload_of(*slot);
}

const PointCloud& Chunk::point_cloud() const {
    const PointCloudAsset* slot = find_slot(point_clouds_, active_point_cloud_id_);
    return slot == nullptr || !slot->data ? empty_point_cloud() : *slot->data;
}

Mesh& Chunk::model() {
    if (models_.empty()) {
        add_model();
    }
    ModelAsset* slot = find_slot(models_, active_model_id_);
    if (slot == nullptr) {
        slot = &models_.front();
    }
    return payload_of(*slot);
}

const Mesh& Chunk::model() const {
    const ModelAsset* slot = find_slot(models_, active_model_id_);
    return slot == nullptr || !slot->data ? empty_model() : *slot->data;
}

std::vector<Image>& Chunk::model_textures() {
    if (models_.empty()) {
        add_model();
    }
    ModelAsset* slot = find_slot(models_, active_model_id_);
    if (slot == nullptr) {
        slot = &models_.front();
    }
    return slot->textures;
}

const std::vector<Image>& Chunk::model_textures() const {
    // The texture pages live in the slot, not in its mesh payload, so they stay readable when
    // the mesh was unloaded; only a missing slot reads as empty.
    const ModelAsset* slot = find_slot(models_, active_model_id_);
    return slot == nullptr ? empty_textures() : slot->textures;
}

Elevation& Chunk::elevation() {
    if (elevations_.empty()) {
        add_elevation();
    }
    ElevationAsset* slot = find_slot(elevations_, active_elevation_id_);
    if (slot == nullptr) {
        slot = &elevations_.front();
    }
    return payload_of(*slot);
}

const Elevation& Chunk::elevation() const {
    const ElevationAsset* slot = find_slot(elevations_, active_elevation_id_);
    return slot == nullptr || !slot->data ? empty_elevation() : *slot->data;
}

Orthomosaic& Chunk::orthomosaic() {
    if (orthomosaics_.empty()) {
        add_orthomosaic();
    }
    OrthomosaicAsset* slot = find_slot(orthomosaics_, active_orthomosaic_id_);
    if (slot == nullptr) {
        slot = &orthomosaics_.front();
    }
    return payload_of(*slot);
}

const Orthomosaic& Chunk::orthomosaic() const {
    const OrthomosaicAsset* slot = find_slot(orthomosaics_, active_orthomosaic_id_);
    return slot == nullptr || !slot->data ? empty_orthomosaic() : *slot->data;
}

KeypointSet* TiePoints::find_by_camera(std::int32_t camera_id) {
    for (auto& set : keypoint_sets) {
        if (set.camera_id == camera_id) {
            return &set;
        }
    }
    return nullptr;
}

const KeypointSet* TiePoints::find_by_camera(std::int32_t camera_id) const {
    for (const auto& set : keypoint_sets) {
        if (set.camera_id == camera_id) {
            return &set;
        }
    }
    return nullptr;
}

std::size_t TiePoints::total_keypoints() const {
    std::size_t total = 0;
    for (const auto& set : keypoint_sets) {
        total += set.keypoints.size();
    }
    return total;
}

bool Chunk::operator==(const Chunk& other) const {
    if (id_ != other.id_ || label_ != other.label_) {
        return false;
    }
    if (sensors_ != other.sensors_ || cameras_ != other.cameras_) {
        return false;
    }
    if (frames_.size() != other.frames_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < frames_.size(); ++i) {
        if (!(*frames_[i] == *other.frames_[i])) {
            return false;
        }
    }
    if (!(tie_points_ == other.tie_points_)) {
        return false;
    }
    if (!(point_clouds_ == other.point_clouds_) ||
        active_point_cloud_id_ != other.active_point_cloud_id_) {
        return false;
    }
    if (!(models_ == other.models_) || active_model_id_ != other.active_model_id_) {
        return false;
    }
    if (!(depth_map_sets_ == other.depth_map_sets_) ||
        active_depth_maps_id_ != other.active_depth_maps_id_) {
        return false;
    }
    if (!(elevations_ == other.elevations_) ||
        active_elevation_id_ != other.active_elevation_id_) {
        return false;
    }
    if (!(orthomosaics_ == other.orthomosaics_) ||
        active_orthomosaic_id_ != other.active_orthomosaic_id_) {
        return false;
    }
    if (next_id_ != other.next_id_ || markers_ != other.markers_ ||
        marker_groups_ != other.marker_groups_ || scalebars_ != other.scalebars_ ||
        camera_groups_ != other.camera_groups_ || !(masks_ == other.masks_) ||
        !(shapes_ == other.shapes_) ||
        !(trajectories_ == other.trajectories_)) {
        return false;
    }
    if (panorama_.has_value() != other.panorama_.has_value()) {
        return false;
    }
    if (panorama_.has_value() && !(*panorama_ == *other.panorama_)) {
        return false;
    }
    if (!(tiled_models_ == other.tiled_models_) ||
        active_tiled_model_id_ != other.active_tiled_model_id_) {
        return false;
    }
    if (!(region_ == other.region_) || !(meta_ == other.meta_) ||
        !(crs_ == other.crs_)) {
        return false;
    }
    return transform_ == other.transform_;
}

// -- coordinate system -----------------------------------------------------

crs::CoordinateSystem Chunk::coordinate_system() const {
#if defined(ALPHASHAPE_USE_PROJ)
    return static_cast<crs::CoordinateSystem>(crs_);
#else
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
#endif
}

void Chunk::set_coordinate_system(const crs::CoordinateSystem& cs) {
#if defined(ALPHASHAPE_USE_PROJ)
    crs_ = CRS(cs);
#else
    static_cast<void>(cs);
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
#endif
}

}  // namespace alphashape
