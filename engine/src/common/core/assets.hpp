#pragma once

// Asset containers (WP-11.1, M11 §4.1): the multi-instance form of every
// derived product. A chunk may hold any number of point clouds, depth-map
// sets, elevations, orthomosaics, tiled models, panoramas and models, each
// with its own label, group and free-form metadata.
//
// NOTE ON INCLUDES: this header deliberately does not include
// `common/core/chunk.hpp` (chunk.hpp includes this one, and DepthMap lives
// there). `DepthMap` is forward-declared, which is enough to declare the
// `DepthMapsAsset` alias; include chunk.hpp before instantiating that alias.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/core/types.hpp"
#include "image/image.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape {

struct DepthMap;    // defined in common/core/chunk.hpp
struct Elevation;   // defined in common/core/chunk.hpp
struct Orthomosaic; // defined in common/core/chunk.hpp

namespace asset_detail {

/// Image carries no operator==; compare geometry and pixel bytes exactly.
bool images_equal(const Image& a, const Image& b);

/// Deep value comparison used by AssetSlot. The generic case defers to the
/// type's own ==.
template <class T>
bool value_equal(const T& a, const T& b) {
    return a == b;
}

inline bool value_equal(const Image& a, const Image& b) {
    return images_equal(a, b);
}

}  // namespace asset_detail

/// One asset instance. `data` is a shared pointer so a slot can be copied
/// cheaply through the undo stack; it is never null (the constructors and the
/// default member initialiser guarantee it).
///
/// Equality is by value: id, label, group, enabled flag, metadata and the
/// pointed-to object compared deeply — never the pointer identity.
template <class T>
struct AssetSlot {
    using value_type = T;

    std::int32_t id = -1;
    std::string label;
    std::int32_t group_id = -1;
    bool enabled = true;
    Meta meta;
    std::shared_ptr<T> data = std::make_shared<T>();

    bool operator==(const AssetSlot& other) const {
        if (id != other.id || label != other.label ||
            group_id != other.group_id || enabled != other.enabled) {
            return false;
        }
        if (!(meta == other.meta)) {
            return false;
        }
        if (!data || !other.data) {
            return data == other.data;
        }
        return asset_detail::value_equal(*data, *other.data);
    }

    bool operator!=(const AssetSlot& other) const { return !(*this == other); }
};

using PointCloudAsset = AssetSlot<PointCloud>;
using DepthMapsAsset = AssetSlot<std::vector<DepthMap>>;
using ElevationAsset = AssetSlot<Elevation>;
using OrthomosaicAsset = AssetSlot<Orthomosaic>;

/// Find an asset by id in one of a chunk's asset vectors; nullptr when no slot
/// carries that id. `id` -1 never matches, which is what makes "active id" work
/// as -1 for an empty container.
template <class Slot>
Slot* find_asset(std::vector<Slot>& slots, std::int32_t id) {
    for (auto& slot : slots) {
        if (slot.id == id) {
            return &slot;
        }
    }
    return nullptr;
}

template <class Slot>
const Slot* find_asset(const std::vector<Slot>& slots, std::int32_t id) {
    for (const auto& slot : slots) {
        if (slot.id == id) {
            return &slot;
        }
    }
    return nullptr;
}

/// Opaque tiled-model payload. The builder and the SLPK/3D-tiles writer land
/// in milestone 21; until then the asset carries bytes through unchanged.
struct TiledModel {
    std::vector<std::uint8_t> payload;

    bool operator==(const TiledModel&) const = default;
};

using TiledModelAsset = AssetSlot<TiledModel>;
using PanoramaAsset = AssetSlot<Image>;

/// A textured model. Extends the generic slot with the texture pages and the
/// id of the asset the mesh was built from (a depth-map set or a point cloud),
/// which is what Metashape's model-source link records.
struct ModelAsset : AssetSlot<Mesh> {
    std::vector<Image> textures;
    std::int64_t source_id = -1;

    bool operator==(const ModelAsset& other) const {
        if (!AssetSlot<Mesh>::operator==(other)) {
            return false;
        }
        if (source_id != other.source_id || textures.size() != other.textures.size()) {
            return false;
        }
        for (std::size_t i = 0; i < textures.size(); ++i) {
            if (!asset_detail::images_equal(textures[i], other.textures[i])) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const ModelAsset& other) const { return !(*this == other); }
};

}  // namespace alphashape
