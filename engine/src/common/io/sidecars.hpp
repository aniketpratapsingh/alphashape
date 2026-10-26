#pragma once

// Magic-tagged binary sidecars stored beside doc.xml inside project.zip.
//
// Every sidecar is self-describing: a four-byte magic carrying its own revision
// letter, then a little-endian payload. The decoder validates the magic, every
// length against the bytes that remain, and any cross-field invariant, so a
// corrupt or hostile entry always raises ProjectIOError rather than producing a
// short or wild object.
//
// The magics are version independent: a v1 and a v2 document point at the same
// encodings, only the entry path differs (v1 uses the legacy flat names, v2
// uses chunks/<chunk_id>/...). New in M11:
//
//   "AMK1"  per-camera masks, u8 bitmap run-length encoded
//   "ASH1"  shapes (features + groups), M11 §4.6
//   "ATJ1"  one trajectory (timed poses)
//   "AFR1"  chunk frames
//
// The validators at the bottom are the single source of truth for the container
// invariants: the writers call them before emitting a byte and the readers call
// them after assembling a container, so anything we write is readable and
// anything we read is something we could have written.

#include <cstdint>
#include <string>
#include <vector>

#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/core/mask.hpp"
#include "common/core/shapes.hpp"
#include "common/core/trajectory.hpp"
#include "common/core/types.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape::docxml::sidecar {

// -- little-endian primitives -----------------------------------------------

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v);
void put_i32(std::vector<std::uint8_t>& out, std::int32_t v);
void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v);
void put_f32(std::vector<std::uint8_t>& out, float f);
void put_f64(std::vector<std::uint8_t>& out, double d);

/// Append a length-prefixed UTF-8 string (u32 byte count, then the bytes).
void put_string(std::vector<std::uint8_t>& out, const std::string& s);

/// A bounds-checked cursor over an untrusted sidecar. Every read checks the
/// bytes that remain; every count is compared with the bytes it implies in
/// 64-bit arithmetic *before* the caller allocates.
class ByteReader {
public:
    ByteReader(const std::vector<std::uint8_t>& in, std::size_t start, const char* what)
        : in_(in), at_(start), what_(what) {}

    std::uint32_t u32();
    std::int32_t i32();
    std::uint8_t u8();
    std::uint64_t u64();
    float f32();
    double f64();
    std::string string();

    /// A u32 count whose items each occupy `item_bytes`, rejected up front when
    /// `count * item_bytes` does not fit in the remaining bytes.
    std::uint32_t count(std::uint64_t item_bytes);

    void f32s(std::uint64_t n, std::vector<float>& out);
    void f64s(std::uint64_t n, std::vector<double>& out);
    void u8s(std::uint64_t n, std::vector<std::uint8_t>& out);

    bool at_end() const { return at_ == in_.size(); }
    std::size_t remaining() const { return in_.size() - at_; }

private:
    const std::vector<std::uint8_t>& in_;
    std::size_t at_;
    const char* what_;
};

// -- magics and entry-name prefixes -----------------------------------------

constexpr char kKptMagic[4] = {'A', 'K', 'P', '1'};
constexpr char kMatchesMagic[4] = {'A', 'M', 'T', '1'};
constexpr char kTracksMagic[4] = {'A', 'T', 'R', '2'};
constexpr char kDepthMapsMagic[4] = {'A', 'D', 'M', '1'};   // legacy, read-only
/// Depth maps with the per-map alignment fingerprint. Doc.xml v1 and v2 both
/// point at the same encoding; "ADM1" is still read, with every fingerprint
/// reported as 0 (unknown), so an old project's maps are never reused.
constexpr char kDepthMapsMagic2[4] = {'A', 'D', 'M', '2'};  // current
constexpr char kPointCloudMagic[4] = {'A', 'P', 'C', '1'};
constexpr char kMeshMagic[4] = {'A', 'M', 'S', '1'};   // legacy, read-only
constexpr char kMeshMagic2[4] = {'A', 'M', 'S', '2'};  // current
constexpr char kOrthoLabelsMagic[4] = {'A', 'O', 'L', '1'};
constexpr char kMasksMagic[4] = {'A', 'M', 'K', '1'};
constexpr char kShapesMagic[4] = {'A', 'S', 'H', '1'};
constexpr char kTrajectoryMagic[4] = {'A', 'T', 'J', '1'};
constexpr char kFramesMagic[4] = {'A', 'F', 'R', '1'};
/// Tiled-model payloads are opaque until M21 builds them; the entry is still
/// magic-tagged so every binary sidecar in the archive is self-describing.
constexpr char kTiledModelMagic[4] = {'A', 'T', 'M', '1'};

// -- keypoints, matches and tracks ------------------------------------------

std::vector<std::uint8_t> encode_keypoints(const KeypointSet& set);
void decode_keypoints(const std::vector<std::uint8_t>& in, KeypointSet& set);

std::vector<std::uint8_t> encode_matches(const std::vector<PairMatches>& pairs);
std::vector<PairMatches> decode_matches(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_tracks(const std::vector<Track>& tracks);
std::vector<Track> decode_tracks(const std::vector<std::uint8_t>& in);

// -- dense products ---------------------------------------------------------

std::vector<std::uint8_t> encode_depth_maps(const std::vector<DepthMap>& maps);
std::vector<DepthMap> decode_depth_maps(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_point_cloud(const PointCloud& cloud);
PointCloud decode_point_cloud(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_mesh(const Mesh& mesh);
Mesh decode_mesh(const std::vector<std::uint8_t>& in);

// -- georeferenced products -------------------------------------------------

std::vector<std::uint8_t> encode_elevation(const Elevation& elevation);
Elevation decode_elevation(const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> encode_orthomosaic(const Orthomosaic& ortho);
Orthomosaic decode_orthomosaic(const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> encode_ortho_labels(const Orthomosaic& ortho);
std::vector<std::int32_t> decode_ortho_labels(const std::vector<std::uint8_t>& in,
                                              int width, int height);

// -- M11 containers ---------------------------------------------------------

std::vector<std::uint8_t> encode_mask(const Mask& mask);
Mask decode_mask(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_shapes(const Shapes& shapes);
Shapes decode_shapes(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_trajectory(const Trajectory& trajectory);
Trajectory decode_trajectory(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_frames(const std::vector<std::shared_ptr<Frame>>& frames);
std::vector<std::shared_ptr<Frame>> decode_frames(const std::vector<std::uint8_t>& in);

std::vector<std::uint8_t> encode_tiled_model(const TiledModel& model);
TiledModel decode_tiled_model(const std::vector<std::uint8_t>& in);

// -- shared validators ------------------------------------------------------

void validate_tie_points(const Chunk& chunk);
void validate_dense(const Chunk& chunk);
void validate_mesh(const Chunk& chunk);
void validate_model_textures(const Chunk& chunk);
void validate_products(const Chunk& chunk);

/// Structural invariants of the M11 containers (masks, shapes, trajectories,
/// frames, markers, groups, scalebars, panoramas, tiled models). The v2 writer
/// runs it before emitting; the reader runs it after assembling the chunk.
void validate_m11_containers(const Chunk& chunk);

}  // namespace alphashape::docxml::sidecar
