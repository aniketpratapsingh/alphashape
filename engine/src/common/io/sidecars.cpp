#include "common/io/sidecars.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape::docxml::sidecar {

// ---------------------------------------------------------------------------
// Little-endian primitives.
// ---------------------------------------------------------------------------
void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}

void put_i32(std::vector<std::uint8_t>& out, std::int32_t v) {
    // A negative int cast to unsigned is defined as the two's-complement bit
    // pattern in C++20, so this round-trips every int32 value exactly.
    put_u32(out, static_cast<std::uint32_t>(v));
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFULL));
    }
}

void put_f32(std::vector<std::uint8_t>& out, float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof bits);
    put_u32(out, bits);
}

void put_f64(std::vector<std::uint8_t>& out, double d) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &d, sizeof bits);
    put_u64(out, bits);
}

void put_string(std::vector<std::uint8_t>& out, const std::string& s) {
    put_u32(out, static_cast<std::uint32_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

namespace {

std::uint32_t get_u32(const std::vector<std::uint8_t>& in, std::size_t at) {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(in[at + i]) << (8U * i);
    }
    return v;
}

std::uint64_t get_u64(const std::vector<std::uint8_t>& in, std::size_t at) {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(in[at + i]) << (8U * i);
    }
    return v;
}

double get_f64(const std::vector<std::uint8_t>& in, std::size_t at) {
    const std::uint64_t bits = get_u64(in, at);
    double d = 0.0;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

float get_f32(const std::vector<std::uint8_t>& in, std::size_t at) {
    const std::uint32_t bits = get_u32(in, at);
    float f = 0.0F;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

bool is_power_of_two(int v) { return v >= 1 && (v & (v - 1)) == 0; }

bool magic_is(const std::vector<std::uint8_t>& in, const char (&magic)[4]) {
    return in.size() >= 4 && std::memcmp(in.data(), magic, 4) == 0;
}

/// Start a sidecar with `magic` and return a writer positioned after it.
std::vector<std::uint8_t> with_magic(const char (&magic)[4]) {
    return std::vector<std::uint8_t>(magic, magic + 4);
}

}  // namespace

// ---------------------------------------------------------------------------
// ByteReader.
// ---------------------------------------------------------------------------
std::uint32_t ByteReader::u32() {
    if (remaining() < 4) {
        throw ProjectIOError(std::string(what_) + ": truncated");
    }
    const std::uint32_t v = get_u32(in_, at_);
    at_ += 4;
    return v;
}

std::int32_t ByteReader::i32() { return static_cast<std::int32_t>(u32()); }

std::uint8_t ByteReader::u8() {
    if (remaining() < 1) {
        throw ProjectIOError(std::string(what_) + ": truncated");
    }
    const std::uint8_t v = in_[at_];
    at_ += 1;
    return v;
}

std::uint64_t ByteReader::u64() {
    if (remaining() < 8) {
        throw ProjectIOError(std::string(what_) + ": truncated");
    }
    const std::uint64_t v = get_u64(in_, at_);
    at_ += 8;
    return v;
}

double ByteReader::f64() {
    if (remaining() < 8) {
        throw ProjectIOError(std::string(what_) + ": truncated");
    }
    const double v = get_f64(in_, at_);
    at_ += 8;
    return v;
}

float ByteReader::f32() {
    if (remaining() < 4) {
        throw ProjectIOError(std::string(what_) + ": truncated");
    }
    const float v = get_f32(in_, at_);
    at_ += 4;
    return v;
}

std::string ByteReader::string() {
    const std::uint32_t n = u32();
    if (static_cast<std::uint64_t>(n) > remaining()) {
        throw ProjectIOError(std::string(what_) + ": string length exceeds the file");
    }
    std::string out(reinterpret_cast<const char*>(in_.data() + at_), n);
    at_ += n;
    return out;
}

std::uint32_t ByteReader::count(std::uint64_t item_bytes) {
    const std::uint32_t n = u32();
    const std::uint64_t need = static_cast<std::uint64_t>(n) * item_bytes;
    if (need > remaining()) {
        throw ProjectIOError(std::string(what_) + ": count exceeds the file length");
    }
    return n;
}

void ByteReader::f32s(std::uint64_t n, std::vector<float>& out) {
    // Compare n against remaining()/4 rather than n*4 against remaining(): a
    // corrupt header can claim a width*height whose *4 overflows uint64, and
    // the wrapped product would slip past a multiplication-based check.
    if (n > remaining() / 4) {
        throw ProjectIOError(std::string(what_) + ": truncated float array");
    }
    out.resize(static_cast<std::size_t>(n));
    for (float& v : out) {
        v = get_f32(in_, at_);
        at_ += 4;
    }
}

void ByteReader::f64s(std::uint64_t n, std::vector<double>& out) {
    if (n > remaining() / 8) {
        throw ProjectIOError(std::string(what_) + ": truncated double array");
    }
    out.resize(static_cast<std::size_t>(n));
    for (double& v : out) {
        v = get_f64(in_, at_);
        at_ += 8;
    }
}

void ByteReader::u8s(std::uint64_t n, std::vector<std::uint8_t>& out) {
    if (n > remaining()) {
        throw ProjectIOError(std::string(what_) + ": truncated byte array");
    }
    out.assign(in_.begin() + static_cast<std::ptrdiff_t>(at_),
               in_.begin() + static_cast<std::ptrdiff_t>(at_ + static_cast<std::size_t>(n)));
    at_ += static_cast<std::size_t>(n);
}

// ---------------------------------------------------------------------------
// Keypoints: "AKP1", u32 count, count * 4 f32 (x, y, scale, orientation), then
// count * Descriptor::kBytes.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_keypoints(const KeypointSet& set) {
    std::vector<std::uint8_t> out = with_magic(kKptMagic);
    put_u32(out, static_cast<std::uint32_t>(set.keypoints.size()));
    for (const auto& kp : set.keypoints) {
        put_f32(out, kp.x);
        put_f32(out, kp.y);
        put_f32(out, kp.scale);
        put_f32(out, kp.orientation);
    }
    for (const auto& d : set.descriptors) {
        out.insert(out.end(), d.bytes.begin(), d.bytes.end());
    }
    return out;
}

void decode_keypoints(const std::vector<std::uint8_t>& in, KeypointSet& set) {
    if (!magic_is(in, kKptMagic) || in.size() < 8) {
        throw ProjectIOError("keypoint file: bad header");
    }
    const std::size_t n = get_u32(in, 4);
    // Computed in 64-bit so a hostile count cannot overflow the size check.
    const std::uint64_t expected = 8 + static_cast<std::uint64_t>(n) * 16U +
                                   static_cast<std::uint64_t>(n) * Descriptor::kBytes;
    if (in.size() != expected) {
        throw ProjectIOError("keypoint file: size does not match its count");
    }
    set.keypoints.resize(n);
    set.descriptors.resize(n);
    std::size_t at = 8;
    for (auto& kp : set.keypoints) {
        kp.x = get_f32(in, at);
        kp.y = get_f32(in, at + 4);
        kp.scale = get_f32(in, at + 8);
        kp.orientation = get_f32(in, at + 12);
        at += 16;
    }
    for (auto& d : set.descriptors) {
        std::memcpy(d.bytes.data(), in.data() + at, Descriptor::kBytes);
        at += Descriptor::kBytes;
    }
}

// ---------------------------------------------------------------------------
// Matches: "AMT1", u32 pair count, then per pair i32 camera_a, i32 camera_b,
// u32 match count and that many (u32 index_a, u32 index_b).
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_matches(const std::vector<PairMatches>& pairs) {
    std::vector<std::uint8_t> out = with_magic(kMatchesMagic);
    put_u32(out, static_cast<std::uint32_t>(pairs.size()));
    for (const PairMatches& pair : pairs) {
        put_i32(out, pair.camera_a);
        put_i32(out, pair.camera_b);
        put_u32(out, static_cast<std::uint32_t>(pair.matches.size()));
        for (const KeypointMatch& m : pair.matches) {
            put_u32(out, m.index_a);
            put_u32(out, m.index_b);
        }
    }
    return out;
}

std::vector<PairMatches> decode_matches(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kMatchesMagic) || in.size() < 8) {
        throw ProjectIOError("matches file: bad header");
    }
    ByteReader r(in, 4, "matches file");
    // Each pair is at least two i32 ids and a u32 count: 12 bytes.
    const std::uint32_t pair_count = r.count(12);
    std::vector<PairMatches> out;
    out.reserve(pair_count);
    for (std::uint32_t i = 0; i < pair_count; ++i) {
        PairMatches pair;
        pair.camera_a = r.i32();
        pair.camera_b = r.i32();
        const std::uint32_t n = r.count(8);  // each match is two u32: 8 bytes
        pair.matches.resize(n);
        for (KeypointMatch& m : pair.matches) {
            m.index_a = r.u32();
            m.index_b = r.u32();
        }
        out.push_back(std::move(pair));
    }
    if (!r.at_end()) {
        throw ProjectIOError("matches file: trailing bytes");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tracks: "ATR2", u32 track count, then per track a u32 observation count, that
// many (i32 camera_id, u32 keypoint_index), a u8 has_point flag and, when 1,
// three f64.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_tracks(const std::vector<Track>& tracks) {
    std::vector<std::uint8_t> out = with_magic(kTracksMagic);
    put_u32(out, static_cast<std::uint32_t>(tracks.size()));
    for (const Track& track : tracks) {
        put_u32(out, static_cast<std::uint32_t>(track.observations.size()));
        for (const Observation& obs : track.observations) {
            put_i32(out, obs.camera_id);
            put_u32(out, obs.keypoint_index);
        }
        if (track.point.has_value()) {
            out.push_back(1);
            put_f64(out, (*track.point).x());
            put_f64(out, (*track.point).y());
            put_f64(out, (*track.point).z());
        } else {
            out.push_back(0);
        }
    }
    return out;
}

std::vector<Track> decode_tracks(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kTracksMagic) || in.size() < 8) {
        throw ProjectIOError("tracks file: bad header");
    }
    ByteReader r(in, 4, "tracks file");
    // Each track is at least its u32 observation count plus the u8 has_point.
    const std::uint32_t track_count = r.count(5);
    std::vector<Track> out;
    out.reserve(track_count);
    for (std::uint32_t i = 0; i < track_count; ++i) {
        const std::uint32_t n = r.count(8);  // each observation is i32 + u32
        Track track;
        track.observations.resize(n);
        for (Observation& obs : track.observations) {
            obs.camera_id = r.i32();
            obs.keypoint_index = r.u32();
        }
        const std::uint8_t has_point = r.u8();
        if (has_point > 1) {
            throw ProjectIOError("tracks file: has_point flag is not 0 or 1");
        }
        if (has_point == 1) {
            const double x = r.f64();
            const double y = r.f64();
            const double z = r.f64();
            track.point = Eigen::Vector3d(x, y, z);
        }
        out.push_back(std::move(track));
    }
    if (!r.at_end()) {
        throw ProjectIOError("tracks file: trailing bytes");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Depth maps. Current "ADM2": u32 map count, then per map u64
// alignment_fingerprint, i32 camera_id, i32 width, i32 height, i32 downscale,
// u8 has_normals, u8 has_confidence, u8 filter_mode (255 = never filtered),
// width*height f32 depths and, for each set flag, 3*width*height f32 normals and
// width*height f32 confidence. Read-only legacy "ADM1" is the same record
// without the fingerprint and the filter mode; its maps decode with fingerprint
// 0 (the "unknown" sentinel, so they are never reused) and filter_mode 1 (a
// pre-ADM2 project only ever stored finished sets, filtered "mild" by default).
// ---------------------------------------------------------------------------
constexpr std::uint8_t kCloudHasNormals = 1U << 0U;
constexpr std::uint8_t kCloudHasColours = 1U << 1U;
constexpr std::uint8_t kCloudHasConfidence = 1U << 2U;

std::vector<std::uint8_t> encode_depth_maps(const std::vector<DepthMap>& maps) {
    std::vector<std::uint8_t> out = with_magic(kDepthMapsMagic2);
    put_u32(out, static_cast<std::uint32_t>(maps.size()));
    for (const DepthMap& map : maps) {
        const bool has_normals = !map.normals.empty();
        const bool has_confidence = !map.confidence.empty();
        put_u64(out, map.alignment_fingerprint);
        put_i32(out, map.camera_id);
        put_i32(out, map.width);
        put_i32(out, map.height);
        put_i32(out, map.downscale);
        out.push_back(has_normals ? 1U : 0U);
        out.push_back(has_confidence ? 1U : 0U);
        // filter_mode: -1 (never filtered) is stored as 255, the four modes as
        // themselves, so a re-run can tell an unfiltered map from a filtered one
        // and from a map filtered with a different strength.
        out.push_back(static_cast<std::uint8_t>(map.filter_mode < 0 ? 255 : map.filter_mode));
        for (const float d : map.depths) put_f32(out, d);
        if (has_normals) {
            for (const float n : map.normals) put_f32(out, n);
        }
        if (has_confidence) {
            for (const float c : map.confidence) put_f32(out, c);
        }
    }
    return out;
}

std::vector<DepthMap> decode_depth_maps(const std::vector<std::uint8_t>& in) {
    const bool current = magic_is(in, kDepthMapsMagic2);
    if ((!current && !magic_is(in, kDepthMapsMagic)) || in.size() < 8) {
        throw ProjectIOError("depth map file: bad header");
    }
    // The per-map fixed part: four i32s, two or three u8 flags, and in ADM2 a
    // leading u64.
    const std::uint64_t head_bytes = current ? 27 : 18;
    ByteReader r(in, 4, "depth map file");
    const std::uint32_t count = r.count(head_bytes);
    std::vector<DepthMap> out;
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        DepthMap map;
        if (current) {
            map.alignment_fingerprint = r.u64();
        } else {
            // A pre-ADM2 project stored only finished, filtered sets, and its
            // default filter mode was "mild".
            map.filter_mode = 1;
        }
        map.camera_id = r.i32();
        map.width = r.i32();
        map.height = r.i32();
        map.downscale = r.i32();
        if (map.width < 0 || map.height < 0) {
            throw ProjectIOError("depth map file: negative dimensions");
        }
        if (!is_power_of_two(map.downscale)) {
            throw ProjectIOError("depth map file: downscale is not a power of two");
        }
        const std::uint8_t has_normals = r.u8();
        const std::uint8_t has_confidence = r.u8();
        if (has_normals > 1 || has_confidence > 1) {
            throw ProjectIOError("depth map file: a flag is not 0 or 1");
        }
        if (current) {
            const std::uint8_t mode = r.u8();
            if (mode != 255U && mode > 3U) {
                throw ProjectIOError("depth map file: unknown filter mode");
            }
            map.filter_mode = mode == 255U ? -1 : static_cast<int>(mode);
        }
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(map.width) * static_cast<std::uint64_t>(map.height);
        r.f32s(pixels, map.depths);
        if (has_normals == 1) {
            r.f32s(pixels * 3U, map.normals);
        }
        if (has_confidence == 1) {
            r.f32s(pixels, map.confidence);
        }
        out.push_back(std::move(map));
    }
    if (!r.at_end()) {
        throw ProjectIOError("depth map file: trailing bytes");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Point cloud: "APC1", u32 point count, a u8 flags byte (bit 0 has_normals,
// bit 1 has_colours, bit 2 has_confidence), count * 3 f64 positions and, for
// each set flag, count * 3 f32 normals, count * 3 u8 colours and count f32
// confidence.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_point_cloud(const PointCloud& cloud) {
    std::vector<std::uint8_t> out = with_magic(kPointCloudMagic);
    put_u32(out, static_cast<std::uint32_t>(cloud.points.size()));
    std::uint8_t flags = 0;
    if (!cloud.normals.empty()) flags |= kCloudHasNormals;
    if (!cloud.colours.empty()) flags |= kCloudHasColours;
    if (!cloud.confidence.empty()) flags |= kCloudHasConfidence;
    out.push_back(flags);
    for (const Eigen::Vector3d& p : cloud.points) {
        put_f64(out, p.x());
        put_f64(out, p.y());
        put_f64(out, p.z());
    }
    if (!cloud.normals.empty()) {
        for (const Eigen::Vector3f& n : cloud.normals) {
            put_f32(out, n.x());
            put_f32(out, n.y());
            put_f32(out, n.z());
        }
    }
    if (!cloud.colours.empty()) {
        out.insert(out.end(), cloud.colours.begin(), cloud.colours.end());
    }
    if (!cloud.confidence.empty()) {
        for (const float c : cloud.confidence) put_f32(out, c);
    }
    return out;
}

PointCloud decode_point_cloud(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kPointCloudMagic) || in.size() < 9) {
        throw ProjectIOError("point cloud file: bad header");
    }
    ByteReader r(in, 4, "point cloud file");
    const std::uint32_t count = r.count(1);  // at least the flags byte follows
    const std::uint8_t flags = r.u8();
    if ((flags & ~(kCloudHasNormals | kCloudHasColours | kCloudHasConfidence)) != 0) {
        throw ProjectIOError("point cloud file: unknown flag bits");
    }
    PointCloud cloud;
    cloud.points.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const double x = r.f64();
        const double y = r.f64();
        const double z = r.f64();
        cloud.points.emplace_back(x, y, z);
    }
    if ((flags & kCloudHasNormals) != 0) {
        std::vector<float> flat;
        r.f32s(static_cast<std::uint64_t>(count) * 3U, flat);
        cloud.normals.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            cloud.normals.emplace_back(flat[i * 3U], flat[i * 3U + 1U], flat[i * 3U + 2U]);
        }
    }
    if ((flags & kCloudHasColours) != 0) {
        r.u8s(static_cast<std::uint64_t>(count) * 3U, cloud.colours);
    }
    if ((flags & kCloudHasConfidence) != 0) {
        r.f32s(count, cloud.confidence);
    }
    if (!r.at_end()) {
        throw ProjectIOError("point cloud file: trailing bytes");
    }
    return cloud;
}

// ---------------------------------------------------------------------------
// Mesh. Two revisions are understood:
//
//   "AMS1" (legacy, read-only): u32 vertex count, u32 face count, u8 flags
//     (bit 0 has_uvs, bit 1 has_colours, bit 2 has_confidence), vertex_count *
//     3 f64 positions, face_count * 3 i32 indices and, for each set flag,
//     vertex_count * 2 f32 uvs (one PER VERTEX), vertex_count * 3 u8 colours
//     and vertex_count f32 confidence.
//
//   "AMS2" (current): same header, the flag set adds bit 3 has_uv_faces, and
//     the uv block becomes a pool: a u32 uv count followed by uv_count * 2 f32
//     coordinates, plus a trailing face_count * 3 i32 uv_faces block when bit 3
//     is set. AMS1's per-vertex uvs cannot express seams, so its values are
//     read for framing and discarded.
// ---------------------------------------------------------------------------
constexpr std::uint8_t kMeshHasUvs = 1U << 0U;
constexpr std::uint8_t kMeshHasColours = 1U << 1U;
constexpr std::uint8_t kMeshHasConfidence = 1U << 2U;
constexpr std::uint8_t kMeshHasUvFaces = 1U << 3U;

std::vector<std::uint8_t> encode_mesh(const Mesh& mesh) {
    std::vector<std::uint8_t> out = with_magic(kMeshMagic2);
    put_u32(out, static_cast<std::uint32_t>(mesh.vertices.size()));
    put_u32(out, static_cast<std::uint32_t>(mesh.faces.size()));
    std::uint8_t flags = 0;
    if (!mesh.uvs.empty()) flags |= kMeshHasUvs;
    if (!mesh.colours.empty()) flags |= kMeshHasColours;
    if (!mesh.confidence.empty()) flags |= kMeshHasConfidence;
    if (!mesh.uv_faces.empty()) flags |= kMeshHasUvFaces;
    out.push_back(flags);
    for (const Eigen::Vector3d& v : mesh.vertices) {
        put_f64(out, v.x());
        put_f64(out, v.y());
        put_f64(out, v.z());
    }
    for (const Eigen::Vector3i& f : mesh.faces) {
        put_i32(out, f.x());
        put_i32(out, f.y());
        put_i32(out, f.z());
    }
    if (!mesh.uvs.empty()) {
        // The pooled form: an explicit count, so the reader no longer has to
        // infer the pool size from the vertex count.
        put_u32(out, static_cast<std::uint32_t>(mesh.uvs.size()));
        for (const Eigen::Vector2f& uv : mesh.uvs) {
            put_f32(out, uv.x());
            put_f32(out, uv.y());
        }
    }
    if (!mesh.colours.empty()) {
        out.insert(out.end(), mesh.colours.begin(), mesh.colours.end());
    }
    if (!mesh.confidence.empty()) {
        for (const float c : mesh.confidence) put_f32(out, c);
    }
    if (!mesh.uv_faces.empty()) {
        for (const Eigen::Vector3i& uv : mesh.uv_faces) {
            put_i32(out, uv.x());
            put_i32(out, uv.y());
            put_i32(out, uv.z());
        }
    }
    return out;
}

Mesh decode_mesh(const std::vector<std::uint8_t>& in) {
    if (in.size() < 4) {
        throw ProjectIOError("mesh file: bad header");
    }
    const bool legacy = std::memcmp(in.data(), kMeshMagic, 4) == 0;
    const bool modern = std::memcmp(in.data(), kMeshMagic2, 4) == 0;
    if (!legacy && !modern) {
        throw ProjectIOError("mesh file: bad header");
    }
    if (in.size() < 13) {
        throw ProjectIOError("mesh file: bad header");
    }
    ByteReader r(in, 4, "mesh file");
    const std::uint32_t vertex_count = r.count(8);  // each vertex is 3 f64
    const std::uint32_t face_count = r.count(12);   // each face is 3 i32
    const std::uint8_t flags = r.u8();
    // AMS2 adds the uv-face flag; AMS1 must not carry it.
    const std::uint8_t allowed =
        modern ? static_cast<std::uint8_t>(kMeshHasUvs | kMeshHasColours |
                                           kMeshHasConfidence | kMeshHasUvFaces)
               : static_cast<std::uint8_t>(kMeshHasUvs | kMeshHasColours |
                                           kMeshHasConfidence);
    if ((flags & ~allowed) != 0) {
        throw ProjectIOError("mesh file: unknown flag bits");
    }
    Mesh mesh;
    std::vector<double> coords;
    r.f64s(static_cast<std::uint64_t>(vertex_count) * 3U, coords);
    mesh.vertices.reserve(vertex_count);
    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        mesh.vertices.emplace_back(coords[i * 3U], coords[i * 3U + 1U],
                                   coords[i * 3U + 2U]);
    }
    mesh.faces.reserve(face_count);
    for (std::uint32_t i = 0; i < face_count; ++i) {
        const std::int32_t a = r.i32();
        const std::int32_t b = r.i32();
        const std::int32_t c = r.i32();
        mesh.faces.emplace_back(a, b, c);
    }
    if ((flags & kMeshHasUvs) != 0) {
        if (modern) {
            const std::uint32_t uv_count = r.count(8);  // each uv is 2 f32
            std::vector<float> flat;
            r.f32s(static_cast<std::uint64_t>(uv_count) * 2U, flat);
            mesh.uvs.reserve(uv_count);
            for (std::uint32_t i = 0; i < uv_count; ++i) {
                mesh.uvs.emplace_back(flat[i * 2U], flat[i * 2U + 1U]);
            }
        } else {
            std::vector<float> flat;
            r.f32s(static_cast<std::uint64_t>(vertex_count) * 2U, flat);
        }
    }
    if ((flags & kMeshHasColours) != 0) {
        r.u8s(static_cast<std::uint64_t>(vertex_count) * 3U, mesh.colours);
    }
    if ((flags & kMeshHasConfidence) != 0) {
        r.f32s(vertex_count, mesh.confidence);
    }
    if (modern && (flags & kMeshHasUvFaces) != 0) {
        mesh.uv_faces.reserve(face_count);
        for (std::uint32_t i = 0; i < face_count; ++i) {
            const std::int32_t a = r.i32();
            const std::int32_t b = r.i32();
            const std::int32_t c = r.i32();
            mesh.uv_faces.emplace_back(a, b, c);
        }
    }
    if (!r.at_end()) {
        throw ProjectIOError("mesh file: trailing bytes");
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// DEM and orthomosaic.
//
// The rasters are GeoTIFF byte streams. GeoTIFF is a file-based codec, so the
// encoder writes to a uniquely named scratch file and reads it back; the
// decoder does the reverse. The name embeds the pid and a per-process counter,
// so two engine processes (ctest -jN) never collide, and the guard removes the
// file on every exit path including an exception from the codec.
//
// The seamline camera labels are a separate sidecar, "AOL1": u32 count, i32
// width, i32 height, then count * i32 labels (row-major, -1 = uncovered).
// ---------------------------------------------------------------------------
namespace {

/// A per-process counter plus the pid names every scratch file.
class ScopedTempFile {
public:
    explicit ScopedTempFile(const char* tag) {
        static std::atomic<unsigned> counter{0};
        std::error_code ec;
        const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
        if (ec) {
            throw ProjectIOError("no temporary directory for a GeoTIFF scratch file");
        }
        const std::string name =
            std::string("alphashape_") + tag + "_" +
            std::to_string(platform::process_id()) + "_" +
            std::to_string(counter.fetch_add(1)) + ".tif";
        path_ = (dir / name).string();
    }
    ~ScopedTempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    ScopedTempFile(const ScopedTempFile&) = delete;
    ScopedTempFile& operator=(const ScopedTempFile&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::vector<std::uint8_t> read_whole_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ProjectIOError("cannot read the GeoTIFF scratch file '" + path + "'");
    }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

void write_whole_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw ProjectIOError("cannot write the GeoTIFF scratch file '" + path + "'");
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw ProjectIOError("short write to the GeoTIFF scratch file '" + path + "'");
    }
}

/// A raster grid must describe a real, north-up, square-cornered grid: positive
/// dimensions, positive cell size, finite origin. Checked by the writer and by
/// the reader after decoding, so a malformed grid never reaches a chunk.
void validate_raster_grid(const RasterGrid& grid, const char* what) {
    if (grid.width <= 0 || grid.height <= 0) {
        throw ProjectIOError(std::string(what) + ": grid has non-positive dimensions");
    }
    if (!(grid.pixel_size_x > 0.0) || !(grid.pixel_size_y > 0.0) ||
        !std::isfinite(grid.pixel_size_x) || !std::isfinite(grid.pixel_size_y)) {
        throw ProjectIOError(std::string(what) +
                             ": grid has a non-positive or non-finite cell size");
    }
    if (!std::isfinite(grid.origin_x) || !std::isfinite(grid.origin_y)) {
        throw ProjectIOError(std::string(what) + ": grid has a non-finite origin");
    }
}

}  // namespace

std::vector<std::uint8_t> encode_elevation(const Elevation& elevation) {
    ScopedTempFile scratch("dem");
    GeoTiffOptions options;
    options.epsg = elevation.epsg();
#if defined(ALPHASHAPE_USE_PROJ)
    try {
        options.crs.emplace(static_cast<crs::CoordinateSystem>(elevation.crs));
    } catch (const ValueError& error) {
        throw ValueError("elevation sidecar: the stored CRS could not be read (" +
                         std::string(error.what()) + ")");
    }
#endif
    options.tiled = true;
    options.tile_size = 256;
    options.deflate = true;
    options.nodata = static_cast<double>(elevation.nodata);
    write_geotiff(scratch.path(), elevation.raster, options);
    return read_whole_file(scratch.path());
}

Elevation decode_elevation(const std::vector<std::uint8_t>& bytes) {
    ScopedTempFile scratch("dem_read");
    write_whole_file(scratch.path(), bytes);
    const GeoTiffInfo info = read_geotiff_info(scratch.path());
    if (!info.is_float || info.channels != 1) {
        throw ProjectIOError("elevation GeoTIFF is not a single-band float raster");
    }
    Elevation elevation;
    elevation.raster = read_geotiff_float(scratch.path());
#if defined(ALPHASHAPE_USE_PROJ)
    // Keep the full CRS the file records, not just the code the legacy
    // accessor exposes: a custom WKT would otherwise be lost on the round trip.
    if (info.crs.has_value()) {
        elevation.crs = CRS(*info.crs);
    } else {
        elevation.set_epsg(info.epsg);
    }
#else
    elevation.set_epsg(info.epsg);
#endif
    elevation.nodata = info.nodata.has_value() ? static_cast<float>(*info.nodata) : -32767.0F;
    return elevation;
}

std::vector<std::uint8_t> encode_orthomosaic(const Orthomosaic& ortho) {
    ScopedTempFile scratch("ortho");
    GeoTiffOptions options;
    options.epsg = ortho.epsg();
#if defined(ALPHASHAPE_USE_PROJ)
    try {
        options.crs.emplace(static_cast<crs::CoordinateSystem>(ortho.crs));
    } catch (const ValueError& error) {
        throw ValueError("orthomosaic sidecar: the stored CRS could not be read (" +
                         std::string(error.what()) + ")");
    }
#endif
    options.tiled = true;
    options.tile_size = 256;
    options.deflate = true;
    write_geotiff(scratch.path(), ortho.raster, options);
    return read_whole_file(scratch.path());
}

Orthomosaic decode_orthomosaic(const std::vector<std::uint8_t>& bytes) {
    ScopedTempFile scratch("ortho_read");
    write_whole_file(scratch.path(), bytes);
    const GeoTiffInfo info = read_geotiff_info(scratch.path());
    if (info.is_float || info.channels != 4) {
        throw ProjectIOError("orthomosaic GeoTIFF is not an 8-bit RGBA raster");
    }
    Orthomosaic ortho;
    ortho.raster = read_geotiff_u8(scratch.path());
#if defined(ALPHASHAPE_USE_PROJ)
    if (info.crs.has_value()) {
        ortho.crs = CRS(*info.crs);
    } else {
        ortho.set_epsg(info.epsg);
    }
#else
    ortho.set_epsg(info.epsg);
#endif
    return ortho;
}

std::vector<std::uint8_t> encode_ortho_labels(const Orthomosaic& ortho) {
    std::vector<std::uint8_t> out = with_magic(kOrthoLabelsMagic);
    put_u32(out, static_cast<std::uint32_t>(ortho.labels.size()));
    put_i32(out, ortho.raster.grid.width);
    put_i32(out, ortho.raster.grid.height);
    for (const std::int32_t label : ortho.labels) {
        put_i32(out, label);
    }
    return out;
}

std::vector<std::int32_t> decode_ortho_labels(const std::vector<std::uint8_t>& in,
                                              int width, int height) {
    if (!magic_is(in, kOrthoLabelsMagic) || in.size() < 16) {
        throw ProjectIOError("orthomosaic labels file: bad header");
    }
    ByteReader r(in, 4, "orthomosaic labels file");
    // A plain u32 rather than ByteReader::count(): the two dimension fields are
    // shared by the file's one array, so count(item_bytes) would over-reserve.
    const std::uint32_t count = r.u32();
    const std::int32_t stored_width = r.i32();
    const std::int32_t stored_height = r.i32();
    if (stored_width != width || stored_height != height) {
        throw ProjectIOError(
            "orthomosaic labels file: dimensions do not match the GeoTIFF");
    }
    const std::uint64_t expected =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (count != expected) {
        throw ProjectIOError(
            "orthomosaic labels file: label count does not match width*height");
    }
    if (static_cast<std::uint64_t>(r.remaining()) != expected * 4U) {
        throw ProjectIOError(
            "orthomosaic labels file: truncated or trailing label bytes");
    }
    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(count));
    for (std::uint32_t i = 0; i < count; ++i) {
        out.push_back(r.i32());
    }
    if (!r.at_end()) {
        throw ProjectIOError("orthomosaic labels file: trailing bytes");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Masks: "AMK1", i32 camera_id, i32 width, i32 height, u8 invert, then the
// width*height bitmap run-length encoded as (u8 run, u8 value) pairs. A run is
// 1..255 and consecutive runs alternate values starting from the first pixel's
// value, so a uniform mask costs two bytes however large it is.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_mask(const Mask& mask) {
    std::vector<std::uint8_t> out = with_magic(kMasksMagic);
    put_i32(out, mask.camera_id);
    put_i32(out, mask.width);
    put_i32(out, mask.height);
    out.push_back(mask.invert ? 1U : 0U);
    std::size_t i = 0;
    while (i < mask.data.size()) {
        const std::uint8_t value = mask.data[i];
        std::size_t run = 1;
        while (i + run < mask.data.size() && mask.data[i + run] == value && run < 255) {
            ++run;
        }
        out.push_back(static_cast<std::uint8_t>(run));
        out.push_back(value);
        i += run;
    }
    return out;
}

Mask decode_mask(const std::vector<std::uint8_t>& in) {
    // 4 magic + i32 camera + i32 width + i32 height + u8 invert.
    if (!magic_is(in, kMasksMagic) || in.size() < 17) {
        throw ProjectIOError("mask file: bad header");
    }
    ByteReader r(in, 4, "mask file");
    Mask mask;
    mask.camera_id = r.i32();
    mask.width = r.i32();
    mask.height = r.i32();
    if (mask.width <= 0 || mask.height <= 0) {
        throw ProjectIOError("mask file: non-positive dimensions");
    }
    const std::uint8_t invert = r.u8();
    if (invert > 1) {
        throw ProjectIOError("mask file: invert flag is not 0 or 1");
    }
    mask.invert = invert == 1;
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(mask.width) * static_cast<std::uint64_t>(mask.height);
    // The bitmap is run-length encoded, so its declared size says nothing about
    // the bytes that follow: a 17-byte header could claim 2^62 pixels and
    // reserve terabytes before the first run is read. A run pair is two bytes
    // and covers at most 255 pixels, which bounds the claim by the data itself.
    constexpr std::uint64_t kMaxPixelsPerRunPair = 255;
    if (pixels > (r.remaining() / 2U) * kMaxPixelsPerRunPair) {
        throw ProjectIOError("mask file: declared size exceeds the encoded data");
    }
    mask.data.reserve(static_cast<std::size_t>(pixels));
    while (mask.data.size() < pixels) {
        const std::uint8_t run = r.u8();
        const std::uint8_t value = r.u8();
        if (run == 0) {
            throw ProjectIOError("mask file: zero-length run");
        }
        if (mask.data.size() + run > pixels) {
            throw ProjectIOError("mask file: a run overflows the bitmap");
        }
        mask.data.insert(mask.data.end(), run, value);
    }
    if (!r.at_end()) {
        throw ProjectIOError("mask file: trailing bytes");
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Shapes: "ASH1" (M11 §4.6). u32 feature count; per feature i32 id, u8 type,
// u8 flags (bit 0 closed, bit 1 has_altitude), a length-prefixed label and
// layer, u32 vertex count, that many 3 f64, a u32 attribute count and that
// many key/value strings; then a u32 group count and per group i32 id, a
// length-prefixed label and a u32 member count with i32 members.
// ---------------------------------------------------------------------------
namespace {

std::uint8_t shape_flags(const Shape& shape) {
    std::uint8_t flags = 0;
    if (shape.closed) flags |= 1U << 0U;
    if (shape.has_altitude) flags |= 1U << 1U;
    return flags;
}

std::uint8_t shape_type_code(ShapeType type) {
    switch (type) {
        case ShapeType::Point: return 0;
        case ShapeType::Polyline: return 1;
        case ShapeType::Polygon: return 2;
    }
    return 0;
}

ShapeType shape_type_from_code(std::uint8_t code) {
    switch (code) {
        case 0: return ShapeType::Point;
        case 1: return ShapeType::Polyline;
        case 2: return ShapeType::Polygon;
        default: throw ProjectIOError("shapes file: unknown shape type code");
    }
}

}  // namespace

std::vector<std::uint8_t> encode_shapes(const Shapes& shapes) {
    std::vector<std::uint8_t> out = with_magic(kShapesMagic);
    put_u32(out, static_cast<std::uint32_t>(shapes.features.size()));
    for (const Shape& shape : shapes.features) {
        put_i32(out, shape.id);
        out.push_back(shape_type_code(shape.type));
        out.push_back(shape_flags(shape));
        put_string(out, shape.label);
        put_string(out, shape.layer);
        put_u32(out, static_cast<std::uint32_t>(shape.vertices.size()));
        for (const Eigen::Vector3d& v : shape.vertices) {
            put_f64(out, v.x());
            put_f64(out, v.y());
            put_f64(out, v.z());
        }
        put_u32(out, static_cast<std::uint32_t>(shape.attributes.size()));
        for (const auto& [key, value] : shape.attributes) {
            put_string(out, key);
            put_string(out, value);
        }
    }
    put_u32(out, static_cast<std::uint32_t>(shapes.groups.size()));
    for (const ShapeGroup& group : shapes.groups) {
        put_i32(out, group.id);
        put_string(out, group.label);
        put_u32(out, static_cast<std::uint32_t>(group.members.size()));
        for (const std::int32_t member : group.members) {
            put_i32(out, member);
        }
    }
    return out;
}

Shapes decode_shapes(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kShapesMagic) || in.size() < 8) {
        throw ProjectIOError("shapes file: bad header");
    }
    ByteReader r(in, 4, "shapes file");
    // Each feature is at least its i32 id, two flag bytes, two empty strings
    // and the vertex/attribute counts: 19 bytes.
    const std::uint32_t feature_count = r.count(19);
    Shapes shapes;
    shapes.features.reserve(feature_count);
    for (std::uint32_t i = 0; i < feature_count; ++i) {
        Shape shape;
        shape.id = r.i32();
        shape.type = shape_type_from_code(r.u8());
        const std::uint8_t flags = r.u8();
        if ((flags & ~(1U | 2U)) != 0) {
            throw ProjectIOError("shapes file: unknown flag bits");
        }
        shape.closed = (flags & 1U) != 0;
        shape.has_altitude = (flags & 2U) != 0;
        shape.label = r.string();
        shape.layer = r.string();
        const std::uint32_t vertex_count = r.count(24);  // each vertex is 3 f64
        shape.vertices.reserve(vertex_count);
        for (std::uint32_t v = 0; v < vertex_count; ++v) {
            const double x = r.f64();
            const double y = r.f64();
            const double z = r.f64();
            shape.vertices.emplace_back(x, y, z);
        }
        const std::uint32_t attribute_count = r.count(8);  // two u32 lengths
        for (std::uint32_t a = 0; a < attribute_count; ++a) {
            const std::string key = r.string();
            shape.attributes.emplace(key, r.string());
        }
        shapes.features.push_back(std::move(shape));
    }
    const std::uint32_t group_count = r.count(12);
    shapes.groups.reserve(group_count);
    for (std::uint32_t i = 0; i < group_count; ++i) {
        ShapeGroup group;
        group.id = r.i32();
        group.label = r.string();
        const std::uint32_t member_count = r.count(4);
        group.members.reserve(member_count);
        for (std::uint32_t m = 0; m < member_count; ++m) {
            group.members.push_back(r.i32());
        }
        shapes.groups.push_back(std::move(group));
    }
    if (!r.at_end()) {
        throw ProjectIOError("shapes file: trailing bytes");
    }
    return shapes;
}

// ---------------------------------------------------------------------------
// Trajectory: "ATJ1", i32 id, a length-prefixed label, u32 point count, then
// per point three f64 position, four f64 quaternion (w, x, y, z), f64
// timestamp and a u8 has_orientation flag.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_trajectory(const Trajectory& trajectory) {
    std::vector<std::uint8_t> out = with_magic(kTrajectoryMagic);
    put_i32(out, trajectory.id);
    put_string(out, trajectory.label);
    put_u32(out, static_cast<std::uint32_t>(trajectory.points.size()));
    for (const TrajectoryPoint& point : trajectory.points) {
        put_f64(out, point.position.x());
        put_f64(out, point.position.y());
        put_f64(out, point.position.z());
        const Eigen::Quaterniond& q = point.orientation;
        put_f64(out, q.w());
        put_f64(out, q.x());
        put_f64(out, q.y());
        put_f64(out, q.z());
        put_f64(out, point.timestamp_s);
        out.push_back(point.has_orientation ? 1U : 0U);
    }
    return out;
}

Trajectory decode_trajectory(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kTrajectoryMagic) || in.size() < 12) {
        throw ProjectIOError("trajectory file: bad header");
    }
    ByteReader r(in, 4, "trajectory file");
    Trajectory trajectory;
    trajectory.id = r.i32();
    trajectory.label = r.string();
    const std::uint32_t count = r.count(65);  // 8 f64 and the flag: 65 bytes
    trajectory.points.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        TrajectoryPoint point;
        const double x = r.f64();
        const double y = r.f64();
        const double z = r.f64();
        point.position = Eigen::Vector3d(x, y, z);
        const double w = r.f64();
        const double qx = r.f64();
        const double qy = r.f64();
        const double qz = r.f64();
        point.orientation = Eigen::Quaterniond(w, qx, qy, qz);
        point.timestamp_s = r.f64();
        const std::uint8_t has_orientation = r.u8();
        if (has_orientation > 1) {
            throw ProjectIOError("trajectory file: has_orientation is not 0 or 1");
        }
        point.has_orientation = has_orientation == 1;
        trajectory.points.push_back(point);
    }
    if (!r.at_end()) {
        throw ProjectIOError("trajectory file: trailing bytes");
    }
    return trajectory;
}

// ---------------------------------------------------------------------------
// Frames: "AFR1", u32 frame count, then per frame i32 id, i32 camera_id, i32
// width, i32 height and a length-prefixed path.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_frames(
    const std::vector<std::shared_ptr<Frame>>& frames) {
    std::vector<std::uint8_t> out = with_magic(kFramesMagic);
    put_u32(out, static_cast<std::uint32_t>(frames.size()));
    for (const auto& frame : frames) {
        put_i32(out, frame->id());
        put_i32(out, frame->camera_id());
        put_i32(out, frame->width());
        put_i32(out, frame->height());
        put_string(out, frame->path());
    }
    return out;
}

std::vector<std::shared_ptr<Frame>> decode_frames(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kFramesMagic) || in.size() < 8) {
        throw ProjectIOError("frames file: bad header");
    }
    ByteReader r(in, 4, "frames file");
    const std::uint32_t count = r.count(20);  // four i32s and a u32 length
    std::vector<std::shared_ptr<Frame>> out;
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::int32_t id = r.i32();
        const std::int32_t camera_id = r.i32();
        const int width = r.i32();
        const int height = r.i32();
        const std::string path = r.string();
        auto frame = std::make_shared<Frame>(path, width, height);
        frame->set_id(id);
        frame->set_camera_id(camera_id);
        out.push_back(std::move(frame));
    }
    if (!r.at_end()) {
        throw ProjectIOError("frames file: trailing bytes");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Validators. Each is the single source of truth for its container's
// invariants: the writer runs it before emitting a byte and the reader runs it
// after assembling the container, so a file the reader would reject can never
// be written and everything written is readable.
// ---------------------------------------------------------------------------
void validate_dense(const Chunk& chunk) {
    std::unordered_set<std::int32_t> camera_ids;
    camera_ids.reserve(chunk.cameras().size());
    for (const Camera& camera : chunk.cameras()) {
        camera_ids.insert(camera.id);
    }
    for (const DepthMapsAsset& asset : chunk.depth_map_sets()) {
        if (asset.data == nullptr) {
            throw ProjectIOError("depth-map set " + std::to_string(asset.id) +
                                 " has a null payload");
        }
        const std::vector<DepthMap>& maps = *asset.data;
        for (std::size_t i = 0; i < maps.size(); ++i) {
            const DepthMap& map = maps[i];
            if (camera_ids.find(map.camera_id) == camera_ids.end()) {
                throw ProjectIOError("depth map " + std::to_string(i) +
                                     " references camera " +
                                     std::to_string(map.camera_id) +
                                     " which is not in the chunk");
            }
            if (map.width < 0 || map.height < 0) {
                throw ProjectIOError("depth map " + std::to_string(i) +
                                     " has negative dimensions");
            }
            if (!is_power_of_two(map.downscale)) {
                throw ProjectIOError("depth map " + std::to_string(i) +
                                     " downscale is not a power of two");
            }
            const std::uint64_t pixels = static_cast<std::uint64_t>(map.width) *
                                         static_cast<std::uint64_t>(map.height);
            if (map.depths.size() != pixels) {
                throw ProjectIOError("depth map " + std::to_string(i) +
                                     ": depths size does not match width*height");
            }
            if (!map.normals.empty() && map.normals.size() != pixels * 3U) {
                throw ProjectIOError("depth map " + std::to_string(i) +
                                     ": normals size is not 3*width*height");
            }
            if (!map.confidence.empty() && map.confidence.size() != pixels) {
                throw ProjectIOError("depth map " + std::to_string(i) +
                                     ": confidence size does not match width*height");
            }
        }
    }

    for (const PointCloudAsset& asset : chunk.point_clouds()) {
        if (asset.data == nullptr) {
            throw ProjectIOError("point cloud " + std::to_string(asset.id) +
                                 " has a null payload");
        }
        const PointCloud& cloud = *asset.data;
        const std::size_t n = cloud.points.size();
        if (!cloud.normals.empty() && cloud.normals.size() != n) {
            throw ProjectIOError("point cloud: normals size does not match the point count");
        }
        if (!cloud.colours.empty() && cloud.colours.size() != n * 3U) {
            throw ProjectIOError("point cloud: colours size is not 3 bytes per point");
        }
        if (!cloud.confidence.empty() && cloud.confidence.size() != n) {
            throw ProjectIOError(
                "point cloud: confidence size does not match the point count");
        }
    }
}

void validate_mesh(const Chunk& chunk) {
    for (const ModelAsset& asset : chunk.models()) {
        if (asset.data == nullptr) {
            throw ProjectIOError("model " + std::to_string(asset.id) +
                                 " has a null payload");
        }
        const Mesh& mesh = *asset.data;
        const std::size_t n = mesh.vertices.size();
        if (!mesh.colours.empty() && mesh.colours.size() != n * 3U) {
            throw ProjectIOError("mesh: colours size is not 3 bytes per vertex");
        }
        if (!mesh.confidence.empty() && mesh.confidence.size() != n) {
            throw ProjectIOError("mesh: confidence size does not match the vertex count");
        }
        // uv_faces is either empty or one triple per face — never a partial list.
        if (!mesh.uv_faces.empty() && mesh.uv_faces.size() != mesh.faces.size()) {
            throw ProjectIOError("mesh: uv_faces size does not match the face count");
        }
        for (std::size_t i = 0; i < mesh.uv_faces.size(); ++i) {
            const Eigen::Vector3i& uv = mesh.uv_faces[i];
            for (int k = 0; k < 3; ++k) {
                const int index = uv[k];
                if (index < 0 || static_cast<std::size_t>(index) >= mesh.uvs.size()) {
                    throw ProjectIOError("mesh: uv_faces[" + std::to_string(i) +
                                         "] has an index out of range of the uv pool");
                }
            }
        }
        for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
            const Eigen::Vector3i& f = mesh.faces[i];
            if (f.x() < 0 || f.y() < 0 || f.z() < 0 ||
                static_cast<std::size_t>(f.x()) >= n ||
                static_cast<std::size_t>(f.y()) >= n ||
                static_cast<std::size_t>(f.z()) >= n) {
                throw ProjectIOError("mesh: face " + std::to_string(i) +
                                     " has a vertex index out of range");
            }
        }
    }
}

void validate_model_textures(const Chunk& chunk) {
    for (const ModelAsset& asset : chunk.models()) {
        for (std::size_t page = 0; page < asset.textures.size(); ++page) {
            const Image& image = asset.textures[page];
            if (image.width() <= 0 || image.height() <= 0) {
                throw ProjectIOError("mesh texture " + std::to_string(page) +
                                     " has non-positive dimensions");
            }
            if (image.channels() != 1 && image.channels() != 3) {
                throw ProjectIOError("mesh texture " + std::to_string(page) +
                                     " has an unsupported channel count");
            }
            const std::uint64_t expected =
                static_cast<std::uint64_t>(image.width()) *
                static_cast<std::uint64_t>(image.height()) *
                static_cast<std::uint64_t>(image.channels());
            if (image.size_bytes() != expected) {
                throw ProjectIOError("mesh texture " + std::to_string(page) +
                                     " byte count does not match its dimensions");
            }
        }
    }
}

void validate_products(const Chunk& chunk) {
    for (const ElevationAsset& asset : chunk.elevations()) {
        if (asset.data == nullptr) {
            throw ProjectIOError("elevation " + std::to_string(asset.id) +
                                 " has a null payload");
        }
        const Elevation& elevation = *asset.data;
        if (elevation.empty()) {
            continue;
        }
        validate_raster_grid(elevation.raster.grid, "elevation");
        if (elevation.raster.channels != 1) {
            throw ProjectIOError("elevation: the raster must have exactly one channel");
        }
        const std::uint64_t expected =
            static_cast<std::uint64_t>(elevation.raster.grid.width) *
            static_cast<std::uint64_t>(elevation.raster.grid.height);
        if (elevation.raster.data.size() != expected) {
            throw ProjectIOError("elevation: heights size does not match width*height");
        }
        if (!std::isfinite(elevation.nodata)) {
            throw ProjectIOError("elevation: nodata must be finite");
        }
        if (elevation.crs.epsg < 0) {
            throw ProjectIOError("elevation: epsg must be >= 0");
        }
    }

    for (const OrthomosaicAsset& asset : chunk.orthomosaics()) {
        if (asset.data == nullptr) {
            throw ProjectIOError("orthomosaic " + std::to_string(asset.id) +
                                 " has a null payload");
        }
        const Orthomosaic& ortho = *asset.data;
        if (ortho.empty()) {
            continue;
        }
        validate_raster_grid(ortho.raster.grid, "orthomosaic");
        if (ortho.raster.channels != 4) {
            throw ProjectIOError("orthomosaic: the raster must be RGBA (4 channels)");
        }
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(ortho.raster.grid.width) *
            static_cast<std::uint64_t>(ortho.raster.grid.height);
        if (ortho.raster.data.size() != pixels * 4U) {
            throw ProjectIOError("orthomosaic: pixel size does not match width*height*4");
        }
        if (ortho.labels.size() != pixels) {
            throw ProjectIOError("orthomosaic: label count does not match width*height");
        }
        if (ortho.crs.epsg < 0) {
            throw ProjectIOError("orthomosaic: epsg must be >= 0");
        }
    }
}

void validate_tie_points(const Chunk& chunk) {
    const TiePoints& tie_points = chunk.tie_points();

    std::unordered_set<std::int32_t> camera_ids;
    camera_ids.reserve(chunk.cameras().size());
    for (const Camera& camera : chunk.cameras()) {
        camera_ids.insert(camera.id);
    }
    // A camera with no KeypointSet at all has none, so it is treated as size 0
    // — every match/track index then fails the range check, which is correct.
    std::unordered_map<std::int32_t, std::size_t> keypoint_counts;
    keypoint_counts.reserve(tie_points.keypoint_sets.size());
    for (const KeypointSet& set : tie_points.keypoint_sets) {
        keypoint_counts.emplace(set.camera_id, set.keypoints.size());
    }
    const auto keypoint_count = [&](std::int32_t id) -> long long {
        const auto it = keypoint_counts.find(id);
        return it == keypoint_counts.end() ? 0 : static_cast<long long>(it->second);
    };
    const auto require_camera = [&](std::int32_t id, const char* what) {
        if (camera_ids.find(id) == camera_ids.end()) {
            throw ProjectIOError(std::string(what) + " references camera " +
                                 std::to_string(id) + " which is not in the chunk");
        }
    };
    const auto require_index = [&](std::int32_t camera_id, std::uint32_t index,
                                   const char* what) {
        if (static_cast<long long>(index) >= keypoint_count(camera_id)) {
            throw ProjectIOError(std::string(what) + ": keypoint index " +
                                 std::to_string(index) + " is out of range for camera " +
                                 std::to_string(camera_id));
        }
    };

    for (std::size_t i = 0; i < tie_points.matches.size(); ++i) {
        const PairMatches& pair = tie_points.matches[i];
        if (!(pair.camera_a < pair.camera_b)) {
            throw ProjectIOError("match pair " + std::to_string(i) +
                                 " is not ordered (camera_a < camera_b)");
        }
        require_camera(pair.camera_a, "match pair");
        require_camera(pair.camera_b, "match pair");
        if (i != 0) {
            const PairMatches& prev = tie_points.matches[i - 1];
            const bool sorted = prev.camera_a < pair.camera_a ||
                                (prev.camera_a == pair.camera_a &&
                                 prev.camera_b < pair.camera_b);
            if (!sorted) {
                throw ProjectIOError("match pairs are not strictly sorted at index " +
                                     std::to_string(i));
            }
        }
        const long long count_a = keypoint_count(pair.camera_a);
        const long long count_b = keypoint_count(pair.camera_b);
        for (const KeypointMatch& m : pair.matches) {
            if (static_cast<long long>(m.index_a) >= count_a) {
                throw ProjectIOError("match pair: keypoint index " +
                                     std::to_string(m.index_a) +
                                     " is out of range for camera " +
                                     std::to_string(pair.camera_a));
            }
            if (static_cast<long long>(m.index_b) >= count_b) {
                throw ProjectIOError("match pair: keypoint index " +
                                     std::to_string(m.index_b) +
                                     " is out of range for camera " +
                                     std::to_string(pair.camera_b));
            }
        }
    }

    for (std::size_t i = 0; i < tie_points.tracks.size(); ++i) {
        const Track& track = tie_points.tracks[i];
        if (track.observations.size() < 2) {
            throw ProjectIOError("track " + std::to_string(i) +
                                 " has fewer than two observations");
        }
        for (std::size_t j = 0; j < track.observations.size(); ++j) {
            const Observation& obs = track.observations[j];
            require_camera(obs.camera_id, "track");
            require_index(obs.camera_id, obs.keypoint_index, "track");
            if (j != 0 && !(track.observations[j - 1].camera_id < obs.camera_id)) {
                throw ProjectIOError("track " + std::to_string(i) +
                                     " observations are not sorted by camera");
            }
        }
        if (track.point.has_value()) {
            const Eigen::Vector3d& p = *track.point;
            if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) {
                throw ProjectIOError("track " + std::to_string(i) +
                                     " has a non-finite point");
            }
        }
    }
}

void validate_m11_containers(const Chunk& chunk) {
    // Masks: a valid bitmap, one per existing camera, sorted by camera id.
    std::unordered_set<std::int32_t> camera_ids;
    for (const Camera& camera : chunk.cameras()) {
        camera_ids.insert(camera.id);
    }
    for (std::size_t i = 0; i < chunk.masks().size(); ++i) {
        const Mask& mask = chunk.masks()[i];
        if (!mask.is_valid()) {
            throw ProjectIOError("mask for camera " + std::to_string(mask.camera_id) +
                                 " has inconsistent dimensions");
        }
        if (camera_ids.find(mask.camera_id) == camera_ids.end()) {
            throw ProjectIOError("mask references camera " +
                                 std::to_string(mask.camera_id) +
                                 " which is not in the chunk");
        }
        if (i != 0 && chunk.masks()[i - 1].camera_id >= mask.camera_id) {
            throw ProjectIOError("masks are not sorted by camera or are duplicated");
        }
    }

    // Markers: sorted unique ids, projections sorted and resolvable.
    std::unordered_set<std::int32_t> marker_ids;
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        const Marker& marker = chunk.markers()[i];
        if (i != 0 && chunk.markers()[i - 1].id >= marker.id) {
            throw ProjectIOError("markers are not sorted by id or are duplicated");
        }
        marker_ids.insert(marker.id);
        for (std::size_t j = 0; j < marker.projections.size(); ++j) {
            const MarkerProjection& p = marker.projections[j];
            if (camera_ids.find(p.camera_id) == camera_ids.end()) {
                throw ProjectIOError("marker " + std::to_string(marker.id) +
                                     " projects into missing camera " +
                                     std::to_string(p.camera_id));
            }
            if (j != 0 && marker.projections[j - 1].camera_id >= p.camera_id) {
                throw ProjectIOError("marker " + std::to_string(marker.id) +
                                     " projections are out of order or duplicated");
            }
        }
    }
    std::unordered_set<std::int32_t> marker_group_ids;
    for (std::size_t i = 0; i < chunk.marker_groups().size(); ++i) {
        const MarkerGroup& group = chunk.marker_groups()[i];
        if (i != 0 && chunk.marker_groups()[i - 1].id >= group.id) {
            throw ProjectIOError("marker groups are not sorted by id or are duplicated");
        }
        marker_group_ids.insert(group.id);
        for (const std::int32_t member : group.members) {
            if (marker_ids.find(member) == marker_ids.end()) {
                throw ProjectIOError("marker group " + std::to_string(group.id) +
                                     " references missing marker " +
                                     std::to_string(member));
            }
        }
    }
    for (const Marker& marker : chunk.markers()) {
        if (marker.group_id >= 0 &&
            marker_group_ids.find(marker.group_id) == marker_group_ids.end()) {
            throw ProjectIOError("marker " + std::to_string(marker.id) +
                                 " references missing group " +
                                 std::to_string(marker.group_id));
        }
    }

    // Scale bars: both endpoints resolve, the length is a real number.
    for (std::size_t i = 0; i < chunk.scalebars().size(); ++i) {
        const Scalebar& bar = chunk.scalebars()[i];
        if (i != 0 && chunk.scalebars()[i - 1].id >= bar.id) {
            throw ProjectIOError("scalebars are not sorted by id or are duplicated");
        }
        for (const ScalebarEndpoint* const endpoint : {&bar.from, &bar.to}) {
            const bool is_marker = endpoint->kind == ScalebarEndpoint::Kind::Marker;
            const bool present =
                is_marker ? marker_ids.count(endpoint->id) != 0
                          : camera_ids.count(endpoint->id) != 0;
            if (!present) {
                throw ProjectIOError("scalebar " + std::to_string(bar.id) +
                                     " references missing " +
                                     (is_marker ? "marker " : "camera ") +
                                     std::to_string(endpoint->id));
            }
        }
        if (!std::isfinite(bar.reference_length_m) || bar.reference_length_m < 0.0) {
            throw ProjectIOError("scalebar " + std::to_string(bar.id) +
                                 " has invalid reference length");
        }
    }

    // Camera groups: members name existing cameras.
    for (std::size_t i = 0; i < chunk.camera_groups().size(); ++i) {
        const CameraGroup& group = chunk.camera_groups()[i];
        if (i != 0 && chunk.camera_groups()[i - 1].id >= group.id) {
            throw ProjectIOError("camera groups are not sorted by id or are duplicated");
        }
        for (const std::int32_t member : group.members) {
            if (camera_ids.find(member) == camera_ids.end()) {
                throw ProjectIOError("camera group " + std::to_string(group.id) +
                                     " references missing camera " +
                                     std::to_string(member));
            }
        }
    }

    // Shapes: sorted unique feature ids, a vertex count the type allows,
    // resolvable group members.
    std::unordered_set<std::int32_t> shape_ids;
    for (std::size_t i = 0; i < chunk.shapes().features.size(); ++i) {
        const Shape& shape = chunk.shapes().features[i];
        if (i != 0 && chunk.shapes().features[i - 1].id >= shape.id) {
            throw ProjectIOError("shapes are not sorted by id or are duplicated");
        }
        const std::size_t minimum = shape.type == ShapeType::Point     ? 1U
                                    : shape.type == ShapeType::Polyline ? 2U
                                                                       : 3U;
        if (shape.vertices.size() < minimum) {
            throw ProjectIOError("shape " + std::to_string(shape.id) +
                                 " has too few vertices for its type");
        }
        shape_ids.insert(shape.id);
    }
    for (std::size_t i = 0; i < chunk.shapes().groups.size(); ++i) {
        const ShapeGroup& group = chunk.shapes().groups[i];
        if (i != 0 && chunk.shapes().groups[i - 1].id >= group.id) {
            throw ProjectIOError("shape groups are not sorted by id or are duplicated");
        }
        for (const std::int32_t member : group.members) {
            if (shape_ids.find(member) == shape_ids.end()) {
                throw ProjectIOError("shape group " + std::to_string(group.id) +
                                     " references missing shape " +
                                     std::to_string(member));
            }
        }
    }

    // Trajectories: sorted unique ids, timestamps non-decreasing.
    for (std::size_t i = 0; i < chunk.trajectories().size(); ++i) {
        const Trajectory& trajectory = chunk.trajectories()[i];
        if (i != 0 && chunk.trajectories()[i - 1].id >= trajectory.id) {
            throw ProjectIOError(
                "trajectories are not sorted by id or are duplicated");
        }
        for (std::size_t j = 1; j < trajectory.points.size(); ++j) {
            if (trajectory.points[j - 1].timestamp_s > trajectory.points[j].timestamp_s) {
                throw ProjectIOError("trajectory " + std::to_string(trajectory.id) +
                                     " points are not in timestamp order");
            }
        }
    }

    // Frames: ids unique.
    std::unordered_set<std::int32_t> frame_ids;
    for (const auto& frame : chunk.frames()) {
        if (frame == nullptr) {
            throw ProjectIOError("a frame has a null payload");
        }
        if (!frame_ids.insert(frame->id()).second) {
            throw ProjectIOError("two frames carry id " + std::to_string(frame->id()));
        }
    }

    // Tiled models and the panorama: payloads exist, ids sorted unique.
    for (std::size_t i = 0; i < chunk.tiled_models().size(); ++i) {
        const TiledModelAsset& asset = chunk.tiled_models()[i];
        if (asset.data == nullptr) {
            throw ProjectIOError("tiled model " + std::to_string(asset.id) +
                                 " has a null payload");
        }
        if (i != 0 && chunk.tiled_models()[i - 1].id >= asset.id) {
            throw ProjectIOError(
                "tiled models are not sorted by id or are duplicated");
        }
    }
    if (chunk.panorama().has_value()) {
        const PanoramaAsset& panorama = *chunk.panorama();
        if (panorama.data == nullptr) {
            throw ProjectIOError("panorama has a null payload");
        }
    }

    // Every active id names an existing asset (or -1 with an empty list).
    const auto check_active = [](const auto& slots, std::int32_t active_id,
                                 const char* kind) {
        if (slots.empty()) {
            if (active_id != -1) {
                throw ProjectIOError(std::string("active ") + kind + " id " +
                                     std::to_string(active_id) + " names nothing");
            }
            return;
        }
        for (const auto& slot : slots) {
            if (slot.id == active_id) {
                return;
            }
        }
        throw ProjectIOError(std::string("active ") + kind + " id " +
                             std::to_string(active_id) + " names nothing");
    };
    check_active(chunk.depth_map_sets(), chunk.active_depth_maps_id(), "depth-map set");
    check_active(chunk.point_clouds(), chunk.active_point_cloud_id(), "point cloud");
    check_active(chunk.models(), chunk.active_model_id(), "model");
    check_active(chunk.elevations(), chunk.active_elevation_id(), "elevation");
    check_active(chunk.orthomosaics(), chunk.active_orthomosaic_id(), "orthomosaic");
    check_active(chunk.tiled_models(), chunk.active_tiled_model_id(), "tiled model");

    // Ids must be unique across every container (one allocator feeds them all).
    std::unordered_set<std::int32_t> seen;
    const auto take = [&](std::int32_t id, const char* kind) {
        if (id >= 0 && !seen.insert(id).second) {
            throw ProjectIOError(std::string("id ") + std::to_string(id) +
                                 " is used by more than one " + kind);
        }
    };
    for (const auto& slot : chunk.depth_map_sets()) take(slot.id, "depth-map set");
    for (const auto& slot : chunk.point_clouds()) take(slot.id, "point cloud");
    for (const auto& slot : chunk.models()) take(slot.id, "model");
    for (const auto& slot : chunk.elevations()) take(slot.id, "elevation");
    for (const auto& slot : chunk.orthomosaics()) take(slot.id, "orthomosaic");
    for (const auto& slot : chunk.tiled_models()) take(slot.id, "tiled model");
    if (chunk.panorama().has_value()) take(chunk.panorama()->id, "panorama");
    for (const auto& marker : chunk.markers()) take(marker.id, "marker");
    for (const auto& group : chunk.marker_groups()) take(group.id, "marker group");
    for (const auto& bar : chunk.scalebars()) take(bar.id, "scalebar");
    for (const auto& group : chunk.camera_groups()) take(group.id, "camera group");
    for (const auto& shape : chunk.shapes().features) take(shape.id, "shape");
    for (const auto& group : chunk.shapes().groups) take(group.id, "shape group");
    for (const auto& trajectory : chunk.trajectories()) take(trajectory.id, "trajectory");
}

// ---------------------------------------------------------------------------
// Tiled model: "ATM1" followed by the opaque payload verbatim. The builder and
// the SLPK/3D-tiles writer land in M21; until then the bytes ride through
// unchanged, but the entry stays self-describing like every other sidecar.
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_tiled_model(const TiledModel& model) {
    std::vector<std::uint8_t> out = with_magic(kTiledModelMagic);
    out.insert(out.end(), model.payload.begin(), model.payload.end());
    return out;
}

TiledModel decode_tiled_model(const std::vector<std::uint8_t>& in) {
    if (!magic_is(in, kTiledModelMagic)) {
        throw ProjectIOError("tiled model file: bad header");
    }
    TiledModel model;
    model.payload.assign(in.begin() + 4, in.end());
    return model;
}

}  // namespace alphashape::docxml::sidecar
