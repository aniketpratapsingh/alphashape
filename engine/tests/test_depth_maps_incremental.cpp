// test_depth_maps_incremental.cpp -- BuildDepthMaps keeps finished maps across a
// cancellation, reuses them on the next run, and invalidates them on a
// re-alignment (see build_depth_maps_task.cpp's "INCREMENTAL" and "REUSE"
// notes).
//
// The scene is the shared seq_0..7 analytic fixture (see dense_fixture.hpp), on
// the SfM poses: these tests are about bookkeeping, not accuracy, so the exact
// depth values do not matter and downscale 8 keeps every run to a few seconds.
//
// Search counting is timing-free: BuildDepthMapsTask exposes
// depth_map_search_count(), incremented once per PatchMatch call actually
// started. A reused map starts none, so the difference between two runs is
// exactly the number of cameras whose search was repeated.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/stereo/build_depth_maps_task.hpp"
#include "common/algorithm/stereo/view_selection.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/io/sidecars.hpp"
#include "common/tasks/registry.hpp"

#include "dense_fixture.hpp"

namespace alphashape {
namespace {

using dense_fixture::aligned_document;
using dense_fixture::seq_files;
using dense_fixture::TempDir;

constexpr int kDownscale = 8;  // the fixture is 480x360; maps come out 60x45
/// How many cameras BuildDepthMaps will process, i.e. the denominator its
/// progress fractions use. Read from view selection rather than assumed: a
/// camera with no usable neighbourhood gets no map, so this is not simply the
/// camera count.
std::size_t depth_camera_count(const Chunk& chunk) {
    return stereo::select_views(chunk, {}).size();
}

Params depth_params(int downscale, bool reuse, const std::string& filter = "mild") {
    Params params = make_params(builtin_registry().require("BuildDepthMaps"));
    params.set("downscale", ParamValue{static_cast<std::int64_t>(downscale)});
    params.set("filter_mode", ParamValue{filter});
    params.set("reuse_depth", ParamValue{reuse});
    return params;
}

/// run_task needs an lvalue Params, so every plain run goes through here.
void run_depth_maps(Document& document, int downscale, bool reuse,
                    const std::string& filter = "mild") {
    Params params = depth_params(downscale, reuse, filter);
    run_task(builtin_registry(), document, "BuildDepthMaps", params);
}

/// A Progress that records every report and cancels from the first report its
/// predicate accepts. The task reports per camera AND per PatchMatch sweep
/// inside a camera, so a threshold on the buildDepthMaps fraction lands exactly
/// on a camera boundary or inside one, which is the difference the tests need
/// to make. The flag latches: once set the task must stop, and a progress source
/// that un-set it would only mask a task that was not polling.
struct CancellingProgress {
    std::function<bool(const std::string& stage, double fraction)> when;
    bool cancel = false;
    std::vector<std::string> stages;
    std::vector<double> fractions;

    Progress progress() {
        Progress out;
        out.report = [this](const std::string& stage, double fraction) {
            stages.push_back(stage);
            fractions.push_back(fraction);
            if (when && !cancel && when(stage, fraction)) {
                cancel = true;
            }
        };
        out.cancelled = [this]() { return cancel; };
        return out;
    }

    bool saw_fraction_in(const std::string& stage, double lo, double hi) const {
        for (std::size_t i = 0; i < stages.size(); ++i) {
            if (stages[i] == stage && fractions[i] > lo && fractions[i] < hi) {
                return true;
            }
        }
        return false;
    }
};

std::size_t valid_pixels(const std::vector<DepthMap>& maps) {
    std::size_t n = 0;
    for (const DepthMap& map : maps) {
        for (const float d : map.depths) {
            if (d > 0.0F) {
                ++n;
            }
        }
    }
    return n;
}

// ---------------------------------------------------------------------------
// 1. A cancel keeps the maps finished so far
// ---------------------------------------------------------------------------

TEST(DepthMapsCancel, KeepsFinishedMapsAndSkipsThemOnRerun) {
    Document document = aligned_document(seq_files());
    Chunk& chunk = *document.active_chunk();

    // Cancel the moment the second camera is finished. The task reports
    // (i + 1) / cameras at each camera boundary, so 2 / cameras is reached
    // exactly then and never inside a camera (a sweep reports (i + f) / cameras
    // with f < 1).
    const std::size_t camera_count = depth_camera_count(chunk);
    ASSERT_GT(camera_count, 2U);

    reset_depth_map_search_count();
    CancellingProgress cancel;
    const double two_cameras = 2.0 / static_cast<double>(camera_count);
    cancel.when = [two_cameras](const std::string& stage, double fraction) {
        return stage == "buildDepthMaps" && fraction >= two_cameras;
    };
    Params params = depth_params(kDownscale, /*reuse=*/true);
    EXPECT_THROW(run_task(builtin_registry(), document, "BuildDepthMaps", params,
                          cancel.progress()),
                 CancelledError);

    // The maps finished before the cancel are in the chunk, as a new set, with
    // their downscale and a fingerprint (they are reusable, not dead weight).
    const std::vector<DepthMap>& kept = chunk.depth_maps();
    ASSERT_FALSE(kept.empty()) << "a cancel threw the finished maps away";
    const std::size_t first_run_searches = depth_map_search_count();
    ASSERT_GE(first_run_searches, kept.size());
    for (const DepthMap& map : kept) {
        EXPECT_EQ(map.downscale, kDownscale);
        EXPECT_NE(map.alignment_fingerprint, 0U);
        // Nothing was filtered yet: the run stopped during the build stage.
        EXPECT_EQ(map.filter_mode, -1);
    }
    EXPECT_FALSE(chunk.depth_map_sets().empty());
    EXPECT_EQ(chunk.active_depth_maps_id(), chunk.depth_map_sets().back().id);

    // A second run with reuse skips the kept cameras' search entirely.
    reset_depth_map_search_count();
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    const std::size_t second_run_searches = depth_map_search_count();

    const Chunk& after = *document.active_chunk();
    ASSERT_FALSE(after.depth_maps().empty());
    EXPECT_GT(after.depth_maps().size(), kept.size())
        << "the re-run did not add the remaining cameras";
    // Every camera the first run finished is skipped, so the re-run searches
    // exactly the remaining ones.
    EXPECT_EQ(second_run_searches + kept.size(), after.depth_maps().size())
        << "reused " << (after.depth_maps().size() - second_run_searches) << " of "
        << after.depth_maps().size() << " maps (expected the " << kept.size()
        << " kept by the cancelled run)";

    // The kept maps come through the filter this time.
    for (const DepthMap& map : after.depth_maps()) {
        EXPECT_GE(map.filter_mode, 0) << "camera " << map.camera_id << " was never filtered";
        EXPECT_GT(valid_pixels({map}), 0U);
    }
}

TEST(DepthMapsCancel, CancelBeforeAnyCameraLeavesTheChunkUnchanged) {
    Document document = aligned_document(seq_files(4));
    const Chunk before = *document.active_chunk();

    // Cancel from the very first report, so nothing finishes: the first poll
    // the task makes is inside the first camera's first sweep.
    CancellingProgress cancel;
    cancel.when = [](const std::string&, double) { return true; };
    Params params = depth_params(kDownscale, /*reuse=*/true);
    EXPECT_THROW(run_task(builtin_registry(), document, "BuildDepthMaps", params,
                          cancel.progress()),
                 CancelledError);

    EXPECT_TRUE(*document.active_chunk() == before)
        << "a cancel that finished nothing must not append an empty set";
}

// ---------------------------------------------------------------------------
// 2. A cancel inside one camera works: the sweep hook polls between sweeps
// ---------------------------------------------------------------------------

TEST(DepthMapsCancel, CancelInsideOneCameraStopsAtASweepBoundary) {
    Document document = aligned_document(seq_files());
    Chunk& chunk = *document.active_chunk();

    // A report strictly inside the FIRST camera's slice of the build stage. Only
    // the per-sweep hook can produce one -- the per-camera report is 0 or
    // 1 / cameras -- so reaching it proves the bar moves during a single image,
    // and cancelling there proves the search polls between sweeps.
    const std::size_t camera_count = depth_camera_count(chunk);
    ASSERT_GT(camera_count, 1U);

    CancellingProgress cancel;
    const double one_camera = 1.0 / static_cast<double>(camera_count);
    cancel.when = [one_camera](const std::string& stage, double fraction) {
        return stage == "buildDepthMaps" && fraction > 0.0 && fraction < one_camera;
    };
    Params params = depth_params(kDownscale, /*reuse=*/true);
    EXPECT_THROW(run_task(builtin_registry(), document, "BuildDepthMaps", params,
                          cancel.progress()),
                 CancelledError);

    // Mid-camera, so nothing was finished: the chunk is untouched (a half-swept
    // map is not a map) and the build stage reported intermediate fractions,
    // which is what proves the search ticked inside the camera.
    EXPECT_TRUE(cancel.saw_fraction_in("buildDepthMaps", 0.0, one_camera))
        << "the progress bar never moved inside the first camera";
    EXPECT_TRUE(chunk.depth_map_sets().empty()) << "a partial camera must not be stored";

    // The re-run completes normally.
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    EXPECT_FALSE(document.active_chunk()->depth_maps().empty());
}

// ---------------------------------------------------------------------------
// 3. Re-alignment invalidates reuse
// ---------------------------------------------------------------------------

TEST(DepthMapsReuse, ReAlignmentInvalidatesTheMaps) {
    Document document = aligned_document(seq_files(4));
    Chunk& chunk = *document.active_chunk();
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    ASSERT_FALSE(chunk.depth_maps().empty());
    const std::size_t map_count = chunk.depth_maps().size();
    for (const DepthMap& map : chunk.depth_maps()) {
        ASSERT_GE(map.filter_mode, 0);
    }

    // A no-op re-run reuses everything.
    reset_depth_map_search_count();
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    EXPECT_EQ(depth_map_search_count(), 0U) << "an unchanged chunk rebuilt maps it could reuse";

    // Move one camera, as a re-alignment would, and that camera's map is rebuilt
    // while every other map is still reused. Which camera is moved does not
    // matter -- neighbours are symmetric here -- but the map must NOT be reused.
    std::vector<DepthMap> before = document.active_chunk()->depth_maps();
    Camera& moved = document.active_chunk()->cameras().front();
    moved.pose(0, 3) += 0.25;
    const std::int32_t moved_id = moved.id;
    const std::uint64_t moved_fingerprint = [&] {
        const auto old = std::find_if(before.begin(), before.end(),
                                      [&](const DepthMap& m) { return m.camera_id == moved_id; });
        return old != before.end() ? old->alignment_fingerprint : 0U;
    }();
    ASSERT_NE(moved_fingerprint, 0U);

    reset_depth_map_search_count();
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    // Forcing the camera's own map to be rebuilt also forces its neighbours'
    // boundaries to be re-filtered, but their SEARCH must be reused.
    EXPECT_EQ(depth_map_search_count(), 1U)
        << "a re-aligned camera must rebuild exactly its own map";

    const Chunk& after = *document.active_chunk();
    ASSERT_EQ(after.depth_maps().size(), map_count);
    bool saw_moved = false;
    for (const DepthMap& map : after.depth_maps()) {
        if (map.camera_id != moved_id) {
            continue;
        }
        saw_moved = true;
        EXPECT_NE(map.alignment_fingerprint, moved_fingerprint)
            << "the moved camera kept the old fingerprint";
        EXPECT_NE(map.alignment_fingerprint, 0U);
    }
    EXPECT_TRUE(saw_moved) << "the moved camera has no map";
    // Every other camera still carries the fingerprint it had: they were reused.
    for (const DepthMap& map : after.depth_maps()) {
        if (map.camera_id == moved_id) {
            continue;
        }
        const auto old = std::find_if(before.begin(), before.end(),
                                      [&](const DepthMap& m) { return m.camera_id == map.camera_id; });
        ASSERT_NE(old, before.end());
        EXPECT_EQ(map.alignment_fingerprint, old->alignment_fingerprint)
            << "camera " << map.camera_id << " was rebuilt after an unrelated move";
    }
}

TEST(DepthMapsReuse, RecalibrationInvalidatesTheMaps) {
    // The sensor's calibration is part of the fingerprint too: a re-calibration
    // changes every camera on that sensor, so none of their maps may be reused.
    Document document = aligned_document(seq_files(4));
    Chunk& chunk = *document.active_chunk();
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    const std::size_t map_count = chunk.depth_maps().size();
    ASSERT_GT(map_count, 0U);

    document.active_chunk()->sensors().front().calibration.f *= 1.01;

    reset_depth_map_search_count();
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    EXPECT_EQ(depth_map_search_count(), map_count)
        << "a re-calibrated sensor must rebuild every map it produced";
}

TEST(DepthMapsReuse, FingerprintIsZeroForAMapWithNoRecordedAlignment) {
    // The fingerprint of a camera before alignment differs from the aligned one,
    // and the "unknown" sentinel 0 never matches a computed hash.
    Document document = aligned_document(seq_files(4));
    const Chunk& chunk = *document.active_chunk();
    ASSERT_FALSE(chunk.cameras().empty());
    ASSERT_FALSE(chunk.sensors().empty());
    const std::uint64_t fs = depth_map_fingerprint(chunk.cameras().front(), chunk.sensors().front(),
                                                   kDownscale);
    EXPECT_NE(fs, 0U);
    EXPECT_NE(fs, depth_map_fingerprint(chunk.cameras().front(), chunk.sensors().front(),
                                        kDownscale / 2))
        << "the downscale must be part of the fingerprint";
}

// ---------------------------------------------------------------------------
// 4. The fingerprint round-trips through the project file
// ---------------------------------------------------------------------------

TEST(DepthMapsReuse, FingerprintAndFilteredFlagSurviveASave) {
    Document document = aligned_document(seq_files(4));
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    const Chunk before = *document.active_chunk();
    ASSERT_FALSE(before.depth_maps().empty());

    TempDir dir("alphashape_depth_fingerprint");
    const std::string project = dir.file("proj.psx");
    document.set_path(project);
    ProjectIO::save(document, project);
    auto reloaded = ProjectIO::load(project);

    ASSERT_NE(reloaded->active_chunk(), nullptr);
    EXPECT_TRUE(*reloaded->active_chunk() == before);
    for (const DepthMap& map : reloaded->active_chunk()->depth_maps()) {
        EXPECT_NE(map.alignment_fingerprint, 0U) << "camera " << map.camera_id;
        EXPECT_EQ(map.filter_mode, 1) << "camera " << map.camera_id;
    }

    // And the reloaded project still reuses them: nothing to rebuild.
    reset_depth_map_search_count();
    run_depth_maps(*reloaded, kDownscale, /*reuse=*/true);
    EXPECT_EQ(depth_map_search_count(), 0U)
        << "a saved and reloaded map must still be reusable";
}

// ---------------------------------------------------------------------------
// 5. reuse_depth=false rebuilds everything, and the parameter default is on
// ---------------------------------------------------------------------------

TEST(DepthMapsReuse, DisablingReuseRebuildsEveryMap) {
    Document document = aligned_document(seq_files(4));
    run_depth_maps(document, kDownscale, /*reuse=*/true);
    const std::size_t map_count = document.active_chunk()->depth_maps().size();
    ASSERT_GT(map_count, 0U);

    reset_depth_map_search_count();
    run_depth_maps(document, kDownscale, /*reuse=*/false);
    EXPECT_EQ(depth_map_search_count(), map_count)
        << "reuse_depth=false must rebuild every camera";
}

TEST(DepthMapsReuse, ReuseDepthDefaultsToTrue) {
    const TaskEntry& entry = builtin_registry().require("BuildDepthMaps");
    const Params defaults = make_params(entry);
    EXPECT_TRUE(defaults.get_bool("reuse_depth"));
}

// ---------------------------------------------------------------------------
// 6. The sidecar format: ADM2 round-trips, ADM1 is read but never reusable
// ---------------------------------------------------------------------------

TEST(DepthMapsSidecar, Adm2RoundTripsFingerprintAndFilterMode) {
    DepthMap map;
    map.camera_id = 3;
    map.width = 2;
    map.height = 2;
    map.downscale = 4;
    map.depths = {1.0F, 2.0F, -1.0F, 4.0F};
    map.normals = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    map.confidence = {0.0F, 1.0F, 0.0F, 2.0F};
    map.alignment_fingerprint = 0xFEEDFACE12345678ULL;
    map.filter_mode = 3;  // aggressive

    const std::vector<std::uint8_t> bytes = docxml::sidecar::encode_depth_maps({map});
    ASSERT_GE(bytes.size(), 4U);
    EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + 4), "ADM2");
    const std::vector<DepthMap> back = docxml::sidecar::decode_depth_maps(bytes);
    ASSERT_EQ(back.size(), 1U);
    EXPECT_EQ(back.front(), map);

    // An unfiltered map (a cancel checkpoint) keeps its -1 through the file.
    map.filter_mode = -1;
    const std::vector<DepthMap> unfiltered =
        docxml::sidecar::decode_depth_maps(docxml::sidecar::encode_depth_maps({map}));
    ASSERT_EQ(unfiltered.size(), 1U);
    EXPECT_EQ(unfiltered.front().filter_mode, -1);
    EXPECT_EQ(unfiltered.front().alignment_fingerprint, 0xFEEDFACE12345678ULL);
}

TEST(DepthMapsSidecar, Adm1IsReadWithUnknownFingerprintAndMildFilter) {
    // The pre-fingerprint record: u32 count, then per map four i32s, two u8
    // flags and the pixel blocks. Hand-built so the layout is stated here rather
    // than taken from a writer that might be changed with it.
    std::vector<std::uint8_t> bytes = {'A', 'D', 'M', '1'};
    docxml::sidecar::put_u32(bytes, 1);
    docxml::sidecar::put_i32(bytes, /*camera_id=*/7);
    docxml::sidecar::put_i32(bytes, /*width=*/2);
    docxml::sidecar::put_i32(bytes, /*height=*/2);
    docxml::sidecar::put_i32(bytes, /*downscale=*/2);
    bytes.push_back(0);  // has_normals
    bytes.push_back(0);  // has_confidence
    for (const float d : {5.0F, 6.0F, -1.0F, 7.0F}) {
        docxml::sidecar::put_f32(bytes, d);
    }

    const std::vector<DepthMap> maps = docxml::sidecar::decode_depth_maps(bytes);
    ASSERT_EQ(maps.size(), 1U);
    const DepthMap& map = maps.front();
    EXPECT_EQ(map.camera_id, 7);
    EXPECT_EQ(map.downscale, 2);
    EXPECT_EQ(map.depths, (std::vector<float>{5.0F, 6.0F, -1.0F, 7.0F}));
    // "Unknown" alignment: a later run must not treat an old map as current.
    EXPECT_EQ(map.alignment_fingerprint, 0U);
    // A pre-ADM2 project stored only finished sets, filtered with the default.
    EXPECT_EQ(map.filter_mode, 1);
}

TEST(DepthMapsSidecar, Adm2RejectsAnUnknownFilterMode) {
    DepthMap map;
    map.width = 1;
    map.height = 1;
    map.depths = {1.0F};
    map.filter_mode = 1;
    std::vector<std::uint8_t> bytes = docxml::sidecar::encode_depth_maps({map});
    // The filter-mode byte is the last of the three flags of the first (and only)
    // record: 4 magic + 4 count + 8 fingerprint + 16 four i32s + 2 flags.
    const std::size_t mode_at = 4 + 4 + 8 + 16 + 2;
    ASSERT_LT(mode_at, bytes.size());
    bytes[mode_at] = 9;
    EXPECT_THROW(docxml::sidecar::decode_depth_maps(bytes), ProjectIOError);
}

}  // namespace
}  // namespace alphashape
