// Tests for track (tie point) construction and the tie-point limit.

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/algorithm/match/tracks.hpp"

namespace alphashape::match {
namespace {

using alphashape::KeypointMatch;
using alphashape::Observation;
using alphashape::PairMatches;
using alphashape::Track;

/// Build a PairMatches with camera_a < camera_b (the invariant callers keep).
PairMatches pair(std::int32_t a, std::int32_t b, std::vector<KeypointMatch> matches) {
    if (a > b) {
        for (KeypointMatch& match : matches) std::swap(match.index_a, match.index_b);
        std::swap(a, b);
    }
    PairMatches result;
    result.camera_a = a;
    result.camera_b = b;
    result.matches = std::move(matches);
    return result;
}

bool has_observation(const Track& track, std::int32_t camera_id, std::uint32_t index) {
    for (const Observation& observation : track.observations) {
        if (observation.camera_id == camera_id && observation.keypoint_index == index) return true;
    }
    return false;
}

// --------------------------------------------------------------------------

TEST(TracksBuild, ChainFormsOneTrack) {
    // 0kp0 - 1kp0 and 1kp0 - 2kp0 link through camera 1 into a single track.
    const std::vector<PairMatches> matches{
        pair(0, 1, {{0, 0}}),
        pair(1, 2, {{0, 0}}),
    };
    TrackStats stats;
    const std::vector<Track> tracks = build_tracks(matches, &stats);
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(stats.tracks, 1U);
    EXPECT_EQ(stats.rejected_inconsistent, 0U);
    ASSERT_EQ(tracks[0].observations.size(), 3U);
    EXPECT_EQ(tracks[0].observations[0], (Observation{0, 0}));
    EXPECT_EQ(tracks[0].observations[1], (Observation{1, 0}));
    EXPECT_EQ(tracks[0].observations[2], (Observation{2, 0}));
}

TEST(TracksBuild, DisjointTracks) {
    const std::vector<PairMatches> matches{
        pair(0, 1, {{0, 0}, {5, 7}}),  // two independent correspondences
        pair(1, 2, {{7, 3}}),          // links 5<-7 with 2kp3
    };
    const std::vector<Track> tracks = build_tracks(matches);
    ASSERT_EQ(tracks.size(), 2U);
    // Track A: 0kp0 - 1kp0 (size 2). Track B: 0kp5 - 1kp7 - 2kp3 (size 3).
    // Sorted by first observation: (0,0) before (0,5).
    EXPECT_TRUE(has_observation(tracks[0], 0, 0));
    EXPECT_TRUE(has_observation(tracks[0], 1, 0));
    EXPECT_EQ(tracks[0].observations.size(), 2U);
    EXPECT_TRUE(has_observation(tracks[1], 0, 5));
    EXPECT_TRUE(has_observation(tracks[1], 1, 7));
    EXPECT_TRUE(has_observation(tracks[1], 2, 3));
    EXPECT_EQ(tracks[1].observations.size(), 3U);
}

TEST(TracksBuild, InconsistentComponentDropped) {
    // camera0 kp1 -- camera1 kp0 -- camera0 kp2: one component with two
    // keypoints of camera 0, which is impossible.
    const std::vector<PairMatches> matches{
        pair(0, 1, {{1, 0}, {2, 0}}),
    };
    TrackStats stats;
    const std::vector<Track> tracks = build_tracks(matches, &stats);
    EXPECT_TRUE(tracks.empty());
    EXPECT_EQ(stats.tracks, 0U);
    EXPECT_EQ(stats.rejected_inconsistent, 1U);
}

TEST(TracksBuild, InconsistentComponentDroppedButOthersSurvive) {
    const std::vector<PairMatches> matches{
        pair(0, 1, {{1, 0}, {2, 0}}),  // inconsistent component
        pair(3, 4, {{0, 0}}),          // clean track
    };
    TrackStats stats;
    const std::vector<Track> tracks = build_tracks(matches, &stats);
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(stats.tracks, 1U);
    EXPECT_EQ(stats.rejected_inconsistent, 1U);
    EXPECT_TRUE(has_observation(tracks[0], 3, 0));
    EXPECT_TRUE(has_observation(tracks[0], 4, 0));
}

TEST(TracksBuild, DeterministicOrdering) {
    // Three independent tracks whose first observations are (0,0), (0,9), (2,1).
    const std::vector<PairMatches> matches{
        pair(2, 3, {{1, 1}}),
        pair(0, 3, {{9, 2}}),
        pair(0, 4, {{0, 5}}),
    };
    const std::vector<Track> first = build_tracks(matches);
    // Rebuild with the input in a different order; the output must be identical.
    const std::vector<PairMatches> shuffled{matches[2], matches[0], matches[1]};
    const std::vector<Track> second = build_tracks(shuffled);
    EXPECT_EQ(first, second);
    ASSERT_EQ(first.size(), 3U);
    EXPECT_EQ(first[0].observations.front(), (Observation{0, 0}));
    EXPECT_EQ(first[1].observations.front(), (Observation{0, 9}));
    EXPECT_EQ(first[2].observations.front(), (Observation{2, 1}));
}

TEST(TracksBuild, EmptyInput) {
    TrackStats stats;
    EXPECT_TRUE(build_tracks({}, &stats).empty());
    EXPECT_EQ(stats.tracks, 0U);
    EXPECT_EQ(stats.rejected_inconsistent, 0U);
}

TEST(TracksLimit, KeepsLongestPerCamera) {
    // Camera 0 observes five tracks of lengths 2..6. Track i is also observed by
    // cameras 1..(i+1), so every camera sees a set whose longest members are
    // tracks 3 and 4, and the limit-2 survivor set is exactly {3, 4}.
    std::vector<Track> tracks;
    for (int i = 0; i < 5; ++i) {
        Track track;
        track.observations.push_back(Observation{0, static_cast<std::uint32_t>(i)});
        for (int camera = 1; camera <= i + 1; ++camera) {
            track.observations.push_back(
                Observation{camera, static_cast<std::uint32_t>(i)});
        }
        tracks.push_back(std::move(track));
    }

    const std::vector<Track> kept = limit_tracks(tracks, 2);
    ASSERT_EQ(kept.size(), 2U);
    // The two longest are tracks 3 (length 5) and 4 (length 6); kept in original
    // order.
    EXPECT_EQ(kept[0], tracks[3]);
    EXPECT_EQ(kept[1], tracks[4]);
}

TEST(TracksLimit, SelectedByAnotherCameraSurvives) {
    // Camera 0 observes two tracks and keeps the longer with limit 1. Camera 9
    // observes only the short one, so it survives too.
    std::vector<Track> tracks;
    Track short_track;
    short_track.observations = {{0, 0}, {9, 0}};
    Track long_track;
    long_track.observations = {{0, 1}, {1, 1}, {2, 1}};
    tracks = {short_track, long_track};

    const std::vector<Track> kept = limit_tracks(tracks, 1);
    ASSERT_EQ(kept.size(), 2U);
    EXPECT_EQ(kept[0], short_track);  // camera 9's pick, in original order
    EXPECT_EQ(kept[1], long_track);   // camera 0's pick
}

TEST(TracksLimit, ZeroMeansNoLimit) {
    std::vector<Track> tracks;
    Track track;
    track.observations = {{0, 0}, {1, 0}};
    tracks.push_back(track);
    EXPECT_EQ(limit_tracks(tracks, 0), tracks);
}

TEST(TracksLimit, HighestLengthTiesByTrackIndex) {
    // Two equal-length tracks, both observed by cameras 0 and 1. With limit 1 the
    // tie is broken by track index in both cameras, so only the earlier survives.
    std::vector<Track> tracks;
    Track first;
    first.observations = {{0, 0}, {1, 0}};
    Track second;
    second.observations = {{0, 1}, {1, 1}};
    tracks = {first, second};

    const std::vector<Track> kept = limit_tracks(tracks, 1);
    ASSERT_EQ(kept.size(), 1U);
    EXPECT_EQ(kept[0], first);
}

}  // namespace
}  // namespace alphashape::match
