// Tests for the model command layer (WP-11.5b): chunk, marker, scale-bar,
// group and camera commands.
//
// Every case drives the same invariant helper: apply changes the document,
// revert restores the exact pre-state (Chunk::operator== over the whole
// model), and a second apply reproduces the first. After every apply and every
// revert the chunk must pass Chunk::validate().

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/scalebar.hpp"

namespace alphashape {
namespace {

std::int32_t add_chunk(Document& document, std::string label) {
    Chunk& chunk = document.workspace().add_chunk(std::move(label));
    document.set_active_chunk_id(chunk.id());
    return chunk.id();
}

Chunk& chunk_of(Document& document, std::int32_t chunk_id) {
    Chunk* chunk = document.workspace().find_chunk(chunk_id);
    if (chunk == nullptr) {
        throw Error("test: no chunk " + std::to_string(chunk_id));
    }
    return *chunk;
}

/// Add `count` cameras (ids 0..count-1) sharing sensor 0, enabled.
void add_cameras(Chunk& chunk, int count) {
    for (int i = 0; i < count; ++i) {
        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(i);
        camera.enabled = true;
        chunk.cameras().push_back(camera);
    }
}

/// Everything the model commands may change: every chunk by value (deep copy,
/// so asset payloads are cloned), their order, the id counters and which chunk
/// is active. Comparing two states is the revert check.
struct ModelState {
    std::vector<Chunk> chunks;
    std::vector<std::int32_t> ids;
    std::int32_t active_chunk_id = -1;
    std::int32_t next_chunk_id = 0;
};

ModelState capture(const Document& document) {
    ModelState state;
    state.active_chunk_id = document.active_chunk_id();
    state.next_chunk_id = document.workspace().next_chunk_id();
    for (const auto& chunk : document.workspace().chunks()) {
        state.chunks.push_back(*chunk);
        state.ids.push_back(chunk->id());
    }
    return state;
}

bool same_state(const ModelState& a, const ModelState& b) {
    if (a.ids != b.ids || a.active_chunk_id != b.active_chunk_id ||
        a.next_chunk_id != b.next_chunk_id ||
        a.chunks.size() != b.chunks.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.chunks.size(); ++i) {
        if (!(a.chunks[i] == b.chunks[i])) {
            return false;
        }
    }
    return true;
}

/// apply -> (validate, state changed) -> revert -> (validate, exact pre-state)
/// -> apply -> (exactly the first post-apply state) -> revert, so the caller
/// gets the document back untouched and can then apply or execute once.
void check_apply_revert_reapply(AbstractCommand& command, Document& document) {
    const ModelState before = capture(document);

    command.apply(document);
    for (const auto& chunk : document.workspace().chunks()) {
        EXPECT_TRUE(chunk->validate().empty())
            << command.name() << ": " << chunk->validate().front();
    }
    const ModelState after = capture(document);
    EXPECT_FALSE(same_state(before, after)) << command.name() << " changed nothing";

    command.revert(document);
    for (const auto& chunk : document.workspace().chunks()) {
        EXPECT_TRUE(chunk->validate().empty())
            << command.name() << ": " << chunk->validate().front();
    }
    EXPECT_TRUE(same_state(before, capture(document)))
        << command.name() << " did not restore the pre-state";

    command.apply(document);
    EXPECT_TRUE(same_state(after, capture(document)))
        << command.name() << " re-apply differs from the first apply";
}

/// Run the full check and leave the document exactly as it was, so the caller
/// can then exercise apply/revert (or the stack) on a clean pre-state.
void check_and_reset(AbstractCommand& command, Document& document) {
    const ModelState before = capture(document);
    check_apply_revert_reapply(command, document);
    command.revert(document);
    EXPECT_TRUE(same_state(before, capture(document)))
        << command.name() << " final revert did not restore the pre-state";
}

// --- chunk commands --------------------------------------------------------

TEST(Undo, AddRemoveChunk) {
    Document document;
    const std::int32_t first = add_chunk(document, "first");
    add_chunk(document, "second");
    UndoStack stack(&document);

    auto add = make_add_chunk_command(document.workspace(), "third", 1);
    check_and_reset(*add, document);
    stack.execute(std::move(add));

    auto& chunks = document.workspace().chunks();
    ASSERT_EQ(chunks.size(), 3U);
    EXPECT_EQ(chunks[1]->label(), "third");
    EXPECT_EQ(document.active_chunk_id(), chunks[1]->id());
    EXPECT_TRUE(chunks[1]->validate().empty());

    stack.undo();
    EXPECT_EQ(document.workspace().chunks().size(), 2U);
    EXPECT_EQ(document.active_chunk_id(), chunks[1]->id());
    // Undo restores the counter, so redo hands the same id out again.
    EXPECT_EQ(document.workspace().next_chunk_id(), 2);

    stack.redo();
    ASSERT_EQ(document.workspace().chunks().size(), 3U);
    EXPECT_EQ(document.workspace().chunks()[1]->label(), "third");
    EXPECT_EQ(document.workspace().next_chunk_id(), 3);

    // Removal of a middle chunk keeps the rest in order; revert puts it back.
    const std::int32_t middle = document.workspace().chunks()[1]->id();
    auto remove = make_remove_chunk_command(middle);
    check_and_reset(*remove, document);
    stack.execute(std::move(remove));
    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    EXPECT_EQ(document.workspace().chunks()[0]->id(), first);
    stack.undo();
    ASSERT_EQ(document.workspace().chunks().size(), 3U);
    EXPECT_EQ(document.workspace().chunks()[1]->id(), middle);

    EXPECT_EQ(document.workspace().chunks()[0]->validate().size(), 0U);
}

TEST(Undo, AddChunkClampsInsertIndexAndRemoveRefusedForLastChunk) {
    Document document;
    const std::int32_t only = add_chunk(document, "only");
    UndoStack stack(&document);

    stack.execute(make_add_chunk_command(document.workspace(), "appended", 99));
    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    EXPECT_EQ(document.workspace().chunks()[1]->label(), "appended");

    stack.undo();
    ASSERT_EQ(document.workspace().chunks().size(), 1U);

    // A document keeps at least one chunk; the command declines before mutating.
    auto remove = make_remove_chunk_command(only);
    const ModelState before = capture(document);
    EXPECT_THROW(remove->apply(document), ValueError);
    EXPECT_TRUE(same_state(before, capture(document)));
}

TEST(Undo, DuplicateChunk) {
    Document document;
    const std::int32_t source = add_chunk(document, "source");
    Chunk& chunk = chunk_of(document, source);
    add_cameras(chunk, 2);
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = "m0";
    chunk.markers().push_back(marker);
    UndoStack stack(&document);

    auto command = make_duplicate_chunk_command(source, "source copy");
    check_and_reset(*command, document);
    stack.execute(std::move(command));

    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    const Chunk& copy = *document.workspace().chunks()[1];
    EXPECT_NE(copy.id(), source);
    EXPECT_EQ(copy.label(), "source copy");
    EXPECT_EQ(copy.cameras().size(), 2U);
    ASSERT_EQ(copy.markers().size(), 1U);
    EXPECT_EQ(copy.markers()[0].label, "m0");
    EXPECT_EQ(copy.next_id(), chunk_of(document, source).next_id());
    EXPECT_EQ(document.active_chunk_id(), copy.id());
    // Distinct objects: editing the copy must not touch the source.
    EXPECT_NE(&copy, &chunk_of(document, source));

    stack.undo();
    ASSERT_EQ(document.workspace().chunks().size(), 1U);
    EXPECT_EQ(document.workspace().next_chunk_id(), 1);
    stack.redo();
    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    EXPECT_EQ(document.workspace().chunks()[1]->label(), "source copy");
}

TEST(Undo, ReorderChunks) {
    Document document;
    const std::int32_t a = add_chunk(document, "a");
    const std::int32_t b = add_chunk(document, "b");
    const std::int32_t c = add_chunk(document, "c");
    UndoStack stack(&document);

    auto command = make_reorder_chunks_command({c, a, b});
    check_and_reset(*command, document);
    stack.execute(std::move(command));

    ASSERT_EQ(document.workspace().chunks().size(), 3U);
    EXPECT_EQ(document.workspace().chunks()[0]->id(), c);
    EXPECT_EQ(document.workspace().chunks()[1]->id(), a);
    EXPECT_EQ(document.workspace().chunks()[2]->id(), b);

    // Labels stay attached to their chunks through the reorder.
    EXPECT_EQ(document.workspace().chunks()[0]->label(), "c");

    stack.undo();
    EXPECT_EQ(document.workspace().chunks()[0]->id(), a);
    EXPECT_EQ(document.workspace().chunks()[2]->id(), c);

    // An order that is not a permutation is refused before mutating.
    auto bad = make_reorder_chunks_command({a, a, b});
    const ModelState before = capture(document);
    EXPECT_THROW(bad->apply(document), ValueError);
    EXPECT_TRUE(same_state(before, capture(document)));

    auto wrong_size = make_reorder_chunks_command({a, b});
    EXPECT_THROW(wrong_size->apply(document), ValueError);

    auto unknown = make_reorder_chunks_command({a, b, 99});
    EXPECT_THROW(unknown->apply(document), ValueError);
    EXPECT_TRUE(same_state(before, capture(document)));
}

// --- markers ---------------------------------------------------------------

TEST(Undo, AddRemoveMarker) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 3);
    UndoStack stack(&document);

    Marker marker;
    marker.label = "m";
    marker.projections.push_back({2, Eigen::Vector2d(10.0, 20.0), true});
    marker.projections.push_back({0, Eigen::Vector2d(1.0, 2.0), true});
    auto add = make_add_marker_command(chunk_id, marker);
    check_and_reset(*add, document);
    stack.execute(std::move(add));

    ASSERT_EQ(chunk.markers().size(), 1U);
    const Marker& added = chunk.markers()[0];
    EXPECT_GE(added.id, 0);
    EXPECT_EQ(added.label, "m");
    // Projections are stored sorted by camera id.
    ASSERT_EQ(added.projections.size(), 2U);
    EXPECT_EQ(added.projections[0].camera_id, 0);
    EXPECT_EQ(added.projections[1].camera_id, 2);
    const std::int32_t marker_id = added.id;

    stack.undo();
    EXPECT_TRUE(chunk.markers().empty());
    stack.redo();
    ASSERT_EQ(chunk.markers().size(), 1U);
    EXPECT_EQ(chunk.markers()[0].id, marker_id);

    auto remove = make_remove_marker_command(chunk_id, marker_id);
    check_and_reset(*remove, document);
    stack.execute(std::move(remove));
    EXPECT_TRUE(chunk.markers().empty());
    stack.undo();
    ASSERT_EQ(chunk.markers().size(), 1U);
    EXPECT_EQ(chunk.markers()[0].label, "m");
}

TEST(Undo, AddMarkerValidatesItsTarget) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 2);

    // Unknown chunk.
    Marker marker;
    marker.label = "m";
    auto no_chunk = make_add_marker_command(99, marker);
    EXPECT_THROW(no_chunk->apply(document), ValueError);

    // Projection into a camera that does not exist.
    Marker bad = marker;
    bad.projections.push_back({7, Eigen::Vector2d::Zero(), true});
    auto bad_camera = make_add_marker_command(chunk_id, bad);
    EXPECT_THROW(bad_camera->apply(document), ValueError);
    EXPECT_TRUE(chunk.markers().empty());

    // Two projections into the same camera violate the container invariant.
    Marker duplicate = marker;
    duplicate.projections.push_back({1, Eigen::Vector2d::Zero(), true});
    duplicate.projections.push_back({1, Eigen::Vector2d(1.0, 1.0), true});
    auto bad_duplicate = make_add_marker_command(chunk_id, duplicate);
    EXPECT_THROW(bad_duplicate->apply(document), ValueError);
    EXPECT_TRUE(chunk.markers().empty());

    // Unsorted input is legal from a caller and is normalised on commit.
    Marker unsorted = marker;
    unsorted.projections.push_back({1, Eigen::Vector2d(3.0, 4.0), true});
    unsorted.projections.push_back({0, Eigen::Vector2d(1.0, 2.0), true});
    auto normalised = make_add_marker_command(chunk_id, unsorted);
    normalised->apply(document);
    ASSERT_EQ(chunk.markers().size(), 1U);
    ASSERT_EQ(chunk.markers()[0].projections.size(), 2U);
    EXPECT_EQ(chunk.markers()[0].projections[0].camera_id, 0);
    EXPECT_EQ(chunk.markers()[0].projections[1].camera_id, 1);
    EXPECT_TRUE(chunk.validate().empty());

    // A non-finite pixel is refused.
    Marker nan_pixel = marker;
    nan_pixel.projections.push_back(
        {0, Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0.0), true});
    EXPECT_THROW(make_add_marker_command(chunk_id, nan_pixel)->apply(document),
                 ValueError);

    // Removing a marker that is not there is refused, not a silent no-op.
    EXPECT_THROW(make_remove_marker_command(chunk_id, 999)->apply(document),
                 ValueError);
    EXPECT_EQ(chunk.markers().size(), 1U);
}

TEST(Undo, MarkerReference) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 1);
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = "ctrl";
    marker.reference = Eigen::Vector3d(10.0, 20.0, 30.0);
    chunk.markers().push_back(marker);
    UndoStack stack(&document);

    auto command = make_set_marker_reference_command(
        chunk_id, marker.id, std::optional<Eigen::Vector3d>(Eigen::Vector3d(1, 2, 3)),
        Eigen::Vector3d(0.1, 0.2, 0.3));
    check_and_reset(*command, document);
    stack.execute(std::move(command));

    ASSERT_EQ(chunk.markers().size(), 1U);
    ASSERT_TRUE(chunk.markers()[0].reference.has_value());
    EXPECT_EQ(*chunk.markers()[0].reference, Eigen::Vector3d(1, 2, 3));
    EXPECT_EQ(chunk.markers()[0].reference_accuracy, Eigen::Vector3d(0.1, 0.2, 0.3));

    stack.undo();
    ASSERT_TRUE(chunk.markers()[0].reference.has_value());
    EXPECT_EQ(*chunk.markers()[0].reference, Eigen::Vector3d(10, 20, 30));

    // Clearing the reference is a legal edit, and revert restores it.
    auto clear = make_set_marker_reference_command(chunk_id, marker.id, std::nullopt,
                                                   Eigen::Vector3d::Constant(0.5));
    check_and_reset(*clear, document);
    clear->apply(document);
    EXPECT_FALSE(chunk.markers()[0].reference.has_value());
    clear->revert(document);
    ASSERT_TRUE(chunk.markers()[0].reference.has_value());

    // A non-positive accuracy is not a valid weight.
    EXPECT_THROW(make_set_marker_reference_command(
                     chunk_id, marker.id, std::optional<Eigen::Vector3d>(),
                     Eigen::Vector3d(0.0, 1.0, 1.0))
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_set_marker_reference_command(
                     chunk_id, 99, std::optional<Eigen::Vector3d>(),
                     Eigen::Vector3d::Constant(1.0))
                     ->apply(document),
                 ValueError);
}

TEST(Undo, MarkerProjectionMove) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 2);
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = "m";
    marker.projections.push_back({1, Eigen::Vector2d(5.0, 6.0), true});
    chunk.markers().push_back(marker);
    UndoStack stack(&document);

    // Moving an existing projection replaces the pixel.
    auto move = make_move_marker_projection_command(chunk_id, marker.id, 1,
                                                    Eigen::Vector2d(7.0, 8.0));
    check_and_reset(*move, document);
    stack.execute(std::move(move));
    ASSERT_EQ(chunk.markers()[0].projections.size(), 1U);
    EXPECT_EQ(chunk.markers()[0].projections[0].pixel, Eigen::Vector2d(7.0, 8.0));
    stack.undo();
    EXPECT_EQ(chunk.markers()[0].projections[0].pixel, Eigen::Vector2d(5.0, 6.0));

    // A projection in a camera that had none is created, sorted into place.
    auto create = make_move_marker_projection_command(chunk_id, marker.id, 0,
                                                      Eigen::Vector2d(1.0, 2.0));
    check_and_reset(*create, document);
    stack.execute(std::move(create));
    ASSERT_EQ(chunk.markers()[0].projections.size(), 2U);
    EXPECT_EQ(chunk.markers()[0].projections[0].camera_id, 0);
    EXPECT_EQ(chunk.markers()[0].projections[1].camera_id, 1);
    stack.undo();
    ASSERT_EQ(chunk.markers()[0].projections.size(), 1U);

    // Unknown camera and unknown marker are refused without mutating.
    EXPECT_THROW(make_move_marker_projection_command(chunk_id, marker.id, 9,
                                                     Eigen::Vector2d::Zero())
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_move_marker_projection_command(chunk_id, 42, 0,
                                                     Eigen::Vector2d::Zero())
                     ->apply(document),
                 ValueError);
}

TEST(Undo, RemoveMarkerProjection) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 3);
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.projections.push_back({0, Eigen::Vector2d(1.0, 1.0), true});
    marker.projections.push_back({1, Eigen::Vector2d(2.0, 2.0), true});
    marker.projections.push_back({2, Eigen::Vector2d(3.0, 3.0), true});
    chunk.markers().push_back(marker);

    auto command = make_remove_marker_projection_command(chunk_id, marker.id, 1);
    check_and_reset(*command, document);
    command->apply(document);
    ASSERT_EQ(chunk.markers()[0].projections.size(), 2U);
    EXPECT_EQ(chunk.markers()[0].projections[0].camera_id, 0);
    EXPECT_EQ(chunk.markers()[0].projections[1].camera_id, 2);
    command->revert(document);
    ASSERT_EQ(chunk.markers()[0].projections.size(), 3U);
    EXPECT_EQ(chunk.markers()[0].projections[1].camera_id, 1);

    // A projection that is not there is a refused edit, not a no-op.
    EXPECT_THROW(make_remove_marker_projection_command(chunk_id, marker.id, 9)
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_remove_marker_projection_command(chunk_id, 99, 0)
                     ->apply(document),
                 ValueError);
    EXPECT_EQ(chunk.markers()[0].projections.size(), 3U);
}

// --- scale bars ------------------------------------------------------------

TEST(Undo, AddScalebar) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 2);
    Marker first;
    first.id = chunk.allocate_id();
    first.label = "a";
    chunk.markers().push_back(first);
    Marker second;
    second.id = chunk.allocate_id();
    second.label = "b";
    chunk.markers().push_back(second);

    Scalebar bar;
    bar.label = "1 m";
    bar.from = {ScalebarEndpoint::Kind::Marker, first.id};
    bar.to = {ScalebarEndpoint::Kind::Marker, second.id};
    bar.reference_length_m = 1.0;
    bar.reference_accuracy_m = 0.01;

    auto command = make_add_scalebar_command(chunk_id, bar);
    check_and_reset(*command, document);
    command->apply(document);
    ASSERT_EQ(chunk.scalebars().size(), 1U);
    EXPECT_EQ(chunk.scalebars()[0].label, "1 m");
    EXPECT_EQ(chunk.scalebars()[0].reference_length_m, 1.0);
    command->revert(document);
    EXPECT_TRUE(chunk.scalebars().empty());

    // A bar whose endpoint does not resolve is refused before mutating.
    Scalebar dangling = bar;
    dangling.from = {ScalebarEndpoint::Kind::Camera, 7};
    EXPECT_THROW(make_add_scalebar_command(chunk_id, dangling)->apply(document),
                 ValueError);
    Scalebar missing_marker = bar;
    missing_marker.to = {ScalebarEndpoint::Kind::Marker, 99};
    EXPECT_THROW(
        make_add_scalebar_command(chunk_id, missing_marker)->apply(document),
        ValueError);
    EXPECT_TRUE(chunk.scalebars().empty());

    // Bars anchored to cameras are legal.
    Scalebar camera_bar = bar;
    camera_bar.from = {ScalebarEndpoint::Kind::Camera, 0};
    camera_bar.to = {ScalebarEndpoint::Kind::Camera, 1};
    auto camera_command = make_add_scalebar_command(chunk_id, camera_bar);
    check_and_reset(*camera_command, document);
}

TEST(Undo, ScalebarReference) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 1);
    Scalebar bar;
    bar.id = chunk.allocate_id();
    bar.label = "s";
    bar.from = {ScalebarEndpoint::Kind::Camera, 0};
    bar.to = {ScalebarEndpoint::Kind::Camera, 0};
    bar.reference_length_m = 2.0;
    bar.reference_accuracy_m = 0.5;
    chunk.scalebars().push_back(bar);

    auto command = make_set_scalebar_reference_command(chunk_id, bar.id, 3.5, 0.02);
    check_and_reset(*command, document);
    command->apply(document);
    EXPECT_EQ(chunk.scalebars()[0].reference_length_m, 3.5);
    EXPECT_EQ(chunk.scalebars()[0].reference_accuracy_m, 0.02);
    command->revert(document);
    EXPECT_EQ(chunk.scalebars()[0].reference_length_m, 2.0);
    EXPECT_EQ(chunk.scalebars()[0].reference_accuracy_m, 0.5);

    EXPECT_THROW(
        make_set_scalebar_reference_command(chunk_id, bar.id, -1.0, 0.01)
            ->apply(document),
        ValueError);
    EXPECT_THROW(
        make_set_scalebar_reference_command(chunk_id, bar.id, 1.0, 0.0)
            ->apply(document),
        ValueError);
    EXPECT_THROW(
        make_set_scalebar_reference_command(chunk_id, 99, 1.0, 0.01)
            ->apply(document),
        ValueError);
    EXPECT_EQ(chunk.scalebars()[0].reference_length_m, 2.0);
}

// --- groups ----------------------------------------------------------------

TEST(Undo, AddGroupAndMoveToGroup) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 4);
    Marker m0;
    m0.id = chunk.allocate_id();
    chunk.markers().push_back(m0);
    Marker m1;
    m1.id = chunk.allocate_id();
    chunk.markers().push_back(m1);

    // Marker group: add, then move markers in, then move one back out.
    auto add_group = make_add_group_command(GroupKind::Marker, chunk_id, "g");
    check_and_reset(*add_group, document);
    add_group->apply(document);
    ASSERT_EQ(chunk.marker_groups().size(), 1U);
    const std::int32_t group_id = chunk.marker_groups()[0].id;
    EXPECT_EQ(chunk.marker_groups()[0].label, "g");
    EXPECT_TRUE(chunk.validate().empty());

    auto move_in = make_move_to_group_command(GroupKind::Marker, chunk_id,
                                              {m0.id, m1.id}, group_id);
    check_and_reset(*move_in, document);
    move_in->apply(document);
    EXPECT_EQ(chunk.marker_groups()[0].members,
              (std::vector<std::int32_t>{m0.id, m1.id}));
    EXPECT_EQ(chunk.markers()[0].group_id, group_id);
    EXPECT_EQ(chunk.markers()[1].group_id, group_id);
    EXPECT_TRUE(chunk.validate().empty());

    auto move_out = make_move_to_group_command(GroupKind::Marker, chunk_id,
                                               {m1.id}, -1);
    check_and_reset(*move_out, document);
    move_out->apply(document);
    EXPECT_EQ(chunk.marker_groups()[0].members, (std::vector<std::int32_t>{m0.id}));
    EXPECT_EQ(chunk.markers()[1].group_id, -1);
    EXPECT_TRUE(chunk.validate().empty());

    // Moving a marker to a second group leaves the first.
    auto second_group =
        make_add_group_command(GroupKind::Marker, chunk_id, "h");
    second_group->apply(document);
    const std::int32_t other_group = chunk.marker_groups()[1].id;
    auto move_across = make_move_to_group_command(GroupKind::Marker, chunk_id,
                                                  {m0.id}, other_group);
    move_across->apply(document);
    EXPECT_TRUE(chunk.marker_groups()[0].members.empty());
    EXPECT_EQ(chunk.marker_groups()[1].members,
              (std::vector<std::int32_t>{m0.id}));
    EXPECT_EQ(chunk.markers()[0].group_id, other_group);
    EXPECT_TRUE(chunk.validate().empty());

    // Removing a group clears its back-references.
    auto remove_group =
        make_remove_group_command(GroupKind::Marker, chunk_id, other_group);
    check_and_reset(*remove_group, document);
    remove_group->apply(document);
    ASSERT_EQ(chunk.marker_groups().size(), 1U);
    EXPECT_EQ(chunk.markers()[0].group_id, -1);
    EXPECT_TRUE(chunk.validate().empty());
    remove_group->revert(document);
    ASSERT_EQ(chunk.marker_groups().size(), 2U);
    EXPECT_EQ(chunk.markers()[0].group_id, other_group);

    EXPECT_THROW(make_move_to_group_command(GroupKind::Marker, chunk_id, {99},
                                            group_id)
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_move_to_group_command(GroupKind::Marker, chunk_id, {m0.id},
                                            99)
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_remove_group_command(GroupKind::Marker, chunk_id, 99)
                     ->apply(document),
                 ValueError);
}

TEST(Undo, CameraGroups) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 3);

    auto add_group = make_add_group_command(GroupKind::Camera, chunk_id, "rig");
    check_and_reset(*add_group, document);
    add_group->apply(document);
    ASSERT_EQ(chunk.camera_groups().size(), 1U);
    const std::int32_t group_id = chunk.camera_groups()[0].id;
    EXPECT_TRUE(chunk.validate().empty());

    auto move = make_move_to_group_command(GroupKind::Camera, chunk_id, {0, 2},
                                           group_id);
    check_and_reset(*move, document);
    move->apply(document);
    EXPECT_EQ(chunk.camera_groups()[0].members, (std::vector<std::int32_t>{0, 2}));
    EXPECT_TRUE(chunk.validate().empty());
    move->revert(document);
    EXPECT_TRUE(chunk.camera_groups()[0].members.empty());

    EXPECT_THROW(make_move_to_group_command(GroupKind::Camera, chunk_id, {9},
                                            group_id)
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_remove_group_command(GroupKind::Camera, chunk_id, 99)
                     ->apply(document),
                 ValueError);

    // Point-cloud and model groups have no backing container yet.
    EXPECT_THROW(
        make_add_group_command(GroupKind::PointCloud, chunk_id, "g")->apply(document),
        ValueError);
    EXPECT_THROW(make_add_group_command(GroupKind::Model, chunk_id, "g")->apply(document),
                 ValueError);
}

// --- cameras ---------------------------------------------------------------

TEST(Undo, CameraEnabledAndSelected) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    add_cameras(chunk, 3);
    UndoStack stack(&document);

    auto disable = make_set_camera_enabled_command(chunk_id, {0, 2}, false);
    check_and_reset(*disable, document);
    stack.execute(std::move(disable));
    EXPECT_FALSE(chunk.cameras()[0].enabled);
    EXPECT_TRUE(chunk.cameras()[1].enabled);
    EXPECT_FALSE(chunk.cameras()[2].enabled);
    stack.undo();
    EXPECT_TRUE(chunk.cameras()[0].enabled);
    EXPECT_TRUE(chunk.cameras()[2].enabled);

    auto select = make_set_camera_selected_command(chunk_id, {1}, true);
    check_and_reset(*select, document);
    stack.execute(std::move(select));
    EXPECT_FALSE(chunk.cameras()[0].selected);
    EXPECT_TRUE(chunk.cameras()[1].selected);
    stack.undo();
    EXPECT_FALSE(chunk.cameras()[1].selected);

    // An unknown camera id is refused before any camera is touched.
    const ModelState before = capture(document);
    EXPECT_THROW(make_set_camera_enabled_command(chunk_id, {0, 9}, false)
                     ->apply(document),
                 ValueError);
    EXPECT_TRUE(same_state(before, capture(document)));
    EXPECT_THROW(make_set_camera_selected_command(99, {0}, true)->apply(document),
                 ValueError);
}

// --- chunk transform and region --------------------------------------------

TEST(Undo, ChunkTransformAndRegion) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");

    Transform transform = Transform::Identity();
    transform(0, 3) = 10.0;
    transform(1, 3) = 20.0;
    Region region;
    region.centre = Eigen::Vector3d(1.0, 2.0, 3.0);
    region.size = Eigen::Vector3d(4.0, 5.0, 6.0);

    auto command = make_set_chunk_transform_command(chunk_id, transform, region);
    check_and_reset(*command, document);
    command->apply(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    EXPECT_EQ(chunk.transform(), transform);
    EXPECT_EQ(chunk.region(), region);
    command->revert(document);
    EXPECT_EQ(chunk.transform(), Transform::Identity());
    EXPECT_EQ(chunk.region(), Region());

    // A non-positive region size and a non-finite transform are illegal.
    Region bad_region = region;
    bad_region.size = Eigen::Vector3d(1.0, 0.0, 1.0);
    EXPECT_THROW(
        make_set_chunk_transform_command(chunk_id, transform, bad_region)
            ->apply(document),
        ValueError);
    Transform bad_transform = transform;
    bad_transform(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        make_set_chunk_transform_command(chunk_id, bad_transform, region)
            ->apply(document),
        ValueError);
    EXPECT_EQ(chunk.transform(), Transform::Identity());

    auto missing = make_set_chunk_transform_command(99, transform, region);
    EXPECT_THROW(missing->apply(document), ValueError);
}

// --- append document -------------------------------------------------------

TEST(Undo, AppendDocument) {
    Document target;
    add_chunk(target, "target");
    const std::int32_t target_id = target.workspace().chunks()[0]->id();

    auto source = std::make_shared<Document>();
    add_chunk(*source, "incoming a");
    add_chunk(*source, "incoming b");
    Chunk& incoming = chunk_of(*source, source->workspace().chunks()[0]->id());
    add_cameras(incoming, 2);
    Marker marker;
    marker.id = incoming.allocate_id();
    marker.label = "m";
    incoming.markers().push_back(marker);

    UndoStack stack(&target);
    auto command = make_append_document_command(source, "/tmp/source.psx");
    check_and_reset(*command, target);
    stack.execute(std::move(command));

    auto& chunks = target.workspace().chunks();
    ASSERT_EQ(chunks.size(), 3U);
    EXPECT_EQ(chunks[0]->id(), target_id);
    // Incoming ids continue the target's counter; chunk-local ids survive.
    EXPECT_EQ(chunks[1]->id(), 1);
    EXPECT_EQ(chunks[2]->id(), 2);
    EXPECT_EQ(chunks[1]->label(), "incoming a");
    EXPECT_EQ(chunks[1]->cameras().size(), 2U);
    ASSERT_EQ(chunks[1]->markers().size(), 1U);
    EXPECT_EQ(chunks[1]->markers()[0].label, "m");
    EXPECT_EQ(target.workspace().next_chunk_id(), 3);
    for (const auto& chunk : chunks) {
        EXPECT_TRUE(chunk->validate().empty());
    }

    stack.undo();
    ASSERT_EQ(target.workspace().chunks().size(), 1U);
    EXPECT_EQ(target.workspace().next_chunk_id(), 1);

    stack.redo();
    ASSERT_EQ(target.workspace().chunks().size(), 3U);
    // A null source is refused.
    EXPECT_THROW(make_append_document_command(nullptr, "")->apply(target),
                 ValueError);
}

/// Scratch directory that removes itself, so the spill test leaves no file in
/// the system temp dir even when it fails mid-way.
class TempDir {
public:
    explicit TempDir(std::string name) {
        std::error_code ec;
        path_ = std::filesystem::temp_directory_path(ec) /
                (ec ? std::string("alphashape-asset-spill") : std::move(name));
        std::filesystem::remove_all(path_, ec);
        std::filesystem::create_directories(path_, ec);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

/// How many regular files a directory currently holds (spill-leak check).
std::size_t count_files(const std::string& dir) {
    std::error_code ec;
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec)) {
            ++count;
        }
    }
    return count;
}

// --- assets ----------------------------------------------------------------

/// A small but non-trivial payload of each kind, so equality after a revert
/// exercises real bytes rather than empty defaults.
PointCloud sample_cloud() {
    PointCloud cloud;
    cloud.points = {Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector3d(-4.0, 5.5, 0.25)};
    cloud.normals = {Eigen::Vector3f(0.0F, 0.0F, 1.0F), Eigen::Vector3f(1.0F, 0.0F, 0.0F)};
    cloud.colours = {10, 20, 30, 40, 50, 60};
    cloud.confidence = {0.5F, 0.9F};
    return cloud;
}

Elevation sample_elevation() {
    Elevation elevation;
    elevation.raster.grid.width = 3;
    elevation.raster.grid.height = 2;
    elevation.raster.grid.origin_x = 500000.0;
    elevation.raster.grid.origin_y = 4649776.0;
    elevation.raster.grid.pixel_size_x = 0.5;
    elevation.raster.grid.pixel_size_y = 0.5;
    elevation.raster.data = {100.0F, 101.0F, 102.0F, 103.0F, 104.0F, 105.0F};
    elevation.set_epsg(32633);
    elevation.source = "depth_maps";
    return elevation;
}

Orthomosaic sample_orthomosaic() {
    Orthomosaic ortho;
    ortho.raster.grid.width = 2;
    ortho.raster.grid.height = 2;
    ortho.raster.grid.pixel_size_x = 0.25;
    ortho.raster.grid.pixel_size_y = 0.25;
    ortho.raster.channels = 4;
    ortho.raster.data = {1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 0, 10, 11, 12, 0};
    ortho.set_epsg(32633);
    ortho.labels = {0, 0, -1, -1};
    ortho.surface = "elevation";
    return ortho;
}

TEST(Undo, AddRemoveAsset) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);
    UndoStack stack(&document);

    // Empty slot, then fill it: the payload is installed by the caller, which
    // is what "wrap a task result" means.
    auto add = make_add_asset_command(AssetKind::Elevation, chunk_id, "DEM");
    check_and_reset(*add, document);
    stack.execute(std::move(add));
    ASSERT_EQ(chunk.elevations().size(), 1U);
    const std::int32_t first = chunk.elevations()[0].id;
    EXPECT_EQ(chunk.elevations()[0].label, "DEM");
    EXPECT_EQ(chunk.active_elevation_id(), first);
    EXPECT_TRUE(chunk.validate().empty());

    // A second add with a payload lands complete in one step.
    auto add_payload = make_add_asset_command(chunk_id, sample_elevation(), "");
    check_and_reset(*add_payload, document);
    stack.execute(std::move(add_payload));
    ASSERT_EQ(chunk.elevations().size(), 2U);
    EXPECT_EQ(chunk.elevations()[1].label, "Elevation 1");
    EXPECT_EQ(*chunk.elevations()[1].data, sample_elevation());
    EXPECT_EQ(chunk.active_elevation_id(), chunk.elevations()[1].id);

    // Removing the active asset moves the active id to the remaining one.
    const std::int32_t second = chunk.elevations()[1].id;
    auto remove = make_remove_asset_command(AssetKind::Elevation, chunk_id, second);
    check_and_reset(*remove, document);
    stack.execute(std::move(remove));
    ASSERT_EQ(chunk.elevations().size(), 1U);
    EXPECT_EQ(chunk.active_elevation_id(), first);
    EXPECT_TRUE(chunk.validate().empty());

    stack.undo();
    ASSERT_EQ(chunk.elevations().size(), 2U);
    EXPECT_EQ(chunk.elevations()[1].id, second);
    EXPECT_EQ(*chunk.elevations()[1].data, sample_elevation());
    EXPECT_EQ(chunk.active_elevation_id(), second);

    stack.redo();
    ASSERT_EQ(chunk.elevations().size(), 1U);
    stack.undo();
    ASSERT_EQ(chunk.elevations().size(), 2U);
    EXPECT_TRUE(chunk.validate().empty());

    // Every kind round-trips the same way.
    for (const AssetKind kind :
         {AssetKind::PointCloud, AssetKind::Model, AssetKind::DepthMaps,
          AssetKind::Orthomosaic, AssetKind::TiledModel, AssetKind::Panorama}) {
        auto add_kind = make_add_asset_command(kind, chunk_id, "");
        check_and_reset(*add_kind, document);
        add_kind->apply(document);
        EXPECT_TRUE(chunk.validate().empty());
    }
    EXPECT_FALSE(chunk.point_clouds().empty());
    EXPECT_FALSE(chunk.models().empty());
    EXPECT_FALSE(chunk.depth_map_sets().empty());
    EXPECT_FALSE(chunk.orthomosaics().empty());
    EXPECT_FALSE(chunk.tiled_models().empty());
    EXPECT_TRUE(chunk.panorama().has_value());

    // Point-cloud payloads survive a remove/revert byte for byte.
    auto add_cloud = make_add_asset_command(chunk_id, sample_cloud(), "cloud");
    add_cloud->apply(document);
    const std::int32_t cloud_id = chunk.point_clouds().back().id;
    EXPECT_EQ(*chunk.point_clouds().back().data, sample_cloud());
    auto remove_cloud =
        make_remove_asset_command(AssetKind::PointCloud, chunk_id, cloud_id);
    remove_cloud->apply(document);
    remove_cloud->revert(document);
    ASSERT_FALSE(chunk.point_clouds().empty());
    EXPECT_EQ(*chunk.point_clouds().back().data, sample_cloud());
    EXPECT_TRUE(chunk.validate().empty());

    EXPECT_THROW(make_remove_asset_command(AssetKind::Elevation, chunk_id, 999)
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_add_asset_command(AssetKind::Panorama, chunk_id, "again")
                     ->apply(document),
                 ValueError);
}

TEST(Undo, SetActiveAsset) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);

    auto first = make_add_asset_command(AssetKind::PointCloud, chunk_id, "a");
    first->apply(document);
    auto second = make_add_asset_command(AssetKind::PointCloud, chunk_id, "b");
    second->apply(document);
    ASSERT_EQ(chunk.point_clouds().size(), 2U);
    const std::int32_t a = chunk.point_clouds()[0].id;
    const std::int32_t b = chunk.point_clouds()[1].id;
    EXPECT_EQ(chunk.active_point_cloud_id(), b);

    auto set_active = make_set_active_asset_command(AssetKind::PointCloud, chunk_id, a);
    check_and_reset(*set_active, document);
    set_active->apply(document);
    EXPECT_EQ(chunk.active_point_cloud_id(), a);
    // The legacy accessor follows the active id.
    EXPECT_EQ(&chunk.point_cloud(), chunk.point_clouds()[0].data.get());
    EXPECT_TRUE(chunk.validate().empty());

    set_active->revert(document);
    EXPECT_EQ(chunk.active_point_cloud_id(), b);
    EXPECT_THROW(make_set_active_asset_command(AssetKind::PointCloud, chunk_id, 999)
                     ->apply(document),
                 ValueError);

    // A removed active asset leaves the remaining one active; revert restores.
    auto remove = make_remove_asset_command(AssetKind::PointCloud, chunk_id, b);
    remove->apply(document);
    EXPECT_EQ(chunk.active_point_cloud_id(), a);
    remove->revert(document);
    EXPECT_EQ(chunk.active_point_cloud_id(), b);
    EXPECT_TRUE(chunk.validate().empty());
}

TEST(Undo, RenameAsset) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);

    auto add = make_add_asset_command(AssetKind::Model, chunk_id, "old");
    add->apply(document);
    const std::int32_t model_id = chunk.models()[0].id;

    auto rename = make_rename_asset_command(AssetKind::Model, chunk_id, model_id,
                                            "new");
    check_and_reset(*rename, document);
    rename->apply(document);
    EXPECT_EQ(chunk.models()[0].label, "new");
    EXPECT_TRUE(chunk.validate().empty());
    rename->revert(document);
    EXPECT_EQ(chunk.models()[0].label, "old");

    EXPECT_THROW(make_rename_asset_command(AssetKind::Model, chunk_id, model_id, "")
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_rename_asset_command(AssetKind::Model, chunk_id, 999, "x")
                     ->apply(document),
                 ValueError);

    // Panorama rename follows the same contract over its single slot.
    auto add_pano = make_add_asset_command(AssetKind::Panorama, chunk_id, "pano");
    add_pano->apply(document);
    auto rename_pano = make_rename_asset_command(AssetKind::Panorama, chunk_id,
                                                 chunk.panorama()->id, "360");
    check_and_reset(*rename_pano, document);
    rename_pano->apply(document);
    EXPECT_EQ(chunk.panorama()->label, "360");
    rename_pano->revert(document);
    EXPECT_EQ(chunk.panorama()->label, "pano");
}

TEST(Undo, AssetPayloadSpillsWhenLarge) {
    // Release only (§4.7): a Debug build's allocator and the 75 MB payload make
    // this the slowest test in the suite by a wide margin.
#if defined(NDEBUG)
    TempDir dir("alphashape-asset-spill");
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);

    Orthomosaic ortho = sample_orthomosaic();
    // 5000 x 5000 x 4 = 100 MB of pixels, above the 64 MiB cap.
    ortho.raster.grid.width = 5000;
    ortho.raster.grid.height = 5000;
    ortho.raster.channels = 4;
    ortho.raster.data.assign(
        static_cast<std::size_t>(ortho.raster.grid.width) *
            static_cast<std::size_t>(ortho.raster.grid.height) * 4U,
        static_cast<std::uint8_t>(0));
    for (std::size_t i = 0; i < ortho.raster.data.size(); i += 4093) {
        ortho.raster.data[i] = static_cast<std::uint8_t>(i % 251);
    }
    ortho.labels.assign(
        static_cast<std::size_t>(ortho.raster.grid.width) *
            static_cast<std::size_t>(ortho.raster.grid.height),
        0);
    ASSERT_GT(ortho.raster.data.size(), kUndoAssetSpillBytes);

    auto add = make_add_asset_command(chunk_id, ortho, "huge");
    add->apply(document);
    const std::int32_t ortho_id = chunk.orthomosaics().back().id;
    const std::vector<std::uint8_t> expected = ortho.raster.data;

    auto remove = make_remove_asset_command(AssetKind::Orthomosaic, chunk_id, ortho_id);
    UndoStack stack(&document);
    stack.set_cache_dir(dir.path());
    stack.execute(std::move(remove));
    ASSERT_TRUE(chunk.orthomosaics().empty());

    stack.undo();
    ASSERT_EQ(chunk.orthomosaics().size(), 1U);
    ASSERT_EQ(chunk.orthomosaics().size(), 1U);
    const Orthomosaic& restored = *chunk.orthomosaics()[0].data;
    EXPECT_EQ(restored.raster.data, expected);
    EXPECT_EQ(restored.labels.size(), ortho.labels.size());
    EXPECT_EQ(restored.epsg(), 32633);
    EXPECT_EQ(restored.surface, "elevation");
    EXPECT_TRUE(chunk.validate().empty());

    // The payload lives on disk while the undo entry is retained...
    const std::size_t spilled = count_files(dir.path());
    EXPECT_EQ(spilled, 1U);

    // ...and no spill file outlives the command that owned it.
    stack.clear();
    EXPECT_EQ(count_files(dir.path()), 0U);

    // Panorama takes the same path: a 100 MB image spills on removal and
    // comes back byte-identical.
    Image big(5000, 5000, 3);
    for (std::size_t i = 0; i < big.size_bytes(); i += 4097) {
        big.data()[i] = static_cast<std::uint8_t>(i % 241);
    }
    ASSERT_GT(big.size_bytes(), kUndoAssetSpillBytes);
    const std::vector<std::uint8_t> pixels(big.data(), big.data() + big.size_bytes());
    auto add_pano = make_add_asset_command(chunk_id, std::move(big), "huge 360");
    add_pano->apply(document);
    ASSERT_TRUE(chunk.panorama().has_value());
    auto remove_pano = make_remove_asset_command(AssetKind::Panorama, chunk_id,
                                                 chunk.panorama()->id);
    stack.execute(std::move(remove_pano));
    EXPECT_FALSE(chunk.panorama().has_value());
    EXPECT_EQ(count_files(dir.path()), 1U);
    stack.undo();
    ASSERT_TRUE(chunk.panorama().has_value());
    ASSERT_EQ(chunk.panorama()->data->size_bytes(), pixels.size());
    EXPECT_EQ(std::memcmp(chunk.panorama()->data->data(), pixels.data(),
                          pixels.size()),
              0);
    stack.clear();
    EXPECT_EQ(count_files(dir.path()), 0U);

    // Without a cache directory the same command is refused *before* the
    // document changes (§4.7, "pre-decided"), and leaves no spill behind.
    auto remove_no_cache =
        make_remove_asset_command(AssetKind::Orthomosaic, chunk_id,
                                  chunk.orthomosaics()[0].id);
    UndoStack uncached(&document);
    EXPECT_THROW(uncached.execute(std::move(remove_no_cache)), Error);
    EXPECT_EQ(chunk.orthomosaics().size(), 1U);
    EXPECT_EQ(count_files(dir.path()), 0U);
#else
    GTEST_SKIP() << "spill test runs in Release only";
#endif
}

TEST(Undo, SetChunkCrs) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document, "chunk");
    Chunk& chunk = chunk_of(document, chunk_id);

    CRS crs;
    crs.epsg = 32633;
    crs.wkt = R"(PROJCS["WGS 84 / UTM zone 33N"])";
    auto command = make_set_chunk_crs_command(chunk_id, crs);
    check_and_reset(*command, document);
    command->apply(document);
    EXPECT_EQ(chunk.crs(), crs);
    EXPECT_TRUE(chunk.validate().empty());
    command->revert(document);
    EXPECT_FALSE(chunk.crs().is_set());

    EXPECT_THROW(make_set_chunk_crs_command(chunk_id, CRS{})->apply(document),
                 ValueError);
}

// --- validation of missing ids everywhere ----------------------------------

TEST(Undo, EveryFactoryRejectsAMissingChunk) {
    Document document;
    add_chunk(document, "only");
    Marker marker;
    Scalebar bar;
    const std::vector<std::unique_ptr<AbstractCommand>> commands =
        [&marker, &bar] {
        std::vector<std::unique_ptr<AbstractCommand>> out;
        out.push_back(make_add_marker_command(77, marker));
        out.push_back(make_remove_marker_command(77, 0));
        out.push_back(make_set_marker_reference_command(
            77, 0, std::optional<Eigen::Vector3d>(), Eigen::Vector3d::Constant(1.0)));
        out.push_back(make_move_marker_projection_command(
            77, 0, 0, Eigen::Vector2d::Zero()));
        out.push_back(make_remove_marker_projection_command(77, 0, 0));
        out.push_back(make_add_scalebar_command(77, bar));
        out.push_back(make_remove_scalebar_command(77, 0));
        out.push_back(make_set_scalebar_reference_command(77, 0, 1.0, 0.01));
        out.push_back(make_add_group_command(GroupKind::Marker, 77, "g"));
        out.push_back(make_remove_group_command(GroupKind::Marker, 77, 0));
        out.push_back(
            make_move_to_group_command(GroupKind::Marker, 77, {0}, -1));
        out.push_back(make_set_camera_enabled_command(77, {0}, false));
        out.push_back(make_set_camera_selected_command(77, {0}, true));
        out.push_back(make_remove_chunk_command(77));
        out.push_back(make_duplicate_chunk_command(77, "copy"));
        out.push_back(make_set_chunk_transform_command(77, Transform::Identity(),
                                                       Region()));
        out.push_back(make_set_chunk_crs_command(77, CRS{}));
        out.push_back(make_add_asset_command(AssetKind::PointCloud, 77, "cloud"));
        out.push_back(make_add_asset_command(77, PointCloud(), "cloud"));
        out.push_back(make_add_asset_command(77, Elevation(), "dem"));
        out.push_back(make_add_asset_command(77, Orthomosaic(), "ortho"));
        out.push_back(make_add_asset_command(AssetKind::Panorama, 77, "pano"));
        out.push_back(make_remove_asset_command(AssetKind::Model, 77, 0));
        out.push_back(make_set_active_asset_command(AssetKind::Model, 77, 0));
        out.push_back(make_rename_asset_command(AssetKind::Model, 77, 0, "x"));
        return out;
    }();

    const ModelState before = capture(document);
    for (const auto& command : commands) {
        EXPECT_THROW(command->apply(document), ValueError) << command->name();
        EXPECT_TRUE(same_state(before, capture(document))) << command->name();
    }
    EXPECT_TRUE(document.workspace().chunks()[0]->validate().empty());
}

}  // namespace
}  // namespace alphashape

