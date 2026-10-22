#pragma once

// Debug dump of undo-stack state, for tests and crash reports (WP-11.5a).
//
// Text only: it is read by a human looking at a failing assertion, and by tests
// that want to assert the stack shape without reaching into private members.
// Deliberately not a serializer — nothing here is ever parsed back.

#include <string>

#include "common/commands/command.hpp"

namespace alphashape {

/// One line per retained entry, oldest first, followed by the position marker.
/// Format:
///
///   undo stack (3/4, dropped 1)
///     [0] Add Marker
///     [1] Move Marker Projection
///     [2] Set Chunk Label
///     [3] * Remove Marker
///
/// The `*` marks the next command to redo; everything before it is applied.
std::string dump_undo_stack(const UndoStack& stack);

}  // namespace alphashape
