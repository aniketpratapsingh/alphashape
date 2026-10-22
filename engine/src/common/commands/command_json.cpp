#include "common/commands/command_json.hpp"

#include <cstddef>
#include <string>

namespace alphashape {

std::string dump_undo_stack(const UndoStack& stack) {
    const std::size_t position = stack.position();
    std::string out = "undo stack (" + std::to_string(position) + "/" +
                      std::to_string(stack.size()) + ", dropped " +
                      std::to_string(stack.dropped_count()) + ")\n";
    const std::vector<std::string> names = stack.command_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        out += "  [";
        out += std::to_string(i);
        out += "] ";
        if (i == position) {
            out += "* ";
        }
        out += names[i];
        out += '\n';
    }
    return out;
}

}  // namespace alphashape
