#include "base/utils/error.hpp"

namespace alphashape {

UnsupportedTaskError::UnsupportedTaskError(const std::string& task)
    : Error("Unsupported task: " + task) {}

InvalidArgumentError::InvalidArgumentError(const std::string& name)
    : Error("Invalid argument name: " + name) {}

}  // namespace alphashape
