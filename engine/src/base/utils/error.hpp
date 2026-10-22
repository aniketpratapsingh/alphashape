#pragma once

// Foundation layer: no photogrammetry concepts.

#include <stdexcept>
#include <string>

namespace alphashape {

/// Base class for all engine errors, so callers can catch one type.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

/// A requested task name is not in the registry.
/// Message: "Unsupported task: <name>".
class UnsupportedTaskError : public Error {
public:
    explicit UnsupportedTaskError(const std::string& task);
};

/// A parameter key was supplied that the task does not declare.
/// Message: "Invalid argument name: <name>".
class InvalidArgumentError : public Error {
public:
    explicit InvalidArgumentError(const std::string& name);
};

/// A value could not be parsed or is out of range for its declared type.
class ValueError : public Error {
public:
    explicit ValueError(const std::string& what) : Error(what) {}
};

/// Project file IO failed (bad container, missing doc.xml, corrupt zip).
class ProjectIOError : public Error {
public:
    explicit ProjectIOError(const std::string& what) : Error(what) {}
};

}  // namespace alphashape
