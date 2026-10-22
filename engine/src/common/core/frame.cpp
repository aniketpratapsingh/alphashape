#include "common/core/frame.hpp"

namespace alphashape {

Frame::Frame(std::string path, int width, int height)
    : path_(std::move(path)), width_(width), height_(height) {}

}  // namespace alphashape
