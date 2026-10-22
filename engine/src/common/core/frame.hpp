#pragma once

// Frame: one input image. A camera references a frame; the frame holds where
// the pixels came from. Frames serialise to /frame.zip inside the project
// container.
//
// v0 stores the reference and geometry only — no pixel decode. Real image
// loading arrives with the image IO milestone; keeping Frame free of pixel
// storage here means the detector can be tested against synthetic images
// without pulling a codec into the core.

#include <cstdint>
#include <string>

#include "common/core/types.hpp"

namespace alphashape {

class Frame {
public:
    Frame() = default;
    Frame(std::string path, int width, int height);

    std::int32_t id() const { return id_; }
    void set_id(std::int32_t id) { id_ = id; }

    /// Path as recorded in the project. May be absolute or relative; the
    /// project format stores a Project/AbsolutePaths flag alongside, which we
    /// write too.
    const std::string& path() const { return path_; }
    void set_path(std::string path) { path_ = std::move(path); }

    int width() const { return width_; }
    int height() const { return height_; }
    void set_size(int width, int height) {
        width_ = width;
        height_ = height;
    }

    /// Which camera views this frame. -1 until linked.
    std::int32_t camera_id() const { return camera_id_; }
    void set_camera_id(std::int32_t id) { camera_id_ = id; }

    bool operator==(const Frame& other) const {
        return id_ == other.id_ && path_ == other.path_ &&
               width_ == other.width_ && height_ == other.height_ &&
               camera_id_ == other.camera_id_;
    }

private:
    std::int32_t id_ = -1;
    std::string path_;
    int width_ = 0;
    int height_ = 0;
    std::int32_t camera_id_ = -1;
};

}  // namespace alphashape
