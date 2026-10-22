#include "common/core/assets.hpp"

#include <cstddef>

namespace alphashape {
namespace asset_detail {

bool images_equal(const Image& a, const Image& b) {
    if (a.width() != b.width() || a.height() != b.height() ||
        a.channels() != b.channels()) {
        return false;
    }
    if (a.size_bytes() != b.size_bytes()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size_bytes(); ++i) {
        if (a.data()[i] != b.data()[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace asset_detail
}  // namespace alphashape
