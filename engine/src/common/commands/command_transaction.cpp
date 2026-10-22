// CommandTransaction and SpillBuffer: the composite command and the §4.7
// memory policy for replaced asset payloads.

#include "common/commands/command.hpp"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <process.h>
#define ALPHASHAPE_GETPID _getpid
#else
#include <unistd.h>
#define ALPHASHAPE_GETPID getpid
#endif

namespace alphashape {

namespace {

/// Header for a spilled payload, so a truncated or foreign file is detected
/// rather than silently restored as garbage.
constexpr char kSpillMagic[4] = {'A', 'U', 'S', 'P'};
constexpr std::uint32_t kSpillVersion = 1;

void write_u32(std::ofstream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint32_t read_u32(std::ifstream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

/// Unique spill-file name: pid plus a process-wide atomic counter, so two
/// buffers created in the same instant never collide. (A clock is never used:
/// two captures in the same millisecond are entirely normal.)
std::string make_spill_path(const std::string& cache_dir) {
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
    std::error_code ec;
    std::filesystem::path dir(cache_dir);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw Error("cannot create undo spill directory '" + cache_dir + "': " + ec.message());
    }
    std::filesystem::path path =
        dir / ("undo-" + std::to_string(static_cast<long long>(ALPHASHAPE_GETPID())) + "-" +
               std::to_string(static_cast<unsigned long long>(n)) + ".img");
    return path.string();
}

}  // namespace

// --- CommandTransaction ----------------------------------------------------

CommandTransaction::CommandTransaction(std::string name) : name_(std::move(name)) {}

void CommandTransaction::add(std::unique_ptr<AbstractCommand> command) {
    if (!command) {
        throw ValueError("cannot add a null command to a transaction");
    }
    commands_.push_back(std::move(command));
}

void CommandTransaction::on_prepare(const UndoContext& context) {
    for (auto& command : commands_) {
        command->on_prepare(context);
    }
}

bool CommandTransaction::is_memory_bounded() const {
    for (const auto& command : commands_) {
        if (!command->is_memory_bounded()) {
            return false;
        }
    }
    return true;
}

std::size_t CommandTransaction::retained_bytes() const {
    std::size_t total = 0;
    for (const auto& command : commands_) {
        total += command->retained_bytes();
    }
    return total;
}

void CommandTransaction::apply(Document& document) {
    // Forward: children in order. On failure undo the children already applied,
    // so a half-applied transaction never reaches the stack.
    std::size_t applied = 0;
    try {
        for (; applied < commands_.size(); ++applied) {
            commands_[applied]->apply(document);
        }
    } catch (...) {
        while (applied > 0) {
            --applied;
            try {
                commands_[applied]->revert(document);
            } catch (...) {
                // A revert that throws here means the document is already
                // inconsistent; keep unwinding so the original error surfaces.
            }
        }
        throw;
    }
}

void CommandTransaction::revert(Document& document) {
    // Reverse order: the inverse of apply.
    for (std::size_t i = commands_.size(); i > 0; --i) {
        commands_[i - 1]->revert(document);
    }
}

// --- SpillBuffer -----------------------------------------------------------

SpillBuffer::SpillBuffer(SpillBuffer&& other) noexcept
    : image_(std::move(other.image_)),
      spill_path_(std::move(other.spill_path_)),
      bytes_(other.bytes_),
      valid_(other.valid_) {
    other.spill_path_.clear();
    other.bytes_ = 0;
    other.valid_ = true;
}

SpillBuffer& SpillBuffer::operator=(SpillBuffer&& other) noexcept {
    if (this != &other) {
        release();
        image_ = std::move(other.image_);
        spill_path_ = std::move(other.spill_path_);
        bytes_ = other.bytes_;
        valid_ = other.valid_;
        other.spill_path_.clear();
        other.bytes_ = 0;
        other.valid_ = true;
    }
    return *this;
}

SpillBuffer::~SpillBuffer() { release(); }

SpillBuffer SpillBuffer::capture(Image image, const std::string& cache_dir) {
    SpillBuffer buffer;
    buffer.bytes_ = image.size_bytes();
    if (buffer.bytes_ <= kUndoAssetSpillBytes) {
        buffer.image_ = std::move(image);
        return buffer;
    }

    if (cache_dir.empty()) {
        // Above the cap with nowhere to spill: the command must be refused.
        buffer.valid_ = false;
        buffer.bytes_ = 0;
        return buffer;
    }

    buffer.spill_path_ = make_spill_path(cache_dir);
    std::ofstream out(buffer.spill_path_, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw Error("cannot open undo spill file '" + buffer.spill_path_ + "'");
    }
    const std::uint32_t width = static_cast<std::uint32_t>(image.width());
    const std::uint32_t height = static_cast<std::uint32_t>(image.height());
    const std::uint32_t channels = static_cast<std::uint32_t>(image.channels());
    out.write(kSpillMagic, sizeof(kSpillMagic));
    write_u32(out, kSpillVersion);
    write_u32(out, width);
    write_u32(out, height);
    write_u32(out, channels);
    out.write(reinterpret_cast<const char*>(image.data()),
              static_cast<std::streamsize>(image.size_bytes()));
    out.close();
    if (!out) {
        // Partial write: never leave a file that restore() would misread.
        std::remove(buffer.spill_path_.c_str());
        buffer.spill_path_.clear();
        throw Error("cannot write undo spill file");
    }
    return buffer;
}

Image SpillBuffer::restore() const {
    if (!spill_path_.empty()) {
        std::ifstream in(spill_path_, std::ios::binary);
        if (!in) {
            throw Error("undo spill file is missing: " + spill_path_);
        }
        char magic[sizeof(kSpillMagic)] = {};
        in.read(magic, sizeof(magic));
        if (!in || std::string(magic, sizeof(magic)) != std::string(kSpillMagic, sizeof(kSpillMagic))) {
            throw Error("undo spill file is corrupt: " + spill_path_);
        }
        if (read_u32(in) != kSpillVersion) {
            throw Error("undo spill file has an unknown version: " + spill_path_);
        }
        const int width = static_cast<int>(read_u32(in));
        const int height = static_cast<int>(read_u32(in));
        const int channels = static_cast<int>(read_u32(in));
        if (!in || width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
            throw Error("undo spill file has a bad header: " + spill_path_);
        }
        Image image(width, height, channels);
        in.read(reinterpret_cast<char*>(image.data()),
                static_cast<std::streamsize>(image.size_bytes()));
        if (!in) {
            throw Error("undo spill file is truncated: " + spill_path_);
        }
        return image;
    }
    return image_;
}

void SpillBuffer::release() {
    if (!spill_path_.empty()) {
        std::remove(spill_path_.c_str());
        spill_path_.clear();
    }
    image_ = Image();
    bytes_ = 0;
    valid_ = true;
}

}  // namespace alphashape
