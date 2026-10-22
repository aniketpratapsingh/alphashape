#pragma once

// A zlib-backed streaming ZIP reader/writer.
//
// We do not write our own zip: the format is standard, so this wraps zlib and the
// documented local-file-header / central-directory layout. Scope for v0:
// store and deflate entries, no encryption, no Zip64 (projects under 4 GiB
// and entries under 4 GiB). Zip64 is a later milestone — noted for honesty.

#include <cstdint>
#include <string>
#include <vector>

namespace alphashape {

struct ZipEntry {
    std::string name;
    std::vector<std::uint8_t> data;
};

/// A zip archive held in memory. The project container is written whole, so
/// streaming to disk incrementally buys nothing at v0 scale.
class ZipArchive {
public:
    ZipArchive() = default;

    /// Add or replace an entry.
    void add(std::string name, const std::vector<std::uint8_t>& data);
    void add(std::string name, const std::string& text);

    bool contains(const std::string& name) const;
    const std::vector<std::uint8_t>& get(const std::string& name) const;
    std::string get_text(const std::string& name) const;

    const std::vector<ZipEntry>& entries() const { return entries_; }

    /// Serialise to zip bytes (deflate compression).
    std::vector<std::uint8_t> serialise() const;

    /// Parse zip bytes. Throws ProjectIOError on a malformed archive.
    static ZipArchive parse(const std::vector<std::uint8_t>& bytes);

private:
    std::vector<ZipEntry> entries_;
};

}  // namespace alphashape
