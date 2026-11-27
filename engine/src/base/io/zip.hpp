#pragma once

// A zlib-backed streaming ZIP reader/writer.
//
// We do not write our own zip: the format is standard, so this wraps zlib and the
// documented local-file-header / central-directory layout. Scope: store and
// deflate entries, no encryption. Zip64 (PKWARE APPNOTE 4.5.3) is emitted only
// when a size, offset or entry count does not fit the classic 32/16-bit fields,
// so an ordinary archive stays byte-for-byte what it was before Zip64 existed.

#include <cstdint>
#include <string>
#include <vector>

namespace alphashape {

struct ZipEntry {
    std::string name;
    std::vector<std::uint8_t> data;
};

/// Writer knobs. The defaults produce the classic archive; nothing below changes
/// the output unless a value actually overflows its 32/16-bit field.
struct ZipWriteOptions {
    /// Emit the Zip64 extended-information extra field for every entry and the
    /// Zip64 end-of-central-directory records, even when the archive would fit
    /// the classic fields. Tests use this to exercise the Zip64 paths without
    /// allocating a 4 GiB entry; applications never set it.
    bool force_zip64 = false;
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

    /// Serialise to zip bytes (deflate compression). Zip64 structures are added
    /// only where a field overflows, unless `options.force_zip64` asks for them.
    std::vector<std::uint8_t> serialise(const ZipWriteOptions& options = {}) const;

    /// Parse zip bytes. Throws ProjectIOError on a malformed archive.
    static ZipArchive parse(const std::vector<std::uint8_t>& bytes);

private:
    std::vector<ZipEntry> entries_;
};

}  // namespace alphashape
