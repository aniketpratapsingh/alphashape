#include "base/io/zip.hpp"

#include <cstring>
#include <limits>

#include <zlib.h>

#include "base/utils/error.hpp"

// Layout follows the PKWARE APPNOTE: per entry a local file header + data,
// then a central directory, then the end-of-central-directory record. Method 8
// is raw deflate (zlib with negative window bits). Integers are little-endian.

namespace alphashape {

namespace {

constexpr std::uint32_t kLocalHeaderSig = 0x04034b50;
constexpr std::uint32_t kCentralHeaderSig = 0x02014b50;
constexpr std::uint32_t kEndOfCentralSig = 0x06054b50;
constexpr std::uint16_t kVersion = 20;       // 2.0: deflate
constexpr std::uint16_t kMethodStore = 0;
constexpr std::uint16_t kMethodDeflate = 8;
constexpr std::uint16_t kDosDate = 0x21;     // 1980-01-01: fixed, so output is reproducible

void put16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8U) & 0xFFU));
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}

class Reader {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    void seek(std::size_t pos) {
        if (pos > bytes_.size()) {
            throw ProjectIOError("zip: offset out of range");
        }
        pos_ = pos;
    }
    std::size_t pos() const { return pos_; }

    std::uint16_t u16() {
        need(2);
        const auto v = static_cast<std::uint16_t>(bytes_[pos_] | (bytes_[pos_ + 1] << 8U));
        pos_ += 2;
        return v;
    }
    std::uint32_t u32() {
        need(4);
        std::uint32_t v = 0;
        for (unsigned i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(bytes_[pos_ + i]) << (8U * i);
        }
        pos_ += 4;
        return v;
    }
    std::string str(std::size_t n) {
        need(n);
        std::string s(reinterpret_cast<const char*>(bytes_.data() + pos_), n);
        pos_ += n;
        return s;
    }
    const std::uint8_t* take(std::size_t n) {
        need(n);
        const std::uint8_t* p = bytes_.data() + pos_;
        pos_ += n;
        return p;
    }
    void skip(std::size_t n) {
        need(n);
        pos_ += n;
    }

private:
    void need(std::size_t n) const {
        if (n > bytes_.size() - pos_) {
            throw ProjectIOError("zip: truncated archive");
        }
    }
    const std::vector<std::uint8_t>& bytes_;
    std::size_t pos_ = 0;
};

std::uint32_t checked32(std::size_t v, const char* what) {
    if (v > std::numeric_limits<std::uint32_t>::max()) {
        throw ProjectIOError(std::string("zip: ") + what + " exceeds 4 GiB (Zip64 not supported yet)");
    }
    return static_cast<std::uint32_t>(v);
}

std::uint16_t checked16(std::size_t v, const char* what) {
    if (v > std::numeric_limits<std::uint16_t>::max()) {
        throw ProjectIOError(std::string("zip: ") + what + " exceeds 65535 (Zip64 not supported yet)");
    }
    return static_cast<std::uint16_t>(v);
}

std::uint32_t crc_of(const std::uint8_t* data, std::size_t size) {
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, data, checked32(size, "entry"));
    return static_cast<std::uint32_t>(crc);
}

std::vector<std::uint8_t> deflate_raw(const std::vector<std::uint8_t>& in) {
    z_stream zs{};
    if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
        throw ProjectIOError("zip: deflateInit failed");
    }
    std::vector<std::uint8_t> out(deflateBound(&zs, checked32(in.size(), "entry")));
    zs.next_in = const_cast<Bytef*>(in.data());
    zs.avail_in = checked32(in.size(), "entry");
    zs.next_out = out.data();
    zs.avail_out = checked32(out.size(), "entry");
    const int rc = deflate(&zs, Z_FINISH);
    const std::size_t produced = zs.total_out;
    deflateEnd(&zs);
    if (rc != Z_STREAM_END) {
        throw ProjectIOError("zip: deflate failed");
    }
    out.resize(produced);
    return out;
}

std::vector<std::uint8_t> inflate_raw(const std::uint8_t* in, std::size_t in_size,
                                      std::size_t out_size) {
    std::vector<std::uint8_t> out(out_size);
    z_stream zs{};
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
        throw ProjectIOError("zip: inflateInit failed");
    }
    zs.next_in = const_cast<Bytef*>(in);
    zs.avail_in = checked32(in_size, "entry");
    zs.next_out = out.data();
    zs.avail_out = checked32(out_size, "entry");
    const int rc = inflate(&zs, Z_FINISH);
    const std::size_t produced = zs.total_out;
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || produced != out_size) {
        throw ProjectIOError("zip: corrupt deflate stream");
    }
    return out;
}

}  // namespace

void ZipArchive::add(std::string name, const std::vector<std::uint8_t>& data) {
    for (auto& entry : entries_) {
        if (entry.name == name) {
            entry.data = data;
            return;
        }
    }
    entries_.push_back(ZipEntry{std::move(name), data});
}

void ZipArchive::add(std::string name, const std::string& text) {
    add(std::move(name), std::vector<std::uint8_t>(text.begin(), text.end()));
}

bool ZipArchive::contains(const std::string& name) const {
    for (const auto& entry : entries_) {
        if (entry.name == name) {
            return true;
        }
    }
    return false;
}

const std::vector<std::uint8_t>& ZipArchive::get(const std::string& name) const {
    for (const auto& entry : entries_) {
        if (entry.name == name) {
            return entry.data;
        }
    }
    throw ProjectIOError("zip: no entry named '" + name + "'");
}

std::string ZipArchive::get_text(const std::string& name) const {
    const auto& data = get(name);
    return std::string(data.begin(), data.end());
}

std::vector<std::uint8_t> ZipArchive::serialise() const {
    std::vector<std::uint8_t> out;
    std::vector<std::uint8_t> central;

    for (const auto& entry : entries_) {
        const std::uint32_t crc = crc_of(entry.data.data(), entry.data.size());
        std::vector<std::uint8_t> packed = deflate_raw(entry.data);
        std::uint16_t method = kMethodDeflate;
        if (packed.size() >= entry.data.size()) {
            packed = entry.data;  // incompressible: store instead
            method = kMethodStore;
        }
        const std::uint32_t offset = checked32(out.size(), "archive");
        const auto name_len = checked16(entry.name.size(), "entry name length");

        put32(out, kLocalHeaderSig);
        put16(out, kVersion);
        put16(out, 0);  // flags
        put16(out, method);
        put16(out, 0);  // time
        put16(out, kDosDate);
        put32(out, crc);
        put32(out, checked32(packed.size(), "entry"));
        put32(out, checked32(entry.data.size(), "entry"));
        put16(out, name_len);
        put16(out, 0);  // extra length
        out.insert(out.end(), entry.name.begin(), entry.name.end());
        out.insert(out.end(), packed.begin(), packed.end());

        put32(central, kCentralHeaderSig);
        put16(central, kVersion);  // made by
        put16(central, kVersion);  // needed
        put16(central, 0);
        put16(central, method);
        put16(central, 0);
        put16(central, kDosDate);
        put32(central, crc);
        put32(central, checked32(packed.size(), "entry"));
        put32(central, checked32(entry.data.size(), "entry"));
        put16(central, name_len);
        put16(central, 0);  // extra
        put16(central, 0);  // comment
        put16(central, 0);  // disk
        put16(central, 0);  // internal attrs
        put32(central, 0);  // external attrs
        put32(central, offset);
        central.insert(central.end(), entry.name.begin(), entry.name.end());
    }

    const std::uint32_t central_offset = checked32(out.size(), "archive");
    out.insert(out.end(), central.begin(), central.end());
    const auto count = checked16(entries_.size(), "entry count");
    put32(out, kEndOfCentralSig);
    put16(out, 0);
    put16(out, 0);
    put16(out, count);
    put16(out, count);
    put32(out, checked32(central.size(), "central directory"));
    put32(out, central_offset);
    put16(out, 0);  // comment length
    return out;
}

ZipArchive ZipArchive::parse(const std::vector<std::uint8_t>& bytes) {
    // The end record sits in the last 22 + (up to 65535 comment) bytes.
    constexpr std::size_t kEocdSize = 22;
    if (bytes.size() < kEocdSize) {
        throw ProjectIOError("zip: file too small");
    }
    std::size_t eocd = std::string::npos;
    const std::size_t lowest = bytes.size() > kEocdSize + 65535 ? bytes.size() - kEocdSize - 65535 : 0;
    for (std::size_t i = bytes.size() - kEocdSize + 1; i-- > lowest;) {
        if (bytes[i] == 0x50 && bytes[i + 1] == 0x4b && bytes[i + 2] == 0x05 && bytes[i + 3] == 0x06) {
            eocd = i;
            break;
        }
    }
    if (eocd == std::string::npos) {
        throw ProjectIOError("zip: end of central directory not found");
    }

    Reader r(bytes);
    r.seek(eocd + 10);
    const std::uint16_t count = r.u16();
    r.skip(4);  // central directory size
    const std::uint32_t central_offset = r.u32();

    ZipArchive archive;
    r.seek(central_offset);
    for (std::uint16_t i = 0; i < count; ++i) {
        if (r.u32() != kCentralHeaderSig) {
            throw ProjectIOError("zip: bad central directory header");
        }
        r.skip(6);
        const std::uint16_t method = r.u16();
        r.skip(4);
        const std::uint32_t crc = r.u32();
        const std::uint32_t packed_size = r.u32();
        const std::uint32_t size = r.u32();
        const std::uint16_t name_len = r.u16();
        const std::uint16_t extra_len = r.u16();
        const std::uint16_t comment_len = r.u16();
        r.skip(8);
        const std::uint32_t local_offset = r.u32();
        std::string name = r.str(name_len);
        r.skip(static_cast<std::size_t>(extra_len) + comment_len);
        const std::size_t next_central = r.pos();

        r.seek(local_offset);
        if (r.u32() != kLocalHeaderSig) {
            throw ProjectIOError("zip: bad local header for '" + name + "'");
        }
        r.skip(22);
        const std::uint16_t local_name_len = r.u16();
        const std::uint16_t local_extra_len = r.u16();
        r.skip(static_cast<std::size_t>(local_name_len) + local_extra_len);
        const std::uint8_t* packed = r.take(packed_size);

        std::vector<std::uint8_t> data;
        if (method == kMethodStore) {
            if (packed_size != size) {
                throw ProjectIOError("zip: stored entry size mismatch");
            }
            data.assign(packed, packed + packed_size);
        } else if (method == kMethodDeflate) {
            // raw deflate cannot expand by more than 1032:1 (the DEFLATE spec's
            // maximum match length of 258 bytes coded in two bits, plus the
            // literals). A corrupt central directory that claims a multi-
            // gigabyte entry from a few packed bytes would otherwise make
            // inflate_raw size a buffer from it before noticing the stream is
            // short, i.e. an unbounded allocation from hostile input.
            constexpr std::uint64_t kMaxDeflateRatio = 1032;
            constexpr std::uint64_t kDeflateSlack = 64;
            if (static_cast<std::uint64_t>(size) >
                static_cast<std::uint64_t>(packed_size) * kMaxDeflateRatio + kDeflateSlack) {
                throw ProjectIOError("zip: declared size of '" + name +
                                     "' is impossible for its compressed length");
            }
            data = inflate_raw(packed, packed_size, size);
        } else {
            throw ProjectIOError("zip: unsupported compression method " + std::to_string(method));
        }
        if (crc_of(data.data(), data.size()) != crc) {
            throw ProjectIOError("zip: CRC mismatch for '" + name + "'");
        }
        archive.entries_.push_back(ZipEntry{std::move(name), std::move(data)});
        r.seek(next_central);
    }
    return archive;
}

}  // namespace alphashape
