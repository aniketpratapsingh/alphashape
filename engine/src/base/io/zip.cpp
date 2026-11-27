#include "base/io/zip.hpp"

#include <algorithm>
#include <limits>

#include <zlib.h>

#include "base/utils/error.hpp"

// Layout follows the PKWARE APPNOTE: per entry a local file header + data,
// then a central directory, then the end-of-central-directory record. Method 8
// is raw deflate (zlib with negative window bits). Integers are little-endian.
//
// Zip64 (APPNOTE 4.5.3, sections 4.3.14-4.3.16) is used only where a value does
// not fit its classic field: a size or offset of 0xFFFFFFFF or more, or an entry
// count of 0xFFFF or more. The overflowing 32/16-bit field then holds the
// sentinel and the real value moves into a Zip64 extended-information extra
// field (header ID 0x0001) or the Zip64 end-of-central-directory records. An
// archive that fits the classic fields is written exactly as before.

namespace alphashape {

namespace {

constexpr std::uint32_t kLocalHeaderSig = 0x04034b50;
constexpr std::uint32_t kCentralHeaderSig = 0x02014b50;
constexpr std::uint32_t kEndOfCentralSig = 0x06054b50;
constexpr std::uint32_t kZip64EndOfCentralSig = 0x06064b50;
constexpr std::uint32_t kZip64LocatorSig = 0x07064b50;
constexpr std::uint16_t kZip64ExtraId = 0x0001;
constexpr std::uint16_t kVersion = 20;       // 2.0: deflate
constexpr std::uint16_t kVersionZip64 = 45;  // 4.5: Zip64
constexpr std::uint16_t kMethodStore = 0;
constexpr std::uint16_t kMethodDeflate = 8;
constexpr std::uint16_t kDosDate = 0x21;     // 1980-01-01: fixed, so output is reproducible
constexpr std::uint32_t kMax32 = 0xFFFFFFFFU;  // sentinel: value lives in Zip64 extra
constexpr std::uint16_t kMax16 = 0xFFFFU;      // sentinel: value lives in Zip64 EOCD

void put16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8U) & 0xFFU));
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}

void put64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
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
    std::uint64_t u64() {
        need(8);
        std::uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i) {
            v |= static_cast<std::uint64_t>(bytes_[pos_ + i]) << (8U * i);
        }
        pos_ += 8;
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

// zlib's uInt is 32-bit (and uLong is 32-bit on Windows), so every call that
// takes a length is fed at most this many bytes at a time. 1 GiB is well below
// the 4 GiB a uInt can hold and keeps the arithmetic in 32 bits on every
// platform.
constexpr std::size_t kZlibChunk = std::size_t(1) << 30;

std::uint32_t crc_of(const std::uint8_t* data, std::size_t size) {
    uLong crc = crc32(0L, Z_NULL, 0);
    std::size_t pos = 0;
    while (pos < size) {
        const std::size_t n = std::min(kZlibChunk, size - pos);
        crc = crc32(crc, data + pos, static_cast<uInt>(n));
        pos += n;
    }
    return static_cast<std::uint32_t>(crc);
}

std::vector<std::uint8_t> deflate_raw(const std::vector<std::uint8_t>& in) {
    z_stream zs{};
    if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
        throw ProjectIOError("zip: deflateInit failed");
    }
    // An input that fits one chunk is compressed in a single Z_FINISH call, the
    // way it always was, so the bytes are unchanged. deflateBound takes a uLong
    // (32-bit on Windows) and a uInt (32-bit everywhere) bounds avail_in, so
    // only a larger input needs the chunked loop below.
    if (in.size() <= kZlibChunk) {
        std::vector<std::uint8_t> out(deflateBound(&zs, static_cast<uLong>(in.size())));
        zs.next_in = const_cast<Bytef*>(in.data());
        zs.avail_in = static_cast<uInt>(in.size());
        zs.next_out = out.data();
        zs.avail_out = static_cast<uInt>(out.size());
        const int rc = deflate(&zs, Z_FINISH);
        const std::size_t produced = zs.total_out;
        deflateEnd(&zs);
        if (rc != Z_STREAM_END) {
            throw ProjectIOError("zip: deflate failed");
        }
        out.resize(produced);
        return out;
    }

    // Larger than one chunk: feed the input a chunk at a time with Z_NO_FLUSH
    // and finish once it is all in. The output grows as it is produced, because
    // deflateBound cannot size a >4 GiB input on Windows. The reserve is capped
    // so a large, highly compressible entry does not reserve its full input size
    // up front; vector growth is amortised, so the cap only trades a few
    // reallocations for a smaller peak.
    std::vector<std::uint8_t> out;
    constexpr std::size_t kReserveCap = std::size_t(64) << 20;
    out.reserve(std::min(in.size(), kReserveCap) + 64);
    constexpr std::size_t kOutChunk = std::size_t(256) << 10;
    std::vector<std::uint8_t> buffer(kOutChunk);
    std::size_t in_pos = 0;
    int rc = Z_OK;
    while (rc != Z_STREAM_END) {
        if (zs.avail_in == 0 && in_pos < in.size()) {
            const std::size_t n = std::min(kZlibChunk, in.size() - in_pos);
            zs.next_in = const_cast<Bytef*>(in.data() + in_pos);
            zs.avail_in = static_cast<uInt>(n);
            in_pos += n;
        }
        zs.next_out = buffer.data();
        zs.avail_out = static_cast<uInt>(buffer.size());
        const int flush = (in_pos >= in.size() && zs.avail_in == 0) ? Z_FINISH : Z_NO_FLUSH;
        rc = deflate(&zs, flush);
        if (rc != Z_OK && rc != Z_STREAM_END) {
            deflateEnd(&zs);
            throw ProjectIOError("zip: deflate failed");
        }
        const std::size_t produced = buffer.size() - zs.avail_out;
        out.insert(out.end(), buffer.data(), buffer.data() + produced);
    }
    deflateEnd(&zs);
    return out;
}

std::vector<std::uint8_t> inflate_raw(const std::uint8_t* in, std::size_t in_size,
                                      std::size_t out_size) {
    std::vector<std::uint8_t> out(out_size);
    z_stream zs{};
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
        throw ProjectIOError("zip: inflateInit failed");
    }
    // A stream and its output that both fit one chunk are inflated in a single
    // Z_FINISH call, as before. Only a larger entry needs the chunked loop; a
    // zero-length output also takes it, because inflate with avail_out == 0
    // cannot report the end of the stream.
    if (out_size > 0 && in_size <= kZlibChunk && out_size <= kZlibChunk) {
        zs.next_in = const_cast<Bytef*>(in);
        zs.avail_in = static_cast<uInt>(in_size);
        zs.next_out = out.data();
        zs.avail_out = static_cast<uInt>(out_size);
        const int rc = inflate(&zs, Z_FINISH);
        const std::size_t produced = zs.total_out;
        inflateEnd(&zs);
        if (rc != Z_STREAM_END || produced != out_size) {
            throw ProjectIOError("zip: corrupt deflate stream");
        }
        return out;
    }

    std::size_t in_pos = 0;
    std::size_t produced = 0;
    std::uint8_t scratch = 0;  // one byte for a zero-length entry's stream
    int rc = Z_OK;
    while (rc != Z_STREAM_END) {
        if (zs.avail_in == 0 && in_pos < in_size) {
            const std::size_t n = std::min(kZlibChunk, in_size - in_pos);
            zs.next_in = const_cast<Bytef*>(in + in_pos);
            zs.avail_in = static_cast<uInt>(n);
            in_pos += n;
        }
        if (zs.avail_out == 0) {
            if (produced < out_size) {
                const std::size_t n = std::min(kZlibChunk, out_size - produced);
                zs.next_out = out.data() + produced;
                zs.avail_out = static_cast<uInt>(n);
            } else {
                // The declared size is exhausted; give zlib one scratch byte so
                // it can reach the end of the stream. A byte written there means
                // the declared size was too small.
                zs.next_out = &scratch;
                zs.avail_out = 1;
            }
        }
        const uInt before = zs.avail_out;
        rc = inflate(&zs, Z_NO_FLUSH);
        const std::size_t wrote = static_cast<std::size_t>(before - zs.avail_out);
        if (produced + wrote > out_size) {
            inflateEnd(&zs);
            throw ProjectIOError("zip: corrupt deflate stream");
        }
        produced += wrote;
        if (rc != Z_OK && rc != Z_STREAM_END) {
            inflateEnd(&zs);
            throw ProjectIOError("zip: corrupt deflate stream");
        }
    }
    inflateEnd(&zs);
    if (produced != out_size) {
        throw ProjectIOError("zip: corrupt deflate stream");
    }
    return out;
}

// Walk a central-directory extra-field block for the Zip64 extended-information
// field (ID 0x0001) and read the 64-bit values it carries. The field holds only
// the values whose classic slot was the 0xFFFFFFFF sentinel, in the fixed order
// uncompressed size, compressed size, local header offset, disk start. Each
// requested value is read in turn; a field too short for one is an error.
void parse_zip64_extra(const std::uint8_t* extra, std::size_t extra_len, bool want_size,
                       bool want_packed, bool want_offset, std::uint64_t& size,
                       std::uint64_t& packed, std::uint64_t& offset) {
    std::size_t pos = 0;
    while (pos + 4 <= extra_len) {
        const std::uint16_t id = static_cast<std::uint16_t>(extra[pos] | (extra[pos + 1] << 8U));
        const std::uint16_t len =
            static_cast<std::uint16_t>(extra[pos + 2] | (extra[pos + 3] << 8U));
        pos += 4;
        const std::size_t field_len = len;
        if (pos + field_len > extra_len) {
            throw ProjectIOError("zip: truncated extra field");
        }
        if (id == kZip64ExtraId) {
            std::size_t p = pos;
            const std::size_t end = pos + field_len;
            const auto read64 = [&]() -> std::uint64_t {
                if (p + 8 > end) {
                    throw ProjectIOError("zip: Zip64 extra field too short");
                }
                std::uint64_t v = 0;
                for (unsigned i = 0; i < 8; ++i) {
                    v |= static_cast<std::uint64_t>(extra[p + i]) << (8U * i);
                }
                p += 8;
                return v;
            };
            if (want_size) {
                size = read64();
            }
            if (want_packed) {
                packed = read64();
            }
            if (want_offset) {
                offset = read64();
            }
            return;
        }
        pos += field_len;
    }
    throw ProjectIOError("zip: Zip64 extra field missing");
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

std::vector<std::uint8_t> ZipArchive::serialise(const ZipWriteOptions& options) const {
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
        const std::uint64_t offset = out.size();
        if (entry.name.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw ProjectIOError("zip: entry name longer than 65535 bytes");
        }
        const auto name_len = static_cast<std::uint16_t>(entry.name.size());

        // A value of 0xFFFFFFFF in a classic field is the Zip64 sentinel, so a
        // real value of exactly 0xFFFFFFFF must also move to the extra field.
        const bool size_zip64 = options.force_zip64 ||
                                packed.size() >= static_cast<std::size_t>(kMax32) ||
                                entry.data.size() >= static_cast<std::size_t>(kMax32);
        const bool offset_zip64 = options.force_zip64 ||
                                  offset >= static_cast<std::uint64_t>(kMax32);
        const bool entry_zip64 = size_zip64 || offset_zip64;
        const auto version = static_cast<std::uint16_t>(entry_zip64 ? kVersionZip64 : kVersion);

        // Local header. When either size overflows, both classic size fields
        // become the sentinel and the Zip64 extra carries both sizes, in the
        // APPNOTE's fixed order (uncompressed, then compressed).
        std::vector<std::uint8_t> local_extra;
        if (size_zip64) {
            put16(local_extra, kZip64ExtraId);
            put16(local_extra, 16U);  // two 8-byte sizes
            put64(local_extra, entry.data.size());
            put64(local_extra, packed.size());
        }

        put32(out, kLocalHeaderSig);
        put16(out, version);
        put16(out, 0);  // flags
        put16(out, method);
        put16(out, 0);  // time
        put16(out, kDosDate);
        put32(out, crc);
        put32(out, size_zip64 ? kMax32 : static_cast<std::uint32_t>(packed.size()));
        put32(out, size_zip64 ? kMax32 : static_cast<std::uint32_t>(entry.data.size()));
        put16(out, name_len);
        put16(out, static_cast<std::uint16_t>(local_extra.size()));
        out.insert(out.end(), entry.name.begin(), entry.name.end());
        out.insert(out.end(), local_extra.begin(), local_extra.end());
        out.insert(out.end(), packed.begin(), packed.end());

        // Central directory. The Zip64 extra carries only the fields whose
        // classic slot is the sentinel, in the fixed order: uncompressed size,
        // compressed size, local header offset.
        std::vector<std::uint8_t> central_extra;
        if (entry_zip64) {
            std::vector<std::uint8_t> values;
            if (size_zip64) {
                put64(values, entry.data.size());
                put64(values, packed.size());
            }
            if (offset_zip64) {
                put64(values, offset);
            }
            put16(central_extra, kZip64ExtraId);
            put16(central_extra, static_cast<std::uint16_t>(values.size()));
            central_extra.insert(central_extra.end(), values.begin(), values.end());
        }

        put32(central, kCentralHeaderSig);
        put16(central, version);  // made by
        put16(central, version);  // needed
        put16(central, 0);
        put16(central, method);
        put16(central, 0);
        put16(central, kDosDate);
        put32(central, crc);
        put32(central, size_zip64 ? kMax32 : static_cast<std::uint32_t>(packed.size()));
        put32(central, size_zip64 ? kMax32 : static_cast<std::uint32_t>(entry.data.size()));
        put16(central, name_len);
        put16(central, static_cast<std::uint16_t>(central_extra.size()));
        put16(central, 0);  // comment
        put16(central, 0);  // disk
        put16(central, 0);  // internal attrs
        put32(central, 0);  // external attrs
        put32(central, offset_zip64 ? kMax32 : static_cast<std::uint32_t>(offset));
        central.insert(central.end(), entry.name.begin(), entry.name.end());
        central.insert(central.end(), central_extra.begin(), central_extra.end());
    }

    const std::uint64_t central_offset = out.size();
    out.insert(out.end(), central.begin(), central.end());
    const std::uint64_t count = entries_.size();
    const std::uint64_t central_size = central.size();

    // The Zip64 end records are needed when the entry count, the central
    // directory size or its offset does not fit the classic fields.
    const bool eocd_zip64 = options.force_zip64 ||
                            count >= static_cast<std::uint64_t>(kMax16) ||
                            central_size >= static_cast<std::uint64_t>(kMax32) ||
                            central_offset >= static_cast<std::uint64_t>(kMax32);
    if (eocd_zip64) {
        const std::uint64_t zip64_eocd_offset = out.size();
        put32(out, kZip64EndOfCentralSig);
        put64(out, 44U);  // size of the rest of this record (no extensible data)
        put16(out, kVersionZip64);  // version made by
        put16(out, kVersionZip64);  // version needed
        put32(out, 0);  // this disk
        put32(out, 0);  // disk with the central directory
        put64(out, count);  // entries on this disk
        put64(out, count);  // entries total
        put64(out, central_size);
        put64(out, central_offset);

        put32(out, kZip64LocatorSig);
        put32(out, 0);  // disk with the Zip64 end record
        put64(out, zip64_eocd_offset);
        put32(out, 1);  // total disks
    }

    // When the Zip64 end records are present the classic EOCD carries the
    // sentinels, so a reader is directed to them; otherwise it carries the real
    // values and the archive is exactly the classic layout.
    const auto eocd_count = static_cast<std::uint16_t>(eocd_zip64 ? kMax16 : count);
    const auto eocd_size = static_cast<std::uint32_t>(eocd_zip64 ? kMax32 : central_size);
    const auto eocd_offset = static_cast<std::uint32_t>(eocd_zip64 ? kMax32 : central_offset);
    put32(out, kEndOfCentralSig);
    put16(out, 0);
    put16(out, 0);
    put16(out, eocd_count);
    put16(out, eocd_count);
    put32(out, eocd_size);
    put32(out, eocd_offset);
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
    std::uint64_t count = r.u16();
    std::uint64_t central_size = r.u32();
    std::uint64_t central_offset = r.u32();

    // A classic field holding its sentinel means the real value is in the Zip64
    // end-of-central-directory record, which the locator (20 bytes before the
    // classic EOCD) points at.
    if (count == static_cast<std::uint64_t>(kMax16) ||
        central_size == static_cast<std::uint64_t>(kMax32) ||
        central_offset == static_cast<std::uint64_t>(kMax32)) {
        if (eocd < 20) {
            throw ProjectIOError("zip: Zip64 locator missing");
        }
        r.seek(eocd - 20);
        if (r.u32() != kZip64LocatorSig) {
            throw ProjectIOError("zip: Zip64 locator missing");
        }
        r.skip(4);  // disk with the Zip64 end record
        const std::uint64_t zip64_eocd = r.u64();
        r.skip(4);  // total disks
        r.seek(static_cast<std::size_t>(zip64_eocd));
        if (r.u32() != kZip64EndOfCentralSig) {
            throw ProjectIOError("zip: bad Zip64 end of central directory");
        }
        r.skip(20);  // record size (8), versions (4), disk numbers (8)
        r.skip(8);   // entries on this disk (same as total for a single-disk archive)
        count = r.u64();
        r.skip(8);   // central directory size (not needed to walk the entries)
        central_offset = r.u64();
    }

    ZipArchive archive;
    r.seek(static_cast<std::size_t>(central_offset));
    for (std::uint64_t i = 0; i < count; ++i) {
        if (r.u32() != kCentralHeaderSig) {
            throw ProjectIOError("zip: bad central directory header");
        }
        r.skip(6);
        const std::uint16_t method = r.u16();
        r.skip(4);
        const std::uint32_t crc = r.u32();
        const std::uint32_t packed_size32 = r.u32();
        const std::uint32_t size32 = r.u32();
        const std::uint16_t name_len = r.u16();
        const std::uint16_t extra_len = r.u16();
        const std::uint16_t comment_len = r.u16();
        r.skip(8);
        const std::uint32_t local_offset32 = r.u32();
        std::string name = r.str(name_len);
        const std::uint8_t* extra = r.take(extra_len);
        r.skip(comment_len);
        const std::size_t next_central = r.pos();

        // Resolve the 64-bit values from the Zip64 extra field for whichever
        // classic fields are sentinels, in the APPNOTE's fixed order.
        std::uint64_t packed_size = packed_size32;
        std::uint64_t size = size32;
        std::uint64_t local_offset = local_offset32;
        if (packed_size32 == kMax32 || size32 == kMax32 || local_offset32 == kMax32) {
            parse_zip64_extra(extra, extra_len, size32 == kMax32, packed_size32 == kMax32,
                              local_offset32 == kMax32, size, packed_size, local_offset);
        }

        r.seek(static_cast<std::size_t>(local_offset));
        if (r.u32() != kLocalHeaderSig) {
            throw ProjectIOError("zip: bad local header for '" + name + "'");
        }
        r.skip(22);
        const std::uint16_t local_name_len = r.u16();
        const std::uint16_t local_extra_len = r.u16();
        r.skip(static_cast<std::size_t>(local_name_len) + local_extra_len);
        const std::uint8_t* packed = r.take(static_cast<std::size_t>(packed_size));

        std::vector<std::uint8_t> data;
        if (method == kMethodStore) {
            if (packed_size != size) {
                throw ProjectIOError("zip: stored entry size mismatch");
            }
            data.assign(packed, packed + static_cast<std::size_t>(packed_size));
        } else if (method == kMethodDeflate) {
            // raw deflate cannot expand by more than 1032:1 (the DEFLATE spec's
            // maximum match length of 258 bytes coded in two bits, plus the
            // literals). A corrupt central directory that claims a multi-
            // gigabyte entry from a few packed bytes would otherwise make
            // inflate_raw size a buffer from it before noticing the stream is
            // short, i.e. an unbounded allocation from hostile input.
            constexpr std::uint64_t kMaxDeflateRatio = 1032;
            constexpr std::uint64_t kDeflateSlack = 64;
            if (size > packed_size * kMaxDeflateRatio + kDeflateSlack) {
                throw ProjectIOError("zip: declared size of '" + name +
                                     "' is impossible for its compressed length");
            }
            data = inflate_raw(packed, static_cast<std::size_t>(packed_size),
                               static_cast<std::size_t>(size));
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
