#include "image/io/exif.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape {

namespace {

// ---------------------------------------------------------------------------
// TIFF/EXIF on-disk layout
// ---------------------------------------------------------------------------
// Every multi-byte integer is in the byte order the header declares. Field
// types and their widths are from the TIFF 6.0 spec.
constexpr std::uint16_t kTypeByte = 1;
constexpr std::uint16_t kTypeAscii = 2;
constexpr std::uint16_t kTypeShort = 3;
constexpr std::uint16_t kTypeLong = 4;
constexpr std::uint16_t kTypeRational = 5;
constexpr std::uint16_t kTypeSByte = 6;
constexpr std::uint16_t kTypeUndefined = 7;
constexpr std::uint16_t kTypeSShort = 8;
constexpr std::uint16_t kTypeSLong = 9;
constexpr std::uint16_t kTypeSRational = 10;
constexpr std::uint16_t kTypeFloat = 11;
constexpr std::uint16_t kTypeDouble = 12;

/// A cap on how many entries one IFD may declare. Real files have tens; a
/// corrupt count easily claims millions. Refusing outright keeps the parser
/// from walking garbage even though every entry is still bounds-checked.
constexpr std::uint32_t kMaxIfdEntries = 4096;
/// A cap on the byte length of any single value we will materialise.
constexpr std::uint32_t kMaxValueBytes = 1U << 20;  // 1 MiB
/// Depth cap for the chain of followed pointers, so a pointer cycle that slips
/// past the visited-set still terminates.
constexpr int kMaxIfdDepth = 8;

/// Size in bytes of one element of a field type, or 0 for an unknown type.
std::size_t type_size(std::uint16_t type) {
    switch (type) {
        case kTypeByte:
        case kTypeAscii:
        case kTypeSByte:
        case kTypeUndefined:
            return 1;
        case kTypeShort:
        case kTypeSShort:
            return 2;
        case kTypeLong:
        case kTypeSLong:
        case kTypeFloat:
            return 4;
        case kTypeRational:
        case kTypeSRational:
        case kTypeDouble:
            return 8;
        default:
            return 0;
    }
}

/// Which IFD we are inside. Tag numbers are only unambiguous relative to this,
/// because the GPS IFD reuses low tag numbers that mean something else in
/// IFD0 and the Exif IFD (0x0001 is both InteropIndex and GPSLatitudeRef).
enum class IfdKind { Root, Exif, Gps };

/// A bounds-checked view over the TIFF block. Every read funnels through here,
/// so no code path can touch memory outside [0, size).
class Block {
public:
    Block(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    bool in_range(std::uint64_t offset, std::uint64_t length) const {
        return offset <= size_ && length <= size_ - offset;
    }

    std::uint8_t u8(std::uint64_t offset) const { return data_[offset]; }

    std::uint16_t u16(std::uint64_t offset, bool little) const {
        const std::uint8_t* p = data_ + offset;
        if (little) {
            return static_cast<std::uint16_t>(p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
        }
        return static_cast<std::uint16_t>((static_cast<std::uint16_t>(p[0]) << 8) | p[1]);
    }

    std::uint32_t u32(std::uint64_t offset, bool little) const {
        const std::uint8_t* p = data_ + offset;
        if (little) {
            return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
                   (static_cast<std::uint32_t>(p[2]) << 16) |
                   (static_cast<std::uint32_t>(p[3]) << 24);
        }
        return (static_cast<std::uint32_t>(p[0]) << 24) |
               (static_cast<std::uint32_t>(p[1]) << 16) |
               (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
    }

    std::int32_t i32(std::uint64_t offset, bool little) const {
        return static_cast<std::int32_t>(u32(offset, little));
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

/// One resolved IFD entry: where its value lives and how it is shaped. The
/// accessors below re-check bounds on every read, so a hostile offset cannot
/// escape even if a later tag handler forgets to.
struct Entry {
    std::uint16_t tag = 0;
    std::uint16_t type = 0;
    std::uint32_t count = 0;
    /// Byte offset of the value inside the block. For a value of 4 bytes or
    /// fewer this is the entry's own value slot, which the parser resolves when
    /// it decodes the record.
    std::uint32_t value_offset = 0;

    /// Total byte length, or 0 if the type is unknown or count*type_size
    /// exceeds the value cap.
    std::uint64_t byte_length() const {
        const std::size_t element = type_size(type);
        if (element == 0) {
            return 0;
        }
        const std::uint64_t total = static_cast<std::uint64_t>(element) * count;
        if (total > kMaxValueBytes) {
            return 0;
        }
        return total;
    }
};

/// The parser: carries the byte order and the visited-IFD set across the
/// recursive descent, and accumulates the GPS sub-fields until the walk is
/// complete.
class Parser {
public:
    Parser(const std::uint8_t* data, std::size_t size, ExifData& out)
        : block_(data, size), out_(out) {}

    void run() {
        if (!block_.in_range(0, 8)) {
            return;  // not even a TIFF header; nothing to parse, no throw
        }
        const std::uint8_t b0 = block_.u8(0);
        const std::uint8_t b1 = block_.u8(1);
        if (b0 == 'I' && b1 == 'I') {
            little_ = true;
        } else if (b0 == 'M' && b1 == 'M') {
            little_ = false;
        } else {
            return;  // not a TIFF block
        }
        // The 0x002A magic confirms this really is a TIFF header rather than
        // two coincidental bytes; a bad magic means "no metadata".
        if (block_.u16(2, little_) != 0x002A) {
            return;
        }
        read_ifd(block_.u32(4, little_), IfdKind::Root, 0);
        finalize_gps();
    }

private:
    /// Reads one IFD and dispatches its entries. Returns on any structural
    /// problem, keeping whatever was already extracted.
    void read_ifd(std::uint32_t offset, IfdKind kind, int depth) {
        if (depth > kMaxIfdDepth) {
            return;
        }
        // An IFD0 offset of 0 is legal-but-empty; a revisit means a pointer
        // loop, which we refuse to follow. Both stop the descent quietly.
        if (offset == 0 || !visited_.insert(offset).second) {
            return;
        }
        if (!block_.in_range(offset, 2)) {
            return;  // offset beyond the end
        }
        const std::uint32_t count = block_.u16(offset, little_);
        if (count > kMaxIfdEntries) {
            return;  // implausible entry count
        }
        const std::uint64_t entries_bytes = static_cast<std::uint64_t>(count) * 12;
        if (!block_.in_range(static_cast<std::uint64_t>(offset) + 2, entries_bytes)) {
            return;  // entries would run past the block
        }

        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint64_t entry_offset =
                static_cast<std::uint64_t>(offset) + 2 + static_cast<std::uint64_t>(i) * 12;
            const Entry entry = parse_entry(entry_offset);
            handle_entry(entry, kind);
            // Sub-IFD pointers live in IFD0 and are followed with the matching
            // context, so their low-numbered tags are interpreted correctly.
            if (kind == IfdKind::Root) {
                if (entry.tag == 0x8769) {
                    if (const std::optional<std::uint32_t> p = read_int(entry)) {
                        read_ifd(*p, IfdKind::Exif, depth + 1);
                    }
                } else if (entry.tag == 0x8825) {
                    if (const std::optional<std::uint32_t> p = read_int(entry)) {
                        read_ifd(*p, IfdKind::Gps, depth + 1);
                    }
                }
            }
        }
        // The next-IFD link is deliberately NOT followed: the interface reads
        // the first directory only.
    }

    /// Decodes an entry's 12-byte record. A type we do not know, or a byte
    /// length over the cap, yields an Entry whose accessors all report "no
    /// value" rather than an exception.
    Entry parse_entry(std::uint64_t entry_offset) const {
        Entry entry;
        entry.tag = block_.u16(entry_offset, little_);
        entry.type = block_.u16(entry_offset + 2, little_);
        entry.count = block_.u32(entry_offset + 4, little_);
        const std::size_t element = type_size(entry.type);
        if (element == 0) {
            return entry;
        }
        const std::uint64_t total = static_cast<std::uint64_t>(element) * entry.count;
        if (total > kMaxValueBytes) {
            return entry;
        }
        if (total <= 4) {
            // Short values live in the entry itself, left-justified in the
            // 4-byte field, which is why the byte order still applies.
            entry.value_offset = static_cast<std::uint32_t>(entry_offset + 8);
        } else {
            entry.value_offset = block_.u32(entry_offset + 8, little_);
        }
        return entry;
    }

    /// A single integer value of BYTE/SHORT/LONG type (or its signed twin).
    /// Used for sub-IFD pointers, dimensions, orientation and altitude ref.
    std::optional<std::uint32_t> read_int(const Entry& entry) const {
        if (entry.byte_length() == 0 || entry.count == 0) {
            return std::nullopt;
        }
        if (!block_.in_range(entry.value_offset, type_size(entry.type))) {
            return std::nullopt;
        }
        switch (entry.type) {
            case kTypeByte:
            case kTypeSByte:
                return static_cast<std::uint32_t>(block_.u8(entry.value_offset));
            case kTypeShort:
            case kTypeSShort:
                return static_cast<std::uint32_t>(block_.u16(entry.value_offset, little_));
            case kTypeLong:
            case kTypeSLong:
                return block_.u32(entry.value_offset, little_);
            default:
                return std::nullopt;
        }
    }

    /// One RATIONAL or SRATIONAL element, rejecting a zero denominator.
    std::optional<double> read_rational(const Entry& entry, std::uint32_t index) const {
        const bool is_signed = entry.type == kTypeSRational;
        if (entry.type != kTypeRational && entry.type != kTypeSRational) {
            return std::nullopt;
        }
        if (index >= entry.count || entry.byte_length() == 0) {
            return std::nullopt;
        }
        const std::uint64_t offset =
            static_cast<std::uint64_t>(entry.value_offset) + static_cast<std::uint64_t>(index) * 8;
        if (!block_.in_range(offset, 8)) {
            return std::nullopt;
        }
        double numerator = 0;
        double denominator = 0;
        if (is_signed) {
            numerator = static_cast<double>(block_.i32(offset, little_));
            denominator = static_cast<double>(block_.i32(offset + 4, little_));
        } else {
            numerator = static_cast<double>(block_.u32(offset, little_));
            denominator = static_cast<double>(block_.u32(offset + 4, little_));
        }
        if (denominator == 0.0) {
            return std::nullopt;
        }
        return numerator / denominator;
    }

    /// ASCII, truncated at the first NUL, with trailing spaces and tabs
    /// trimmed. A missing terminator is fine: the count bounds the string.
    std::optional<std::string> read_ascii(const Entry& entry) const {
        if (entry.type != kTypeAscii || entry.byte_length() == 0) {
            return std::nullopt;
        }
        if (!block_.in_range(entry.value_offset, entry.byte_length())) {
            return std::nullopt;
        }
        std::string value;
        value.reserve(entry.count);
        for (std::uint32_t i = 0; i < entry.count; ++i) {
            const char c = static_cast<char>(block_.u8(entry.value_offset + i));
            if (c == '\0') {
                break;
            }
            value.push_back(c);
        }
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        if (value.empty()) {
            return std::nullopt;
        }
        return value;
    }

    /// The first byte of an ASCII field, for the one-character GPS refs.
    std::optional<char> read_ascii_char(const Entry& entry) const {
        const std::optional<std::string> text = read_ascii(entry);
        if (!text.has_value() || text->empty()) {
            return std::nullopt;
        }
        return (*text)[0];
    }

    /// A GPS coordinate: three RATIONALs (degrees, minutes, seconds) folded to
    /// decimal degrees.
    std::optional<double> read_gps_coordinate(const Entry& entry) const {
        if (entry.count < 3) {
            return std::nullopt;
        }
        const std::optional<double> degrees = read_rational(entry, 0);
        const std::optional<double> minutes = read_rational(entry, 1);
        const std::optional<double> seconds = read_rational(entry, 2);
        if (!degrees.has_value() || !minutes.has_value() || !seconds.has_value()) {
            return std::nullopt;
        }
        return *degrees + *minutes / 60.0 + *seconds / 3600.0;
    }

    void handle_entry(const Entry& entry, IfdKind kind);

    /// Folds the accumulated GPS sub-fields into a GpsPosition, applying the
    /// hemisphere signs. Only a latitude AND a longitude make a position.
    void finalize_gps() {
        if (!gps_lat_.has_value() || !gps_lon_.has_value()) {
            return;
        }
        GpsPosition position;
        const double lat = *gps_lat_;
        const double lon = *gps_lon_;
        position.latitude_deg = (gps_lat_ref_ == 'S') ? -lat : lat;
        position.longitude_deg = (gps_lon_ref_ == 'W') ? -lon : lon;
        if (gps_alt_.has_value()) {
            // GPSAltitudeRef 1 means "below sea level", so the value is the
            // magnitude and we negate it.
            position.altitude_m = (gps_alt_ref_ == 1) ? -*gps_alt_ : *gps_alt_;
        }
        out_.gps = position;
    }

    Block block_;
    ExifData& out_;
    bool little_ = true;
    std::set<std::uint32_t> visited_;

    // GPS sub-fields, valid only inside the GPS IFD. Kept here so the tags
    // cannot collide with the Interop tags in the others.
    std::optional<char> gps_lat_ref_;
    std::optional<char> gps_lon_ref_;
    std::optional<double> gps_lat_;
    std::optional<double> gps_lon_;
    std::optional<double> gps_alt_;
    std::optional<std::uint32_t> gps_alt_ref_;
};

void Parser::handle_entry(const Entry& entry, IfdKind kind) {
    if (kind == IfdKind::Gps) {
        switch (entry.tag) {
            case 0x0001:
                if (auto v = read_ascii_char(entry)) {
                    gps_lat_ref_ = *v;
                }
                break;
            case 0x0002:
                if (auto v = read_gps_coordinate(entry)) {
                    gps_lat_ = *v;
                }
                break;
            case 0x0003:
                if (auto v = read_ascii_char(entry)) {
                    gps_lon_ref_ = *v;
                }
                break;
            case 0x0004:
                if (auto v = read_gps_coordinate(entry)) {
                    gps_lon_ = *v;
                }
                break;
            case 0x0005:
                if (auto v = read_int(entry)) {
                    gps_alt_ref_ = *v;
                }
                break;
            case 0x0006:
                if (auto v = read_rational(entry, 0)) {
                    gps_alt_ = *v;
                }
                break;
            default:
                break;
        }
        return;
    }

    switch (entry.tag) {
        case 0x010F:  // Make
            if (auto v = read_ascii(entry)) out_.make = *v;
            break;
        case 0x0110:  // Model
            if (auto v = read_ascii(entry)) out_.model = *v;
            break;
        case 0x0112: {  // Orientation
            const std::optional<std::uint32_t> v = read_int(entry);
            if (v.has_value() && *v >= 1 && *v <= 8) {
                out_.orientation = static_cast<int>(*v);
            }
            break;
        }
        default:
            break;
    }

    // The Exif IFD carries everything else. Its tags do not collide with
    // IFD0's, so this block runs for both kinds.
    switch (entry.tag) {
        case 0x9003:  // DateTimeOriginal
            if (auto v = read_ascii(entry)) out_.datetime_original = *v;
            break;
        case 0x920A:  // FocalLength
            if (auto v = read_rational(entry, 0)) out_.focal_length_mm = *v;
            break;
        case 0xA002: {  // PixelXDimension: SHORT or LONG
            const std::optional<std::uint32_t> v = read_int(entry);
            if (v.has_value()) out_.pixel_x_dimension = static_cast<int>(*v);
            break;
        }
        case 0xA003: {  // PixelYDimension
            const std::optional<std::uint32_t> v = read_int(entry);
            if (v.has_value()) out_.pixel_y_dimension = static_cast<int>(*v);
            break;
        }
        case 0xA20E:  // FocalPlaneXResolution
            if (auto v = read_rational(entry, 0)) out_.focal_plane_x_resolution = *v;
            break;
        case 0xA20F:  // FocalPlaneYResolution
            if (auto v = read_rational(entry, 0)) out_.focal_plane_y_resolution = *v;
            break;
        case 0xA210: {  // FocalPlaneResolutionUnit
            const std::optional<std::uint32_t> v = read_int(entry);
            if (v.has_value()) out_.focal_plane_resolution_unit = static_cast<int>(*v);
            break;
        }
        case 0xA405: {  // FocalLengthIn35mmFilm: SHORT, stored as double
            const std::optional<std::uint32_t> v = read_int(entry);
            if (v.has_value()) out_.focal_length_35mm = static_cast<double>(*v);
            break;
        }
        case 0xA431:  // BodySerialNumber
            if (auto v = read_ascii(entry)) out_.body_serial = *v;
            break;
        case 0xA434:  // LensModel
            if (auto v = read_ascii(entry)) out_.lens_model = *v;
            break;
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Container scanning
// ---------------------------------------------------------------------------

/// Walks the JPEG marker segments and hands the first Exif APP1 payload to the
/// parser. Stops at SOS, skips APP1 segments that are not Exif (XMP, etc.) and
/// treats any structural damage as "no metadata".
void scan_jpeg(const std::uint8_t* data, std::size_t size, ExifData& out) {
    // SOI must be the first two bytes; the caller has already checked the
    // magic, but re-checking keeps this function self-contained.
    std::size_t pos = 2;
    while (pos + 4 <= size) {
        if (data[pos] != 0xFF) {
            return;  // resynchronisation is not attempted: a broken stream ends the scan
        }
        // Fill bytes (0xFF padding) are legal before a marker.
        std::size_t marker_pos = pos;
        while (marker_pos + 1 < size && data[marker_pos] == 0xFF && data[marker_pos + 1] == 0xFF) {
            ++marker_pos;
        }
        if (marker_pos + 1 >= size) {
            return;
        }
        const std::uint8_t marker = data[marker_pos + 1];
        if (marker == 0xD8) {  // SOI again: nothing to do, skip
            pos = marker_pos + 2;
            continue;
        }
        if (marker == 0xDA) {  // SOS: entropy-coded data follows, stop
            return;
        }
        // Standalone markers carry no length.
        if (marker == 0xD0 || marker == 0xD1 || marker == 0xD2 || marker == 0xD3 || marker == 0xD4 ||
            marker == 0xD5 || marker == 0xD6 || marker == 0xD7 || marker == 0xD9) {
            pos = marker_pos + 2;
            continue;
        }
        if (marker_pos + 4 > size) {
            return;
        }
        const std::uint16_t length = static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(data[marker_pos + 2]) << 8) | data[marker_pos + 3]);
        if (length < 2) {
            return;  // a segment length includes its own two bytes
        }
        const std::size_t segment_start = marker_pos + 4;
        const std::size_t payload_length = static_cast<std::size_t>(length) - 2;
        if (segment_start + payload_length > size) {
            return;  // segment runs past the file
        }
        if (marker == 0xE1 && payload_length >= 6 &&
            std::memcmp(data + segment_start, "Exif\0\0", 6) == 0) {
            const std::uint8_t* tiff = data + segment_start + 6;
            out = parse_exif_tiff_block(tiff, payload_length - 6);
            return;
        }
        // Any other APPn (including the XMP APP1) is skipped.
        pos = segment_start + payload_length;
    }
}

/// Reads a PNG's eXIf chunk when present. Chunk layout: 4-byte length, 4-byte
/// type, payload, 4-byte CRC. The eXIf payload is a raw TIFF block.
void scan_png(const std::uint8_t* data, std::size_t size, ExifData& out) {
    std::size_t pos = 8;  // skip the signature
    while (pos + 8 <= size) {
        const auto be32 = [&](std::size_t at) {
            return (static_cast<std::uint32_t>(data[at]) << 24) |
                   (static_cast<std::uint32_t>(data[at + 1]) << 16) |
                   (static_cast<std::uint32_t>(data[at + 2]) << 8) |
                   static_cast<std::uint32_t>(data[at + 3]);
        };
        const std::uint32_t length = be32(pos);
        const std::size_t payload = pos + 8;
        if (payload + length + 4 > size) {
            return;  // truncated or hostile
        }
        if (std::memcmp(data + pos + 4, "eXIf", 4) == 0) {
            out = parse_exif_tiff_block(data + payload, length);
            return;
        }
        if (std::memcmp(data + pos + 4, "IEND", 4) == 0) {
            return;
        }
        pos = payload + length + 4;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

std::optional<double> ExifData::pixel_size_um() const {
    if (!focal_plane_x_resolution.has_value() || !focal_plane_resolution_unit.has_value()) {
        return std::nullopt;
    }
    const double resolution = *focal_plane_x_resolution;
    if (resolution <= 0.0) {
        return std::nullopt;
    }
    double unit_um = 0;
    switch (*focal_plane_resolution_unit) {
        case 2:
            unit_um = 25400.0;  // per inch
            break;
        case 3:
            unit_um = 10000.0;  // per centimetre
            break;
        case 4:
            unit_um = 1000.0;  // per millimetre
            break;
        case 5:
            unit_um = 1.0;  // per micrometre
            break;
        default:
            return std::nullopt;
    }
    return unit_um / resolution;
}

ExifData parse_exif_tiff_block(const std::uint8_t* data, std::size_t size) {
    ExifData out;
    if (data == nullptr || size == 0) {
        return out;
    }
    Parser parser(data, size, out);
    parser.run();
    return out;
}

ExifData read_exif(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw ProjectIOError("cannot open image '" + path + "'");
    }
    const std::streamoff end = in.tellg();
    if (end < 0) {
        throw ProjectIOError("cannot determine size of '" + path + "'");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (static_cast<std::size_t>(in.gcount()) != bytes.size()) {
            throw ProjectIOError("short read from '" + path + "'");
        }
    }

    ExifData out;
    if (bytes.size() >= 4 && bytes[0] == 0xFF && bytes[1] == 0xD8) {
        scan_jpeg(bytes.data(), bytes.size(), out);
    } else if (bytes.size() >= 8 && bytes[0] == 0x89 && bytes[1] == 'P' && bytes[2] == 'N' &&
               bytes[3] == 'G') {
        scan_png(bytes.data(), bytes.size(), out);
    } else if (bytes.size() >= 8 &&
               ((bytes[0] == 'I' && bytes[1] == 'I') || (bytes[0] == 'M' && bytes[1] == 'M'))) {
        out = parse_exif_tiff_block(bytes.data(), bytes.size());
    }
    // Any other container yields a default ExifData; malformed metadata never
    // throws, only a failure to open the file does (handled above).
    return out;
}

}  // namespace alphashape
