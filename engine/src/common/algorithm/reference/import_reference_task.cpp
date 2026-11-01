#include "common/algorithm/reference/import_reference_task.hpp"

#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/reference/reference_csv.hpp"
#include "common/algorithm/reference/reference_table.hpp"
#include "common/algorithm/reference/reference_xml.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/scalebar.hpp"
#include "common/core/types.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/euler.hpp"
#include "model/crs/transform.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;
using reference::DelimitedRow;

enum class Format { Csv, Txt, Xml };

std::string lower(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(begin, end - begin);
}

/// One field of the record: which logical column it fills.
enum class Column {
    Skip,
    Label,
    X,
    Y,
    Z,
    SigmaX,
    SigmaY,
    SigmaZ,
    Sigma,
    Rot0,
    Rot1,
    Rot2,
};

/// Map a header name (or an explicit `columns` entry) to a logical column.
/// `angle_convention_xyz` is set when omega/phi/kappa is seen, which names the
/// classic XYZ sequence rather than yaw/pitch/roll in the requested order.
Column column_from_name(const std::string& raw, bool* angle_convention_xyz) {
    const std::string name = lower(trim(raw));
    if (name.empty() || name == "skip" || name == "ignore" || name == "-") {
        return Column::Skip;
    }
    if (name == "label" || name == "name" || name == "id" || name == "point" ||
        name == "point_name" || name == "gcp") {
        return Column::Label;
    }
    if (name == "x" || name == "easting" || name == "east" || name == "lon" ||
        name == "longitude") {
        return Column::X;
    }
    if (name == "y" || name == "northing" || name == "north" || name == "lat" ||
        name == "latitude") {
        return Column::Y;
    }
    if (name == "z" || name == "height" || name == "alt" || name == "altitude" ||
        name == "elevation" || name == "h") {
        return Column::Z;
    }
    if (name == "sx" || name == "stdx" || name == "std_x" || name == "sigma_x" ||
        name == "accuracy_x" || name == "acc_x") {
        return Column::SigmaX;
    }
    if (name == "sy" || name == "stdy" || name == "std_y" || name == "sigma_y" ||
        name == "accuracy_y" || name == "acc_y") {
        return Column::SigmaY;
    }
    if (name == "sz" || name == "stdz" || name == "std_z" || name == "sigma_z" ||
        name == "accuracy_z" || name == "acc_z") {
        return Column::SigmaZ;
    }
    if (name == "sigma" || name == "sxyz" || name == "accuracy" || name == "acc") {
        return Column::Sigma;
    }
    if (name == "yaw" || name == "omega") {
        if (name == "omega" && angle_convention_xyz != nullptr) {
            *angle_convention_xyz = true;
        }
        return Column::Rot0;
    }
    if (name == "pitch" || name == "phi") {
        return Column::Rot1;
    }
    if (name == "roll" || name == "kappa") {
        return Column::Rot2;
    }
    return Column::Skip;
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

bool is_skip_name(const std::string& raw) {
    const std::string name = lower(trim(raw));
    return name.empty() || name == "skip" || name == "ignore" || name == "-";
}

Format format_from_extension(const std::string& path) {
    const std::string ext = lower(fs::path(path).extension().string());
    if (ext == ".csv") {
        return Format::Csv;
    }
    if (ext == ".xml") {
        return Format::Xml;
    }
    if (ext == ".txt" || ext == ".dat" || ext == ".tsv") {
        return Format::Txt;
    }
    throw ValueError("ImportReference: cannot infer the format of '" + path +
                     "'; set the `format` parameter to csv, txt or xml");
}

Format parse_format(const std::string& name, const std::string& path) {
    const std::string text = lower(trim(name));
    if (text.empty()) {
        return format_from_extension(path);
    }
    if (text == "csv") {
        return Format::Csv;
    }
    if (text == "txt" || text == "text") {
        return Format::Txt;
    }
    if (text == "xml") {
        return Format::Xml;
    }
    throw ValueError("ImportReference: format must be csv, txt or xml, got '" + name + "'");
}

/// Resolve the delimiter parameter. The two-character forms are what a command
/// line or Python string carries; the literal character is accepted too.
std::string delimiter_text(const std::string& raw) {
    if (raw == "\\s+" || raw == "\\s") {
        return "\\s+";
    }
    if (raw == "\\t") {
        return "\t";
    }
    if (raw.empty()) {
        throw ValueError("ImportReference: delimiter must not be empty");
    }
    if (raw.size() != 1) {
        throw ValueError("ImportReference: delimiter must be one character or \"\\\\s+\", got '" +
                         raw + "'");
    }
    return raw;
}

/// The effective field splitter: whitespace runs for TXT with the default
/// comma, otherwise the named single character.
bool uses_whitespace(Format format, const std::string& delimiter) {
    return format == Format::Txt && delimiter == ",";
}

/// Which field of a record feeds each logical column.
struct Mapping {
    std::vector<std::pair<std::size_t, Column>> fields;
    bool angle_convention_xyz = false;

    int index_of(Column wanted) const {
        for (const auto& [index, kind] : fields) {
            if (kind == wanted) {
                return static_cast<int>(index);
            }
        }
        return -1;
    }

    bool complete() const {
        return index_of(Column::Label) >= 0 && index_of(Column::X) >= 0 &&
               index_of(Column::Y) >= 0 && index_of(Column::Z) >= 0;
    }
};

const char* column_name(Column column) {
    switch (column) {
        case Column::Label: return "label";
        case Column::X: return "x";
        case Column::Y: return "y";
        case Column::Z: return "z";
        case Column::SigmaX: return "sigma_x";
        case Column::SigmaY: return "sigma_y";
        case Column::SigmaZ: return "sigma_z";
        case Column::Sigma: return "sigma";
        case Column::Rot0: return "rotation 1";
        case Column::Rot1: return "rotation 2";
        case Column::Rot2: return "rotation 3";
        case Column::Skip: return "skip";
    }
    return "?";
}

void add_mapping(Mapping& mapping, std::size_t index, Column column) {
    if (column == Column::Skip) {
        return;
    }
    if (mapping.index_of(column) >= 0) {
        throw ValueError(std::string("ImportReference: column '") + column_name(column) +
                         "' is mapped more than once");
    }
    mapping.fields.emplace_back(index, column);
}

/// Build the mapping from an explicit `columns` list; every entry must be a
/// known column name or an explicit skip.
Mapping explicit_mapping(const std::vector<std::string>& columns) {
    Mapping mapping;
    for (std::size_t i = 0; i < columns.size(); ++i) {
        const std::string name = lower(trim(columns[i]));
        if (is_skip_name(name)) {
            continue;
        }
        const Column column = column_from_name(name, &mapping.angle_convention_xyz);
        if (column == Column::Skip) {
            throw ValueError("ImportReference: unknown column name '" + columns[i] +
                             "' in `columns`");
        }
        add_mapping(mapping, i, column);
    }
    if (!mapping.complete()) {
        throw ValueError(
            "ImportReference: `columns` must name label, x, y and z");
    }
    return mapping;
}

/// Auto-detect a header row or the four-column headerless layout.
Mapping automatic_mapping(const DelimitedRow& first, bool decimal_comma) {
    Mapping mapping;
    const bool numeric_body =
        first.fields.size() >= 4 &&
        reference::is_number(trim(first.fields[1]), decimal_comma) &&
        reference::is_number(trim(first.fields[2]), decimal_comma) &&
        reference::is_number(trim(first.fields[3]), decimal_comma);
    if (numeric_body) {
        add_mapping(mapping, 0, Column::Label);
        add_mapping(mapping, 1, Column::X);
        add_mapping(mapping, 2, Column::Y);
        add_mapping(mapping, 3, Column::Z);
        return mapping;
    }
    for (std::size_t i = 0; i < first.fields.size(); ++i) {
        const Column column = column_from_name(first.fields[i], &mapping.angle_convention_xyz);
        // An unrecognised header column is ignored: a survey export may carry
        // extra fields, and the import only needs the four it understands.
        add_mapping(mapping, i, column);
    }
    if (!mapping.complete()) {
        throw ValueError(
            "ImportReference: could not find label, x, y and z columns in the header; "
            "set `columns` explicitly");
    }
    return mapping;
}

struct RowAccuracy {
    std::optional<double> x;
    std::optional<double> y;
    std::optional<double> z;
    std::optional<double> single;
};

/// One parsed record, still in the file's CRS and rotation convention.
struct ParsedRow {
    std::size_t line = 0;
    std::string label;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    RowAccuracy accuracy;
    std::optional<Eigen::Vector3d> angles_deg;
    bool enabled = true;
};

bool looks_like_headerless(const DelimitedRow& row, bool decimal_comma) {
    return row.fields.size() >= 4 &&
           reference::is_number(trim(row.fields[1]), decimal_comma) &&
           reference::is_number(trim(row.fields[2]), decimal_comma) &&
           reference::is_number(trim(row.fields[3]), decimal_comma);
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw Error("ImportReference: cannot open '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (in.bad()) {
        throw Error("ImportReference: failed while reading '" + path + "'");
    }
    return buffer.str();
}

double parse_field(const std::string& text, bool decimal_comma, const std::string& context) {
    return reference::parse_number(text, decimal_comma, context);
}

Eigen::Vector3d rotation_to_chunk_order(const Eigen::Vector3d& angles_deg,
                                        crs::RotationOrder file_order,
                                        crs::RotationOrder chunk_order) {
    if (file_order == chunk_order) {
        return angles_deg;
    }
    const crs::EulerAngles in{angles_deg.x(), angles_deg.y(), angles_deg.z(), file_order};
    const crs::EulerAngles out = crs::from_matrix(crs::to_matrix(in), chunk_order);
    return Eigen::Vector3d(out.yaw, out.pitch, out.roll);
}

crs::RotationOrder parse_rotation_order(const std::string& text, const char* what) {
    crs::RotationOrder order = crs::RotationOrder::ZYX;
    if (!crs::parse_rotation_order(text, &order)) {
        throw ValueError(std::string("ImportReference: ") + what + " '" + text +
                         "' is not a rotation order (XYZ, XZY, YXZ, YZX, ZXY, ZYX)");
    }
    return order;
}

crs::RotationOrder chunk_rotation_order(const Chunk& chunk) {
    const std::string* value = chunk.meta().find(reference::kRotationOrderKey);
    return value == nullptr ? crs::RotationOrder::ZYX
                            : parse_rotation_order(*value, "chunk rotation order");
}

std::vector<ParsedRow> parse_records(const std::vector<DelimitedRow>& rows,
                                     const Mapping& mapping, const std::string& path,
                                     bool decimal_comma, bool radians) {
    constexpr double kRadToDeg = 57.295779513082320876798154814105;
    std::vector<ParsedRow> parsed;
    parsed.reserve(rows.size());
    for (const DelimitedRow& row : rows) {
        const std::string context =
            "ImportReference: " + path + ": line " + std::to_string(row.line);
        const auto field = [&](Column column) -> std::optional<std::string> {
            const int index = mapping.index_of(column);
            if (index < 0) {
                return std::nullopt;
            }
            const std::size_t at = static_cast<std::size_t>(index);
            if (at >= row.fields.size()) {
                throw ValueError(context + ": missing column '" + column_name(column) + "'");
            }
            return trim(row.fields[at]);
        };
        const auto number = [&](Column column) -> std::optional<double> {
            const std::optional<std::string> text = field(column);
            if (!text.has_value()) {
                return std::nullopt;
            }
            return parse_field(*text, decimal_comma,
                               context + ": column '" + column_name(column) + "'");
        };

        ParsedRow out;
        out.line = row.line;
        const std::optional<std::string> label = field(Column::Label);
        const std::optional<double> x = number(Column::X);
        const std::optional<double> y = number(Column::Y);
        const std::optional<double> z = number(Column::Z);
        if (!label.has_value() || !x.has_value() || !y.has_value() || !z.has_value()) {
            throw ValueError(context + ": missing label, x, y or z");
        }
        out.label = *label;
        out.position = Eigen::Vector3d(*x, *y, *z);
        out.accuracy.x = number(Column::SigmaX);
        out.accuracy.y = number(Column::SigmaY);
        out.accuracy.z = number(Column::SigmaZ);
        out.accuracy.single = number(Column::Sigma);
        const std::optional<double> a = number(Column::Rot0);
        const std::optional<double> b = number(Column::Rot1);
        const std::optional<double> c = number(Column::Rot2);
        const bool any_angle = a.has_value() || b.has_value() || c.has_value();
        if (any_angle) {
            const double scale = radians ? kRadToDeg : 1.0;
            out.angles_deg = Eigen::Vector3d(a.value_or(0.0) * scale,
                                             b.value_or(0.0) * scale,
                                             c.value_or(0.0) * scale);
        }
        parsed.push_back(std::move(out));
    }
    return parsed;
}

/// Reject a label that occurs twice, naming both lines.
void check_duplicate_labels(const std::vector<ParsedRow>& rows) {
    std::map<std::string, std::size_t> seen;
    for (const ParsedRow& row : rows) {
        const auto found = seen.find(row.label);
        if (found != seen.end()) {
            throw ValueError("ImportReference: duplicate label '" + row.label +
                             "' on lines " + std::to_string(found->second) + " and " +
                             std::to_string(row.line));
        }
        seen.emplace(row.label, row.line);
    }
}

crs::CoordinateSystem parse_crs_text(const std::string& raw) {
    const std::string text = trim(raw);
    const std::string prefix = "EPSG:";
    if (text.size() > prefix.size() &&
        lower(text.substr(0, prefix.size())) == lower(prefix)) {
        const std::string code = text.substr(prefix.size());
        int epsg = 0;
        const auto result = std::from_chars(code.data(), code.data() + code.size(), epsg);
        if (result.ec != std::errc{} || result.ptr != code.data() + code.size() || epsg <= 0) {
            throw ValueError("ImportReference: '" + raw + "' is not an EPSG:<code> CRS");
        }
        return crs::CoordinateSystem::from_epsg(epsg);
    }
    try {
        return crs::CoordinateSystem::from_wkt(text);
    } catch (const ValueError&) {
        return crs::CoordinateSystem::from_proj(text);
    }
}

#if defined(ALPHASHAPE_USE_PROJ)
/// Transform from the file CRS into the chunk CRS, or nullopt when the file is
/// already expressed in it. The chunk CRS is the target because the reference
/// slots default to it and an explicit slot is the caller's own contract.
std::optional<crs::CrsTransform> make_import_transform(const std::string& crs_text,
                                                        const Chunk& chunk) {
    if (trim(crs_text).empty()) {
        return std::nullopt;
    }
    const crs::CoordinateSystem source = parse_crs_text(crs_text);
    const crs::CoordinateSystem target = chunk.coordinate_system();
    if (!target.is_set() || target.is_local()) {
        throw ValueError(
            "ImportReference: `crs` was given but the chunk has no coordinate system; "
            "set the chunk CRS first or leave `crs` empty");
    }
    return crs::CrsTransform::create(source, target);
}
#endif

Eigen::Vector3d convert_position(const std::optional<crs::CrsTransform>& transform,
                                 const Eigen::Vector3d& position,
                                 const std::string& label) {
    if (!transform.has_value()) {
        return position;
    }
    const std::optional<crs::CrsPoint> out =
        transform->forward(crs::CrsPoint{position.x(), position.y(), position.z()});
    if (!out.has_value()) {
        throw ValueError("ImportReference: '" + label +
                         "' is outside the area of use of the requested CRS");
    }
    return Eigen::Vector3d(out->x, out->y, out->z);
}

Eigen::Vector3d accuracy_of(const RowAccuracy& accuracy, double fallback) {
    const double x = accuracy.x.value_or(accuracy.single.value_or(fallback));
    const double y = accuracy.y.value_or(accuracy.single.value_or(fallback));
    const double z = accuracy.z.value_or(accuracy.single.value_or(fallback));
    return Eigen::Vector3d(x, y, z);
}

void apply_marker_type(Marker& marker, const std::string& marker_type) {
    const std::string type = lower(trim(marker_type));
    if (type == "check_point" || type == "checkpoint") {
        reference::set_marker_role(marker, reference::MarkerReferenceRole::CheckPoint);
    } else if (type == "gcp" || type == "control") {
        reference::set_marker_role(marker, reference::MarkerReferenceRole::Gcp);
    } else if (type == "marker") {
        // A plain marker: leave the role unstated, which WP-13.1 reads as gcp
        // once a reference exists.
    } else {
        throw ValueError("ImportReference: marker_type must be gcp, check_point or marker, got '" +
                         marker_type + "'");
    }
}

std::vector<ParsedRow> records_from_xml(const reference::XmlReferenceDocument& document,
                                        const std::string& path) {
    std::vector<ParsedRow> rows;
    std::map<std::string, std::string> seen;
    const auto append = [&](const reference::XmlReferencePosition& item) {
        if (seen.find(item.label) != seen.end()) {
            throw ValueError("ImportReference: duplicate label '" + item.label +
                             "' in reference XML '" + path + "'");
        }
        seen.emplace(item.label, item.label);
        ParsedRow row;
        row.line = 0;
        row.label = item.label;
        row.position = item.position;
        row.accuracy.single = item.accuracy > 0.0 ? std::optional<double>(item.accuracy)
                                                  : std::nullopt;
        row.enabled = item.enabled;
        rows.push_back(std::move(row));
    };
    for (const auto& camera : document.cameras) {
        append(camera);
    }
    for (const auto& marker : document.markers) {
        append(marker);
    }
    return rows;
}

}  // namespace
}  // namespace alphashape

namespace alphashape {

void ImportReferenceTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::string{},
                    "Input file. Required. The format is inferred from the extension "
                    "when `format` is empty."});
    ParamSpec format{"format", ParamType::String, std::string{},
                     "csv, txt or xml. Empty infers it from the path extension."};
    format.choices = {"", "csv", "txt", "xml"};
    params.declare(format);
    params.declare({"delimiter", ParamType::String, std::string{","},
                    "CSV field separator: \",\", \";\", \"\\t\" or \" \". "
                    "\"\\\\s+\" splits on whitespace runs. TXT defaults to whitespace."});
    params.declare({"skip_rows", ParamType::Int, std::int64_t{0},
                    "Records to drop before the header or first data row."});
    params.declare({"columns", ParamType::StringList, std::vector<std::string>{},
                    "Explicit column mapping, e.g. [\"label\", \"x\", \"y\", \"z\"]. "
                    "Empty auto-detects a header or a headerless label,x,y,z layout."});
    params.declare({"crs", ParamType::String, std::string{},
                    "CRS of the file's coordinates as \"EPSG:<code>\", WKT or a PROJ "
                    "string. Empty means the chunk CRS. EPSG:4326 inputs (lon, lat, "
                    "alt) are converted into the chunk CRS."});
    ParamSpec order{"rotation_order", ParamType::String, std::string{"ZYX"},
                    "Rotation order of yaw/pitch/roll in the file."};
    order.choices = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};
    params.declare(order);
    ParamSpec units{"angle_units", ParamType::String, std::string{"deg"},
                    "Units of the rotation columns: deg or rad."};
    units.choices = {"deg", "rad"};
    params.declare(units);
    ParamSpec role{"marker_type", ParamType::String, std::string{"gcp"},
                   "Role given to imported markers: gcp, check_point or marker."};
    role.choices = {"gcp", "check_point", "marker"};
    params.declare(role);
    params.declare({"decimal_comma", ParamType::Bool, false,
                    "Read numeric fields with a decimal comma."});
    ParamSpec encoding{"encoding", ParamType::String, std::string{"utf-8"},
                       "Text encoding of the file: utf-8 or latin1."};
    encoding.choices = {"utf-8", "latin1"};
    params.declare(encoding);
    params.declare({"apply", ParamType::Bool, true,
                    "Parse and validate only when false."});
    params.declare({"create_markers", ParamType::Bool, true,
                    "Create markers for labels that match no camera or marker. When "
                    "false, unmatched labels are reported in a warning."});
}

std::vector<Stage> ImportReferenceTask::stages() const {
    return {Stage{"parse", 0}, Stage{"apply", 0}};
}

void ImportReferenceTask::apply(Document& document, Params& params, const Progress& progress) {
    progress.update("parse", 0.0);
    const std::string path = trim(params.get_string("path"));
    if (path.empty()) {
        throw ValueError("ImportReference: path is required");
    }
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ImportReference needs an active chunk");
    }
    const Format format = parse_format(params.get_string("format"), path);
    const std::string delimiter = delimiter_text(params.get_string("delimiter"));
    const bool decimal_comma = params.get_bool("decimal_comma");
    const std::string encoding = lower(trim(params.get_string("encoding")));
    const bool latin1 = encoding == "latin1" || encoding == "iso-8859-1";
    if (!latin1 && encoding != "utf-8" && encoding != "utf8") {
        throw ValueError("ImportReference: encoding must be utf-8 or latin1, got '" +
                         params.get_string("encoding") + "'");
    }
    if (params.get_int("skip_rows") < 0) {
        throw ValueError("ImportReference: skip_rows must not be negative");
    }

    std::string text = read_file(path);
    if (latin1) {
        text = reference::latin1_to_utf8(text);
    } else if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEFU &&
               static_cast<unsigned char>(text[1]) == 0xBBU &&
               static_cast<unsigned char>(text[2]) == 0xBFU) {
        text.erase(0, 3);  // UTF-8 BOM
    }

    std::vector<ParsedRow> rows;
    std::optional<reference::XmlReferenceDocument> xml;
    bool angle_convention_xyz = false;
    if (format == Format::Xml) {
        xml = reference::parse_reference_xml(text);
        rows = records_from_xml(*xml, path);
    } else {
        const std::vector<DelimitedRow> all =
            uses_whitespace(format, delimiter)
                ? reference::parse_whitespace(text)
                : reference::parse_delimited(text, delimiter[0]);
        const std::size_t skipped =
            static_cast<std::size_t>(params.get_int("skip_rows"));
        if (skipped >= all.size()) {
            throw ValueError("ImportReference: '" + path +
                             "' has no records after skip_rows=" +
                             std::to_string(skipped));
        }
        const std::vector<DelimitedRow> kept(all.begin() + static_cast<std::ptrdiff_t>(skipped),
                                             all.end());
        const std::vector<std::string> columns = params.get_string_list("columns");
        Mapping mapping;
        std::size_t first_data = 0;
        if (!columns.empty()) {
            mapping = explicit_mapping(columns);
        } else {
            mapping = automatic_mapping(kept.front(), decimal_comma);
            first_data = looks_like_headerless(kept.front(), decimal_comma) ? 0 : 1;
        }
        if (first_data >= kept.size()) {
            throw ValueError("ImportReference: '" + path + "' has a header but no data rows");
        }
        const std::vector<DelimitedRow> data(
            kept.begin() + static_cast<std::ptrdiff_t>(first_data), kept.end());
        const std::string units = lower(trim(params.get_string("angle_units")));
        if (units != "deg" && units != "rad") {
            throw ValueError("ImportReference: angle_units must be deg or rad, got '" +
                             params.get_string("angle_units") + "'");
        }
        rows = parse_records(data, mapping, path, decimal_comma, units == "rad");
        angle_convention_xyz = mapping.angle_convention_xyz;
    }
    check_duplicate_labels(rows);
    progress.update("parse", 1.0);

    progress.update("apply", 0.0);
    const std::string crs_text = params.get_string("crs");
#if defined(ALPHASHAPE_USE_PROJ)
    const std::optional<crs::CrsTransform> transform = make_import_transform(crs_text, *chunk);
#else
    if (!trim(crs_text).empty()) {
        throw Error("ImportReference: converting from another CRS needs a build with PROJ "
                    "(ALPHASHAPE_USE_PROJ=ON)");
    }
    const std::optional<crs::CrsTransform> transform;
#endif

    const bool apply_changes = params.get_bool("apply");
    const bool create_markers = params.get_bool("create_markers");
    const std::string marker_type = params.get_string("marker_type");
    {
        Marker probe;  // validate the role name before any chunk mutation
        apply_marker_type(probe, marker_type);
    }
    const crs::RotationOrder chunk_order = chunk_rotation_order(*chunk);
    const crs::RotationOrder file_order =
        angle_convention_xyz
            ? crs::RotationOrder::XYZ
            : parse_rotation_order(params.get_string("rotation_order"), "rotation_order");

    const ReferenceSettings settings = chunk->reference_settings();
    std::size_t camera_count = 0;
    std::size_t marker_count = 0;
    std::size_t created_count = 0;
    // Label lookups are built once: a 100 000-row file against a 100 000-marker
    // chunk would otherwise be quadratic. First occurrence wins if a chunk
    // already carries duplicate labels.
    // Indexes, not pointers: adding markers reallocates the container.
    std::map<std::string, std::size_t> camera_index;
    for (std::size_t i = 0; i < chunk->cameras().size(); ++i) {
        camera_index.emplace(chunk->cameras()[i].label, i);
    }
    std::map<std::string, std::size_t> marker_index;
    for (std::size_t i = 0; i < chunk->markers().size(); ++i) {
        marker_index.emplace(chunk->markers()[i].label, i);
    }
    std::vector<std::string> unmatched;
    for (const ParsedRow& row : rows) {
        Camera* camera = nullptr;
        if (const auto found = camera_index.find(row.label);
            found != camera_index.end()) {
            camera = &chunk->cameras()[found->second];
        }
        Marker* marker = nullptr;
        if (const auto found = marker_index.find(row.label);
            found != marker_index.end()) {
            marker = &chunk->markers()[found->second];
        }
        const std::string where =
            row.line == 0 ? std::string{} : "line " + std::to_string(row.line) + ": ";
        if (row.angles_deg.has_value() && camera == nullptr) {
            throw ValueError("ImportReference: " + where + "'" + row.label +
                             "' carries rotation angles but matches no camera");
        }
        const Eigen::Vector3d position = convert_position(transform, row.position, row.label);
        if (!apply_changes) {
            if (camera != nullptr) {
                ++camera_count;
            } else {
                ++marker_count;
            }
            continue;
        }
        if (camera != nullptr) {
            camera->reference.location = position;
            camera->reference.accuracy =
                accuracy_of(row.accuracy, settings.camera_location_accuracy_m);
            camera->reference.enabled = row.enabled;
            if (row.angles_deg.has_value()) {
                camera->reference.orientation_deg =
                    rotation_to_chunk_order(*row.angles_deg, file_order, chunk_order);
            }
            ++camera_count;
        } else if (marker != nullptr) {
            marker->reference = position;
            marker->reference_accuracy =
                accuracy_of(row.accuracy, settings.marker_location_accuracy_m);
            apply_marker_type(*marker, marker_type);
            ++marker_count;
        } else if (create_markers) {
            Marker& created = chunk->markers().emplace_back();
            created.id = chunk->allocate_id();
            created.label = row.label;
            created.reference = position;
            created.reference_accuracy =
                accuracy_of(row.accuracy, settings.marker_location_accuracy_m);
            apply_marker_type(created, marker_type);
            ++marker_count;
            ++created_count;
        } else {
            unmatched.push_back(row.label);
        }
    }

    if (apply_changes && xml.has_value()) {
        for (const reference::XmlReferenceScalebar& imported : xml->scalebars) {
            bool matched = false;
            for (Scalebar& bar : chunk->scalebars()) {
                if (bar.label != imported.label) {
                    continue;
                }
                bar.reference_length_m = imported.distance;
                bar.reference_accuracy_m = imported.accuracy > 0.0
                                               ? imported.accuracy
                                               : settings.scalebar_accuracy_m;
                matched = true;
                break;
            }
            if (!matched) {
                unmatched.push_back(imported.label);
            }
        }
    }
    progress.update("apply", 1.0);

    if (!unmatched.empty()) {
        std::string labels;
        const std::size_t shown = unmatched.size() < 10 ? unmatched.size() : 10;
        for (std::size_t i = 0; i < shown; ++i) {
            if (i != 0) {
                labels += ", ";
            }
            labels += unmatched[i];
        }
        if (unmatched.size() > shown) {
            labels += ", ...";
        }
        if (Log::enabled(LogLevel::Warn)) {
            log(LogLevel::Warn)
                << "ImportReference: " << unmatched.size()
                << " label(s) matched no camera, marker or scale bar and were not imported: "
                << labels;
        }
    }
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ImportReference: " << (apply_changes ? "imported " : "validated ")
                            << camera_count << " camera(s), " << marker_count
                            << " marker(s)";
        if (created_count > 0) {
            log(LogLevel::Info) << "ImportReference: created " << created_count
                                << " new marker(s)";
        }
    }
}

}  // namespace alphashape
