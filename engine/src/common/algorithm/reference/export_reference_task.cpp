#include "common/algorithm/reference/export_reference_task.hpp"

#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/reference/reference_csv.hpp"
#include "common/algorithm/reference/reference_table.hpp"
#include "common/algorithm/reference/reference_xml.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/scalebar.hpp"
#include "model/crs/coordinate_system.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

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

Format parse_format(const std::string& name, const std::string& path) {
    const std::string text = lower(trim(name));
    if (!text.empty()) {
        if (text == "csv") {
            return Format::Csv;
        }
        if (text == "txt" || text == "text") {
            return Format::Txt;
        }
        if (text == "xml") {
            return Format::Xml;
        }
        throw ValueError("ExportReference: format must be csv, txt or xml, got '" + name + "'");
    }
    const std::string ext = lower(fs::path(path).extension().string());
    if (ext == ".csv") {
        return Format::Csv;
    }
    if (ext == ".xml") {
        return Format::Xml;
    }
    if (ext == ".txt" || ext == ".dat") {
        return Format::Txt;
    }
    throw ValueError("ExportReference: cannot infer the format of '" + path +
                     "'; set the `format` parameter to csv, txt or xml");
}

char delimiter_char(const std::string& raw, Format format) {
    if (format == Format::Txt && raw == ",") {
        return ' ';
    }
    if (raw == "\\t") {
        return '\t';
    }
    if (raw == "\\s+" || raw == "\\s") {
        return ' ';
    }
    if (raw.size() != 1) {
        throw ValueError("ExportReference: delimiter must be one character, got '" + raw + "'");
    }
    return raw[0];
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

crs::CoordinateSystem parse_crs_text(const std::string& raw) {
    const std::string text = trim(raw);
    const std::string prefix = "EPSG:";
    if (text.size() > prefix.size() &&
        lower(text.substr(0, prefix.size())) == lower(prefix)) {
        const std::string code = text.substr(prefix.size());
        int epsg = 0;
        const auto result = std::from_chars(code.data(), code.data() + code.size(), epsg);
        if (result.ec != std::errc{} || result.ptr != code.data() + code.size() || epsg <= 0) {
            throw ValueError("ExportReference: '" + raw + "' is not an EPSG:<code> CRS");
        }
        return crs::CoordinateSystem::from_epsg(epsg);
    }
    try {
        return crs::CoordinateSystem::from_wkt(text);
    } catch (const ValueError&) {
        return crs::CoordinateSystem::from_proj(text);
    }
}

const char* kind_name(reference::ReferenceRowKind kind) {
    switch (kind) {
        case reference::ReferenceRowKind::Camera: return "camera";
        case reference::ReferenceRowKind::Marker: return "marker";
        case reference::ReferenceRowKind::CheckPoint: return "check_point";
    }
    return "marker";
}

std::optional<double> component(const std::optional<Eigen::Vector3d>& value, int axis) {
    if (!value.has_value()) {
        return std::nullopt;
    }
    return axis == 0 ? value->x() : (axis == 1 ? value->y() : value->z());
}

bool axis_suffix(const std::string& column, const std::string& prefix, int* axis) {
    const std::string x = prefix + "_x";
    const std::string y = prefix + "_y";
    const std::string z = prefix + "_z";
    if (column == x) {
        *axis = 0;
        return true;
    }
    if (column == y) {
        *axis = 1;
        return true;
    }
    if (column == z) {
        *axis = 2;
        return true;
    }
    return false;
}

/// A value cell for one reference-table row, or nullopt when the column does
/// not apply (JSON-style empty field in the output).
std::optional<double> camera_marker_value(const std::string& column,
                                          const reference::ReferenceRow& row) {
    int axis = 0;
    if (axis_suffix(column, "estimated", &axis)) {
        return component(row.estimated, axis);
    }
    if (axis_suffix(column, "error", &axis)) {
        return component(row.error, axis);
    }
    if (column == "x") {
        return component(row.source, 0);
    }
    if (column == "y") {
        return component(row.source, 1);
    }
    if (column == "z") {
        return component(row.source, 2);
    }
    if (column == "sigma" || column == "sigma_x" || column == "sigma_y" ||
        column == "sigma_z") {
        return row.sigma;
    }
    if (column == "capture_distance") {
        return row.capture_distance_m;
    }
    if (column == "enabled") {
        return row.enabled ? 1.0 : 0.0;
    }
    return std::nullopt;
}

bool camera_marker_has_column(const std::string& column) {
    return column == "label" || column == "kind" || column == "x" || column == "y" ||
           column == "z" || column == "sigma" || column == "sigma_x" ||
           column == "sigma_y" || column == "sigma_z" || column == "capture_distance" ||
           column == "enabled" ||
           [&] {
               int axis = 0;
               return axis_suffix(column, "estimated", &axis) ||
                      axis_suffix(column, "error", &axis);
           }();
}

bool scalebar_has_column(const std::string& column) {
    return column == "label" || column == "kind" || column == "distance" ||
           column == "estimated_distance" || column == "accuracy" || column == "enabled";
}

std::optional<double> scalebar_value(const std::string& column, const Scalebar& bar) {
    if (column == "distance") {
        return bar.reference_length_m;
    }
    if (column == "estimated_distance") {
        return bar.estimated_length_m;
    }
    if (column == "accuracy") {
        return bar.reference_accuracy_m;
    }
    if (column == "enabled") {
        return 1.0;
    }
    return std::nullopt;
}

}  // namespace
}  // namespace alphashape

namespace alphashape {

void ExportReferenceTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::string{}, "Output file. Required."});
    ParamSpec format{"format", ParamType::String, std::string{},
                     "csv, txt or xml. Empty infers it from the path extension."};
    format.choices = {"", "csv", "txt", "xml"};
    params.declare(format);
    params.declare({"delimiter", ParamType::String, std::string{","},
                    "CSV field separator; TXT defaults to a space."});
    params.declare({"columns_order", ParamType::StringList,
                    std::vector<std::string>{"label", "x", "y", "z", "estimated_x",
                                             "estimated_y", "estimated_z", "error_x",
                                             "error_y", "error_z", "sigma_x", "sigma_y",
                                             "sigma_z"},
                    "Columns of the CSV/TXT header and every row. Recognised names: "
                    "label, kind, x, y, z, estimated_x/y/z, error_x/y/z, "
                    "sigma_x/y/z, sigma, capture_distance, enabled (scale bars: "
                    "distance, estimated_distance, accuracy)."});
    params.declare({"crs", ParamType::String, std::string{},
                    "CRS of the written coordinates as \"EPSG:<code>\", WKT or a PROJ "
                    "string. Empty writes in the chunk CRS."});
    params.declare({"precision", ParamType::Int, std::int64_t{6},
                    "Digits after the decimal point in CSV/TXT output."});
    params.declare({"include_cameras", ParamType::Bool, true, "Write camera rows."});
    params.declare({"include_markers", ParamType::Bool, true,
                    "Write marker and check-point rows."});
    params.declare({"include_scalebars", ParamType::Bool, true, "Write scale-bar rows."});
}

std::vector<Stage> ExportReferenceTask::stages() const {
    return {Stage{"build", 0}, Stage{"write", 0}};
}

void ExportReferenceTask::apply(Document& document, Params& params, const Progress& progress) {
    progress.update("build", 0.0);
    const std::string path = trim(params.get_string("path"));
    if (path.empty()) {
        throw ValueError("ExportReference: path is required");
    }
    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportReference needs an active chunk");
    }
    const Format format = parse_format(params.get_string("format"), path);
    const char delimiter = delimiter_char(params.get_string("delimiter"), format);
    const std::int64_t requested_precision = params.get_int("precision");
    if (requested_precision < 0 || requested_precision > 15) {
        throw ValueError("ExportReference: precision must be between 0 and 15");
    }
    const int precision = static_cast<int>(requested_precision);

    std::vector<std::string> columns;
    for (const std::string& raw : params.get_string_list("columns_order")) {
        const std::string column = lower(trim(raw));
        if (column.empty()) {
            continue;
        }
        if (!camera_marker_has_column(column) && !scalebar_has_column(column)) {
            throw ValueError("ExportReference: unknown column name '" + raw +
                             "' in columns_order");
        }
        columns.push_back(column);
    }
    if (columns.empty()) {
        throw ValueError("ExportReference: columns_order must name at least one column");
    }

    const std::string crs_text = trim(params.get_string("crs"));
    const crs::CoordinateSystem display =
        crs_text.empty() ? chunk->coordinate_system() : parse_crs_text(crs_text);
    const reference::ReferenceTable table =
        reference::build_reference_table(*chunk, display);
    const bool include_cameras = params.get_bool("include_cameras");
    const bool include_markers = params.get_bool("include_markers");
    const bool include_scalebars = params.get_bool("include_scalebars");
    progress.update("build", 1.0);

    if (format == Format::Xml) {
        reference::XmlReferenceDocument document_xml;
        for (const reference::ReferenceRow& row : table.rows) {
            if (row.kind == reference::ReferenceRowKind::Camera && !include_cameras) {
                continue;
            }
            if (row.kind != reference::ReferenceRowKind::Camera && !include_markers) {
                continue;
            }
            // The XML schema requires x/y/z, so a row without a measured source
            // cannot be represented there (the CSV form keeps it with empty
            // position cells and its estimated/error columns).
            if (!row.source.has_value()) {
                continue;
            }
            reference::XmlReferencePosition item;
            item.label = row.label;
            item.position = *row.source;
            item.accuracy = row.sigma;
            item.enabled = row.enabled;
            if (row.kind == reference::ReferenceRowKind::Camera) {
                document_xml.cameras.push_back(std::move(item));
            } else {
                document_xml.markers.push_back(std::move(item));
            }
        }
        if (include_scalebars) {
            for (const Scalebar& bar : chunk->scalebars()) {
                reference::XmlReferenceScalebar item;
                item.label = bar.label;
                item.distance = bar.reference_length_m;
                item.accuracy = bar.reference_accuracy_m;
                document_xml.scalebars.push_back(std::move(item));
            }
        }
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw Error("ExportReference: cannot open '" + path + "' for writing");
        }
        out << reference::write_reference_xml(document_xml);
        out.close();
        if (!out) {
            throw Error("ExportReference: failed while writing '" + path + "'");
        }
        progress.update("write", 1.0);
        if (Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info) << "ExportReference: " << document_xml.cameras.size()
                                << " camera(s), " << document_xml.markers.size()
                                << " marker(s), " << document_xml.scalebars.size()
                                << " scale bar(s) -> " << path;
        }
        return;
    }

    const auto camera_marker_cell = [&](const std::string& column,
                                        const reference::ReferenceRow& row) {
        if (column == "label") {
            return row.label;
        }
        if (column == "kind") {
            return std::string(kind_name(row.kind));
        }
        const std::optional<double> value = camera_marker_value(column, row);
        return value.has_value() ? reference::format_number(*value, precision)
                                 : std::string{};
    };
    const auto scalebar_cell = [&](const std::string& column, const Scalebar& bar) {
        if (column == "label") {
            return bar.label;
        }
        if (column == "kind") {
            return std::string("scalebar");
        }
        const std::optional<double> value = scalebar_value(column, bar);
        return value.has_value() ? reference::format_number(*value, precision)
                                 : std::string{};
    };

    std::string out;
    out += reference::format_delimited(columns, delimiter);
    std::size_t row_count = 0;
    for (const reference::ReferenceRow& row : table.rows) {
        if (row.kind == reference::ReferenceRowKind::Camera && !include_cameras) {
            continue;
        }
        if (row.kind != reference::ReferenceRowKind::Camera && !include_markers) {
            continue;
        }
        std::vector<std::string> cells;
        cells.reserve(columns.size());
        for (const std::string& column : columns) {
            cells.push_back(
                camera_marker_has_column(column)
                    ? camera_marker_cell(column, row)
                    : std::string{});  // a scale-bar-only column does not apply
        }
        out += reference::format_delimited(cells, delimiter);
        ++row_count;
    }
    std::size_t bar_count = 0;
    if (include_scalebars) {
        for (const Scalebar& bar : chunk->scalebars()) {
            std::vector<std::string> cells;
            cells.reserve(columns.size());
            for (const std::string& column : columns) {
                cells.push_back(scalebar_has_column(column) ? scalebar_cell(column, bar)
                                                            : std::string{});
            }
            out += reference::format_delimited(cells, delimiter);
            ++bar_count;
        }
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw Error("ExportReference: cannot open '" + path + "' for writing");
    }
    file << out;
    file.close();
    if (!file) {
        throw Error("ExportReference: failed while writing '" + path + "'");
    }
    progress.update("write", 1.0);
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ExportReference: " << row_count << " row(s), " << bar_count
                            << " scale bar(s) -> " << path;
    }
}

}  // namespace alphashape
