#pragma once

/// @file export_reference_task.hpp
/// @brief ExportReference: write the chunk reference table (WP-13.2).
///
/// Emits the measured, estimated and error positions from the WP-13.1
/// reference table as CSV, whitespace-delimited TXT or the common
/// photogrammetry reference XML. The CSV/TXT form carries a header line named
/// by `columns_order`; the XML form carries the format's fixed attributes.
///
/// Camera and marker rows can be selected independently, and scale bars are
/// written when `include_scalebars` is set (CSV/TXT always, XML only when
/// there are any).

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

class ExportReferenceTask : public AbstractTask {
public:
    std::string name() const override { return "ExportReference"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
