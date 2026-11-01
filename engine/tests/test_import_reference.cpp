// WP-13.2: reference import/export.
//
// Each case writes a real fixture into the platform temporary directory and
// runs ImportReference/ExportReference through the task registry, so the
// parameter schema, the parser and the chunk mutation are all exercised
// together. The round-trip cases check the contract that makes the writer worth
// having: what ExportReference writes, ImportReference reads back.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/reference/import_reference_task.hpp"
#include "common/algorithm/reference/reference_table.hpp"
#include "common/commands/command.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/scalebar.hpp"
#include "common/core/types.hpp"
#include "common/tasks/registry.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/transform.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// A path in the temporary directory that the destructor removes.
class TempFile {
public:
    explicit TempFile(const std::string& suffix)
        : path_(platform::unique_temp_path("alphashape_reference_", suffix)) {}
    ~TempFile() {
        std::error_code error;
        fs::remove(path_, error);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

void write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out) << "cannot create " << path;
    out << text;
    out.close();
    ASSERT_TRUE(out) << "cannot write " << path;
}

Document make_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

Camera& add_camera(Chunk& chunk, const std::string& label) {
    chunk.cameras().push_back(Camera{});
    Camera& camera = chunk.cameras().back();
    camera.id = chunk.allocate_id();
    camera.label = label;
    return camera;
}

Marker& add_marker(Chunk& chunk, const std::string& label) {
    chunk.markers().push_back(Marker{});
    Marker& marker = chunk.markers().back();
    marker.id = chunk.allocate_id();
    marker.label = label;
    return marker;
}

Params params_for(const std::string& task) {
    return make_params(builtin_registry().require(task));
}

void run_task_named(Document& document, const std::string& task, Params& params) {
    run_task(builtin_registry(), document, task, params);
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

TEST(Reference, ImportCsvWithMapping) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(),
               "CAM1,500000.125,5100000.25,100.5,0.02\n"
               "GCP1,500010.5,5100010.75,101.25,0.01\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z", "sigma"});
    run_task_named(document, "ImportReference", params);

    ASSERT_EQ(chunk->cameras().size(), 1U);
    const CameraReference& reference = chunk->cameras()[0].reference;
    ASSERT_TRUE(reference.location.has_value());
    EXPECT_NEAR(reference.location->x(), 500000.125, 1e-9);
    EXPECT_NEAR(reference.location->y(), 5100000.25, 1e-9);
    EXPECT_NEAR(reference.location->z(), 100.5, 1e-9);
    EXPECT_NEAR(reference.accuracy.x(), 0.02, 1e-12);

    ASSERT_EQ(chunk->markers().size(), 1U);
    ASSERT_TRUE(chunk->markers()[0].reference.has_value());
    EXPECT_NEAR(chunk->markers()[0].reference->x(), 500010.5, 1e-9);
    EXPECT_NEAR(chunk->markers()[0].reference_accuracy.z(), 0.01, 1e-12);
}

TEST(Reference, ImportSemicolonWithHeader) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(),
               "Name;Easting;Northing;Height;Accuracy\n"
               "CAM1;1.5;2.5;3.5;0.3\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("delimiter", ";");
    run_task_named(document, "ImportReference", params);

    ASSERT_TRUE(chunk->cameras()[0].reference.location.has_value());
    EXPECT_NEAR(chunk->cameras()[0].reference.location->x(), 1.5, 1e-12);
    EXPECT_NEAR(chunk->cameras()[0].reference.location->y(), 2.5, 1e-12);
    EXPECT_NEAR(chunk->cameras()[0].reference.location->z(), 3.5, 1e-12);
    EXPECT_NEAR(chunk->cameras()[0].reference.accuracy.y(), 0.3, 1e-12);
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

TEST(Reference, ImportWhitespaceNoHeader) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".txt");
    write_text(file.path(), "CAM1 10 20 30\nP1 40 50 60\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("format", "txt");
    run_task_named(document, "ImportReference", params);

    ASSERT_TRUE(chunk->cameras()[0].reference.location.has_value());
    EXPECT_NEAR(chunk->cameras()[0].reference.location->x(), 10.0, 1e-12);
    EXPECT_NEAR(chunk->cameras()[0].reference.location->z(), 30.0, 1e-12);
    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].label, "P1");
}

TEST(Reference, ImportSkipsRows) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(),
               "# survey export\n"
               "# date 2026-10-31\n"
               "CAM1,7,8,9\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("skip_rows", std::int64_t{2});
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    run_task_named(document, "ImportReference", params);

    ASSERT_TRUE(chunk->cameras()[0].reference.location.has_value());
    EXPECT_NEAR(chunk->cameras()[0].reference.location->x(), 7.0, 1e-12);
}

TEST(Reference, ImportRotationsZyx) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(), "CAM1,1,2,3,10,20,30\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns",
               std::vector<std::string>{"label", "x", "y", "z", "yaw", "pitch", "roll"});
    params.set("rotation_order", "ZYX");
    run_task_named(document, "ImportReference", params);

    const CameraReference& reference = chunk->cameras()[0].reference;
    ASSERT_TRUE(reference.orientation_deg.has_value());
    EXPECT_NEAR(reference.orientation_deg->x(), 10.0, 1e-9);
    EXPECT_NEAR(reference.orientation_deg->y(), 20.0, 1e-9);
    EXPECT_NEAR(reference.orientation_deg->z(), 30.0, 1e-9);
}

TEST(Reference, ImportCreatesMissingMarkers) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    TempFile file(".csv");
    write_text(file.path(), "NEW1,1,2,3\n");

    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    run_task_named(document, "ImportReference", params);
    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].label, "NEW1");

    Document skipped = make_document();
    Chunk* skipped_chunk = skipped.active_chunk();
    Params skip_params = params_for("ImportReference");
    skip_params.set("path", file.path());
    skip_params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    skip_params.set("create_markers", false);
    run_task_named(skipped, "ImportReference", skip_params);
    EXPECT_TRUE(skipped_chunk->markers().empty());
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

#if defined(ALPHASHAPE_USE_PROJ)
TEST(Reference, ImportLatLonAltIntoUtm) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    chunk->set_coordinate_system(crs::CoordinateSystem::from_epsg(32632));

    TempFile file(".csv");
    write_text(file.path(), "GCP1,12.0,55.0,100.0\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    params.set("crs", "EPSG:4326");
    run_task_named(document, "ImportReference", params);

    ASSERT_EQ(chunk->markers().size(), 1U);
    ASSERT_TRUE(chunk->markers()[0].reference.has_value());
    const crs::CrsTransform transform = crs::CrsTransform::create(
        crs::CoordinateSystem::from_epsg(4326), crs::CoordinateSystem::from_epsg(32632));
    const std::optional<crs::CrsPoint> expected =
        transform.forward(crs::CrsPoint{12.0, 55.0, 100.0});
    ASSERT_TRUE(expected.has_value());
    EXPECT_NEAR(chunk->markers()[0].reference->x(), expected->x, 0.01);
    EXPECT_NEAR(chunk->markers()[0].reference->y(), expected->y, 0.01);
    EXPECT_NEAR(chunk->markers()[0].reference->z(), expected->z, 0.01);
}
#endif  // ALPHASHAPE_USE_PROJ

TEST(Reference, ExportThenImportIsExact) {
    Document source = make_document();
    Chunk* source_chunk = source.active_chunk();
    Camera& camera = add_camera(*source_chunk, "CAM1");
    camera.reference.location = Eigen::Vector3d(123.123456, 456.654321, 50.987654);
    camera.reference.accuracy = Eigen::Vector3d(0.25, 0.25, 0.5);

    TempFile file(".csv");
    Params export_params = params_for("ExportReference");
    export_params.set("path", file.path());
    export_params.set("precision", std::int64_t{6});
    run_task_named(source, "ExportReference", export_params);

    Document target = make_document();
    Chunk* target_chunk = target.active_chunk();
    add_camera(*target_chunk, "CAM1");
    Params import_params = params_for("ImportReference");
    import_params.set("path", file.path());
    run_task_named(target, "ImportReference", import_params);

    ASSERT_TRUE(target_chunk->cameras()[0].reference.location.has_value());
    const Eigen::Vector3d& imported = *target_chunk->cameras()[0].reference.location;
    EXPECT_NEAR(imported.x(), 123.123456, 1e-6);
    EXPECT_NEAR(imported.y(), 456.654321, 1e-6);
    EXPECT_NEAR(imported.z(), 50.987654, 1e-6);
}

TEST(Reference, XmlRoundTrip) {
    Document source = make_document();
    Chunk* source_chunk = source.active_chunk();
    add_camera(*source_chunk, "CAM1").reference.location =
        Eigen::Vector3d(1.25, 2.5, 3.75);
    add_marker(*source_chunk, "GCP1").reference = Eigen::Vector3d(4.5, 5.25, 6.125);
    // The marker needs a role that export/import preserves.
    reference::set_marker_role(source_chunk->markers()[0],
                               reference::MarkerReferenceRole::Gcp);
    Scalebar& bar = source_chunk->scalebars().emplace_back();
    bar.id = source_chunk->allocate_id();
    bar.label = "S1";
    bar.reference_length_m = 12.5;
    bar.reference_accuracy_m = 0.002;

    TempFile file(".xml");
    Params export_params = params_for("ExportReference");
    export_params.set("path", file.path());
    run_task_named(source, "ExportReference", export_params);

    Document target = make_document();
    Chunk* target_chunk = target.active_chunk();
    add_camera(*target_chunk, "CAM1");
    add_marker(*target_chunk, "GCP1");
    Scalebar& target_bar = target_chunk->scalebars().emplace_back();
    target_bar.id = target_chunk->allocate_id();
    target_bar.label = "S1";
    Params import_params = params_for("ImportReference");
    import_params.set("path", file.path());
    run_task_named(target, "ImportReference", import_params);

    ASSERT_TRUE(target_chunk->cameras()[0].reference.location.has_value());
    EXPECT_NEAR(target_chunk->cameras()[0].reference.location->x(), 1.25, 1e-12);
    ASSERT_TRUE(target_chunk->markers()[0].reference.has_value());
    EXPECT_NEAR(target_chunk->markers()[0].reference->z(), 6.125, 1e-12);
    EXPECT_NEAR(target_chunk->scalebars()[0].reference_length_m, 12.5, 1e-12);
    EXPECT_NEAR(target_chunk->scalebars()[0].reference_accuracy_m, 0.002, 1e-12);
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

TEST(Reference, BadNumberReportsLine) {
    Document document = make_document();
    TempFile file(".csv");
    write_text(file.path(), "CAM1,1,2,3\nCAM1B,oops,2,3\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    try {
        run_task_named(document, "ImportReference", params);
        FAIL() << "a malformed number must be rejected";
    } catch (const ValueError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("line 2"), std::string::npos) << message;
    }
}

TEST(Reference, DuplicateLabelReportsBothLines) {
    Document document = make_document();
    TempFile file(".csv");
    write_text(file.path(), "GCP1,1,2,3\nGCP1,4,5,6\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    try {
        run_task_named(document, "ImportReference", params);
        FAIL() << "a duplicate label must be rejected";
    } catch (const ValueError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("lines 1 and 2"), std::string::npos) << message;
    }
}

TEST(Reference, ImportHundredThousandRows) {
    const std::string path = platform::unique_temp_path("alphashape_reference_bulk_", ".csv");
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out << "label,x,y,z\n";
        for (int i = 0; i < 100000; ++i) {
            out << "P" << i << ',' << (i % 1000) << ".5," << (i / 1000) << ".25,"
                << (i % 97) << ".125\n";
        }
    }

    Document document = make_document();
    Params params = params_for("ImportReference");
    params.set("path", path);
    const auto start = std::chrono::steady_clock::now();
    run_task_named(document, "ImportReference", params);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double seconds =
        std::chrono::duration<double>(elapsed).count();

    Chunk* chunk = document.active_chunk();
    ASSERT_EQ(chunk->markers().size(), 100000U);
    EXPECT_EQ(chunk->markers()[0].label, "P0");
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(seconds, 5.0) << "100k-row import took " << seconds << " s";
    }
    std::error_code error;
    fs::remove(path, error);
}

TEST(Reference, ApplyFalseDoesNotMutate) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");
    const Chunk pre_state = *chunk;

    TempFile file(".csv");
    write_text(file.path(), "CAM1,1,2,3\nNEW1,4,5,6\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    params.set("apply", false);
    run_task_named(document, "ImportReference", params);

    EXPECT_TRUE(*chunk == pre_state) << "apply=false changed the chunk";
    EXPECT_EQ(chunk->markers().size(), 0U);
}

TEST(Reference, BadRowLeavesChunkUnchanged) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "C1");
    add_camera(*chunk, "C2");
    add_camera(*chunk, "C3");
    const Chunk pre_state = *chunk;

    TempFile file(".csv");
    write_text(file.path(), "C1,1,2,3,0.1\nC2,4,5,6,0.2\nC3,7,8,9,-1\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z", "sigma"});
    try {
        run_task_named(document, "ImportReference", params);
        FAIL() << "a non-positive sigma must be rejected";
    } catch (const ValueError& error) {
        EXPECT_NE(std::string(error.what()).find("line 3"), std::string::npos);
    }
    // Rows 1 and 2 validated first; neither may have been applied.
    EXPECT_TRUE(*chunk == pre_state) << "a bad row left earlier rows applied";
    EXPECT_EQ(chunk->markers().size(), 0U);
}

TEST(Reference, ImportIsUndoable) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");
    const Chunk pre_state = *chunk;

    std::vector<reference::ImportRowPlan> plan;
    reference::ImportRowPlan camera_row;
    camera_row.target = reference::ImportRowPlan::Target::Camera;
    camera_row.index = 0;
    camera_row.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    camera_row.accuracy = Eigen::Vector3d(0.1, 0.2, 0.3);
    camera_row.orientation_deg = Eigen::Vector3d(10.0, 20.0, 30.0);
    plan.push_back(camera_row);
    reference::ImportRowPlan marker_row;
    marker_row.target = reference::ImportRowPlan::Target::NewMarker;
    marker_row.label = "NEW1";
    marker_row.position = Eigen::Vector3d(4.0, 5.0, 6.0);
    marker_row.accuracy = Eigen::Vector3d(0.5, 0.5, 0.5);
    marker_row.role = reference::MarkerReferenceRole::Gcp;
    plan.push_back(marker_row);

    UndoStack stack(&document);
    stack.execute(reference::make_import_reference_command(chunk->id(), std::move(plan)));
    ASSERT_TRUE(chunk->cameras()[0].reference.location.has_value());
    EXPECT_EQ(chunk->markers().size(), 1U);
    const Chunk after = *chunk;

    stack.undo();
    EXPECT_TRUE(*chunk == pre_state) << "undo did not restore the pre-state";
    stack.redo();
    EXPECT_TRUE(*chunk == after) << "redo differs from the first apply";
}

TEST(Reference, DecimalCommaWithSemicolon) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(), "CAM1;1,5;2,5;3,5\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("delimiter", ";");
    params.set("decimal_comma", true);
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    run_task_named(document, "ImportReference", params);

    ASSERT_TRUE(chunk->cameras()[0].reference.location.has_value());
    EXPECT_NEAR(chunk->cameras()[0].reference.location->x(), 1.5, 1e-12);
    EXPECT_NEAR(chunk->cameras()[0].reference.location->y(), 2.5, 1e-12);
    EXPECT_NEAR(chunk->cameras()[0].reference.location->z(), 3.5, 1e-12);
}

TEST(Reference, DecimalCommaWithCommaDelimiterRejected) {
    Document document = make_document();
    TempFile file(".csv");
    write_text(file.path(), "CAM1,1,2,3\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("decimal_comma", true);
    EXPECT_THROW(run_task_named(document, "ImportReference", params), ValueError);
}

TEST(Reference, CrlfAndBomAccepted) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(), "\xEF\xBB\xBF" "CAM1,1,2,3\r\nP1,4,5,6\r\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    run_task_named(document, "ImportReference", params);

    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].label, "P1");
    ASSERT_TRUE(chunk->cameras()[0].reference.location.has_value());
    EXPECT_NEAR(chunk->cameras()[0].reference.location->z(), 3.0, 1e-12);
}

TEST(Reference, QuotedDelimiterInLabel) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();

    TempFile file(".csv");
    write_text(file.path(), "\"GCP,1\",1,2,3\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    run_task_named(document, "ImportReference", params);

    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].label, "GCP,1");
}

TEST(Reference, NonPositiveSigmaRejected) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");

    TempFile file(".csv");
    write_text(file.path(), "CAM1,1,2,3,0\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z", "sigma"});
    EXPECT_THROW(run_task_named(document, "ImportReference", params), ValueError);
    EXPECT_FALSE(chunk->cameras()[0].reference.location.has_value());
}

TEST(Reference, ExistingCheckPointKeepsRole) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Marker& checkpoint = add_marker(*chunk, "CP1");
    checkpoint.reference = Eigen::Vector3d(1.0, 2.0, 3.0);
    reference::set_marker_role(checkpoint, reference::MarkerReferenceRole::CheckPoint);

    TempFile file(".csv");
    write_text(file.path(), "CP1,4,5,6\nNEW1,7,8,9\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    run_task_named(document, "ImportReference", params);

    // The default `gcp` must not retype an existing check point...
    EXPECT_EQ(reference::marker_role(chunk->markers()[0]),
              reference::MarkerReferenceRole::CheckPoint);
    // ...while a created marker becomes a control point.
    ASSERT_EQ(chunk->markers().size(), 2U);
    EXPECT_EQ(reference::marker_role(chunk->markers()[1]),
              reference::MarkerReferenceRole::Gcp);

    // An explicit marker_type does overwrite it.
    Params explicit_params = params_for("ImportReference");
    explicit_params.set("path", file.path());
    explicit_params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    explicit_params.set("marker_type", "gcp");
    run_task_named(document, "ImportReference", explicit_params);
    EXPECT_EQ(reference::marker_role(chunk->markers()[0]),
              reference::MarkerReferenceRole::Gcp);
}

TEST(Reference, ExportWritesEmptyForNaN) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Camera& camera = add_camera(*chunk, "CAM1");
    camera.reference.location =
        Eigen::Vector3d(1.0, 2.0, std::numeric_limits<double>::quiet_NaN());
    camera.reference.accuracy = Eigen::Vector3d(0.1, 0.2, 0.3);

    TempFile file(".csv");
    Params params = params_for("ExportReference");
    params.set("path", file.path());
    params.set("columns_order", std::vector<std::string>{"label", "x", "y", "z", "sigma_x",
                                                         "sigma_y", "sigma_z"});
    run_task_named(document, "ExportReference", params);

    std::ifstream in(file.path(), std::ios::binary);
    ASSERT_TRUE(in);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    EXPECT_EQ(text.find("nan"), std::string::npos) << text;
    EXPECT_EQ(text.find("inf"), std::string::npos) << text;
    // Empty z; each per-axis sigma is its own value, not the row average.
    EXPECT_NE(text.find("CAM1,1.000000,2.000000,,0.100000,0.200000,0.300000"),
              std::string::npos)
        << text;
}

#if defined(ALPHASHAPE_USE_PROJ)
TEST(Reference, ImportIntoExplicitMarkerCrs) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    // The chunk CRS is unset/local; only the marker slot names a CRS.
    chunk->set_marker_crs(crs::CoordinateSystem::from_epsg(32632));

    TempFile file(".csv");
    write_text(file.path(), "GCP1,12.0,55.0,100.0\n");
    Params params = params_for("ImportReference");
    params.set("path", file.path());
    params.set("columns", std::vector<std::string>{"label", "x", "y", "z"});
    params.set("crs", "EPSG:4326");
    run_task_named(document, "ImportReference", params);

    ASSERT_EQ(chunk->markers().size(), 1U);
    ASSERT_TRUE(chunk->markers()[0].reference.has_value());
    const crs::CrsTransform transform = crs::CrsTransform::create(
        crs::CoordinateSystem::from_epsg(4326), crs::CoordinateSystem::from_epsg(32632));
    const std::optional<crs::CrsPoint> expected =
        transform.forward(crs::CrsPoint{12.0, 55.0, 100.0});
    ASSERT_TRUE(expected.has_value());
    EXPECT_NEAR(chunk->markers()[0].reference->x(), expected->x, 0.01);
    EXPECT_NEAR(chunk->markers()[0].reference->y(), expected->y, 0.01);
    EXPECT_NEAR(chunk->markers()[0].reference->z(), expected->z, 0.01);
}
#endif  // ALPHASHAPE_USE_PROJ

}  // namespace
}  // namespace alphashape
