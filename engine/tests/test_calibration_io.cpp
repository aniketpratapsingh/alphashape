// WP-13.16: calibration file I/O tests.
//
// Every format is round-tripped through a real file. The OpenCV projection
// test re-implements the published pinhole model independently of the engine,
// which is what guards the p1/p2 crossing and the half-pixel principal-point
// shift in the mapping.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "calibration_io_fixture.hpp"
#include "common/algorithm/calibration/calibration_io.hpp"
#include "common/io/doc_xml_common.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 960;

/// A calibration with all thirteen coefficients non-zero, so a round trip
/// cannot pass by leaving a default in place.
Calibration full_calibration(CameraType type) {
    Calibration c;
    c.type = type;
    c.width = kWidth;
    c.height = kHeight;
    c.f = 1150.25;
    c.cx = 3.5;
    c.cy = -7.25;
    c.b1 = 0.75;
    c.b2 = -0.125;
    c.k1 = 0.031;
    c.k2 = -0.007;
    c.k3 = 0.0011;
    c.k4 = -0.00013;
    c.p1 = 0.0021;
    c.p2 = -0.0017;
    c.p3 = 0.00031;
    c.p4 = -0.00023;
    return c;
}

/// A calibration representable in the five-term OpenCV distortion model.
Calibration opencv_calibration() {
    Calibration c = full_calibration(CameraType::Frame);
    c.k4 = 0.0;
    c.p3 = 0.0;
    c.p4 = 0.0;
    return c;
}

void expect_relative(double actual, double expected, const char* what) {
    const double scale = std::max(1.0, std::max(std::abs(actual), std::abs(expected)));
    EXPECT_NEAR(actual, expected, 1e-9 * scale) << what;
}

std::string read_file_text(const std::string& path) {
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

/// Every number in `text`, skipping the whitespace, commas and brackets of an
/// OpenCV matrix payload.
std::vector<double> numbers_in(const std::string& text) {
    std::vector<double> values;
    std::size_t i = 0;
    while (i < text.size()) {
        const auto space = [](char ch) {
            return std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == '[' ||
                   ch == ']' || ch == ',';
        };
        while (i < text.size() && space(text[i])) {
            ++i;
        }
        const std::size_t start = i;
        while (i < text.size() && !space(text[i])) {
            ++i;
        }
        if (i > start) {
            values.push_back(docxml::parse_double(text.substr(start, i - start)));
        }
    }
    return values;
}

/// The published OpenCV pinhole model, evaluated from the raw numbers as they
/// were written. `k` is the row-major 3x3 camera matrix and `d` the
/// distortion vector (k1, k2, P1, P2, k3), so a symmetric writer/reader error
/// cannot cancel itself the way deriving the parameters from our Calibration
/// would allow. The returned point is in our pixel convention (centres at
/// (0.5, 0.5)); OpenCV's is shifted by half a pixel.
Eigen::Vector2d project_raw(const std::vector<double>& k, const std::vector<double>& d,
                            const Eigen::Vector3d& p) {
    const double x = p.x() / p.z();
    const double y = p.y() / p.z();
    const double r2 = x * x + y * y;
    const double radial = 1.0 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2;
    const double p1 = d[2];
    const double p2 = d[3];
    const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    return {k[0] * xd + k[1] * yd + k[2] + 0.5, k[4] * yd + k[5] + 0.5};
}

TEST_F(CalibrationIO, OpenCvProjectionAgrees) {
    const Calibration truth = opencv_calibration();
    const std::string file = path(".xml");
    (void)save_calibration(file, truth, CalibrationFormat::OpenCvXml);

    const docxml::XmlNode root = docxml::parse_xml(read_file_text(file));
    const docxml::XmlNode* matrix = docxml::find_child(root, "camera_matrix");
    ASSERT_NE(matrix, nullptr);
    const docxml::XmlNode* matrix_data = docxml::find_child(*matrix, "data");
    ASSERT_NE(matrix_data, nullptr);
    const std::vector<double> k = numbers_in(matrix_data->text);
    const docxml::XmlNode* distortion = docxml::find_child(root, "distortion_coefficients");
    ASSERT_NE(distortion, nullptr);
    const docxml::XmlNode* distortion_data = docxml::find_child(*distortion, "data");
    ASSERT_NE(distortion_data, nullptr);
    const std::vector<double> d = numbers_in(distortion_data->text);
    ASSERT_EQ(k.size(), 9U);
    ASSERT_GE(d.size(), 5U);

    std::mt19937_64 engine(20261105);
    for (int i = 0; i < 20; ++i) {
        const double x = rng::uniform_real(engine, -1.5, 1.5);
        const double y = rng::uniform_real(engine, -1.5, 1.5);
        const double z = rng::uniform_real(engine, 1.0, 4.0);
        const Eigen::Vector3d point(x, y, z);
        const Eigen::Vector2d ours = truth.project(point);
        const Eigen::Vector2d theirs = project_raw(k, d, point);
        EXPECT_NEAR(ours.x(), theirs.x(), 1e-9) << "sample " << i;
        EXPECT_NEAR(ours.y(), theirs.y(), 1e-9) << "sample " << i;
    }
}

TEST_F(CalibrationIO, OpenCvYamlProjectionAgrees) {
    const Calibration truth = opencv_calibration();
    const std::string file = path(".yml");
    (void)save_calibration(file, truth, CalibrationFormat::OpenCvYaml);
    const std::string text = read_file_text(file);

    const std::size_t camera_key = text.find("camera_matrix:");
    ASSERT_NE(camera_key, std::string::npos);
    const std::size_t camera_open = text.find("data: [", camera_key);
    ASSERT_NE(camera_open, std::string::npos);
    const std::size_t camera_close = text.find(']', camera_open);
    ASSERT_NE(camera_close, std::string::npos);
    const std::size_t camera_data = camera_open + std::string("data: [").size();
    const std::vector<double> k =
        numbers_in(text.substr(camera_data, camera_close - camera_data));
    const std::size_t distortion_key =
        text.find("distortion_coefficients:", camera_close);
    ASSERT_NE(distortion_key, std::string::npos);
    const std::size_t distortion_open = text.find("data: [", distortion_key);
    ASSERT_NE(distortion_open, std::string::npos);
    const std::size_t distortion_close = text.find(']', distortion_open);
    ASSERT_NE(distortion_close, std::string::npos);
    const std::size_t distortion_data =
        distortion_open + std::string("data: [").size();
    const std::vector<double> d =
        numbers_in(text.substr(distortion_data, distortion_close - distortion_data));
    ASSERT_EQ(k.size(), 9U);
    ASSERT_GE(d.size(), 5U);

    std::mt19937_64 engine(20261105);
    for (int i = 0; i < 20; ++i) {
        const double x = rng::uniform_real(engine, -1.5, 1.5);
        const double y = rng::uniform_real(engine, -1.5, 1.5);
        const double z = rng::uniform_real(engine, 1.0, 4.0);
        const Eigen::Vector3d point(x, y, z);
        const Eigen::Vector2d ours = truth.project(point);
        const Eigen::Vector2d theirs = project_raw(k, d, point);
        EXPECT_NEAR(ours.x(), theirs.x(), 1e-9) << "sample " << i;
        EXPECT_NEAR(ours.y(), theirs.y(), 1e-9) << "sample " << i;
    }
}

TEST_F(CalibrationIO, RoundTripsXml) {
    for (const CameraType type : {CameraType::Frame, CameraType::Fisheye}) {
        const Calibration c = full_calibration(type);
        const std::string file = path(".xml");
        const std::vector<std::string> warnings =
            save_calibration(file, c, CalibrationFormat::Xml);
        EXPECT_TRUE(warnings.empty());
        const CalibrationFileResult loaded = load_calibration(file, CalibrationFormat::Xml);
        EXPECT_TRUE(loaded.warnings.empty());
        EXPECT_TRUE(loaded.calibration == c) << "xml round trip changed a field";
    }
}

TEST_F(CalibrationIO, RoundTripsJson) {
    for (const CameraType type : {CameraType::Frame, CameraType::Fisheye}) {
        const Calibration c = full_calibration(type);
        const std::string file = path(".json");
        const std::vector<std::string> warnings =
            save_calibration(file, c, CalibrationFormat::Json);
        EXPECT_TRUE(warnings.empty());
        const CalibrationFileResult loaded = load_calibration(file, CalibrationFormat::Json);
        EXPECT_TRUE(loaded.warnings.empty());
        EXPECT_TRUE(loaded.calibration == c) << "json round trip changed a field";
    }
}

TEST_F(CalibrationIO, RoundTripsOpenCvXml) {
    const Calibration c = opencv_calibration();
    const std::string file = path(".xml");
    const std::vector<std::string> warnings =
        save_calibration(file, c, CalibrationFormat::OpenCvXml);
    EXPECT_TRUE(warnings.empty());
    const CalibrationFileResult loaded =
        load_calibration(file, CalibrationFormat::OpenCvXml);
    EXPECT_TRUE(loaded.warnings.empty());
    EXPECT_EQ(loaded.calibration.type, CameraType::Frame);
    EXPECT_EQ(loaded.calibration.width, c.width);
    EXPECT_EQ(loaded.calibration.height, c.height);
    expect_relative(loaded.calibration.f, c.f, "f");
    expect_relative(loaded.calibration.cx, c.cx, "cx");
    expect_relative(loaded.calibration.cy, c.cy, "cy");
    expect_relative(loaded.calibration.b1, c.b1, "b1");
    expect_relative(loaded.calibration.b2, c.b2, "b2");
    expect_relative(loaded.calibration.k1, c.k1, "k1");
    expect_relative(loaded.calibration.k2, c.k2, "k2");
    expect_relative(loaded.calibration.k3, c.k3, "k3");
    expect_relative(loaded.calibration.p1, c.p1, "p1");
    expect_relative(loaded.calibration.p2, c.p2, "p2");
}

TEST_F(CalibrationIO, RoundTripsOpenCvYaml) {
    const Calibration c = opencv_calibration();
    const std::string file = path(".yml");
    const std::vector<std::string> warnings =
        save_calibration(file, c, CalibrationFormat::OpenCvYaml);
    EXPECT_TRUE(warnings.empty());
    const CalibrationFileResult loaded =
        load_calibration(file, CalibrationFormat::OpenCvYaml);
    EXPECT_TRUE(loaded.warnings.empty());
    EXPECT_EQ(loaded.calibration.width, c.width);
    EXPECT_EQ(loaded.calibration.height, c.height);
    expect_relative(loaded.calibration.f, c.f, "f");
    expect_relative(loaded.calibration.cx, c.cx, "cx");
    expect_relative(loaded.calibration.cy, c.cy, "cy");
    expect_relative(loaded.calibration.b1, c.b1, "b1");
    expect_relative(loaded.calibration.b2, c.b2, "b2");
    expect_relative(loaded.calibration.k1, c.k1, "k1");
    expect_relative(loaded.calibration.k2, c.k2, "k2");
    expect_relative(loaded.calibration.k3, c.k3, "k3");
    expect_relative(loaded.calibration.p1, c.p1, "p1");
    expect_relative(loaded.calibration.p2, c.p2, "p2");
}

TEST_F(CalibrationIO, OpenCvHigherOrderDiscardedWithWarning) {
    const std::string yaml =
        "%YAML:1.0\n---\n"
        "image_width: 1000\n"
        "image_height: 800\n"
        "camera_matrix: !!opencv-matrix\n"
        "   rows: 3\n   cols: 3\n   dt: d\n"
        "   data: [ 900.0, 0.0, 495.0, 0.0, 900.0, 405.0, 0.0, 0.0, 1.0 ]\n"
        "distortion_coefficients: !!opencv-matrix\n"
        "   rows: 1\n   cols: 8\n   dt: d\n"
        "   data: [ 0.1, 0.01, 0.001, 0.002, 0.0001, 0.5, 0.0, 0.0 ]\n";
    const std::string file = write(".yml", yaml);
    const CalibrationFileResult loaded =
        load_calibration(file, CalibrationFormat::OpenCvYaml);
    ASSERT_FALSE(loaded.warnings.empty());
    EXPECT_NE(loaded.warnings.front().find("not representable"), std::string::npos);
    expect_relative(loaded.calibration.k1, 0.1, "k1");
    expect_relative(loaded.calibration.k2, 0.01, "k2");
    expect_relative(loaded.calibration.p1, 0.002, "p1");
    expect_relative(loaded.calibration.p2, 0.001, "p2");
    expect_relative(loaded.calibration.k3, 0.0001, "k3");
    EXPECT_EQ(loaded.calibration.k4, 0.0);
}

TEST_F(CalibrationIO, OpenCvSaveWarnsForK4) {
    Calibration c = opencv_calibration();
    c.k4 = 0.0125;
    const std::string file = path(".xml");
    const std::vector<std::string> warnings =
        save_calibration(file, c, CalibrationFormat::OpenCvXml);
    ASSERT_FALSE(warnings.empty());
    EXPECT_NE(warnings.front().find("k4"), std::string::npos);
    const CalibrationFileResult loaded =
        load_calibration(file, CalibrationFormat::OpenCvXml);
    EXPECT_EQ(loaded.calibration.k4, 0.0);
}

TEST_F(CalibrationIO, OpenCvRefusesFisheye) {
    const std::string file = path(".xml");
    EXPECT_THROW(save_calibration(file, full_calibration(CameraType::Fisheye),
                                  CalibrationFormat::OpenCvXml),
                 ValueError);
}

TEST_F(CalibrationIO, JsonRejectsUnknownKey) {
    const std::string json =
        "{\"format\":\"alphashape-calibration\",\"version\":1,\"type\":\"frame\","
        "\"width\":640,\"height\":480,\"f\":500,\"bogus\":1}\n";
    const std::string file = write(".json", json);
    EXPECT_THROW(load_calibration(file, CalibrationFormat::Json), ValueError);
}

TEST_F(CalibrationIO, YamlRejectsGarbage) {
    const std::string yaml = "%YAML:1.0\n---\nimage_width: 10\ngarbage line\n";
    const std::string file = write(".yml", yaml);
    try {
        (void)load_calibration(file, CalibrationFormat::OpenCvYaml);
        FAIL() << "expected ValueError";
    } catch (const ValueError& e) {
        EXPECT_NE(std::string(e.what()).find("line 4"), std::string::npos) << e.what();
    }
}

TEST_F(CalibrationIO, OpenCvUsesFallbackSize) {
    const std::string yaml =
        "%YAML:1.0\n---\n"
        "camera_matrix: !!opencv-matrix\n"
        "   rows: 3\n   cols: 3\n   dt: d\n"
        "   data: [ 900.0, 0.0, 495.0, 0.0, 900.0, 405.0, 0.0, 0.0, 1.0 ]\n"
        "distortion_coefficients: !!opencv-matrix\n"
        "   rows: 1\n   cols: 5\n   dt: d\n"
        "   data: [ 0.0, 0.0, 0.0, 0.0, 0.0 ]\n";
    const std::string file = write(".yml", yaml);
    const CalibrationFileResult loaded =
        load_calibration(file, CalibrationFormat::OpenCvYaml, 640, 480);
    EXPECT_EQ(loaded.calibration.width, 640);
    EXPECT_EQ(loaded.calibration.height, 480);
    EXPECT_THROW(load_calibration(file, CalibrationFormat::OpenCvYaml), ValueError);
}

// A non-finite coefficient cannot describe a lens; every writer refuses it
// before touching the file.
TEST_F(CalibrationIO, RefusesToSaveNonFinite) {
    Calibration c = opencv_calibration();
    c.k1 = std::nan("");
    for (const CalibrationFormat format :
         {CalibrationFormat::Xml, CalibrationFormat::Json, CalibrationFormat::OpenCvXml,
          CalibrationFormat::OpenCvYaml}) {
        const std::string file = path(".out");
        EXPECT_THROW(save_calibration(file, c, format), ValueError) << file;
        EXPECT_FALSE(std::filesystem::exists(file)) << file;
    }
}

// The doc.xml reader shared with the project format must reject NaN rather
// than let it into an optimisation.
TEST_F(CalibrationIO, XmlRejectsNan) {
    const std::string xml =
        "<?xml version=\"1.0\"?>\n<calibration type=\"frame\" width=\"640\" height=\"480\" "
        "f=\"500\" cx=\"0\" cy=\"0\" b1=\"0\" b2=\"0\" k1=\"nan\" k2=\"0\" k3=\"0\" "
        "k4=\"0\" p1=\"0\" p2=\"0\"/>\n";
    const std::string file = write(".xml", xml);
    EXPECT_THROW(load_calibration(file, CalibrationFormat::Xml), ValueError);
}

}  // namespace
}  // namespace alphashape
