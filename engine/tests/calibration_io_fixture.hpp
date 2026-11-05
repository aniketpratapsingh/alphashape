#pragma once

// Shared scratch-file fixture for the CalibrationIO test suite (WP-13.16).
//
// The suite spans two translation units: the file-format tests and the
// OptimizeCameras master-sharing test. gtest requires every test in one suite
// to use the same fixture class, so the class lives here and both files
// include it.

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"

namespace alphashape {

/// Unique scratch files under the system temp directory, removed on teardown.
class CalibrationIO : public ::testing::Test {
protected:
    void TearDown() override {
        for (const std::string& path : files_) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    std::string path(const std::string& suffix) {
        std::string value = platform::unique_temp_path("alphashape_calio_", suffix);
        files_.push_back(value);
        return value;
    }

    std::string write(const std::string& suffix, const std::string& text) {
        const std::string file = path(suffix);
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        EXPECT_TRUE(out.good());
        return file;
    }

private:
    std::vector<std::string> files_;
};

}  // namespace alphashape
