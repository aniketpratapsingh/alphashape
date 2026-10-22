// tests/make_v1_fixture.cpp -- writes tests/fixtures/v1_project.zip.
//
// Not part of the test binary: a one-shot fixture generator, built only on
// demand so the repository carries the bytes rather than a build step. Run it
// after any change to tests/v1_fixture.hpp:
//
//     cmake --build build-... --target make_v1_fixture
//     build-.../tests/make_v1_fixture > <repo>/engine/tests/fixtures/v1_project.zip
//
// The reader test (ProjectV2.LoadsV1Fixture) regenerates the same bytes in
// memory and asserts they match the checked-in file, so a stale fixture fails
// loudly instead of quietly testing the wrong thing.

#include <cstdio>

#include <iostream>

#include "v1_fixture.hpp"
#include "v1_writer.hpp"

int main() {
    const std::vector<std::uint8_t> bytes =
        alphashape::test_v1::write_project(alphashape::test_v1::make_v1_fixture_document());
    std::cout.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
    if (!std::cout) {
        std::fprintf(stderr, "make_v1_fixture: short write\n");
        return 1;
    }
    std::fprintf(stderr, "wrote %zu bytes\n", bytes.size());
    return 0;
}
