#pragma once

// GPU-vs-CPU speed-up floors only mean something on dedicated hardware. Shared
// cloud runners (virtualised GPUs, noisy neighbours) set ALPHASHAPE_PERF_ASSERTS=0
// so the timing tests still run and report numbers but do not fail on them.

#include <cstdlib>
#include <cstring>

namespace alphashape::test {

inline bool perf_asserts_enabled() {
    const char* value = std::getenv("ALPHASHAPE_PERF_ASSERTS");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

}  // namespace alphashape::test
