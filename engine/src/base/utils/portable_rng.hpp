#pragma once

// Portable, seeded random draws for the RANSAC samplers.
//
// The standard library's *distributions* are not portable: std::uniform_int_
// distribution is specified only by its output range, so libstdc++ and libc++
// consume the engine's bit stream differently and a fixed seed yields different
// samples on macOS and Linux. The engines are portable, but the mapping from
// bits to indices is not, which made seeded reconstructions diverge between the
// two platforms (the M1 and x86-64 builds of the same test produced different
// scenes for the same seed).
//
// Everything here is built from the engines' raw output plus plain integer
// arithmetic, so a seed selects the same samples on every implementation.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace alphashape::rng {

/// Uniform integer in [0, bound], unbiased by rejection.
///
/// Lemire's multiply-shift reduction with rejection (Lemire, "Fast Random
/// Integer Generation in an Interval", ACM TACO 2019). The engine supplies 64
/// bits; the low 32 are multiplied by the exclusive bound and the high half is
/// the candidate. A candidate below the rejection threshold is redrawn, which
/// removes the modulo bias without the slow per-draw division.
template <class Engine>
std::uint32_t uniform_below(Engine& engine, std::uint32_t bound) {
    if (bound == 0) {
        return 0;
    }
    // Threshold = 2^32 mod (bound + 1): the bottom of the biased tail.
    const std::uint64_t range = std::uint64_t{bound} + 1U;
    const std::uint64_t threshold =
        (std::uint64_t{1} << 32) % range;  // 0 when range divides 2^32 exactly
    std::uint64_t product = 0;
    std::uint32_t candidate = 0;
    do {
        product = (static_cast<std::uint64_t>(engine()) & 0xFFFFFFFFULL) * range;
        candidate = static_cast<std::uint32_t>(product >> 32);
    } while (static_cast<std::uint64_t>(candidate) < threshold);
    return candidate;
}

/// Uniform index in [0, count), for count > 0.
template <class Engine>
std::size_t index_below(Engine& engine, std::size_t count) {
    // Indices here are camera/point positions in a handful of thousands, so the
    // 32-bit draw is exact; the guard keeps a larger container honest.
    if (count == 0) {
        return 0;
    }
    if (count > 0xFFFFFFFFULL) {
        // Never reached by these samplers; combine two draws rather than
        // truncate silently.
        const std::uint64_t high = uniform_below(engine, 0xFFFFFFFFU);
        const std::uint64_t low = uniform_below(engine, 0xFFFFFFFFU);
        return static_cast<std::size_t>(((high << 32) | low) % count);
    }
    return static_cast<std::size_t>(uniform_below(engine, static_cast<std::uint32_t>(count - 1)));
}

/// A double in [0, 1) built from 53 engine bits (two draws). std::uniform_real_
/// distribution is also implementation-defined, so a seeded test scene built
/// with it differs between libc++ and libstdc++.
template <class Engine>
double unit_interval(Engine& engine) {
    // Two 32-bit draws form one 64-bit word; the top 53 bits scale exactly by
    // 2^-53, which is the full mantissa precision and needs no rejection.
    //
    // The mask matters: the engines' result_type is uint_fast32_t, which is
    // 64-bit under libstdc++ and 32-bit under libc++, so shifting the raw
    // return value would keep a different number of bits on each platform.
    const std::uint64_t high = static_cast<std::uint64_t>(engine()) & 0xFFFFFFFFULL;
    const std::uint64_t low = static_cast<std::uint64_t>(engine()) & 0xFFFFFFFFULL;
    const std::uint64_t bits = (high << 32) | low;
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

/// A double uniformly in [low, high).
template <class Engine>
double uniform_real(Engine& engine, double low, double high) {
    return low + (high - low) * unit_interval(engine);
}

/// A standard-normal deviate by the Box-Muller transform (Marsaglia's polar
/// form). Each call consumes draws until a point lands inside the unit disc;
/// both coordinates of that point are used, so no cache state is needed and a
/// seed reproduces exactly the same sequence on every implementation.
template <class Engine>
double normal(Engine& engine, double mean = 0.0, double deviation = 1.0) {
    double u = 0.0;
    double v = 0.0;
    double s = 0.0;
    do {
        u = 2.0 * unit_interval(engine) - 1.0;
        v = 2.0 * unit_interval(engine) - 1.0;
        s = u * u + v * v;
    } while (s == 0.0 || s >= 1.0);
    const double factor = std::sqrt(-2.0 * std::log(s) / s);
    return mean + deviation * u * factor;
}

/// Draws `pick` distinct indices in [0, count) by partial Fisher-Yates.
///
/// The caller's scratch is reused across RANSAC iterations. Every index has an
/// equal chance of being chosen and the order matches the draw order, which is
/// what the samplers rely on for reproducible samples.
template <class Engine>
void distinct_indices(Engine& engine, std::size_t count, std::size_t pick,
                      std::size_t* out, std::size_t* scratch) {
    for (std::size_t i = 0; i < count; ++i) {
        scratch[i] = i;
    }
    for (std::size_t i = 0; i < pick; ++i) {
        const std::size_t j = i + index_below(engine, count - i);
        const std::size_t swap = scratch[i];
        scratch[i] = scratch[j];
        scratch[j] = swap;
        out[i] = scratch[i];
    }
}

}  // namespace alphashape::rng
