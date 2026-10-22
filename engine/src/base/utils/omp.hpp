#pragma once

// OpenMP port surface.
//
// The engine uses only the classic worksharing forms -- `#pragma omp parallel
// for` (with schedule(static) / schedule(dynamic,4)), `#pragma omp parallel`
// regions and `collapse(n)` -- which LLVM's libomp and MSVC's /openmp:llvm both
// implement. `simd` is the one construct whose support differs: MSVC's legacy
// /openmp (2.0) ignores it and GCC/Clang accept both spellings.
//
// The supported form is `#pragma omp ALPHASHAPE_OMP_SIMD`, which expands to
// `omp simd` everywhere except a pure-MSVC build, where it expands to a bare
// `#pragma omp` (parsed, ignored) -- so a site that is added later still
// compiles and stays semantically correct.
//
// Include this header wherever an OpenMP pragma is written. Nothing in it
// requires the OpenMP runtime: parallel loops are already guarded by
// ALPHASHAPE_HAVE_OPENMP, which the build sets only when find_package(OpenMP)
// succeeded. A build without OpenMP compiles every pragma away.

// The tree has no construct that needs a gate today; the header exists so a
// future one has exactly one place to declare it (WP-29.1 acceptance: the
// abstraction is present even while unused).
#if defined(_MSC_VER) && !defined(__clang__)
#define ALPHASHAPE_OMP_SIMD /* /openmp 2.0 ignores simd; /openmp:llvm supports it */
#else
#define ALPHASHAPE_OMP_SIMD simd
#endif
