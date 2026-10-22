# Contributing to alphashape

Thanks for your interest. Bug reports, test datasets, fixes and features are all welcome.

## Ground rules

- **Own implementation only.** Implement algorithms from published papers and cite them in a comment
  next to the code. Do not copy code, constants or assets from other photogrammetry products or from
  disassembled/decompiled software. Code from compatible open-source projects is fine when its licence
  allows it and the licence is recorded.
- **Zero warnings.** CI builds with `-DALPHASHAPE_WARNINGS_AS_ERRORS=ON` on GCC, Clang and clang-cl.
- **Tests with every change.** New features get C++ tests (GoogleTest); new tasks also get a Python
  binding test and CLI access through the task registry. Never loosen an existing test bound without
  multi-seed evidence in the pull request.
- **Portable randomness.** Seeded tests draw through `base/utils/portable_rng.hpp`, never `<random>`
  distributions, and never make two draws in one expression (`rng_order_check` enforces this).
- **Honest limitations.** Anything knowingly partial goes into `engine/LIMITATIONS.md`.
- **Naming.** Task and parameter names follow Metashape's public Python API naming where an equivalent
  exists, so scripts port easily.

## Workflow

1. Fork, create a branch, make focused commits (`Area: summary`, imperative mood, no emojis).
2. Build and test locally (see `README.md` / `engine/TESTING.md`), then run
   `tools/check_repo_hygiene.sh`.
3. Open a pull request against `main`. CI must be green (the single required check is **CI OK**).
4. Sign off your commits (`git commit -s`) to certify the
   [Developer Certificate of Origin](https://developercertificate.org/).

Never commit datasets, build trees, SDKs (`tools/` except scripts) or project files (`*.psx`, `*.files/`).

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/aniketpratapsingh/alphashape/issues/new/choose).
A good bug report has the alphashape version or commit, operating system, GPU and
driver, the exact command or script, and the smallest input that reproduces the problem. Please do
not attach private imagery. Security problems go through the private process in
[SECURITY.md](SECURITY.md), not a public issue.

## Coding standards

- C++20, built warning-free on GCC, Clang and clang-cl; no compiler extensions.
- Match the style of the surrounding file (4-space indent, `snake_case` functions and variables,
  `CamelCase` types, headers use `#pragma once`). Keep headers minimal and include what you use.
- Prefer RAII and value types over raw ownership; validate all data read from files before using it
  (project files, images, rasters and point clouds are untrusted input).
- Cite the source paper next to any non-obvious algorithm, and keep results deterministic for a given
  seed and thread count.

## Testing

Run the whole suite with `ctest` (see `engine/TESTING.md`). Every behaviour change needs a test
that fails without it, and CI must pass on Linux, macOS and Windows before a pull request is merged.

## Licence

By contributing you agree that your contribution is licensed under GPL-3.0-or-later.
