#pragma once

// Temporary-file helpers. The engine writes GeoTIFF scratch files and test
// fixtures to the platform temporary directory; before this header those sites
// spelled "/tmp" or relied on POSIX-only calls.

#include <string>

namespace alphashape::platform {

/// The platform temporary directory, without a trailing separator. Throws
/// std::runtime_error when the platform cannot name one. On Windows this is
/// GetTempPathW (respects TMP/TEMP, and unlike TMPDIR it is what the CRT and
/// installers agree on); elsewhere TMPDIR then /tmp.
std::string temp_directory();

/// A path inside temp_directory() that no other caller in this process, and no
/// other process, will receive: "<temp>/<prefix><pid>_<counter><suffix>".
///
/// The counter is a process-wide atomic and the pid is included, so two engine
/// processes running in parallel (ctest -jN) never collide. This does *not*
/// create the file; callers that need the name reserved must create it.
std::string unique_temp_path(const std::string& prefix, const std::string& suffix);

/// The current process id, or 0 if the platform offers none.
long long process_id();

}  // namespace alphashape::platform
