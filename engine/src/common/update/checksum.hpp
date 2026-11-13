#pragma once

// SHA-256 checksums for downloaded installers.
//
// The release pipeline publishes a `SHA256SUMS` file in sha256sum format
// ("<64 hex digits>  <file name>"). The updater refuses to install a file whose
// checksum is missing or does not match, so a truncated download or a tampered
// asset cannot reach the installer. Signature verification is future work:
// releases are currently unsigned, and the checksum is integrity only.
//
// The SHA-256 implementation lives here (Qt-free, so the core is testable on
// its own); the GUI hashes the same bytes with QCryptographicHash and compares
// through checksum_matches().

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace alphashape::update {

struct ChecksumEntry {
    std::string sha256;  ///< 64 lowercase hex digits
    std::string name;    ///< bare file name
};

/// Parse a SHA256SUMS file. Malformed lines are skipped; an empty result means
/// the file had no usable entry.
std::vector<ChecksumEntry> parse_sha256sums(std::string_view text);

/// The entry for exactly `name`, or nullptr.
const ChecksumEntry* find_checksum(const std::vector<ChecksumEntry>& entries,
                                   std::string_view name);

/// Lowercase hex digest of `bytes`.
std::string sha256_hex(std::string_view bytes);

/// Lowercase hex digest of a file, or nullopt when it cannot be read.
std::optional<std::string> sha256_hex_file(const std::filesystem::path& path);

/// Case-insensitive comparison; both sides must be 64 hex digits.
bool checksum_matches(std::string_view expected_hex, std::string_view actual_hex);

enum class VerifyStatus {
    Ok,
    MissingEntry,  ///< the asset is not listed in SHA256SUMS
    Mismatch,
    ReadError,     ///< the downloaded file could not be read
};

struct VerifyResult {
    VerifyStatus status = VerifyStatus::MissingEntry;
    std::string expected;
    std::string actual;
    std::string error;  ///< human-readable reason, empty for Ok
};

/// Read `path` and compare it with the SHA256SUMS entry for `name`.
VerifyResult verify_file_sha256(const std::filesystem::path& path, std::string_view name,
                                std::string_view sha256sums_text);

}  // namespace alphashape::update
