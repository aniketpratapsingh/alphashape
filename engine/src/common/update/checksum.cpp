#include "common/update/checksum.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <utility>

namespace alphashape::update {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

std::uint32_t rotr(std::uint32_t value, int count) {
    return (value >> count) | (value << (32 - count));
}

class Sha256 {
public:
    void update(const unsigned char* data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            buffer_[buffer_size_++] = data[i];
            if (buffer_size_ == 64) {
                transform(buffer_.data());
                buffer_size_ = 0;
                total_ += 64;
            }
        }
    }

    std::array<unsigned char, 32> finish() {
        const std::uint64_t bit_length = (total_ + buffer_size_) * 8;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) {
                buffer_[buffer_size_++] = 0;
            }
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) {
            buffer_[buffer_size_++] = 0;
        }
        for (int i = 7; i >= 0; --i) {
            buffer_[buffer_size_++] = static_cast<unsigned char>((bit_length >> (i * 8)) & 0xFF);
        }
        transform(buffer_.data());

        std::array<unsigned char, 32> digest{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            digest[i * 4 + 0] = static_cast<unsigned char>((state_[i] >> 24) & 0xFF);
            digest[i * 4 + 1] = static_cast<unsigned char>((state_[i] >> 16) & 0xFF);
            digest[i * 4 + 2] = static_cast<unsigned char>((state_[i] >> 8) & 0xFF);
            digest[i * 4 + 3] = static_cast<unsigned char>(state_[i] & 0xFF);
        }
        return digest;
    }

private:
    void transform(const unsigned char* block) {
        std::array<std::uint32_t, 64> words{};
        for (int i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                       (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                       (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                       static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^
                                     (words[i - 15] >> 3);
            const std::uint32_t s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^
                                     (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + s1 + choose + kRoundConstants[i] + words[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                           0xa54ff53a, 0x510e527f, 0x9b05688c,
                                           0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t total_ = 0;
};

std::string to_hex(const std::array<unsigned char, 32>& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(64);
    for (const unsigned char byte : digest) {
        text.push_back(digits[byte >> 4]);
        text.push_back(digits[byte & 0x0F]);
    }
    return text;
}

bool is_hex64(std::string_view text) {
    if (text.size() != 64) {
        return false;
    }
    for (const char c : text) {
        const bool digit = c >= '0' && c <= '9';
        const bool lower = c >= 'a' && c <= 'f';
        const bool upper = c >= 'A' && c <= 'F';
        if (!digit && !lower && !upper) {
            return false;
        }
    }
    return true;
}

std::string lowercase(std::string_view text) {
    std::string result(text);
    for (char& c : result) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return result;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' ||
                             text.front() == '\r' || text.front() == '\n')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' ||
                             text.back() == '\r' || text.back() == '\n')) {
        text.remove_suffix(1);
    }
    return text;
}

}  // namespace

std::vector<ChecksumEntry> parse_sha256sums(std::string_view text) {
    std::vector<ChecksumEntry> entries;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find('\n', start);
        std::string_view line =
            end == std::string_view::npos ? text.substr(start) : text.substr(start, end - start);
        start = end == std::string_view::npos ? text.size() + 1 : end + 1;

        line = trim(line);
        if (line.size() < 66) {
            continue;
        }
        const std::string_view hash = line.substr(0, 64);
        if (!is_hex64(hash)) {
            continue;
        }
        // sha256sum writes "<hash>  <name>" (two spaces) or "<hash> *<name>"; a
        // single space is also tolerated so hand-edited files still work.
        std::size_t name_start = 64;
        while (name_start < line.size() && (line[name_start] == ' ' || line[name_start] == '\t')) {
            ++name_start;
        }
        if (name_start < line.size() && line[name_start] == '*') {
            ++name_start;
        }
        const std::string_view name = line.substr(name_start);
        if (name.empty()) {
            continue;
        }
        ChecksumEntry entry;
        entry.sha256 = lowercase(hash);
        entry.name = std::string(name);
        entries.push_back(std::move(entry));
    }
    return entries;
}

const ChecksumEntry* find_checksum(const std::vector<ChecksumEntry>& entries,
                                   std::string_view name) {
    for (const ChecksumEntry& entry : entries) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

std::string sha256_hex(std::string_view bytes) {
    Sha256 hash;
    hash.update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
    return to_hex(hash.finish());
}

std::optional<std::string> sha256_hex_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    Sha256 hash;
    std::array<char, 64 * 1024> buffer{};
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = file.gcount();
        if (got > 0) {
            hash.update(reinterpret_cast<const unsigned char*>(buffer.data()),
                        static_cast<std::size_t>(got));
        }
    }
    if (file.bad()) {
        return std::nullopt;
    }
    return to_hex(hash.finish());
}

bool checksum_matches(std::string_view expected_hex, std::string_view actual_hex) {
    if (!is_hex64(expected_hex) || !is_hex64(actual_hex)) {
        return false;
    }
    return lowercase(expected_hex) == lowercase(actual_hex);
}

VerifyResult verify_file_sha256(const std::filesystem::path& path, std::string_view name,
                                std::string_view sha256sums_text) {
    VerifyResult result;
    const auto entries = parse_sha256sums(sha256sums_text);
    const ChecksumEntry* entry = find_checksum(entries, name);
    if (entry == nullptr) {
        result.status = VerifyStatus::MissingEntry;
        result.error = "SHA256SUMS does not list " + std::string(name) +
                       "; refusing to install an unverifiable download";
        return result;
    }
    result.expected = entry->sha256;
    const auto actual = sha256_hex_file(path);
    if (!actual) {
        result.status = VerifyStatus::ReadError;
        result.error = "the downloaded file could not be read for checksum verification";
        return result;
    }
    result.actual = *actual;
    if (!checksum_matches(result.expected, result.actual)) {
        result.status = VerifyStatus::Mismatch;
        result.error = "checksum mismatch for " + std::string(name) + ": expected " +
                       result.expected + ", got " + result.actual;
        return result;
    }
    result.status = VerifyStatus::Ok;
    return result;
}

}  // namespace alphashape::update
