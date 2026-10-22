#pragma once

// WKT2:2019 structural normalisation for the CRS golden fixtures (WP-12.2).
//
// WKT2 text is not a stable identity. Between PROJ data releases the same EPSG
// code renders differently in at least four ways, all of which this normaliser
// removes:
//
//   1. The CRS name and datum name change (EPSG:3006 is "SWEREF99 TM" in PROJ
//      9.4 and "ETRS89-SWE [SWEREF 99 TM]" in 9.9; EPSG:3035 gained
//      "-extended").
//   2. The WGS 84 datum gained a MEMBER per realisation (G2296 arrived after
//      G2139), so the ENSEMBLE block length differs.
//   3. ID[] blocks are only emitted for objects that have an authority code;
//      a re-parsed WKT may or may not carry them (the 3857 reprojection case).
//   4. USAGE/SCOPE/AREA/BBOX and ANCHOREPOCH blocks appear and disappear as the
//      database records change.
//
// What remains — the construct (GEOGCRS/PROJCRS/GEODCRS/ENGCRS), the datum's
// ellipsoid axes and inverse flattening, the projection method's parameters in
// order, the CS dimension and the axis directions and order, and unit text — is
// what the golden fixture pins. It is byte-identical on PROJ 9.4 (Ubuntu 24.04)
// and 9.9 (Homebrew), which is what lets one fixture serve both test machines.
//
// The generator that writes tests/fixtures/crs/builtin_wkt_skeletons.txt lives
// beside it (tests/fixtures/crs/make_skeletons.cpp) and includes this header,
// so the fixture and the assertion cannot drift apart.

#include <string>

namespace crs_fixture {

/// Element keywords whose contents vary with the PROJ data release; the whole
/// bracketed block is dropped.
inline bool dropped_keyword(const std::string& keyword) {
    if (keyword == "MEMBER" || keyword == "USAGE" || keyword == "BBOX" || keyword == "SCOPE" ||
        keyword == "AREA" || keyword == "REMARK" || keyword == "ID" || keyword == "EPOCH" ||
        keyword == "ENSEMBLEACCURACY" || keyword == "DYNAMIC" || keyword == "ANCHOREPOCH") {
        return true;
    }
    return false;
}

/// Recursive-descent reader for the WKT subset PROJ emits: KEYWORD[child,...],
/// quoted names, bare words and numbers. All quoted strings are dropped (names
/// are what change), dropped blocks are removed, and ENSEMBLE is renamed to
/// DATUM so a datum and an ensemble describing the same ellipsoid compare equal.
class SkeletonParser {
public:
    explicit SkeletonParser(const std::string& text) : text_(text) {}

    /// The normalised form, e.g. `PROJCRS[BASEGEOGCRS[DATUM[ELLIPSOID[6378137,
    /// 298.257223563,LENGTHUNIT[1]],...]]]`.
    std::string parse() {
        skip_space();
        return node();
    }

private:
    void skip_space() {
        while (index_ < text_.size() && (text_[index_] == ' ' || text_[index_] == '\n' ||
                                         text_[index_] == '\r' || text_[index_] == '\t')) {
            ++index_;
        }
    }

    std::string word() {
        skip_space();
        const std::size_t start = index_;
        while (index_ < text_.size() && text_[index_] != '[' && text_[index_] != ']' &&
               text_[index_] != ',' && text_[index_] != '"' && text_[index_] != ' ' &&
               text_[index_] != '\n' && text_[index_] != '\r' && text_[index_] != '\t') {
            ++index_;
        }
        return text_.substr(start, index_ - start);
    }

    void skip_string() {
        skip_space();
        if (index_ >= text_.size() || text_[index_] != '"') {
            return;
        }
        ++index_;
        while (index_ < text_.size() && text_[index_] != '"') {
            ++index_;
        }
        ++index_;  // closing quote
    }

    void skip_block() {
        int depth = 0;
        while (index_ < text_.size()) {
            const char c = text_[index_];
            if (c == '[') {
                ++depth;
            } else if (c == ']') {
                --depth;
                if (depth == 0) {
                    ++index_;
                    return;
                }
            }
            ++index_;
        }
    }

    std::string node() {
        std::string keyword = word();
        skip_space();
        if (index_ >= text_.size() || text_[index_] != '[') {
            return keyword;
        }
        if (dropped_keyword(keyword)) {
            skip_block();
            return std::string();
        }
        if (keyword == "ENSEMBLE") {
            keyword = "DATUM";
        }
        ++index_;  // '['
        std::string out = keyword + "[";
        bool first = true;
        for (;;) {
            skip_space();
            if (index_ >= text_.size()) {
                break;
            }
            const char c = text_[index_];
            if (c == ']') {
                ++index_;
                break;
            }
            if (c == ',') {
                ++index_;
                continue;
            }
            std::string child;
            if (c == '"') {
                skip_string();  // names are version-dependent: drop them
            } else {
                const std::size_t save = index_;
                const std::string bare = word();
                skip_space();
                if (index_ < text_.size() && text_[index_] == '[') {
                    index_ = save;
                    child = node();
                } else {
                    child = bare;
                }
            }
            if (child.empty()) {
                continue;
            }
            if (!first) {
                out += ',';
            }
            out += child;
            first = false;
        }
        out += ']';
        return out;
    }

    const std::string& text_;
    std::size_t index_ = 0;
};

inline std::string wkt_skeleton(const std::string& wkt) {
    return SkeletonParser(wkt).parse();
}

}  // namespace crs_fixture
