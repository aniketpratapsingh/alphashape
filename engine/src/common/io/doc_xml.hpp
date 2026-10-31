#pragma once

// doc.xml: the document description stored inside project.zip.
//
// This header is the only public entry point; the implementation is split by
// format version (see common/io/doc_xml.cpp, which is the dispatcher):
//
//   doc_xml_common.{hpp,cpp}  the XML tree and parser, escaping, numeric and
//                             value formatting, calibration/meta/CRS helpers
//   sidecars.{hpp,cpp}        every magic-tagged binary codec and the container
//                             validators shared by both readers
//   doc_xml_v1.cpp            the pre-M11 reader (flat entry names, one asset
//                             per kind, no M11 containers)
//   doc_xml_v2.cpp            the M11 writer and its reader (§4.4-4.6)
//
// WRITTEN FORMAT (v2). Root <document version="2" next_id active_id> with
// global_crs, meta and <chunks>/<chunk>. A chunk carries the full container set
// of §4.4 -- sensors, cameras, camera_groups, frames, markers (+marker_groups),
// scalebars, tie_points, mask_sets/masks, depth_map_sets/depth_maps,
// dense_clouds/dense_cloud, models/model (+texture/pages), tiled_models,
// elevations/elevation, orthomosaics/orthomosaic, trajectories/trajectory,
// shapes, panorama, region, transform, crs, meta -- and every asset of every
// kind, not just the active one. A container is written only when non-empty.
//
// Bulk data never goes into the XML. Each chunk's binary entries live under
// chunks/<chunk_id>/ (tie_points/, depth_maps/, point_clouds/, models/,
// elevations/, orthomosaics/, masks/, shapes.shp, trajectories/, frames.bin,
// tiled_models/). Every entry carries a four-byte magic and a version
// independent decoder, so v1 and v2 point at the same encodings through
// different paths. See sidecars.hpp for the layouts.
//
// READER. version == 1 loads through the v1 reader exactly as before;
// version 2, 3 or 4 loads through the v2 reader (which is strict about unknown
// elements inside <chunk>; v3 and v4 differ only in the depth-map sidecar
// encoding, ADM1 -> ADM2 -> ADM3, which the codec keys off its own magic); a
// missing version or one newer than Document::kFormatVersion raises
// ProjectIOError.

#include <memory>
#include <string>
#include <vector>

#include "base/io/zip.hpp"
#include "common/core/document.hpp"

namespace alphashape {

/// Name of the XML entry inside project.zip.
inline constexpr const char* kDocXmlEntry = "doc.xml";

/// Side information collected while reading a project (§4.5).
///
/// `version` is the doc.xml format version that was read (0 when the archive
/// was rejected before the version could be determined). `warnings` are
/// non-fatal problems: unknown attributes, elements of a newer minor revision
/// the reader deliberately skipped, missing optional sidecars, and frames whose
/// recorded paths could not be resolved. Warnings never abort loading; every
/// condition that does abort loading raises ProjectIOError.
struct ReadReport {
    int version = 0;
    std::vector<std::string> warnings;
};

/// Write doc.xml and every binary sidecar into `archive`.
///
/// Only the current format (Document::kFormatVersion, v4) can be written; a
/// different `version` raises ProjectIOError rather than silently upgrading.
/// Throws ProjectIOError for content the format cannot represent, rather than
/// silently dropping it.
void write_document(const Document& document, ZipArchive& archive,
                    int version = Document::kFormatVersion);

/// Rebuild a Document from an archive written by write_document.
///
/// When `report` is non-null it receives the format version that was read and
/// any non-fatal warnings. Throws ProjectIOError on malformed or unsupported
/// input.
std::shared_ptr<Document> read_document(const ZipArchive& archive,
                                        ReadReport* report = nullptr);

}  // namespace alphashape
