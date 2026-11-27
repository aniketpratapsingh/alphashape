#pragma once

// AddPhotos: import image files into the active chunk as frames + cameras.
//
// ---------------------------------------------------------------------------
// Initial focal length, in priority order
// ---------------------------------------------------------------------------
// The sensor's calibration needs a focal length in PIXELS, but the metadata we
// read gives millimetres (and only when the camera wrote EXIF). Three routes,
// tried in this order:
//
//   1. FocalLength (0x920A) with a known pixel pitch:
//          f_px = focal_mm * 1000 / pixel_size_um
//      This is exact metric geometry — the one route with no convention in it.
//      The pixel pitch comes from FocalPlaneXResolution/Unit (0xA20E/0xA210).
//
//   2. FocalLengthIn35mmFilm (0xA405):
//          f_px = f35 * hypot(w, h) / hypot(36, 24)
//      The 35mm tag is by definition the focal length that would produce the
//      same field of view on a full-frame 36x24 sensor, so scaling by the ratio
//      of frame diagonals recovers the pixel focal length without any knowledge
//      of the real sensor size. The full-frame diagonal convention
//      (hypot(36,24) = 43.267 mm) is the standard reading of that tag.
//
//   3. Nothing usable:
//          f_px = 1.2 * max(w, h)
//      OUR DEFAULT, and the only constant in this file that is a choice rather
//      than a derived value. 1.2 * max(w, h) is a focal length of roughly one
//      and a fifth of the long edge, which is about a 50-degree horizontal
//      field of view on a 4:3 frame. It is the convention used by several
//      open-source SfM packages for a lens with no recorded focal length, and
//      picking the same one that they picked means our default projects
//      similarly. The VALUE is ours to change; the ORDER above is not, because
//      routes 1 and 2 are dictated by what the tags mean.
//
// The result seeds Calibration::f. Distortion terms stay zero: a bundle
// adjustment milestone refines all of it later.
//
// ---------------------------------------------------------------------------
// XMP lens calibration
// ---------------------------------------------------------------------------
// When `load_xmp_calibration` is set and a photo's XMP carries a DJI factory
// calibration (`DewarpData`, or at least `CalibratedFocalLength`), the sensor's
// INITIAL calibration is taken from it instead of the EXIF focal length: the
// focal length, principal point and Brown-Conrady distortion, mapped to the
// engine's model (see calibration_from_dewarp in the .cpp for the p1/p2 swap).
// Photos of one sensor share the first photo's values; a disagreement keeps the
// first and logs a warning. `DewarpFlag=1` (the camera already undistorted the
// JPEG) keeps the focal length and principal point but zeroes the distortion.
// The option is off by default, so a non-DJI photo and a DJI photo imported
// without it keep the EXIF-seeded calibration above.
//
// ---------------------------------------------------------------------------
// Undo
// ---------------------------------------------------------------------------
// Like every other task, AddPhotos edits the chunk in place and its changes are
// NOT placed on the undo stack. A caller that needs a reversible import wraps
// the task in a staged command; the GUI does exactly that where it offers one.

#include "common/tasks/task.hpp"

namespace alphashape {

class AddPhotosTask : public AbstractTask {
public:
    std::string name() const override { return "AddPhotos"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
