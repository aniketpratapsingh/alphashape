#pragma once

// The camera model: intrinsics, distortion, and the pixel<->ray maps.
//
// Pixel convention. The centre of the top-left pixel is (0.5, 0.5) and the
// image spans [0, W] x [0, H]; the principal point therefore sits at
// (W/2 + cx, H/2 + cy) when the calibration is perfect (cx = cy = 0 means the
// optical axis meets the exact image centre). Camera frame: +X right, +Y down,
// +Z forward (the camera looks along +Z).
//
// Frame projection is the Brown-Conrady model with an affine (aspect +
// skew) correction. With x = X/Z, y = Y/Z and r^2 = x^2 + y^2:
//
//     x' = x (1 + k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8)
//              + p1 (r^2 + 2 x^2) + 2 p2 x y
//              + p3 r^2 (r^2 + 2 x^2) + 2 p4 r^2 x y
//     y' = y (1 + k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8)
//              + p2 (r^2 + 2 y^2) + 2 p1 x y
//              + p4 r^2 (r^2 + 2 y^2) + 2 p3 r^2 x y
//     u  = W/2 + cx + x' (f + b1) + y' b2        (b1: affinity, b2: skew)
//     v  = H/2 + cy + y' f
//
// Fisheye (Kannala-Brandt style, equidistant) works on the same pipeline. The
// incoming direction is reduced to the angle off the optical axis with atan2
// (so it is valid behind the image plane) and the normalised-plane point is
// scaled by theta/r0 before the identical distortion polynomial and affine map
// are applied:
//
//     r3d = sqrt(X^2 + Y^2),  theta = atan2(r3d, Z)
//     x   = X theta / r3d,    y = Y theta / r3d      (theta/r0 scaling)
//     then the Frame polynomial and affine map above, with r^2 = theta^2.
//
// The forward model has exactly one implementation: the try_project_point()
// template below. Calibration::project(), Calibration::unproject() and the
// bundle-adjustment ReprojectionCost all route through it (the last with a
// Ceres Jet scalar), so autodiff differentiates exactly the code the runtime
// uses.
//
// try_project_point() never throws: it returns false exactly where the model
// is undefined (a Frame point at or behind the image plane, a Fisheye point at
// the camera origin). A caller can therefore drive it from a hot loop or from
// Ceres autodiff without a throw and without a second copy of the maths.
// project_point() is the thin throwing wrapper kept for callers that want the
// ValueError semantics.

#include <string>
#include <type_traits>

#include <Eigen/Core>

#include "base/utils/error.hpp"

namespace alphashape {

enum class CameraType { Frame, Fisheye };  // spherical: later milestone

/// Intrinsics and lens distortion for one sensor.
struct Calibration {
    CameraType type = CameraType::Frame;
    int width = 0;
    int height = 0;
    double f = 0.0;       ///< focal length, pixels
    double cx = 0.0;      ///< principal point X offset from image centre, pixels
    double cy = 0.0;      ///< principal point Y offset from image centre, pixels
    double b1 = 0.0;      ///< affinity (aspect), pixels
    double b2 = 0.0;      ///< non-orthogonality (skew), pixels
    double k1 = 0.0;      ///< radial
    double k2 = 0.0;      ///< radial
    double k3 = 0.0;      ///< radial
    double k4 = 0.0;      ///< radial
    double p1 = 0.0;      ///< tangential (decentering)
    double p2 = 0.0;      ///< tangential (decentering)
    double p3 = 0.0;      ///< tangential (decentering), r^2-weighted
    double p4 = 0.0;      ///< tangential (decentering), r^2-weighted

    /// A calibration of the given geometry with every distortion term zero.
    static Calibration initial(CameraType type, int width, int height, double f_pixels);

    /// Exact (bitwise) comparison of every field.
    bool operator==(const Calibration& other) const;

    /// Project a camera-frame point to pixels. Throws ValueError for a Frame
    /// sensor when z <= 0. Fisheye accepts z <= 0 (up to the antipode).
    Eigen::Vector2d project(const Eigen::Vector3d& p_cam) const;

    /// Recover a unit-length ray (camera frame) through the given pixel.
    /// Never throws: see the inversion notes in calibration.cpp.
    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel) const;
};

/// The calibration of an image produced by box-averaging `downscale` x
/// `downscale` blocks of a `full`-resolution image (the operator in
/// image/operators/downsample.hpp). Only the floor(W/ds) * floor(H/ds) top-left
/// region is covered; the trailing strip is ignored.
///
/// With our pixel convention (top-left pixel centre at (0.5, 0.5)) a full-res
/// coordinate u maps to u' = u / ds exactly, so
///     width' = W / ds, height' = H / ds   (integer division)
///     f' = f / ds,  b1' = b1 / ds,  b2' = b2 / ds
///     k1..k4, p1..p4 unchanged;  type unchanged
/// and the principal point is placed exactly: the absolute pp (W/2 + cx) lands
/// at (W/2 + cx)/ds, so
///     cx' = (W/2.0 + cx) / ds - width'/2.0,   cy' likewise.
///
/// Throws ValueError when `downscale` is not a power of two >= 1.
Calibration scaled_calibration(const Calibration& full, int downscale);

/// Doubles in the packed parameter vector, order:
/// f, cx, cy, b1, b2, k1, k2, k3, k4, p1, p2, p3, p4.
/// p3/p4 are appended so every pre-existing index (and every stored 11-double
/// array) keeps its meaning; old data loads with p3 = p4 = 0.
inline constexpr int kCalibrationParamCount = 13;

/// Force-inline marker for the forward model. The stereo inner loop evaluates
/// it once per window sample (millions of times per image) in a loop whose only
/// alternative is a call: left to the heuristics the compiler emits it
/// out-of-line (the fisheye branch's atan2 makes it look large), which costs a
/// call plus register spill on the frame path, so the attribute is explicit.
#if defined(_MSC_VER)
#define ALPHASHAPE_PROJECT_INLINE __forceinline
#else
#define ALPHASHAPE_PROJECT_INLINE inline __attribute__((always_inline))
#endif

/// The forward model, templated so it can be instantiated with an autodiff
/// scalar type. `params` holds kCalibrationParamCount values in the order
/// above; `p_cam` is three values (X, Y, Z) in the camera frame; `pixel_out`
/// receives (u, v).
///
/// Returns false (writing nothing) exactly where the model is undefined: a
/// Frame point with Z <= 0, or a Fisheye point at the camera origin. Never
/// throws, so it is safe to call from a hot loop and from Ceres autodiff.
template <typename T>
ALPHASHAPE_PROJECT_INLINE bool try_project_point(CameraType type, int width, int height,
                                                 const T* params, const T* p_cam,
                                                 T* pixel_out) {
    using std::atan2;
    using std::sqrt;

    const T f = params[0];
    const T cx = params[1];
    const T cy = params[2];
    const T b1 = params[3];
    const T b2 = params[4];
    const T k1 = params[5];
    const T k2 = params[6];
    const T k3 = params[7];
    const T k4 = params[8];
    const T p1 = params[9];
    const T p2 = params[10];
    const T p3 = params[11];
    const T p4 = params[12];

    const T X = p_cam[0];
    const T Y = p_cam[1];
    const T Z = p_cam[2];

    // Normalised-plane coordinates (xs, ys): the input the shared distortion
    // polynomial below operates on.
    T xs = T(0);
    T ys = T(0);
    if (type == CameraType::Frame) {
        if (!(Z > T(0))) {
            return false;  // behind the image plane: no projection
        }
        xs = X / Z;
        ys = Y / Z;
    } else {
        const T r3d = sqrt(X * X + Y * Y);
        if (r3d == T(0)) {
            if (Z == T(0)) {
                return false;  // at the origin: no direction to project
            }
            xs = T(0);
            ys = T(0);
        } else {
            const T theta = atan2(r3d, Z);  // [0, pi]; works behind the image plane
            const T scale = theta / r3d;    // theta/r0, the equidistant scaling
            xs = X * scale;
            ys = Y * scale;
        }
    }

    // Undistorted shortcut. With k1..k4 = p1..p4 = 0 the polynomial below is
    // exactly the identity (radial = 1 and both tangential terms are 0), so this
    // branch reproduces its output value for value while skipping the powers and
    // the polynomial -- the overwhelming case for a clean sensor, and the case
    // the stereo inner loop hits millions of times per image. It is a shortcut
    // inside the single model, not a second implementation: the general path is
    // still the only place the maths lives.
    //
    // Restricted to plain scalar types. Under autodiff the shortcut would make
    // the projection independent of k1..k4 / p1..p4, so the gradient with
    // respect to those coefficients would be identically zero at a zero start
    // and self-calibration could never move them off zero; an autodiff scalar
    // therefore always takes the full polynomial (whose value at zero
    // distortion is bit-identical to the shortcut anyway).
    const bool undistorted = std::is_arithmetic<T>::value && k1 == T(0) && k2 == T(0) &&
                             k3 == T(0) && k4 == T(0) && p1 == T(0) && p2 == T(0) &&
                             p3 == T(0) && p4 == T(0);
    T xd = T(0);
    T yd = T(0);
    if (undistorted) {
        xd = xs;
        yd = ys;
    } else {
        const T r2 = xs * xs + ys * ys;
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;
        const T r8 = r4 * r4;
        const T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        const T two = T(2);
        // Higher-order tangential pair: the same p1/p2 template pre-multiplied
        // by r^2, i.e. p3 (r^2 + 2 x^2) r^2 + 2 p4 x y r^2, mirrored for y.
        // Folding them into p1/p2 keeps the p3 = p4 = 0 result bit-identical.
        const T q1 = p1 + p3 * r2;
        const T q2 = p2 + p4 * r2;
        xd = xs * radial + q1 * (r2 + two * xs * xs) + two * q2 * xs * ys;
        yd = ys * radial + q2 * (r2 + two * ys * ys) + two * q1 * xs * ys;
    }

    const T half_w = static_cast<T>(width) * T(0.5);
    const T half_h = static_cast<T>(height) * T(0.5);
    pixel_out[0] = half_w + cx + xd * (f + b1) + yd * b2;
    pixel_out[1] = half_h + cy + yd * f;
    return true;
}

/// The forward model with the historical throwing contract. A thin wrapper
/// over try_project_point() so there is exactly one implementation of the
/// maths: it reproduces the ValueError messages the model used to throw
/// inline (Frame Z <= 0, Fisheye at the origin) when the try returns false.
template <typename T>
void project_point(CameraType type, int width, int height, const T* params,
                   const T* p_cam, T* pixel_out) {
    if (try_project_point<T>(type, width, height, params, p_cam, pixel_out)) {
        return;
    }
    if (type == CameraType::Frame) {
        throw ValueError("camera model: point is behind the image plane (z <= 0)");
    }
    throw ValueError("camera model: fisheye point at the origin has no direction");
}

}  // namespace alphashape
