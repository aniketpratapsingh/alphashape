#include "model/camera/calibration.hpp"

#include <cmath>
#include <limits>

namespace alphashape {

namespace {

/// The forward distortion polynomial on the normalised plane, evaluated
/// through the shared templated model so the inverse can never drift from it.
/// With width = height = 0, cx = cy = b1 = b2 = 0 and f = 1, project_point()
/// collapses to exactly
///     xd = x (1 + k1 r^2 + ...) + p1 (r^2 + 2 x^2) + 2 p2 x y
///     yd = y (1 + k1 r^2 + ...) + p2 (r^2 + 2 y^2) + 2 p1 x y
/// (the point is fed at z = 1, which Frame projection accepts).
struct DistortionPoly {
    double params[kCalibrationParamCount];

    explicit DistortionPoly(const Calibration& c)
        : params{c.f,      c.cx, c.cy, c.b1, c.b2, c.k1,
                 c.k2,     c.k3, c.k4, c.p1, c.p2, c.p3, c.p4} {
        params[0] = 1.0;  // f
        params[1] = 0.0;  // cx
        params[2] = 0.0;  // cy
        params[3] = 0.0;  // b1
        params[4] = 0.0;  // b2
    }

    void eval(double x, double y, double& xd, double& yd) const {
        const double p[3] = {x, y, 1.0};
        double out[2];
        project_point<double>(CameraType::Frame, 0, 0, params, p, out);
        xd = out[0];
        yd = out[1];
    }
};

/// Invert the distortion polynomial: given distorted normalised-plane
/// coordinates (xd, yd), recover the undistorted (x, y).
///
/// Primary method is Newton's method on the 2x2 Jacobian; if it stalls (a
/// singular Jacobian, or the distortion is beyond its monotone region) we fall
/// back to fixed-point iteration and finally return whichever estimate has the
/// smaller residual. Convergence is capped at kMaxIter iterations per method
/// with a kTol tolerance on the normalised plane. Nothing here throws: strong
/// distortion can make the inverse locally non-unique, and a best-effort
/// estimate is more useful to a caller than an exception.
void invert_distortion(const DistortionPoly& poly, double xd, double yd, double& x,
                       double& y) {
    constexpr int kMaxIter = 20;
    constexpr double kTol = 1e-12;
    const double k1 = poly.params[5];
    const double k2 = poly.params[6];
    const double k3 = poly.params[7];
    const double k4 = poly.params[8];
    const double p1 = poly.params[9];
    const double p2 = poly.params[10];
    const double p3 = poly.params[11];
    const double p4 = poly.params[12];

    // -- Newton -------------------------------------------------------------
    double nx = xd;
    double ny = yd;
    double best_x = nx;
    double best_y = ny;
    double best_residual = std::numeric_limits<double>::infinity();
    for (int i = 0; i < kMaxIter; ++i) {
        const double r2 = nx * nx + ny * ny;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double s = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        const double q = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;

        double fx_model = 0.0;
        double fy_model = 0.0;
        poly.eval(nx, ny, fx_model, fy_model);
        const double fx = fx_model - xd;
        const double fy = fy_model - yd;
        const double residual = std::hypot(fx, fy);
        if (residual < best_residual) {
            best_residual = residual;
            best_x = nx;
            best_y = ny;
        }
        if (residual < kTol) {
            x = nx;
            y = ny;
            return;
        }

        // Higher-order tangential pair folded the same way as the forward
        // model: the tangential block uses q1 = p1 + p3 r^2, q2 = p2 + p4 r^2.
        const double q1 = p1 + p3 * r2;
        const double q2 = p2 + p4 * r2;
        const double j00 = s + 2.0 * nx * nx * q + 6.0 * nx * q1 + 2.0 * ny * q2 +
                           2.0 * p3 * nx * (r2 + 2.0 * nx * nx) +
                           4.0 * p4 * nx * nx * ny;
        const double j01 = 2.0 * nx * ny * q + 2.0 * ny * q1 + 2.0 * nx * q2 +
                           2.0 * p3 * ny * (r2 + 2.0 * nx * nx) +
                           4.0 * p4 * nx * ny * ny;
        const double j10 = 2.0 * nx * ny * q + 2.0 * nx * q2 + 2.0 * ny * q1 +
                           2.0 * p4 * nx * (r2 + 2.0 * ny * ny) +
                           4.0 * p3 * nx * nx * ny;
        const double j11 = s + 2.0 * ny * ny * q + 6.0 * ny * q2 + 2.0 * nx * q1 +
                           2.0 * p4 * ny * (r2 + 2.0 * ny * ny) +
                           4.0 * p3 * nx * ny * ny;
        const double det = j00 * j11 - j01 * j10;
        if (std::abs(det) < 1e-300) {
            break;
        }
        const double step_x = (j11 * fx - j01 * fy) / det;
        const double step_y = (j00 * fy - j10 * fx) / det;
        nx -= step_x;
        ny -= step_y;
    }

    // -- Fixed point --------------------------------------------------------
    // Split the model as (x s + tangential) = target and iterate
    //     x <- (x_target - tangential(x, y)) / s.
    double px = xd;
    double py = yd;
    for (int i = 0; i < kMaxIter; ++i) {
        const double r2 = px * px + py * py;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double s = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        if (std::abs(s) < 1e-300) {
            break;
        }
        const double q1 = p1 + p3 * r2;
        const double q2 = p2 + p4 * r2;
        const double tx = q1 * (r2 + 2.0 * px * px) + 2.0 * q2 * px * py;
        const double ty = q2 * (r2 + 2.0 * py * py) + 2.0 * q1 * px * py;
        const double ox = (xd - tx) / s;
        const double oy = (yd - ty) / s;

        double exd = 0.0;
        double eyd = 0.0;
        poly.eval(ox, oy, exd, eyd);
        const double residual = std::hypot(exd - xd, eyd - yd);
        if (residual < best_residual) {
            best_residual = residual;
            best_x = ox;
            best_y = oy;
        }
        const bool settled = std::abs(ox - px) < kTol && std::abs(oy - py) < kTol;
        px = ox;
        py = oy;
        if (settled || residual < kTol) {
            break;
        }
    }

    x = best_x;
    y = best_y;
}

}  // namespace

Calibration scaled_calibration(const Calibration& full, int downscale) {
    if (downscale < 1 || (downscale & (downscale - 1)) != 0) {
        throw ValueError("scaled_calibration: downscale must be a power of two >= 1");
    }
    Calibration out = full;
    if (downscale == 1) {
        return out;
    }
    const double ds = static_cast<double>(downscale);
    out.width = full.width / downscale;
    out.height = full.height / downscale;
    out.f = full.f / ds;
    out.b1 = full.b1 / ds;
    out.b2 = full.b2 / ds;
    // Principal point stays at the same absolute location: the full-res pp
    // (W/2 + cx) must land at (W/2 + cx)/ds in the reduced image, so express it
    // relative to the new image centre. Computed in double for exactness.
    out.cx = (static_cast<double>(full.width) * 0.5 + full.cx) / ds -
             static_cast<double>(out.width) * 0.5;
    out.cy = (static_cast<double>(full.height) * 0.5 + full.cy) / ds -
             static_cast<double>(out.height) * 0.5;
    // Distortion coefficients and the camera type are resolution independent.
    return out;
}

Calibration Calibration::initial(CameraType type, int width, int height,
                                 double f_pixels) {
    Calibration c;
    c.type = type;
    c.width = width;
    c.height = height;
    c.f = f_pixels;
    // Every distortion term stays at its default of zero.
    return c;
}

bool Calibration::operator==(const Calibration& other) const {
    return type == other.type && width == other.width && height == other.height &&
           f == other.f && cx == other.cx && cy == other.cy && b1 == other.b1 &&
           b2 == other.b2 && k1 == other.k1 && k2 == other.k2 && k3 == other.k3 &&
           k4 == other.k4 && p1 == other.p1 && p2 == other.p2 && p3 == other.p3 &&
           p4 == other.p4;
}

Eigen::Vector2d Calibration::project(const Eigen::Vector3d& p_cam) const {
    const double params[kCalibrationParamCount] = {f,  cx, cy, b1, b2, k1, k2,
                                                   k3, k4, p1, p2, p3, p4};
    double out[2] = {0.0, 0.0};
    project_point<double>(type, width, height, params, p_cam.data(), out);
    return Eigen::Vector2d(out[0], out[1]);
}

Eigen::Vector3d Calibration::unproject(const Eigen::Vector2d& pixel) const {
    // 1. Invert the affine map exactly. From
    //        u = W/2 + cx + x'(f + b1) + y' b2
    //        v = H/2 + cy + y' f
    //    solve v for y', then u for x'. A zero focal length (f or f+b1) makes
    //    the map singular; return a straight-ahead ray rather than dividing by
    //    zero, since unproject must not throw.
    const double half_w = static_cast<double>(width) * 0.5;
    const double half_h = static_cast<double>(height) * 0.5;
    const double denom_y = f;
    const double denom_x = f + b1;
    if (denom_y == 0.0 || denom_x == 0.0) {
        return Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    const double yd = (pixel.y() - half_h - cy) / denom_y;
    const double xd = (pixel.x() - half_w - cx - yd * b2) / denom_x;

    // 2. Invert the distortion.
    const DistortionPoly poly(*this);
    double xs = xd;
    double ys = yd;
    invert_distortion(poly, xd, yd, xs, ys);

    // 3. Map back to a camera-frame direction.
    if (type == CameraType::Frame) {
        // (xs, ys) are X/Z, Y/Z. The z = 1 point normalises to the ray.
        return Eigen::Vector3d(xs, ys, 1.0).normalized();
    }

    // Fisheye: r0 = theta. Build the ray from the angle and the azimuth so it
    // stays valid behind the image plane.
    const double theta = std::hypot(xs, ys);
    if (theta == 0.0) {
        return Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    const double phi = std::atan2(ys, xs);
    Eigen::Vector3d ray;
    if (theta < 0.5 * 3.14159265358979323846) {
        // Near the axis the z = 1 form (tan theta) is finite and best
        // conditioned; this is the same direction as the general form below.
        const double t = std::tan(theta);
        ray = Eigen::Vector3d(t * std::cos(phi), t * std::sin(phi), 1.0);
    } else {
        // At and beyond 90 degrees, go straight through the angle.
        ray = Eigen::Vector3d(std::sin(theta) * std::cos(phi),
                              std::sin(theta) * std::sin(phi), std::cos(theta));
    }
    return ray.normalized();
}

}  // namespace alphashape
