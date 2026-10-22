#include "model/geo/transverse_mercator.hpp"

#include <array>
#include <cmath>

#include "base/utils/error.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape::geo {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr int kMaxIterations = 5;

/// Krüger series coefficients evaluated once for the WGS84 ellipsoid.
struct KrugerSeries {
    double n = 0.0;           ///< third flattening
    double A = 0.0;           ///< rectifying radius scale
    double c = 0.0;           ///< 2 sqrt(n) / (1 + n) conformal-latitude factor
    std::array<double, 3> alpha{};  ///< forward series coefficients
    std::array<double, 3> beta{};   ///< inverse series coefficients
    std::array<double, 3> delta{};  ///< rectifying -> geodetic latitude
};

/// Lazily-built WGS84 Krüger coefficients (thread-safe function-local static).
const KrugerSeries& kruger() {
    static const KrugerSeries s = [] {
        KrugerSeries k;
        k.n = kWgs84F / (2.0 - kWgs84F);
        const double n = k.n;
        const double n2 = n * n;
        const double n3 = n2 * n;
        const double n4 = n2 * n2;
        k.A = kWgs84A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0);
        k.c = 2.0 * std::sqrt(n) / (1.0 + n);
        k.alpha = {n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0,
                   13.0 * n2 / 48.0 - 3.0 * n3 / 5.0,
                   61.0 * n3 / 240.0};
        k.beta = {n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0,
                  n2 / 48.0 + n3 / 15.0,
                  17.0 * n3 / 480.0};
        k.delta = {2.0 * n - 2.0 * n2 / 3.0 - 2.0 * n3,
                   7.0 * n2 / 3.0 - 8.0 * n3 / 5.0,
                   56.0 * n3 / 15.0};
        return k;
    }();
    return s;
}

/// Wrap a longitude difference in degrees into (-180, 180].
double wrap_lon_deg(double d) {
    double w = std::fmod(d + 180.0, 360.0);
    if (w < 0.0) {
        w += 360.0;
    }
    return w - 180.0;
}

}  // namespace

int UtmZone::epsg() const {
    return (north ? 32600 : 32700) + zone;
}

UtmZone utm_zone_for(double longitude_deg, double latitude_deg) {
    const double lon = wrap_lon_deg(longitude_deg);
    int zone = static_cast<int>(std::floor((lon + 180.0) / 6.0)) + 1;
    if (zone < 1) {
        zone = 1;
    }
    if (zone > 60) {
        zone = 60;
    }
    UtmZone z;
    z.zone = zone;
    z.north = latitude_deg >= 0.0;
    return z;
}

std::optional<UtmZone> utm_zone_from_epsg(int epsg) {
    UtmZone z;
    if (epsg >= 32601 && epsg <= 32660) {
        z.zone = epsg - 32600;
        z.north = true;
        return z;
    }
    if (epsg >= 32701 && epsg <= 32760) {
        z.zone = epsg - 32700;
        z.north = false;
        return z;
    }
    return std::nullopt;
}

Eigen::Vector2d tm_forward(double lon_deg, double lat_deg, double lon0_deg,
                           double k0, double false_e, double false_n) {
    const KrugerSeries& s = kruger();
    const double phi = lat_deg * kDeg2Rad;
    const double dlam = wrap_lon_deg(lon_deg - lon0_deg) * kDeg2Rad;
    const double sin_phi = std::sin(phi);
    const double cos_lam = std::cos(dlam);
    const double sin_lam = std::sin(dlam);

    // Conformal latitude via the Kruger n-series.
    const double t = std::sinh(std::atanh(sin_phi) -
                               s.c * std::atanh(s.c * sin_phi));
    const double xi = std::atan2(t, cos_lam);
    const double eta = std::atanh(sin_lam / std::sqrt(1.0 + t * t));

    double sum_e = 0.0;
    double sum_n = 0.0;
    for (int j = 1; j <= 3; ++j) {
        const double a = s.alpha[static_cast<std::size_t>(j - 1)];
        const double jd = static_cast<double>(j);
        const double x2 = 2.0 * jd * xi;
        const double y2 = 2.0 * jd * eta;
        sum_e += a * std::cos(x2) * std::sinh(y2);
        sum_n += a * std::sin(x2) * std::cosh(y2);
    }

    const double scale = k0 * s.A;
    return {false_e + scale * (eta + sum_e), false_n + scale * (xi + sum_n)};
}

double tm_point_scale(double lon_deg, double lat_deg, double lon0_deg,
                      double k0) {
    const KrugerSeries& s = kruger();
    const double phi = lat_deg * kDeg2Rad;
    const double dlam = wrap_lon_deg(lon_deg - lon0_deg) * kDeg2Rad;
    const double sin_phi = std::sin(phi);
    const double cos_phi = std::cos(phi);
    const double sin_lam = std::sin(dlam);
    const double cos_lam = std::cos(dlam);

    const double t = std::sinh(std::atanh(sin_phi) -
                               s.c * std::atanh(s.c * sin_phi));
    const double xi = std::atan2(t, cos_lam);
    const double eta = std::atanh(sin_lam / std::sqrt(1.0 + t * t));

    double sigma = 1.0;
    double tau = 0.0;
    for (int j = 1; j <= 3; ++j) {
        const double jd = static_cast<double>(j);
        const double x2 = 2.0 * jd * xi;
        const double y2 = 2.0 * jd * eta;
        const double a2j = 2.0 * jd * s.alpha[static_cast<std::size_t>(j - 1)];
        sigma += a2j * std::cos(x2) * std::cosh(y2);
        tau += a2j * std::sin(x2) * std::sinh(y2);
    }

    const double tan_phi = cos_phi != 0.0 ? sin_phi / cos_phi : 0.0;
    const double m = (1.0 - s.n) / (1.0 + s.n) * tan_phi;
    const double denom = t * t + cos_lam * cos_lam;
    const double num = (1.0 + m * m) * (sigma * sigma + tau * tau);
    return (k0 * s.A / kWgs84A) * std::sqrt(num / denom);
}

Eigen::Vector2d tm_inverse(double easting, double northing, double lon0_deg,
                           double k0, double false_e, double false_n) {
    const KrugerSeries& s = kruger();
    const double scale = k0 * s.A;

    // Rectifying coordinates xi, eta.
    const double xi = (northing - false_n) / scale;
    const double eta = (easting - false_e) / scale;

    // First guess via the beta series.
    double xi_p = xi;
    double eta_p = eta;
    for (int j = 1; j <= 3; ++j) {
        const double jd = static_cast<double>(j);
        const double b = s.beta[static_cast<std::size_t>(j - 1)];
        const double x2 = 2.0 * jd * xi;
        const double y2 = 2.0 * jd * eta;
        xi_p -= b * std::sin(x2) * std::cosh(y2);
        eta_p -= b * std::cos(x2) * std::sinh(y2);
    }
    const double chi = std::asin(std::sin(xi_p) / std::cosh(eta_p));
    double lat_rad = chi + s.delta[0] * std::sin(2.0 * chi) +
                     s.delta[1] * std::sin(4.0 * chi) +
                     s.delta[2] * std::sin(6.0 * chi);
    double lon_rad = (lon0_deg * kDeg2Rad) +
                     std::atan2(std::sinh(eta_p), std::cos(xi_p));

    // Newton refinement on the forward map: solve f(lon,lat) = (E,N).
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        const double lon_d = lon_rad / kDeg2Rad;
        const double lat_d = lat_rad / kDeg2Rad;
        const Eigen::Vector2d f = tm_forward(lon_d, lat_d, lon0_deg, k0,
                                             false_e, false_n);
        const double re = f.x() - easting;
        const double rn = f.y() - northing;
        if (std::abs(re) < 1e-9 && std::abs(rn) < 1e-9) {
            break;
        }
        // Numerical 2x2 Jacobian w.r.t. (lon_deg, lat_deg); step in degrees.
        constexpr double h = 1e-7;
        const Eigen::Vector2d f_lon =
            tm_forward(lon_d + h, lat_d, lon0_deg, k0, false_e, false_n);
        const Eigen::Vector2d f_lat =
            tm_forward(lon_d, lat_d + h, lon0_deg, k0, false_e, false_n);
        const double j00 = (f_lon.x() - f.x()) / h;
        const double j10 = (f_lon.y() - f.y()) / h;
        const double j01 = (f_lat.x() - f.x()) / h;
        const double j11 = (f_lat.y() - f.y()) / h;
        const double det = j00 * j11 - j01 * j10;
        if (det == 0.0) {
            break;
        }
        const double dlon = (-re * j11 + rn * j01) / det;
        const double dlat = (-rn * j00 + re * j10) / det;
        lon_rad += dlon * kDeg2Rad;
        lat_rad += dlat * kDeg2Rad;
    }

    return {lon_rad / kDeg2Rad, lat_rad / kDeg2Rad};
}

Eigen::Vector2d geodetic_to_utm(double lon_deg, double lat_deg,
                                const UtmZone& zone) {
    const double lon0 = 6.0 * static_cast<double>(zone.zone) - 183.0;
    const double false_n = zone.north ? 0.0 : 10000000.0;
    return tm_forward(lon_deg, lat_deg, lon0, 0.9996, 500000.0, false_n);
}

Eigen::Vector2d utm_to_geodetic(double easting, double northing,
                                const UtmZone& zone) {
    const double lon0 = 6.0 * static_cast<double>(zone.zone) - 183.0;
    const double false_n = zone.north ? 0.0 : 10000000.0;
    return tm_inverse(easting, northing, lon0, 0.9996, 500000.0, false_n);
}

}  // namespace alphashape::geo
