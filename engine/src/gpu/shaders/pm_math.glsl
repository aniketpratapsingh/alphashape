// Shared declarations for the GPU PatchMatch kernels: the push-constant block,
// the buffer bindings, the counter-based splitmix64 stream (the only source of
// random numbers), the forward camera model from
// model/camera/calibration.hpp and the bilinear sampler that mirrors
// patch_match.cpp's sample_image().

// One lane per pixel of the active checkerboard colour. 64 lanes (one 8x16
// tile's worth of one colour) and 49 window samples (radius 3 at step 1, or
// radius 5 at step 2) are the kernel limits; inputs beyond them take the CPU
// path. The per-lane scratch is a 49-float register array in the cost core.
const int PM_MAX_SAMPLES = 49;
const int PM_MAX_NEIGHBOURS = 32;
const int PM_LANES = 64;
const float PM_FACING_EPS = 1e-6;
const float PM_CAMERA_Z_EPS = 1e-9;
const float PM_NO_MATCH_COST = 2.0;
const float PM_TWO_PI = 6.283185307179586;
const float PM_CONE_HALF_ANGLE = 1.0471975511965976;  // pi / 3
const float PM_MAX_NORMAL_ANGLE = 1.0471975511965976; // pi / 3

// Why `precise` appears on selected expressions below and in pm_core.glsl.
//
// These kernels are built without the SPIR-V optimiser (see the CMakeLists and
// alphashape_add_shader), so nothing here is currently contracted. The marks are
// kept because they encode a property of the algorithm that must survive if a
// kernel is ever optimised again, and because they cost nothing at -O0: under -O
// glslang/SPIRV-Tools is free to contract `a + b * u` into a fused multiply-add,
// and NVIDIA hardware does so.
//
// The CPU reference in patch_match.cpp computes every such expression in double
// as two separately rounded operations, and this kernel's output is not smooth in
// those last bits: sample_image() floors the projected coordinate, so a single-ulp
// difference near an integer boundary reads a different texel, which changes a
// neighbour's ZNCC, which flips the winner-take-all selection for the pixel.
// Measured on the dense fixture: optimised pm_refine alone dropped GPU/CPU depth
// agreement from 100% to 93.4% purely from that, and marking only the final
// projected coordinates did not restore it.
//
// `precise` is therefore applied along the whole *chain* that decides which texel
// is sampled, not just to its last two statements. Declaring only the projected
// pixel `precise` is not enough: that stops the final `+` from folding into an
// FMA but leaves the operands free to be contracted in their own defining
// expressions, and a single differing bit anywhere on the path moves the
// floor() just the same. The marks, in data-flow order:
//
//   nb = dot(n, b_p), rho = depth * nb     pm_core.glsl evaluate_hypothesis()
//   nb = dot(n, b_q), depth_q = rho / nb   pm_core.glsl candidate_depth()
//   x_n = depth_q * (M b_q) + c            pm_core.glsl neighbour_cost()
//   rho_q, denom; depth_p = rho_q / denom  pm_core.glsl propagate()
//   xd, yd, r2, radial_scale, u, v         project_point(), below
//   perturbed_inv                          pm_refine.comp
//
// The rule is "mark the expression, not the variable's uses", so only operations
// that *have* a fused form are marked: a dot, an `a + b * c`, a scale-then-add.
// Divisions, `sqrt` and plain multiplies are already exact as written, and
// marking them would only enlarge the set of expressions an optimiser must keep
// non-reassociable -- which is what made this kernel expensive to compile.
// `precise` propagates into an initialiser, so one declaration per stage covers
// the expression that computes it, and a value stored through `out` or an array
// carries its non-contracted definition with it.
//
// Everything off that chain -- the ZNCC accumulators (which have their own
// precise declarations), the normals and the bilinear weights -- stays free to
// contract, which is what keeps the 49 x 32 cost loop affordable to optimise.
//
// The *draws* (rng_uniform64's split sum, random_depth's inverse-depth lerp,
// random_cone_normal's trig) are deliberately left contractible, even though they
// are upstream of rho. `precise` is a much weaker guarantee for them than it
// looks: they already evaluate a double formula in float32, so a last-bit
// difference from the CPU is inherent to the algorithm and NoContraction does not
// change that. What `precise` buys is the *absence* of a difference the CPU does
// not have -- a contraction its separate rounding rules out -- and that is what
// the chain above needed, sitting immediately in front of floor().
//
// MEASUREMENTS AND THE OPEN ISSUE (why both search kernels are built -O0).
// The chain above is what makes the *arithmetic* exact. It does not make the
// optimised module match the CPU on NVIDIA, and that is why the shaders are
// shipped unoptimised. On a Tesla T4 (driver 535.161.07), against the same CPU
// reference, over this branch's history:
//
//   pm_init  at -O   : depth 100.00%  normal  0.92%  cost  31.81%   <- wrong
//   pm_init  at -O0  : depth 100.00%  normal 100.00%  cost 100.00%   <- correct
//   pm_refine at -O  : dense agreement 93.4% (with only u/v marked):
//                      85.6% before the shader defects were fixed
//   pm_refine at -O0 : dense agreement 100.0%, aerial 100.0%         <- main
//
// A deeper problem sits behind those numbers: with `precise` on the whole chain the
// -O pipeline does not finish at all. glslc needs tens of gigabytes for pm_refine
// and a CI runner OOM-kills it (exit 137: 7 GB macOS runner, and the GitLab T4
// lane). NoContraction inside the 49-sample x 32-neighbour cost loop is what
// stops the optimiser reassociating, and that is where the memory goes. -Os does
// not help: it was tried and is killed the same way.
//
// So the accurate chain and an optimised module are mutually exclusive with this
// toolchain, and the measured fallback -- -O on pm_refine with no contraction --
// is 93.4%, not the 100.0% main has. Building both kernels at -O0 is therefore
// the only configuration that is correct on every lane, and it is what ships.
// That was confirmed the hard way: the mutable RNG cursor, stateless keyed
// streams with a loop, and straight-line keyed draws all produced the same
// 0.92%, so the difference is not state, draw order or the shaders' own undefined
// behaviour (all three of which were found and fixed independently). It is not
// the shared cost core either: pm_init's depths, drawn from that same stream at
// index 0, are 100% exact while its normals, drawn at indices 1 and 2, are not.
// Diagnosing further needs T4 hardware and spirv-dis of both modules.
//
// What this means for a future attempt: an optimised module changes results on
// NVIDIA, so it cannot be enabled without a hardware lane that gates it. The
// `PatchMatchGpu.RefineSweepsMatchCpu` test exists for exactly that, at a 99%
// floor on hardware -- loose enough not to assert bit-identity of another
// library's sin/cos, tight enough that the 93.4% regression fails it -- and
// `TvFusionGpu.BuildModelMeetsCpuBounds` asserts the fused mesh's boundary-edge
// count is equal, a discontinuous function of that agreement and the sharpest
// available check.

layout(push_constant) uniform PmParams {
    uint width;
    uint height;
    uint colour;        // checkerboard colour of the active half-sweep
    uint iteration;     // sweep index, low word
    uint iteration_hi;  // sweep index, high word (0xFFFFFFFF for the initial pass)
    uint n_neighbours;
    uint n_samples;
    uint best_k;        // already clamped to [1, n_neighbours]
    float depth_min;
    float depth_max;
    float inv_span;     // 1/depth_min - 1/depth_max
    float normal_angle;
    float depth_delta;
    float min_texture;
    float max_cost;
    int radius;         // interior margin (window_radius)
    uint seed_lo;       // PatchMatchOptions::seed, low word
    uint seed_hi;       // PatchMatchOptions::seed, high word
    uint touch;         // always 0: keeps every binding alive through -O
    uint row_offset;    // first output row of this dispatch's row band
} pc;

layout(binding = 0) buffer RefImage { float ref_img[]; };
layout(binding = 1) buffer NeighbourImages { float nb_img[]; };
layout(binding = 2) buffer ViewParams { float views[]; };
layout(binding = 3) buffer ViewInts { uint view_ints[]; };
layout(binding = 4) buffer Offsets { int offsets[]; };
layout(binding = 5) buffer StateDepth { float state_depth[]; };
layout(binding = 6) buffer StateNormal { float state_normal[]; };
layout(binding = 7) buffer StateCost { float state_cost[]; };
layout(binding = 8) buffer StateActive { uint state_active[]; };layout(binding = 9) buffer RefWindows { float ref_windows[]; };
layout(binding = 10) buffer Bearings { float bearings[]; };
layout(binding = 11) buffer OutDepth { float out_depth[]; };
layout(binding = 12) buffer OutNormals { float out_normals[]; };
layout(binding = 13) buffer OutCost { float out_cost[]; };

// The search kernels do not write the outputs, and glslc -O removes unused
// interface variables, which would make the binding count differ per kernel.
// This never-taken branch (pc.touch is always 0) keeps all 14 bindings in every
// module, so one buffer list works for every dispatch.
//
// Element 0 of every binding, deliberately: the branch is dead, but an
// optimiser may speculatively hoist a load out of it, and an out-of-range
// storage-buffer access is undefined behaviour in Vulkan. Indexing by the lane's
// pixel was out of range for views/view_ints/offsets on every image wider than
// the neighbour or window count (i.e. almost always), which is exactly the kind
// of latent UB that behaves differently at -O0 and -O.
void keep_bindings_alive() {
    if (pc.touch != 0u) {
        float keep = ref_img[0] + nb_img[0] + views[0] + view_ints[0] + offsets[0] +
                     state_depth[0] + state_normal[0] + state_cost[0] + state_active[0] +
                     ref_windows[0] + bearings[0];
        out_depth[0] = keep;
        out_normals[0] = keep;
        out_normals[1] = keep;
        out_normals[2] = keep;
        out_cost[0] = keep;
    }
}

// View record layout, one per neighbour.
const int PM_VIEW_PARAMS = 0;    // 11 floats: f, cx, cy, b1, b2, k1..k4, p1, p2
const int PM_VIEW_AFFINE = 11;   // 9 floats: M, row-major
const int PM_VIEW_CENTRE = 20;   // 3 floats: c
const int PM_VIEW_STRIDE = 24;
const int PM_VIEW_INT_STRIDE = 4;  // width, height, type, image pixel offset

// ---------------------------------------------------------------------------
// Randomness: the CPU's stateless splitmix64 stream (patch_match.cpp's
// RngStream), emulated with pairs of 32-bit words because the M1 has no 64-bit
// integers. Counter-based and keyed by (seed, pixel, iteration, stream), so the
// GPU draws the same hypotheses as the CPU and no result depends on scheduling.
// ---------------------------------------------------------------------------

struct U64 {
    uint lo;
    uint hi;
};

U64 u64_add(U64 a, uint value_lo, uint value_hi) {
    U64 r;
    r.lo = a.lo + value_lo;
    r.hi = a.hi + value_hi + (r.lo < a.lo ? 1u : 0u);
    return r;
}

U64 u64_xor(U64 a, U64 b) {
    U64 r;
    r.lo = a.lo ^ b.lo;
    r.hi = a.hi ^ b.hi;
    return r;
}

// (a.hi,a.lo) >> shift for shift in [0, 63].
U64 u64_shr(U64 a, uint shift) {
    U64 r;
    if (shift == 0u) {
        return a;
    }
    if (shift >= 32u) {
        r.lo = a.hi >> (shift - 32u);
        r.hi = 0u;
    } else {
        r.lo = (a.lo >> shift) | (a.hi << (32u - shift));
        r.hi = a.hi >> shift;
    }
    return r;
}

// Truncated 64x64 -> 64 multiply from 32-bit partial products.
U64 u64_mul(U64 a, U64 b) {
    uint ll_hi, ll_lo, lh_hi, lh_lo, hl_hi, hl_lo;
    umulExtended(a.lo, b.lo, ll_hi, ll_lo);
    umulExtended(a.lo, b.hi, lh_hi, lh_lo);
    umulExtended(a.hi, b.lo, hl_hi, hl_lo);
    U64 r;
    r.lo = ll_lo;
    r.hi = ll_hi + lh_lo + hl_lo;
    return r;
}

U64 u64_from_u32(uint value) {
    U64 r;
    r.lo = value;
    r.hi = 0u;
    return r;
}

U64 splitmix64(U64 x) {
    x = u64_add(x, 0x7F4A7C15u, 0x9E3779B9u);
    x = u64_mul(u64_xor(x, u64_shr(x, 30u)), U64(0x1CE4E5B9u, 0xBF58476Du));
    x = u64_mul(u64_xor(x, u64_shr(x, 27u)), U64(0x133111EBu, 0x94D049BBu));
    return u64_xor(x, u64_shr(x, 31u));
}

// A stream key, not a mutable cursor. The nth draw of a stream is
// splitmix64^(n+1) of its key, so every draw is a pure function of
// (key, draw index) with nothing carried between calls.
//
// This shape is deliberate. The first version carried the state in an `inout`
// struct passed down through rng_next/random_unit_vector/random_cone_normal/
// perturb_normal/random_depth. glslang lowers an `inout` aggregate parameter to
// a pointer into Function storage with an explicit copy-in and copy-out, and the
// copy-out is the part an inliner/scalar-replacement pass is least likely to
// preserve -- undefined behaviour the optimiser is entitled to exploit. That was
// fixed on its own merits, not because it explained the NVIDIA numbers: with the
// keyed stream the T4 figure did not move (see the OPEN ISSUE below). The keyed
// form also removes the whole class of "two draws in one argument list"
// ordering hazards, since the draw index is written out at every call site.
struct RngKey {
    U64 state;  // already folded: seed, pixel, iteration and stream
};

RngKey rng_key(U64 seed, uint pixel, U64 iteration, uint stream) {
    U64 h = splitmix64(u64_add(seed, 0x7F4A7C15u, 0x9E3779B9u));
    // pixel * 0x100000001B3
    h = splitmix64(u64_xor(h, u64_mul(u64_from_u32(pixel), U64(0x000001B3u, 0x00000100u))));
    // iteration * 0xC2B2AE3D27D4EB4F
    h = splitmix64(u64_xor(h, u64_mul(iteration, U64(0x27D4EB4Fu, 0xC2B2AE3Du))));
    // stream * 0x9E3779B97F4A7C15
    h = splitmix64(u64_xor(h, u64_mul(U64(stream, 0u), U64(0x7F4A7C15u, 0x9E3779B9u))));
    RngKey key;
    key.state = h;
    return key;
}

// The nth raw draw of a stream (n = 0 is the CPU's first next()).
//
// Written as straight-line calls rather than a `for (i = 0; i <= index; ++i)`
// chain, so no run-time trip count exists for an optimiser to get wrong. That is
// a correctness improvement in its own right -- a fixed-trip-count loop returning
// one value for every index is exactly the shape an aggressive optimiser can get
// wrong -- but it was NOT the cause of the NVIDIA difference: the straight-line
// form and the loop form produced the same 0.92% normal agreement on the T4, as
// did the mutable-`inout` generator before them. See the OPEN ISSUE at the top of
// this file for the measurements and the ruled-out list.
//
// Callers must use PM_DRAW_0/1/2 and never build an index at run time.
U64 rng_draw0(RngKey key) { return splitmix64(key.state); }
U64 rng_draw1(RngKey key) { return splitmix64(rng_draw0(key)); }
U64 rng_draw2(RngKey key) { return splitmix64(rng_draw1(key)); }

// Draw indices are part of the stream contract, not runtime data.
#define PM_DRAW_0 0u
#define PM_DRAW_1 1u
#define PM_DRAW_2 2u

// Dispatch on a *compile-time* index. `index` must be a literal at the call
// site; anything else is a programming error, and the fallback keeps the result
// defined (draw 0) rather than wrong in a way that looks like a bad normal.
U64 rng_draw(RngKey key, uint index) {
    if (index == PM_DRAW_0) {
        return rng_draw0(key);
    }
    if (index == PM_DRAW_1) {
        return rng_draw1(key);
    }
    if (index == PM_DRAW_2) {
        return rng_draw2(key);
    }
    U64 state = rng_draw2(key);
    for (uint i = 3u; i <= index; ++i) {
        state = splitmix64(state);
    }
    return state;
}

// 53-bit uniform in [0, 1) from one draw, identical to uniform01().
float rng_uniform64(U64 bits) {
    // (bits >> 11) * 2^-53. The expression is written so the two operands never
    // exceed 2^24 (exact in float32): the high 29 bits scaled by 2^-29 plus the
    // remaining 24 bits scaled by 2^-53.
    float hi_part = float(bits.hi >> 3u) * (1.0 / 536870912.0);
    float lo_part = float((bits.hi & 7u) << 21u | (bits.lo >> 11u)) * (1.0 / 9007199254740992.0);
    return hi_part + lo_part;
}

float rng_uniform(RngKey key, uint index) { return rng_uniform64(rng_draw(key, index)); }

// Eigen's Vector3d::unitOrthogonal(), reproduced exactly: it picks the
// (-y, x, 0) basis unless x and y are both negligible next to z, in which case
// it takes (0, -z, y). The random cone and the normal perturbation both build
// their tangent frame this way, so matching the basis is what keeps the GPU and
// CPU random planes the same.
vec3 eigen_unit_orthogonal(vec3 v) {
    const float eps = 1e-12;
    if (!(abs(v.x) <= abs(v.z) * eps) || !(abs(v.y) <= abs(v.z) * eps)) {
        float inv = 1.0 / sqrt(v.x * v.x + v.y * v.y);
        return vec3(-v.y * inv, v.x * inv, 0.0);
    }
    float inv = 1.0 / sqrt(v.y * v.y + v.z * v.z);
    return vec3(0.0, -v.z * inv, v.y * inv);
}

vec3 any_orthogonal(vec3 v) { return eigen_unit_orthogonal(v); }

vec3 random_unit_vector(RngKey key, uint index) {
    float z = 2.0 * rng_uniform(key, index + 0u) - 1.0;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = PM_TWO_PI * rng_uniform(key, index + 1u);
    return vec3(r * cos(phi), r * sin(phi), z);
}

// Unit normal inside the 60-degree cone about -b_p: it faces the reference camera.
//
// This is where the unresolved NVIDIA difference shows: the optimised pm_init
// module reports normal agreement 0.92% on a Tesla T4 while the depths drawn from
// the same stream are exact. See the OPEN ISSUE at the top of this file; it is not
// a draw-order or state-loss problem (three different generator shapes produced
// the same number). Do not reach for `precise` on the trig to chase it:
// NoContraction in the cost loop makes glslc -O need tens of gigabytes and CI
// OOM-kills it, which is the other half of why both kernels ship at -O0.
vec3 random_cone_normal(RngKey key, uint index, vec3 b_p) {
    vec3 axis = normalize(-b_p);
    float cos_min = cos(PM_CONE_HALF_ANGLE);
    float ct = cos_min + (1.0 - cos_min) * rng_uniform(key, index + 0u);
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float phi = PM_TWO_PI * rng_uniform(key, index + 1u);
    vec3 u = any_orthogonal(axis);
    vec3 v = cross(axis, u);
    return normalize(st * cos(phi) * u + st * sin(phi) * v + ct * axis);
}

// n rotated by a random angle in [0, max_angle] about a random axis.
vec3 perturb_normal(RngKey key, uint index, vec3 n, float max_angle) {
    float angle = max_angle * rng_uniform(key, index + 0u);
    vec3 axis = random_unit_vector(key, index + 1u);
    if (abs(dot(axis, n)) > 0.999) {
        axis = any_orthogonal(n);
    }
    float c = cos(angle);
    float s = sin(angle);
    return normalize(n * c + cross(axis, n) * s + axis * dot(axis, n) * (1.0 - c));
}

// Depth sampled uniformly in inverse depth over [depth_min, depth_max].
float random_depth(RngKey key, uint index) {
    float lo = 1.0 / pc.depth_max;
    float hi = 1.0 / pc.depth_min;
    float inv = lo + (hi - lo) * rng_uniform(key, index + 0u);
    return 1.0 / inv;
}

// The forward model of model/camera/calibration.hpp in float32 (Brown-Conrady
// plus the affine map; the theta/r0 equidistant scaling for a fisheye).
// Returns false exactly where the CPU model does (Frame z <= 0, Fisheye origin).
bool project_point(int type, int vw, int vh, vec3 p_cam, vec3 fcx, vec2 b, vec4 radial,
                   vec2 tangential, out vec2 pixel) {
    // See the note at the top of this file. Only the operations that *can* be
    // contracted into a fused multiply-add are marked: a division, a sqrt and a
    // plain multiply have no fused form, so their values are already exact as
    // written and marking them would only enlarge the set of expressions the
    // optimiser must keep non-reassociable. That set is what makes glslc -O
    // expensive, and this function runs once per window sample per candidate.
    float xs = 0.0;
    float ys = 0.0;
    if (type == 0) {
        if (!(p_cam.z > 0.0)) {
            return false;
        }
        xs = p_cam.x / p_cam.z;
        ys = p_cam.y / p_cam.z;
    } else {
        // The only fusable expression on the fisheye path is this sum.
        precise float r3d = sqrt(p_cam.x * p_cam.x + p_cam.y * p_cam.y);
        if (r3d == 0.0) {
            if (p_cam.z == 0.0) {
                return false;
            }
        } else {
            float theta = atan(r3d, p_cam.z);
            float scale = theta / r3d;
            xs = p_cam.x * scale;
            ys = p_cam.y * scale;
        }
    }
    float k1 = radial.x, k2 = radial.y, k3 = radial.z, k4 = radial.w;
    float p1 = tangential.x, p2 = tangential.y;
    // Declared precise, so both the copy and the polynomial assignment below are
    // non-contractible: the undistorted copy is exact either way, the polynomial
    // is a chain of multiply-adds.
    precise float xd = xs;
    precise float yd = ys;
    if (k1 != 0.0 || k2 != 0.0 || k3 != 0.0 || k4 != 0.0 || p1 != 0.0 || p2 != 0.0) {
        // r2 and radial_scale are sums of products (fusable); r4/r6/r8 are plain
        // multiplies and need no mark.
        precise float r2 = xs * xs + ys * ys;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float r8 = r4 * r4;
        precise float radial_scale = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        xd = xs * radial_scale + p1 * (r2 + 2.0 * xs * xs) + 2.0 * p2 * xs * ys;
        yd = ys * radial_scale + p2 * (r2 + 2.0 * ys * ys) + 2.0 * p1 * xs * ys;
    }
    float f = fcx.x, cx = fcx.y, cy = fcx.z;
    // Same operation sequence as try_project_point(), no contraction: the
    // sampler's floor() below turns a 1-ulp difference into a different texel.
    precise float u = float(vw) * 0.5 + cx + xd * (f + b.x) + yd * b.y;
    precise float v = float(vh) * 0.5 + cy + yd * f;
    pixel = vec2(u, v);
    return true;
}

// Bilinear sample at integer-centred coordinates, identical to sample_image():
// the two lower indices are clamped to the image independently.
float sample_image(int image_base, int vw, int vh, float x, float y) {
    float fx = floor(x);
    float fy = floor(y);
    int xr = int(fx);
    int yr = int(fy);
    float ax = x - fx;
    float ay = y - fy;
    // Both indices are clamped independently: at a coordinate in [-0.5, 0)
    // this replicates the edge pixel, which is what sample() does on the CPU.
    int xa = clamp(xr, 0, vw - 1);
    int xb = clamp(xr + 1, 0, vw - 1);
    int ya = clamp(yr, 0, vh - 1);
    int yb = clamp(yr + 1, 0, vh - 1);
    int row_a = image_base + ya * vw;
    int row_b = image_base + yb * vw;
    float v00 = nb_img[row_a + xa];
    float v10 = nb_img[row_a + xb];
    float v01 = nb_img[row_b + xa];
    float v11 = nb_img[row_b + xb];
    // Deliberately NOT `precise`: this runs once per window sample per
    // neighbour (49 x 32 in the worst case), and NoContraction there is what
    // made glslc -O blow past the memory of a CI runner. Contracting this mix
    // perturbs the sampled value in its last bit only; it does not move the
    // sample to another texel, which is the effect that actually flips a match
    // (see project_point below).
    float top = v00 * (1.0 - ax) + v10 * ax;
    float bottom = v01 * (1.0 - ax) + v11 * ax;
    return top * (1.0 - ay) + bottom * ay;
}
