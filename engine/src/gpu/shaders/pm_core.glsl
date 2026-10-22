// PatchMatch cost core, shared by the two pm_* kernels.
//
// One lane owns one *active* pixel of the checkerboard colour: the 64 lanes of
// a workgroup cover the 64 pixels of one colour in an 8x16 output tile, so no
// lane is ever idle. Only the neighbour view records are staged in shared
// memory (1.5 KB); everything else is read from global memory, where the tile's
// overlapping windows are served from cache. That keeps shared-memory pressure
// low enough for several workgroups per core, which the branchy, divergent
// inner loop needs to hide latency.
//
// The cost is the CPU's multiview_cost(), term for term and in the same order:
// ZNCC per neighbour over the same sampled window, the 75%-of-window visibility
// quorum that makes k_eff = min(best_k, votes), per-neighbour costs clamped to
// [0, 2] and a sentinel of 2.0.

const int PM_TILE_X = 8;
const int PM_TILE_Y = 16;  // 64 colour pixels = one lane each

shared float s_views[PM_MAX_NEIGHBOURS * PM_VIEW_STRIDE];
shared uint s_view_ints[PM_MAX_NEIGHBOURS * PM_VIEW_INT_STRIDE];
// Window offsets are read by every candidate of every neighbour; staging them
// removes ~49 global loads per candidate.
shared ivec2 s_offsets[PM_MAX_SAMPLES];

// Pixel of one lane: the n-th pixel of the checkerboard colour inside the
// workgroup's 8x16 tile. A tile's first pixel has colour (origin.x + origin.y) &
// 1; within the tile the column parity flips with the row, so the 64
// same-colour pixels are four per row over sixteen rows.
ivec2 lane_pixel(int lane) {
    ivec2 origin = ivec2(int(gl_WorkGroupID.x) * PM_TILE_X, int(gl_WorkGroupID.y) * PM_TILE_Y);
    int local_parity = (int(pc.colour) ^ ((origin.x + origin.y) & 1));
    int ly = lane / (PM_TILE_X / 2);
    int ordinal = lane % (PM_TILE_X / 2);
    // (lx + ly) & 1 must equal local_parity, so the column parity flips per row.
    int lx_parity = local_parity ^ (ly & 1);
    return ivec2(origin.x + 2 * ordinal + lx_parity, origin.y + ly);
}

// Stages the neighbour view records and the window offsets: a few hundred
// bytes, loaded once per workgroup instead of once per candidate per lane.
void stage_views() {
    int nn = int(pc.n_neighbours);
    int total = nn * PM_VIEW_STRIDE;
    for (int i = int(gl_LocalInvocationID.x); i < total; i += PM_LANES) {
        s_views[i] = views[i];
    }
    total = nn * PM_VIEW_INT_STRIDE;
    for (int i = int(gl_LocalInvocationID.x); i < total; i += PM_LANES) {
        s_view_ints[i] = view_ints[i];
    }
    for (int i = int(gl_LocalInvocationID.x); i < int(pc.n_samples); i += PM_LANES) {
        s_offsets[i] = ivec2(offsets[2 * i], offsets[2 * i + 1]);
    }
    memoryBarrierShared();
    barrier();
}

// Reference texture test: the mean over the full sampled window, as the CPU
// measures it (the precomputed array holds the CPU's edge-clamped values).
bool window_is_textured(int pixel) {
    int n = int(pc.n_samples);
    int window_base = pixel * n;
    float sum = 0.0;
    float sq = 0.0;
    for (int k = 0; k < n; ++k) {
        float v = ref_windows[window_base + k];
        sum += v;
        sq += v * v;
    }
    float inv = 1.0 / float(n);
    float mean = sum * inv;
    float var = max(0.0, sq - sum * mean);
    return sqrt(var * inv) >= pc.min_texture;
}

// Plane depth at window sample k for a candidate plane (rho, n), with the
// CPU's validity rule: a sample is skipped when the plane is edge-on or behind
// the reference camera.
float candidate_depth(int x, int y, int k, float rho, vec3 n) {
    ivec2 off = s_offsets[k];
    int b = ((y + off.y) * int(pc.width) + (x + off.x)) * 3;
    vec3 bearing = vec3(bearings[b + 0], bearings[b + 1], bearings[b + 2]);
    float nb = dot(n, bearing);
    if (nb < -PM_FACING_EPS && rho / nb > 0.0) {
        return rho / nb;
    }
    return 0.0;
}

// ZNCC over the window for one neighbour, mirroring neighbour_cost().
// Per-neighbour constants, loaded once per pixel instead of once per candidate
// (they never change across the ~9 candidates of a half-sweep).
struct ViewData {
    vec3 fcx;
    vec2 b;
    vec4 radial;
    vec2 tangential;
    float m[9];
    vec3 c;
    int vw;
    int vh;
    int type;
    int image_base;
    float max_x;
    float max_y;
};

ViewData load_view(int view) {
    int vint = view * PM_VIEW_INT_STRIDE;
    int vpar = view * PM_VIEW_STRIDE;
    ViewData v;
    v.vw = int(s_view_ints[vint + 0]);
    v.vh = int(s_view_ints[vint + 1]);
    v.type = int(s_view_ints[vint + 2]);
    v.image_base = int(s_view_ints[vint + 3]);
    v.fcx = vec3(s_views[vpar + 0], s_views[vpar + 1], s_views[vpar + 2]);
    v.b = vec2(s_views[vpar + 3], s_views[vpar + 4]);
    v.radial = vec4(s_views[vpar + 5], s_views[vpar + 6], s_views[vpar + 7], s_views[vpar + 8]);
    v.tangential = vec2(s_views[vpar + 9], s_views[vpar + 10]);
    for (int i = 0; i < 9; ++i) {
        v.m[i] = s_views[vpar + 11 + i];
    }
    v.c = vec3(s_views[vpar + 20], s_views[vpar + 21], s_views[vpar + 22]);
    v.max_x = float(v.vw - 1);
    v.max_y = float(v.vh - 1);
    return v;
}

float neighbour_cost(int x, int y, int min_valid, float dq[PM_MAX_SAMPLES], ViewData v,
                     out int valid_out) {
    int n = int(pc.n_samples);
    int window_base = (y * int(pc.width) + x) * n;

    precise float ref_sum = 0.0, ref_sq = 0.0, nb_sum = 0.0, nb_sq = 0.0, cross_sum = 0.0;
    int valid = 0;
    // Sums are accumulated about the first valid sample so the variance and
    // covariance below do not cancel catastrophically; contracted multiply-adds
    // (NVIDIA fuses them) otherwise change the result near ties.
    float r0 = 0.0, m0 = 0.0;
    for (int k = 0; k < n; ++k) {
        float depth_q = dq[k];
        if (!(depth_q > 0.0)) {
            continue;  // edge-on or behind the reference camera for every view
        }
        ivec2 off = s_offsets[k];
        int bi = ((y + off.y) * int(pc.width) + (x + off.x)) * 3;
        vec3 bq = vec3(bearings[bi + 0], bearings[bi + 1], bearings[bi + 2]);
        vec3 x_n = depth_q * vec3(v.m[0] * bq.x + v.m[1] * bq.y + v.m[2] * bq.z,
                                  v.m[3] * bq.x + v.m[4] * bq.y + v.m[5] * bq.z,
                                  v.m[6] * bq.x + v.m[7] * bq.y + v.m[8] * bq.z) +
                   v.c;
        if (!(x_n.z > PM_CAMERA_Z_EPS)) {
            continue;
        }
        vec2 pixel;
        if (!project_point(v.type, v.vw, v.vh, x_n, v.fcx, v.b, v.radial, v.tangential,
                           pixel)) {
            continue;
        }
        // Convention bridge: project() puts the top-left pixel centre at
        // (0.5, 0.5), sample_image() at (0, 0).
        float sx = pixel.x - 0.5;
        float sy = pixel.y - 0.5;
        if (sx < -0.5 || sy < -0.5 || sx > v.max_x + 0.5 || sy > v.max_y + 0.5) {
            continue;
        }
        float r = ref_windows[window_base + k];
        float m = sample_image(v.image_base, v.vw, v.vh, sx, sy);
        if (valid == 0) {
            r0 = r;
            m0 = m;
        }
        r -= r0;
        m -= m0;
        ref_sum += r;
        ref_sq += r * r;
        nb_sum += m;
        nb_sq += m * m;
        cross_sum += r * m;
        ++valid;
    }
    valid_out = valid;
    if (valid < min_valid) {
        return PM_NO_MATCH_COST;
    }
    float inv = 1.0 / float(valid);
    float ref_mean = ref_sum * inv;
    float nb_mean = nb_sum * inv;
    precise float ref_var = max(0.0, ref_sq - ref_sum * ref_mean);
    precise float nb_var = max(0.0, nb_sq - nb_sum * nb_mean);
    if (ref_var <= 1e-12 || nb_var <= 1e-12) {
        return PM_NO_MATCH_COST;
    }
    precise float covariance = cross_sum - ref_sum * nb_mean;
    float z = clamp(covariance / sqrt(ref_var * nb_var), -1.0, 1.0);
    return 1.0 - z;
}

// Multi-view cost: mean of the best_k smallest voting costs, with
// k_eff = min(best_k, votes). Only neighbours that see at least 75% of the
// window vote; geometric invisibility is never folded in as a phantom 2.0.
float multiview_cost(int x, int y, float rho, vec3 plane_normal) {
    int n_samples = int(pc.n_samples);
    int min_valid = (3 * n_samples + 3) / 4;  // ceil(0.75 * n_samples)
    // Candidate-level plane depths, computed once and shared by every
    // neighbour (the CPU's prepare_candidate), and the validity bound the
    // quorum is judged against.
    float dq[PM_MAX_SAMPLES];
    int candidate_valid = 0;
    for (int k = 0; k < n_samples; ++k) {
        dq[k] = candidate_depth(x, y, k, rho, plane_normal);
        if (dq[k] > 0.0) {
            ++candidate_valid;
        }
    }
    if (candidate_valid < min_valid) {
        return PM_NO_MATCH_COST;
    }
    const int want = int(pc.best_k);
    float best[PM_MAX_NEIGHBOURS];
    int filled = 0;
    int votes = 0;
    int nn = int(pc.n_neighbours);
    for (int j = 0; j < nn; ++j) {
        int valid = 0;
        float cost = neighbour_cost(x, y, min_valid, dq, load_view(j), valid);
        if (valid < min_valid) {
            continue;
        }
        ++votes;
        if (filled < want) {
            int i = filled;
            while (i > 0 && best[i - 1] > cost) {
                best[i] = best[i - 1];
                --i;
            }
            best[i] = cost;
            ++filled;
        } else if (cost < best[want - 1]) {
            int i = want - 1;
            while (i > 0 && best[i - 1] > cost) {
                best[i] = best[i - 1];
                --i;
            }
            best[i] = cost;
        }
    }
    if (votes == 0) {
        return PM_NO_MATCH_COST;
    }
    int k_eff = min(want, votes);
    float total = 0.0;
    for (int t = 0; t < k_eff; ++t) {
        total += best[t];
    }
    return total / float(k_eff);
}

// The best hypothesis found for one pixel.
struct Best {
    float depth;
    vec3 normal;
    float cost;
};

// Snap a hypothesis into a valid plane and evaluate it, as
// evaluate_hypothesis(). A candidate equal to the incumbent reproduces its cost
// and can never be strictly cheaper, so it is skipped.
void evaluate_hypothesis(int x, int y, float depth, vec3 n, vec3 b_p, inout Best best) {
    if (!(depth > pc.depth_min && depth < pc.depth_max)) {
        return;
    }
    float nb = dot(n, b_p);
    if (nb >= -PM_FACING_EPS) {
        return;
    }
    if (depth == best.depth && n == best.normal) {
        return;
    }
    float cost = multiview_cost(x, y, depth * nb, n);
    if (cost < best.cost - pc.cost_epsilon) {
        best.cost = cost;
        best.depth = depth;
        best.normal = n;
    }
}

vec3 pixel_bearing(int pixel) {
    return vec3(bearings[pixel * 3 + 0], bearings[pixel * 3 + 1], bearings[pixel * 3 + 2]);
}

// Propagate pixel q's plane onto pixel p (as propagate()), reading q from the
// state buffers of the previous half-sweep.
void propagate(int x, int y, int q, vec3 b_p, inout Best best) {
    float qd = state_depth[q];
    if (!(qd > 0.0)) {
        return;
    }
    vec3 n = vec3(state_normal[q * 3 + 0], state_normal[q * 3 + 1], state_normal[q * 3 + 2]);
    if (dot(n, n) < 1e-12) {
        return;
    }
    float rho_q = qd * dot(n, pixel_bearing(q));
    float denom = dot(n, b_p);
    if (denom >= -PM_FACING_EPS) {
        return;
    }
    evaluate_hypothesis(x, y, rho_q / denom, n, b_p, best);
}
