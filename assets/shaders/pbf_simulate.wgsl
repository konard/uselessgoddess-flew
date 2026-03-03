// assets/shaders/pbf_simulate.wgsl
// Position-Based Fluids (PBF) simulation shader
// Based on: Macklin & Müller 2013 "Position Based Fluids"
// GPU implementation with Spatial Hashing for neighbor search

// ---- Bindings ----
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;       // xyz=pos, w=density
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;      // xyz=vel, w=lambda
@group(0) @binding(2) var<storage, read_write> predicted: array<vec4<f32>>;       // predicted positions
@group(0) @binding(3) var<storage, read> hash_values: array<u32>;                 // sorted particle indices
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;                  // cell start indices
@group(0) @binding(5) var<storage, read> cell_count_arr: array<u32>;              // particles per cell
@group(0) @binding(6) var<uniform> sim: SimParams;
@group(0) @binding(7) var<storage, read> volume_sdf: array<f32>;                  // SDF from CT volume (boundary)
@group(0) @binding(8) var<uniform> sdf_params: SdfParams;

struct SimParams {
    particle_count: u32,
    dt: f32,
    h: f32,               // smoothing radius
    rho0: f32,            // rest density
    eps_cfm: f32,         // relaxation (CFM epsilon)
    k_corr: f32,          // surface tension coefficient
    delta_q: f32,         // surface tension delta_q (fraction of h)
    n_corr: f32,          // surface tension exponent
    c_xsph: f32,          // XSPH viscosity coefficient
    gravity: vec3<f32>,
    solver_iterations: u32,
    cell_count: u32,      // power-of-2 hash table size
    _pad0: u32,
    _pad1: u32,
}

struct SdfParams {
    origin: vec3<f32>,
    voxel_size: f32,
    grid_size: vec3<u32>,
    _pad: u32,
}

// Large prime numbers for spatial hashing
const P1: u32 = 73856093u;
const P2: u32 = 19349663u;
const P3: u32 = 83492791u;

fn cell_hash(cell: vec3<i32>) -> u32 {
    let ux = u32(cell.x) * P1;
    let uy = u32(cell.y) * P2;
    let uz = u32(cell.z) * P3;
    return (ux ^ uy ^ uz) & (sim.cell_count - 1u);
}

fn world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(pos / sim.h));
}

// Poly6 kernel (density estimation)
fn W_poly6(r_sq: f32) -> f32 {
    let h_sq = sim.h * sim.h;
    if r_sq >= h_sq { return 0.0; }
    let x = h_sq - r_sq;
    // k = 315 / (64 * pi * h^9)
    let k = 315.0 / (64.0 * 3.14159265 * pow(sim.h, 9.0));
    return k * x * x * x;
}

// Spiky gradient kernel (pressure solving)
fn W_spiky_grad(r: vec3<f32>, r_len: f32) -> vec3<f32> {
    if r_len >= sim.h || r_len < 0.0001 { return vec3<f32>(0.0); }
    // k = -45 / (pi * h^6)
    let k = -45.0 / (3.14159265 * pow(sim.h, 6.0));
    let x = sim.h - r_len;
    return k * x * x * (r / r_len);
}

// Sample volume SDF at world position
fn sample_sdf(pos: vec3<f32>) -> f32 {
    let local = (pos - sdf_params.origin) / sdf_params.voxel_size;
    let grid = vec3<i32>(floor(local));
    let gs = vec3<i32>(sdf_params.grid_size);

    if any(grid < vec3<i32>(0)) || any(grid >= gs - vec3<i32>(1)) {
        // Outside SDF: use wall plane at boundaries as fallback
        let half = vec3<f32>(sdf_params.grid_size) * 0.5 * sdf_params.voxel_size + sdf_params.origin;
        let box_size = vec3<f32>(sdf_params.grid_size) * sdf_params.voxel_size;
        let d = abs(pos - half) - box_size * 0.5;
        return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3<f32>(0.0)));
    }

    let t = fract(local);
    let idx = grid.x + grid.y * gs.x + grid.z * gs.x * gs.y;

    // Trilinear interpolation
    let d000 = volume_sdf[u32(idx)];
    let d100 = volume_sdf[u32(idx + 1)];
    let d010 = volume_sdf[u32(idx + gs.x)];
    let d110 = volume_sdf[u32(idx + gs.x + 1)];
    let d001 = volume_sdf[u32(idx + gs.x * gs.y)];
    let d101 = volume_sdf[u32(idx + gs.x * gs.y + 1)];
    let d011 = volume_sdf[u32(idx + gs.x * gs.y + gs.x)];
    let d111 = volume_sdf[u32(idx + gs.x * gs.y + gs.x + 1)];

    let d00 = mix(d000, d100, t.x);
    let d01 = mix(d001, d101, t.x);
    let d10 = mix(d010, d110, t.x);
    let d11 = mix(d011, d111, t.x);
    let d0 = mix(d00, d10, t.y);
    let d1 = mix(d01, d11, t.y);
    return mix(d0, d1, t.z);
}

// SDF gradient (approximate normal) via finite differences
fn sdf_normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = sdf_params.voxel_size;
    let dx = sample_sdf(pos + vec3(eps, 0.0, 0.0)) - sample_sdf(pos - vec3(eps, 0.0, 0.0));
    let dy = sample_sdf(pos + vec3(0.0, eps, 0.0)) - sample_sdf(pos - vec3(0.0, eps, 0.0));
    let dz = sample_sdf(pos + vec3(0.0, 0.0, eps)) - sample_sdf(pos - vec3(0.0, 0.0, eps));
    return normalize(vec3<f32>(dx, dy, dz));
}

// ---- Pass 1: Predict positions under gravity ----
@compute @workgroup_size(64)
fn predict_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= sim.particle_count { return; }

    var vel = velocities[idx].xyz;
    var pos = positions[idx].xyz;

    // Apply gravity
    vel += sim.gravity * sim.dt;
    var pred = pos + vel * sim.dt;

    // Boundary collision with SDF (pre-clip to domain box)
    let sdf_val = sample_sdf(pred);
    if sdf_val < 0.0 {
        let n = sdf_normal(pred);
        pred -= sdf_val * n;
    }

    predicted[idx] = vec4<f32>(pred, positions[idx].w);
    velocities[idx] = vec4<f32>(vel, velocities[idx].w);
}

// ---- Pass 2: Compute density and lambda (constraint) ----
@compute @workgroup_size(64)
fn compute_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= sim.particle_count { return; }

    let pos_i = predicted[idx].xyz;
    let cell_i = world_to_cell(pos_i);

    var rho_i = 0.0;
    var grad_sum_sq = 0.0;
    var grad_i = vec3<f32>(0.0);

    // Iterate over 3x3x3 neighborhood
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);
                let h = cell_hash(neighbor_cell);
                let start = cell_start[h];
                let count = cell_count_arr[h];

                for (var k = 0u; k < count; k++) {
                    let j = hash_values[start + k];
                    if j >= sim.particle_count { continue; }

                    let pos_j = predicted[j].xyz;
                    let r = pos_i - pos_j;
                    let r_sq = dot(r, r);

                    rho_i += W_poly6(r_sq);

                    if j != idx {
                        let spiky = W_spiky_grad(r, sqrt(r_sq));
                        grad_i += spiky;
                        grad_sum_sq += dot(spiky, spiky);
                    }
                }
            }
        }
    }

    // Constraint: C_i = rho_i/rho0 - 1
    let C_i = rho_i / sim.rho0 - 1.0;

    // Sum of squared gradients (k = i term)
    grad_sum_sq += dot(grad_i, grad_i);
    grad_sum_sq /= (sim.rho0 * sim.rho0);

    // Lambda (Lagrange multiplier)
    let lambda = -C_i / (grad_sum_sq + sim.eps_cfm);

    // Store density in w, lambda in velocities.w
    predicted[idx].w = rho_i;
    velocities[idx].w = lambda;
}

// ---- Pass 3: Compute position corrections ----
@compute @workgroup_size(64)
fn compute_delta_pos(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= sim.particle_count { return; }

    let pos_i = predicted[idx].xyz;
    let lambda_i = velocities[idx].w;
    let cell_i = world_to_cell(pos_i);

    var delta_p = vec3<f32>(0.0);

    // Reference kernel value for surface tension correction
    let dq = sim.delta_q * sim.h;
    let W_dq = W_poly6(dq * dq);

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);
                let h = cell_hash(neighbor_cell);
                let start = cell_start[h];
                let count = cell_count_arr[h];

                for (var k = 0u; k < count; k++) {
                    let j = hash_values[start + k];
                    if j == idx || j >= sim.particle_count { continue; }

                    let pos_j = predicted[j].xyz;
                    let r = pos_i - pos_j;
                    let r_len = length(r);

                    // Surface tension (tensile instability correction)
                    let W_r = W_poly6(dot(r, r));
                    var s_corr = 0.0;
                    if W_dq > 0.0001 {
                        s_corr = -sim.k_corr * pow(W_r / W_dq, sim.n_corr);
                    }

                    let lambda_j = velocities[j].w;
                    delta_p += (lambda_i + lambda_j + s_corr) * W_spiky_grad(r, r_len);
                }
            }
        }
    }

    delta_p /= sim.rho0;

    // Boundary constraint via SDF
    var new_pred = pos_i + delta_p;
    let sdf_val = sample_sdf(new_pred);
    if sdf_val < 0.0 {
        let n = sdf_normal(new_pred);
        new_pred -= sdf_val * n;
    }

    predicted[idx] = vec4<f32>(new_pred, predicted[idx].w);
}

// ---- Pass 4: Update velocities and apply XSPH viscosity ----
@compute @workgroup_size(64)
fn update_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= sim.particle_count { return; }

    let pos_new = predicted[idx].xyz;
    let pos_old = positions[idx].xyz;
    var vel = (pos_new - pos_old) / sim.dt;

    let cell_i = world_to_cell(pos_new);

    // XSPH viscosity: smooth velocity field
    var vel_correction = vec3<f32>(0.0);

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);
                let h = cell_hash(neighbor_cell);
                let start = cell_start[h];
                let count = cell_count_arr[h];

                for (var k = 0u; k < count; k++) {
                    let j = hash_values[start + k];
                    if j == idx || j >= sim.particle_count { continue; }

                    let pos_j = predicted[j].xyz;
                    let vel_j = velocities[j].xyz;
                    let r = pos_new - pos_j;
                    let W = W_poly6(dot(r, r));

                    vel_correction += (vel_j - vel) * W;
                }
            }
        }
    }

    vel += sim.c_xsph * vel_correction;

    positions[idx] = vec4<f32>(pos_new, positions[idx].w);
    velocities[idx] = vec4<f32>(vel, velocities[idx].w);
}
