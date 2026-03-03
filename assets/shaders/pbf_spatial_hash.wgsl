// assets/shaders/pbf_spatial_hash.wgsl
// Spatial hashing compute shader for PBF neighbor search
// Uses a compact uniform grid (spatial hash) for O(1) average neighbor lookup

// ---- Bindings ----
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;         // particle positions (w=unused)
@group(0) @binding(1) var<storage, read_write> hash_keys: array<atomic<u32>>; // per-particle cell hash
@group(0) @binding(2) var<storage, read_write> hash_values: array<u32>;       // particle indices (sorted by cell)
@group(0) @binding(3) var<storage, read_write> cell_start: array<atomic<u32>>;// start index in hash_values for each cell
@group(0) @binding(4) var<uniform> params: SpatialHashParams;

struct SpatialHashParams {
    particle_count: u32,
    cell_count: u32,      // must be power of 2
    cell_size: f32,       // = smoothing radius h
    // padding
    _pad: u32,
}

// Large prime numbers for hashing
const P1: u32 = 73856093u;
const P2: u32 = 19349663u;
const P3: u32 = 83492791u;

fn cell_hash(cell: vec3<i32>) -> u32 {
    let ux = u32(cell.x) * P1;
    let uy = u32(cell.y) * P2;
    let uz = u32(cell.z) * P3;
    return (ux ^ uy ^ uz) & (params.cell_count - 1u);
}

fn world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(pos / params.cell_size));
}

// Pass 1: Compute hash keys for all particles
@compute @workgroup_size(64)
fn compute_hashes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.particle_count { return; }

    let pos = positions[idx].xyz;
    let cell = world_to_cell(pos);
    let h = cell_hash(cell);
    hash_keys[idx] = h;
    // Increment cell counter
    atomicAdd(&cell_start[h], 1u);
}

// Pass 2: After prefix sum on cell_start, write particle indices
// (called after CPU-side or GPU prefix scan)
@compute @workgroup_size(64)
fn scatter_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.particle_count { return; }

    let h = hash_keys[idx];
    let slot = atomicSub(&cell_start[h], 1u) - 1u;
    hash_values[slot] = idx;
}
