// src/pbf.rs
// Position-Based Fluids (PBF) simulation - CPU-side Bevy integration
// GPU compute dispatch is set up here; the actual simulation runs in WGSL shaders.
//
// Architecture:
//  - PbfState: resource holding particle data + simulation parameters
//  - PbfPlugin: registers systems and resources
//  - Spatial hashing runs on GPU (pbf_spatial_hash.wgsl)
//  - Solver passes run on GPU (pbf_simulate.wgsl)
//  - ParticleEntity: component linking Bevy entities to particle indices

use bevy::prelude::*;
use rand::Rng;

// ---- Public API ----

pub struct PbfPlugin;

impl Plugin for PbfPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PbfState>()
            .add_systems(Update, (simulate_step, sync_particle_transforms).chain());
    }
}

/// Marks an entity as a fluid particle; `index` references PbfState::particles.
#[derive(Component)]
pub struct ParticleEntity {
    pub index: u32,
}

// ---- Particle Data ----

#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub predicted: Vec3,
    pub density: f32,
    pub lambda: f32,
}

impl Particle {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            predicted: position,
            density: 0.0,
            lambda: 0.0,
        }
    }
}

// ---- Simulation State ----

#[derive(Resource)]
pub struct PbfState {
    pub particles: Vec<Particle>,

    // Simulation parameters (tweakable at runtime)
    pub dt: f32,
    pub h: f32,            // smoothing radius
    pub rho0: f32,         // rest density
    pub eps_cfm: f32,      // CFM relaxation
    pub k_corr: f32,       // surface tension
    pub c_xsph: f32,       // XSPH viscosity
    pub gravity: Vec3,
    pub solver_iters: u32,

    // Exposed for UI
    pub viscosity: f32,    // maps to c_xsph
    pub pressure_scale: f32,

    pub particle_radius: f32,

    // Spatial hash table (CPU-side for now; GPU dispatch in full impl)
    hash_table: Vec<Vec<u32>>,
    cell_count: usize,
}

impl Default for PbfState {
    fn default() -> Self {
        Self {
            particles: Vec::new(),
            dt: 1.0 / 60.0,
            h: 0.05,           // 5cm smoothing radius
            rho0: 6378.0,      // rest density for water-like fluid
            eps_cfm: 100.0,
            k_corr: 0.001,
            c_xsph: 0.01,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver_iters: 3,
            viscosity: 0.01,
            pressure_scale: 1.0,
            particle_radius: 0.012,
            hash_table: Vec::new(),
            cell_count: 0,
        }
    }
}

impl PbfState {
    /// Initialize particles in a block inside the volume (roughly in the cranial cavity)
    pub fn initialize(&mut self, count: usize) {
        let mut rng = rand::rng();
        self.particles.clear();

        // Spawn particles in a small cube centered at origin
        // (within the normalized [-0.5, 0.5]^3 volume space)
        let spawn_region = 0.15_f32; // half-width of spawn box
        for _ in 0..count {
            let x = rng.random_range(-spawn_region..spawn_region);
            let y = rng.random_range(0.0..spawn_region * 2.0); // bias upward
            let z = rng.random_range(-spawn_region..spawn_region);
            self.particles.push(Particle::new(Vec3::new(x, y, z)));
        }

        // Build spatial hash
        self.rebuild_hash();
    }

    /// Rebuild the CPU spatial hash table
    fn rebuild_hash(&mut self) {
        // Power-of-2 cell count for bitwise modulo
        self.cell_count = next_power_of_two(self.particles.len() * 4);
        self.hash_table = vec![Vec::new(); self.cell_count];

        for (i, p) in self.particles.iter().enumerate() {
            let h = spatial_hash(p.position, self.h, self.cell_count);
            self.hash_table[h].push(i as u32);
        }
    }

    /// Collect neighbors of particle `idx` within smoothing radius `h`
    pub fn get_neighbors(&self, idx: usize) -> Vec<usize> {
        let pos = self.particles[idx].position;
        let cell = world_to_cell(pos, self.h);
        let mut neighbors = Vec::with_capacity(64);

        for dz in -1..=1_i32 {
            for dy in -1..=1_i32 {
                for dx in -1..=1_i32 {
                    let neighbor_cell = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    let h = cell_hash(neighbor_cell, self.cell_count);
                    for &j in &self.hash_table[h] {
                        let j = j as usize;
                        let r = pos - self.particles[j].position;
                        if dot_v(r) < self.h * self.h {
                            neighbors.push(j);
                        }
                    }
                }
            }
        }
        neighbors
    }

    /// Run one PBF simulation step (CPU fallback; GPU version dispatched via render graph)
    pub fn step(&mut self) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }

        let dt = self.dt;
        let h = self.h;
        let rho0 = self.rho0;
        let eps = self.eps_cfm;
        let gravity = self.gravity;

        // ---- 1. Predict positions ----
        for p in self.particles.iter_mut() {
            p.velocity += gravity * dt;
            p.predicted = p.position + p.velocity * dt;

            // Simple AABB boundary (volume cube [-0.5, 0.5])
            p.predicted = p.predicted.clamp(Vec3::splat(-0.49), Vec3::splat(0.49));
        }

        // Rebuild hash with predicted positions
        self.cell_count = next_power_of_two(n * 4);
        self.hash_table = vec![Vec::new(); self.cell_count];
        for (i, p) in self.particles.iter().enumerate() {
            let hh = spatial_hash(p.predicted, h, self.cell_count);
            self.hash_table[hh].push(i as u32);
        }

        // ---- 2. Solver iterations ----
        for _ in 0..self.solver_iters {
            // Compute lambda for each particle
            let mut lambdas = vec![0.0_f32; n];
            for i in 0..n {
                let pos_i = self.particles[i].predicted;
                let cell = world_to_cell(pos_i, h);

                let mut rho = 0.0_f32;
                let mut grad_sum_sq = 0.0_f32;
                let mut grad_i = Vec3::ZERO;

                for dz in -1..=1_i32 {
                    for dy in -1..=1_i32 {
                        for dx in -1..=1_i32 {
                            let nc = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                            let hh = cell_hash(nc, self.cell_count);
                            for &j_u in &self.hash_table[hh] {
                                let j = j_u as usize;
                                let r = pos_i - self.particles[j].predicted;
                                let r2 = dot_v(r);
                                rho += w_poly6(r2, h);
                                if j != i {
                                    let spiky = w_spiky_grad(r, r2.sqrt(), h);
                                    grad_i += spiky;
                                    grad_sum_sq += dot_v(spiky);
                                }
                            }
                        }
                    }
                }

                let c_i = rho / rho0 - 1.0;
                grad_sum_sq += dot_v(grad_i);
                grad_sum_sq /= rho0 * rho0;
                lambdas[i] = -c_i / (grad_sum_sq + eps);
                self.particles[i].density = rho;
                self.particles[i].lambda = lambdas[i];
            }

            // Compute position corrections
            let mut deltas = vec![Vec3::ZERO; n];
            for i in 0..n {
                let pos_i = self.particles[i].predicted;
                let lambda_i = lambdas[i];
                let cell = world_to_cell(pos_i, h);

                let dq = 0.1 * h;
                let w_dq = w_poly6(dq * dq, h);

                for dz in -1..=1_i32 {
                    for dy in -1..=1_i32 {
                        for dx in -1..=1_i32 {
                            let nc = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                            let hh = cell_hash(nc, self.cell_count);
                            for &j_u in &self.hash_table[hh] {
                                let j = j_u as usize;
                                if i == j {
                                    continue;
                                }
                                let r = pos_i - self.particles[j].predicted;
                                let r_len = r.length();

                                // Surface tension correction
                                let w_r = w_poly6(dot_v(r), h);
                                let s_corr = if w_dq > 0.0001 {
                                    -self.k_corr * (w_r / w_dq).powi(4)
                                } else {
                                    0.0
                                };

                                let lambda_j = lambdas[j];
                                deltas[i] += (lambda_i + lambda_j + s_corr)
                                    * w_spiky_grad(r, r_len, h);
                            }
                        }
                    }
                }
                deltas[i] /= rho0;
            }

            // Apply corrections + boundary clamping
            for i in 0..n {
                self.particles[i].predicted += deltas[i];
                self.particles[i].predicted =
                    self.particles[i].predicted.clamp(Vec3::splat(-0.49), Vec3::splat(0.49));
            }
        }

        // ---- 3. Update velocities + XSPH viscosity ----
        let c_xsph = self.c_xsph;

        // Copy predicted → velocity corrections
        let mut vel_corrections = vec![Vec3::ZERO; n];
        for i in 0..n {
            let pos_i = self.particles[i].predicted;
            let vel_i = (pos_i - self.particles[i].position) / dt;
            let cell = world_to_cell(pos_i, h);

            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        let nc = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                        let hh = cell_hash(nc, self.cell_count);
                        for &j_u in &self.hash_table[hh] {
                            let j = j_u as usize;
                            if i == j {
                                continue;
                            }
                            let pos_j = self.particles[j].predicted;
                            let vel_j = (pos_j - self.particles[j].position) / dt;
                            let r = pos_i - pos_j;
                            let w = w_poly6(dot_v(r), h);
                            vel_corrections[i] += (vel_j - vel_i) * w;
                        }
                    }
                }
            }
        }

        for i in 0..n {
            let pos_new = self.particles[i].predicted;
            let pos_old = self.particles[i].position;
            let mut vel = (pos_new - pos_old) / dt;
            vel += c_xsph * vel_corrections[i];
            self.particles[i].velocity = vel;
            self.particles[i].position = pos_new;
        }

        // Rebuild hash for next frame
        self.rebuild_hash();
    }
}

// ---- Kernel functions ----

fn w_poly6(r_sq: f32, h: f32) -> f32 {
    let h_sq = h * h;
    if r_sq >= h_sq {
        return 0.0;
    }
    let x = h_sq - r_sq;
    let k = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
    k * x * x * x
}

fn w_spiky_grad(r: Vec3, r_len: f32, h: f32) -> Vec3 {
    if r_len >= h || r_len < 0.0001 {
        return Vec3::ZERO;
    }
    let k = -45.0 / (std::f32::consts::PI * h.powi(6));
    let x = h - r_len;
    k * x * x * (r / r_len)
}

// ---- Spatial hashing helpers ----

const PRIME1: u64 = 73856093;
const PRIME2: u64 = 19349663;
const PRIME3: u64 = 83492791;

fn world_to_cell(pos: Vec3, h: f32) -> [i32; 3] {
    [
        (pos.x / h).floor() as i32,
        (pos.y / h).floor() as i32,
        (pos.z / h).floor() as i32,
    ]
}

fn cell_hash(cell: [i32; 3], table_size: usize) -> usize {
    let x = (cell[0] as i64 * PRIME1 as i64) as u64;
    let y = (cell[1] as i64 * PRIME2 as i64) as u64;
    let z = (cell[2] as i64 * PRIME3 as i64) as u64;
    ((x ^ y ^ z) as usize) & (table_size - 1)
}

fn spatial_hash(pos: Vec3, h: f32, table_size: usize) -> usize {
    cell_hash(world_to_cell(pos, h), table_size)
}

fn next_power_of_two(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    }
}

fn dot_v(v: Vec3) -> f32 {
    v.dot(v)
}

// ---- Bevy Systems ----

/// Advance the PBF simulation one step per frame.
pub fn simulate_step(mut pbf: ResMut<PbfState>) {
    pbf.step();
}

/// Sync particle positions to Bevy Transform components.
pub fn sync_particle_transforms(
    pbf: Res<PbfState>,
    mut query: Query<(&ParticleEntity, &mut Transform)>,
) {
    for (pe, mut transform) in query.iter_mut() {
        let idx = pe.index as usize;
        if idx < pbf.particles.len() {
            transform.translation = pbf.particles[idx].position;
        }
    }
}
