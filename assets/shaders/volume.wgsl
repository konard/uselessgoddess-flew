// Volume Raymarching Shader for Medical CT Visualization
// ========================================================
// Features:
// - Ray-box intersection supporting camera inside the volume
// - Front-to-back alpha compositing with early ray termination
// - Gradient-based Phong shading for surface depth cues
// - Jittered sampling to suppress banding artefacts
// - Medical CT transfer function (air transparent, soft tissue reddish, bone white)

#import bevy_pbr::{
    mesh_view_bindings::view,
    mesh_bindings::mesh,
    forward_io::VertexOutput,
}

@group(2) @binding(0) var volume_texture: texture_3d<f32>;
@group(2) @binding(1) var volume_sampler: sampler;

struct MaterialConfig {
    threshold: f32,
    step_count: f32,
    density_scale: f32,
    jitter_strength: f32,
}
@group(2) @binding(2) var<uniform> config: MaterialConfig;

const MAX_STEPS: i32 = 512;
const MIN_STEP_SIZE: f32 = 0.0005;
const GRADIENT_DELTA: f32 = 0.005;
const EARLY_TERMINATION_ALPHA: f32 = 0.98;

// Ray-box intersection via the slab method.
// The volume occupies the unit cube centered at the origin: [-0.5, 0.5]^3.
// Returns vec2(t_near, t_far); ray misses when t_near > t_far.
fn ray_box_intersection(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / (ray_dir + vec3<f32>(1e-10));

    let t0 = (-0.5 - ray_origin) * inv_dir;
    let t1 = (0.5 - ray_origin) * inv_dir;

    let t_min = min(t0, t1);
    let t_max = max(t0, t1);

    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far  = min(min(t_max.x, t_max.y), t_max.z);

    return vec2<f32>(t_near, t_far);
}

// Map local cube coordinates [-0.5, 0.5]^3 to texture UVW [0, 1]^3.
fn local_to_uvw(local_pos: vec3<f32>) -> vec3<f32> {
    return local_pos + 0.5;
}

fn sample_volume(uvw: vec3<f32>) -> f32 {
    return textureSampleLevel(volume_texture, volume_sampler, uvw, 0.0).r;
}

// Central-difference gradient, normalised. Returns zero vector in flat regions.
fn compute_gradient(uvw: vec3<f32>) -> vec3<f32> {
    let d = GRADIENT_DELTA;
    let gx = sample_volume(uvw + vec3<f32>(d, 0.0, 0.0))
           - sample_volume(uvw - vec3<f32>(d, 0.0, 0.0));
    let gy = sample_volume(uvw + vec3<f32>(0.0, d, 0.0))
           - sample_volume(uvw - vec3<f32>(0.0, d, 0.0));
    let gz = sample_volume(uvw + vec3<f32>(0.0, 0.0, d))
           - sample_volume(uvw - vec3<f32>(0.0, 0.0, d));

    let grad = vec3<f32>(gx, gy, gz);
    let len  = length(grad);
    if len < 0.0001 {
        return vec3<f32>(0.0);
    }
    return grad / len;
}

// Medical transfer function mapping HU-normalised density → RGBA.
//   air / noise:  transparent
//   soft tissue:  semi-transparent reddish
//   bone:         opaque warm white, gradient-shaded for depth
fn transfer_function(
    density: f32,
    threshold: f32,
    gradient: vec3<f32>,
    light_dir: vec3<f32>,
) -> vec4<f32> {
    let noise_floor = threshold * 0.4;

    if density < noise_floor {
        return vec4<f32>(0.0);
    }

    if density < threshold {
        let t = (density - noise_floor) / (threshold - noise_floor);
        let color = vec3<f32>(0.75, 0.25, 0.25);
        let alpha = t * t * 0.08;
        return vec4<f32>(color * t, alpha);
    }

    let bone_t = (density - threshold) / (1.0 - threshold + 0.001);

    let bone_base = mix(
        vec3<f32>(0.88, 0.86, 0.82),
        vec3<f32>(1.0,  0.99, 0.97),
        bone_t,
    );

    var shading = 1.0;
    if length(gradient) > 0.1 {
        let diffuse = max(dot(-gradient, light_dir), 0.0);
        shading = 0.4 + 0.6 * diffuse;
    }

    let alpha = mix(0.35, 0.97, bone_t * bone_t);
    return vec4<f32>(bone_base * shading, alpha);
}

// Simple hash for per-pixel jitter (anti-banding).
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

// Transform a world-space ray into the mesh's local cube space.
// Correctly handles arbitrary translation, rotation, and non-uniform scale.
fn transform_ray_to_local(
    world_origin: vec3<f32>,
    world_dir:   vec3<f32>,
    model:       mat4x4<f32>,
) -> array<vec3<f32>, 2> {
    let scale = vec3<f32>(
        length(model[0].xyz),
        length(model[1].xyz),
        length(model[2].xyz),
    );

    let rot_x = model[0].xyz / scale.x;
    let rot_y = model[1].xyz / scale.y;
    let rot_z = model[2].xyz / scale.z;

    // Inverse rotation = transpose for orthonormal basis.
    let inv_rot = mat3x3<f32>(
        vec3<f32>(rot_x.x, rot_y.x, rot_z.x),
        vec3<f32>(rot_x.y, rot_y.y, rot_z.y),
        vec3<f32>(rot_x.z, rot_y.z, rot_z.z),
    );

    let translation  = model[3].xyz;
    let local_origin = inv_rot * (world_origin - translation) / scale;
    let local_dir    = normalize(inv_rot * world_dir / scale);

    return array<vec3<f32>, 2>(local_origin, local_dir);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_frag = in.world_position.xyz;
    let camera_pos = view.world_position;
    let model      = mesh[in.instance_index].world_from_local;

    let world_dir = normalize(world_frag - camera_pos);

    let local_ray    = transform_ray_to_local(camera_pos, world_dir, model);
    let local_origin = local_ray[0];
    let local_dir    = local_ray[1];

    let t_hit = ray_box_intersection(local_origin, local_dir);

    if t_hit.x > t_hit.y {
        discard;
    }

    // Support camera inside the volume by clamping t_near to 0.
    let t_near = max(t_hit.x, 0.0);
    let t_far  = t_hit.y;

    if t_far < 0.0 {
        discard;
    }

    let ray_length = t_far - t_near;
    let base_steps = i32(config.step_count);
    let step_size  = max(ray_length / f32(base_steps), MIN_STEP_SIZE);

    let jitter = hash(in.position.xy) * config.jitter_strength;
    var t = t_near + jitter * step_size;

    let light_dir = normalize(vec3<f32>(0.2, 0.5, 1.0));

    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    var prev_density      = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        if t > t_far || accumulated_alpha > EARLY_TERMINATION_ALPHA {
            break;
        }
        if i >= base_steps {
            break;
        }

        let sample_pos = local_origin + local_dir * t;
        let uvw        = local_to_uvw(sample_pos);

        if any(uvw < vec3<f32>(-0.001)) || any(uvw > vec3<f32>(1.001)) {
            t += step_size;
            continue;
        }

        let safe_uvw = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0));
        let density  = sample_volume(safe_uvw);

        var gradient = vec3<f32>(0.0);
        if density > config.threshold * 0.5 {
            gradient = compute_gradient(safe_uvw);
        }

        let sample_rgba = transfer_function(density, config.threshold, gradient, light_dir);

        if sample_rgba.a > 0.001 {
            let scaled_alpha = sample_rgba.a * step_size * config.density_scale;
            let weight       = (1.0 - accumulated_alpha) * scaled_alpha;
            accumulated_color += sample_rgba.rgb * weight;
            accumulated_alpha += weight;
        }

        // Halve the step near density transitions for sharper surfaces.
        var current_step = step_size;
        if abs(density - prev_density) > 0.1 && density > config.threshold * 0.3 {
            current_step = step_size * 0.5;
        }
        prev_density = density;

        t += current_step;
    }

    if accumulated_alpha < 0.005 {
        discard;
    }

    let bg_color    = vec3<f32>(0.08, 0.08, 0.10);
    let final_color = accumulated_color + bg_color * (1.0 - accumulated_alpha);

    return vec4<f32>(final_color, min(accumulated_alpha, 1.0));
}
