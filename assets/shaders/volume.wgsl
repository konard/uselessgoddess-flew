// Volume Raymarching Shader for Medical CT Visualization
// ========================================================
// SOTA implementation featuring:
// - Proper ray-box intersection with camera inside volume support
// - Front-to-back alpha compositing with early ray termination
// - Gradient-based shading for depth perception
// - Adaptive step refinement near surfaces
// - Jittered sampling for anti-banding
// - Medical transfer function (bones white, soft tissue reddish)

#import bevy_pbr::{
    mesh_view_bindings::view,
    mesh_bindings::mesh,
    forward_io::VertexOutput,
}

// --- Material bindings ---
@group(2) @binding(0) var volume_texture: texture_3d<f32>;
@group(2) @binding(1) var volume_sampler: sampler;

struct MaterialConfig {
    threshold: f32,       // Density threshold for bone visualization (0.0-1.0)
    step_count: f32,      // Base raymarching steps (32-512)
    density_scale: f32,   // Opacity multiplier (1.0-50.0)
    jitter_strength: f32, // Anti-banding jitter (0.0-1.0)
}
@group(2) @binding(2) var<uniform> config: MaterialConfig;

// --- Constants ---
const MAX_STEPS: i32 = 512;
const MIN_STEP_SIZE: f32 = 0.0005;
const GRADIENT_DELTA: f32 = 0.005;
const EARLY_TERMINATION_ALPHA: f32 = 0.98;

// =============================================================================
// Ray-Box Intersection (Slab method)
// For unit cube centered at origin: bounds = [-0.5, 0.5]
// Returns vec2(t_near, t_far). If t_near > t_far, ray misses the box.
// =============================================================================
fn ray_box_intersection(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    // Avoid division by zero with small epsilon
    let inv_dir = 1.0 / (ray_dir + vec3<f32>(1e-10));

    let t0 = (-0.5 - ray_origin) * inv_dir;
    let t1 = (0.5 - ray_origin) * inv_dir;

    let t_min = min(t0, t1);
    let t_max = max(t0, t1);

    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);

    return vec2<f32>(t_near, t_far);
}

// =============================================================================
// Coordinate conversion: local cube space [-0.5, 0.5] -> texture UV [0, 1]
// =============================================================================
fn local_to_uv(local_pos: vec3<f32>) -> vec3<f32> {
    return local_pos + 0.5;
}

// =============================================================================
// Sample volume density at UV coordinate
// =============================================================================
fn sample_volume(uv: vec3<f32>) -> f32 {
    return textureSampleLevel(volume_texture, volume_sampler, uv, 0.0).r;
}

// =============================================================================
// Compute gradient (central differences) for lighting
// Returns normalized gradient direction
// =============================================================================
fn compute_gradient(uv: vec3<f32>) -> vec3<f32> {
    let d = GRADIENT_DELTA;

    let gx = sample_volume(uv + vec3<f32>(d, 0.0, 0.0)) -
             sample_volume(uv - vec3<f32>(d, 0.0, 0.0));
    let gy = sample_volume(uv + vec3<f32>(0.0, d, 0.0)) -
             sample_volume(uv - vec3<f32>(0.0, d, 0.0));
    let gz = sample_volume(uv + vec3<f32>(0.0, 0.0, d)) -
             sample_volume(uv - vec3<f32>(0.0, 0.0, d));

    let grad = vec3<f32>(gx, gy, gz);
    let len = length(grad);

    // Return normalized gradient, or zero if too small
    if len < 0.0001 {
        return vec3<f32>(0.0);
    }
    return grad / len;
}

// =============================================================================
// Medical Transfer Function
// Maps CT density values to RGBA colors
// - Air/noise: transparent
// - Soft tissue: semi-transparent reddish
// - Bone: opaque white/gray with gradient shading
// =============================================================================
fn transfer_function(density: f32, threshold: f32, gradient: vec3<f32>, light_dir: vec3<f32>) -> vec4<f32> {
    // Region 1: Below noise floor - fully transparent
    let noise_floor = threshold * 0.4;
    if density < noise_floor {
        return vec4<f32>(0.0);
    }

    // Region 2: Soft tissue [noise_floor, threshold]
    if density < threshold {
        let t = (density - noise_floor) / (threshold - noise_floor);
        let soft_tissue_color = vec3<f32>(0.75, 0.25, 0.25); // Reddish
        let alpha = t * t * 0.08; // Quadratic falloff, very transparent
        return vec4<f32>(soft_tissue_color * t, alpha);
    }

    // Region 3: Bone [threshold, 1.0]
    let bone_t = (density - threshold) / (1.0 - threshold + 0.001);

    // Base bone color: warm white gradient
    let bone_base = mix(
        vec3<f32>(0.88, 0.86, 0.82), // Light bone (slight warm tint)
        vec3<f32>(1.0, 0.99, 0.97),  // Dense bone (near white)
        bone_t
    );

    // Apply gradient-based shading for depth perception
    var shading = 1.0;
    if length(gradient) > 0.1 {
        // Diffuse lighting: dot(normal, light_dir)
        // Normal points in direction of increasing density (into the bone)
        let diffuse = max(dot(-gradient, light_dir), 0.0);
        // Add ambient and diffuse components
        shading = 0.4 + 0.6 * diffuse;
    }

    let bone_color = bone_base * shading;

    // Opacity: higher for denser bone
    let alpha = mix(0.35, 0.97, bone_t * bone_t);

    return vec4<f32>(bone_color, alpha);
}

// =============================================================================
// Pseudo-random hash for jittering (anti-banding)
// =============================================================================
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// =============================================================================
// Transform world-space ray to local cube space
// Properly handles mesh transform (translation, rotation, scale)
// =============================================================================
fn transform_ray_to_local(
    world_origin: vec3<f32>,
    world_dir: vec3<f32>,
    model_matrix: mat4x4<f32>
) -> array<vec3<f32>, 2> {
    // Extract scale from model matrix columns
    let scale = vec3<f32>(
        length(model_matrix[0].xyz),
        length(model_matrix[1].xyz),
        length(model_matrix[2].xyz)
    );

    // Extract rotation matrix (normalized columns)
    let rot_x = model_matrix[0].xyz / scale.x;
    let rot_y = model_matrix[1].xyz / scale.y;
    let rot_z = model_matrix[2].xyz / scale.z;

    // Build inverse rotation (transpose for orthogonal matrix)
    let inv_rot = mat3x3<f32>(
        vec3<f32>(rot_x.x, rot_y.x, rot_z.x),
        vec3<f32>(rot_x.y, rot_y.y, rot_z.y),
        vec3<f32>(rot_x.z, rot_y.z, rot_z.z)
    );

    // Translation
    let translation = model_matrix[3].xyz;

    // Transform origin: translate then rotate then scale
    let local_origin = inv_rot * (world_origin - translation) / scale;

    // Transform direction: rotate then scale (direction doesn't translate)
    let local_dir = normalize(inv_rot * world_dir / scale);

    return array<vec3<f32>, 2>(local_origin, local_dir);
}

// =============================================================================
// Fragment shader - Main raymarching entry point
// =============================================================================
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get world-space fragment position (on cube surface)
    let world_frag_pos = in.world_position.xyz;

    // Get camera position in world space
    let camera_world = view.world_position;

    // Get mesh transform matrix
    let model_matrix = mesh[in.instance_index].world_from_local;

    // Compute world-space ray direction
    let world_ray_dir = normalize(world_frag_pos - camera_world);

    // Transform ray to local cube space (unit cube centered at origin)
    let local_ray = transform_ray_to_local(camera_world, world_ray_dir, model_matrix);
    let local_origin = local_ray[0];
    let local_dir = local_ray[1];

    // Ray-box intersection in local space
    let t_hit = ray_box_intersection(local_origin, local_dir);

    // Check if ray misses the volume entirely
    if t_hit.x > t_hit.y {
        discard;
    }

    // Handle camera inside volume: clamp t_near to 0
    let t_near = max(t_hit.x, 0.0);
    let t_far = t_hit.y;

    // Early exit if volume is behind camera
    if t_far < 0.0 {
        discard;
    }

    // Calculate adaptive step size
    let ray_length = t_far - t_near;
    let base_steps = i32(config.step_count);
    var step_size = ray_length / f32(base_steps);
    step_size = max(step_size, MIN_STEP_SIZE);

    // Jitter ray start to reduce banding artifacts
    let jitter = hash(in.position.xy) * config.jitter_strength;
    var t = t_near + jitter * step_size;

    // Light direction for shading (from camera, slightly above)
    let light_dir = normalize(vec3<f32>(0.2, 0.5, 1.0));

    // Front-to-back compositing accumulators
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    // Previous density for adaptive stepping
    var prev_density = 0.0;

    // Raymarching loop
    for (var i = 0; i < MAX_STEPS; i++) {
        // Stop conditions: past volume or nearly opaque
        if t > t_far || accumulated_alpha > EARLY_TERMINATION_ALPHA {
            break;
        }

        // Stop if we've exceeded configured step count
        if i >= base_steps {
            break;
        }

        // Current sample position in local space
        let sample_pos = local_origin + local_dir * t;

        // Convert to texture UV coordinates
        let uv = local_to_uv(sample_pos);

        // Bounds check (should be [0, 1] but check for numerical precision)
        if any(uv < vec3<f32>(-0.001)) || any(uv > vec3<f32>(1.001)) {
            t += step_size;
            continue;
        }

        // Clamp UV to valid range
        let safe_uv = clamp(uv, vec3<f32>(0.0), vec3<f32>(1.0));

        // Sample volume density
        let density = sample_volume(safe_uv);

        // Compute gradient for shading (only for visible regions)
        var gradient = vec3<f32>(0.0);
        if density > config.threshold * 0.5 {
            gradient = compute_gradient(safe_uv);
        }

        // Apply transfer function
        let sample_rgba = transfer_function(density, config.threshold, gradient, light_dir);

        // Skip fully transparent samples
        if sample_rgba.a > 0.001 {
            // Scale opacity by step size and density scale for correct accumulation
            let scaled_alpha = sample_rgba.a * step_size * config.density_scale;

            // Front-to-back compositing formula
            let weight = (1.0 - accumulated_alpha) * scaled_alpha;
            accumulated_color += sample_rgba.rgb * weight;
            accumulated_alpha += weight;
        }

        // Adaptive stepping: take smaller steps near surfaces
        var current_step = step_size;
        let density_change = abs(density - prev_density);
        if density_change > 0.1 && density > config.threshold * 0.3 {
            // Near a surface transition - use half step for better quality
            current_step = step_size * 0.5;
        }
        prev_density = density;

        t += current_step;
    }

    // If nothing visible was accumulated, discard fragment
    if accumulated_alpha < 0.005 {
        discard;
    }

    // Background color (subtle gradient for depth)
    let bg_color = vec3<f32>(0.08, 0.08, 0.10);
    let final_color = accumulated_color + bg_color * (1.0 - accumulated_alpha);

    return vec4<f32>(final_color, min(accumulated_alpha, 1.0));
}
