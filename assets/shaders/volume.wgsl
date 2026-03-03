// Volume Raymarching Shader for Medical Visualization
// SOTA implementation with proper ray-box intersection, early termination,
// and support for camera flying inside the volume

// Import Bevy's standard bindings
#import bevy_pbr::{
    mesh_view_bindings::view,
    mesh_bindings::mesh,
    forward_io::VertexOutput,
}

// Material bindings
@group(2) @binding(0) var volume_texture: texture_3d<f32>;
@group(2) @binding(1) var volume_sampler: sampler;

struct MaterialConfig {
    threshold: f32,          // Density threshold for bone visualization
    step_count: f32,         // Number of raymarching steps
    density_scale: f32,      // Density multiplier for transparency
    jitter_strength: f32,    // Jitter to reduce banding artifacts
}
@group(2) @binding(2) var<uniform> config: MaterialConfig;

// Constants
const MAX_STEPS: i32 = 256;
const MIN_STEP_SIZE: f32 = 0.001;

// Ray-box intersection for unit cube centered at origin [-0.5, 0.5]
// Returns (t_near, t_far) - intersection distances along ray
// t_near > t_far means no intersection
fn ray_box_intersection(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;

    let t0 = (-0.5 - ray_origin) * inv_dir;
    let t1 = (0.5 - ray_origin) * inv_dir;

    let t_min = min(t0, t1);
    let t_max = max(t0, t1);

    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);

    return vec2<f32>(t_near, t_far);
}

// Convert world position to texture UV coordinates [0, 1]
fn world_to_uv(world_pos: vec3<f32>) -> vec3<f32> {
    return world_pos + 0.5; // Map [-0.5, 0.5] to [0, 1]
}

// Sample volume density at given UV coordinate
fn sample_volume(uv: vec3<f32>) -> f32 {
    return textureSampleLevel(volume_texture, volume_sampler, uv, 0.0).r;
}

// Transfer function: maps density to color and opacity
// For medical CT data: bones are bright (high density), soft tissue is darker
fn transfer_function(density: f32, threshold: f32) -> vec4<f32> {
    // Skip below threshold (mostly air/noise)
    if density < threshold * 0.5 {
        return vec4<f32>(0.0);
    }

    // Soft tissue region [threshold*0.5, threshold]
    if density < threshold {
        let t = (density - threshold * 0.5) / (threshold * 0.5);
        let color = vec3<f32>(0.8, 0.3, 0.3) * t; // Reddish for soft tissue
        let alpha = t * 0.1; // Very transparent
        return vec4<f32>(color, alpha);
    }

    // Bone region [threshold, 1.0]
    let bone_t = (density - threshold) / (1.0 - threshold);

    // Color gradient: light gray to white for bones
    let bone_color = mix(
        vec3<f32>(0.85, 0.85, 0.82),  // Light bone
        vec3<f32>(1.0, 1.0, 0.98),     // Dense bone (white)
        bone_t
    );

    // Opacity increases with density
    let alpha = mix(0.3, 0.95, bone_t);

    return vec4<f32>(bone_color, alpha);
}

// Pseudo-random for jittering (to reduce banding)
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get world-space position of fragment on the cube surface
    let world_pos = in.world_position.xyz;

    // Get mesh transform (inverse needed to convert to local space)
    let model_matrix = mesh[in.instance_index].world_from_local;
    let inv_model = transpose(mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    ));

    // Camera position in world space
    let camera_world = view.world_position;

    // Transform to local/object space (unit cube centered at origin)
    // We need to account for the mesh transform
    let local_pos = (vec4<f32>(world_pos, 1.0) * transpose(model_matrix)).xyz;
    let camera_local = (vec4<f32>(camera_world, 1.0) * transpose(model_matrix)).xyz;

    // Actually, let's simplify: assume unit cube at origin with identity transform
    // The world_pos IS already the local position for a unit cube centered at origin
    let ray_origin = camera_world;
    let ray_dir = normalize(world_pos - camera_world);

    // Transform ray to local space of unit cube
    // For a cube at origin with scale 1, local = world (after mesh transform)
    // We need to use the mesh's world_from_local matrix
    let scale = vec3<f32>(
        length(model_matrix[0].xyz),
        length(model_matrix[1].xyz),
        length(model_matrix[2].xyz)
    );
    let translation = model_matrix[3].xyz;

    // Transform ray to local cube space
    let local_ray_origin = (ray_origin - translation) / scale;
    let local_ray_dir = normalize(ray_dir / scale);

    // Ray-box intersection
    let t_hit = ray_box_intersection(local_ray_origin, local_ray_dir);

    // Check if ray misses the volume
    if t_hit.x > t_hit.y {
        discard;
    }

    // Clamp near plane to 0 if camera is inside the volume
    let t_near = max(t_hit.x, 0.0);
    let t_far = t_hit.y;

    // Early exit if behind camera
    if t_far < 0.0 {
        discard;
    }

    // Calculate step size based on config
    let step_count = i32(config.step_count);
    let ray_length = t_far - t_near;
    var step_size = ray_length / f32(step_count);
    step_size = max(step_size, MIN_STEP_SIZE);

    // Jitter starting position to reduce banding artifacts
    let jitter = hash(in.position.xy) * config.jitter_strength;
    var t = t_near + jitter * step_size;

    // Front-to-back compositing
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    // Raymarching loop
    for (var i = 0; i < MAX_STEPS && i < step_count; i++) {
        if t > t_far || accumulated_alpha > 0.95 {
            break; // Early ray termination
        }

        // Current sample position in local space
        let sample_pos = local_ray_origin + local_ray_dir * t;

        // Convert to UV coordinates [0, 1]
        let uv = world_to_uv(sample_pos);

        // Check bounds (should be inside [0, 1])
        if any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0)) {
            t += step_size;
            continue;
        }

        // Sample volume density
        let density = sample_volume(uv);

        // Apply transfer function
        let sample_color = transfer_function(density, config.threshold);

        // Scale opacity by step size and density scale
        let scaled_alpha = sample_color.a * step_size * config.density_scale;

        // Front-to-back compositing
        let weight = (1.0 - accumulated_alpha) * scaled_alpha;
        accumulated_color += sample_color.rgb * weight;
        accumulated_alpha += weight;

        t += step_size;
    }

    // Apply some ambient/background lighting
    let ambient = vec3<f32>(0.1, 0.1, 0.12);
    let final_color = accumulated_color + ambient * (1.0 - accumulated_alpha);

    // If nothing was accumulated, discard the fragment
    if accumulated_alpha < 0.01 {
        discard;
    }

    return vec4<f32>(final_color, accumulated_alpha);
}
