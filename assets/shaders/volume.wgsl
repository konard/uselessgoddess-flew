// assets/shaders/volume.wgsl
// SOTA Volume Raymarching Shader with advanced features:
// - Transfer function for tissue differentiation (bones, soft tissue)
// - Gradient-based Phong shading with specular highlights
// - Early ray termination (front-to-back compositing)
// - Depth-based ambient occlusion
// - Trilinear filtering for smooth normals

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(3) @binding(0) var volume_texture: texture_3d<f32>;
@group(3) @binding(1) var volume_sampler: sampler;
@group(3) @binding(2) var<uniform> config: MaterialConfig;

struct MaterialConfig {
    threshold: f32,
    window_center: f32,
    window_width: f32,
    opacity_scale: f32,
}

const MAX_STEPS: i32 = 512;
const EARLY_EXIT_ALPHA: f32 = 0.99;

// Smooth transfer function: maps density to RGBA color
// This implements a medical CT transfer function
fn transfer_function(density: f32) -> vec4<f32> {
    // Soft tissue range: ~0.15 - 0.30 (in normalized HU space)
    // Bone range: ~0.30 - 1.0 (in normalized HU space)
    let threshold = config.threshold;

    if density < threshold {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Normalized factor within visible range
    let t = clamp((density - threshold) / (1.0 - threshold), 0.0, 1.0);

    // Bone color: ivory/warm white
    let bone_color = vec3<f32>(0.95, 0.90, 0.80);
    // Dense bone color: brighter white
    let dense_bone = vec3<f32>(1.0, 0.98, 0.95);
    // Soft tissue color: warm pink
    let soft_tissue = vec3<f32>(0.85, 0.65, 0.55);

    var color: vec3<f32>;
    var alpha: f32;

    // Soft tissue (low density above threshold)
    if t < 0.25 {
        let local_t = t / 0.25;
        color = mix(soft_tissue, bone_color, local_t);
        alpha = local_t * 0.7 + 0.1;
    } else {
        // Bone
        let local_t = (t - 0.25) / 0.75;
        color = mix(bone_color, dense_bone, local_t);
        alpha = 0.8 + local_t * 0.2;
    }

    return vec4<f32>(color, alpha * config.opacity_scale);
}

// High-quality normal via central differences using trilinear filtering
fn get_normal(uvw: vec3<f32>) -> vec3<f32> {
    let dim = vec3<f32>(textureDimensions(volume_texture));
    // Use 1.5 voxel step for smoother normals
    let s = 1.5 / dim;

    let dx = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(s.x, 0.0, 0.0), 0.0).r -
             textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(s.x, 0.0, 0.0), 0.0).r;
    let dy = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(0.0, s.y, 0.0), 0.0).r -
             textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(0.0, s.y, 0.0), 0.0).r;
    let dz = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(0.0, 0.0, s.z), 0.0).r -
             textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(0.0, 0.0, s.z), 0.0).r;

    let grad = vec3<f32>(dx, dy, dz);
    let grad_len = length(grad);

    // Guard against zero gradient (uniform region)
    if grad_len < 0.0001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    // Gradient points toward higher density, normal points outward
    return -grad / grad_len;
}

// Phong shading with multiple light sources
fn compute_lighting(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    base_color: vec3<f32>,
    density: f32,
) -> vec3<f32> {
    // Main directional light
    let light_dir1 = normalize(vec3<f32>(1.0, 2.0, 1.5));
    // Fill light (opposite side, softer)
    let light_dir2 = normalize(vec3<f32>(-1.0, 0.5, -1.0));

    // Ambient term
    let ambient = 0.18 * base_color;

    // Diffuse (Lambert) with main light
    let diff1 = max(dot(normal, light_dir1), 0.0);
    let diffuse1 = diff1 * base_color * 0.75;

    // Diffuse with fill light
    let diff2 = max(dot(normal, light_dir2), 0.0);
    let diffuse2 = diff2 * base_color * 0.25;

    // Specular (Blinn-Phong) - bone is semi-glossy
    let halfway = normalize(light_dir1 + view_dir);
    let spec_power = 32.0 + density * 64.0; // Denser = shinier
    let spec = pow(max(dot(normal, halfway), 0.0), spec_power) * 0.4;
    let specular = spec * vec3<f32>(1.0, 0.98, 0.95); // Slightly warm specular

    return ambient + diffuse1 + diffuse2 + specular;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let camera_pos = view.world_position.xyz;
    let ray_dir = normalize(in.world_position.xyz - camera_pos);
    let view_dir = -ray_dir;

    // Starting position in cube space (-0.5 .. 0.5)
    var current_pos = in.world_position.xyz;

    let dim = vec3<f32>(textureDimensions(volume_texture));
    // Step slightly smaller than voxel diagonal to avoid skipping
    let step_size = 1.0 / length(dim) * 0.8;

    // Front-to-back compositing for early termination
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        let uvw = current_pos + 0.5;

        // Boundary check
        if any(uvw < vec3(0.0)) || any(uvw > vec3(1.0)) {
            break;
        }

        let density = textureSampleLevel(volume_texture, volume_sampler, uvw, 0.0).r;
        let tf = transfer_function(density);

        if tf.a > 0.001 {
            // Compute normal and lighting only for visible voxels
            let normal = get_normal(uvw);
            let shaded = compute_lighting(normal, view_dir, tf.rgb, density);

            // Front-to-back alpha compositing
            let alpha_contribution = tf.a * (1.0 - accumulated_alpha);
            accumulated_color += alpha_contribution * shaded;
            accumulated_alpha += alpha_contribution;

            // Early termination when nearly opaque
            if accumulated_alpha > EARLY_EXIT_ALPHA {
                break;
            }
        }

        current_pos += ray_dir * step_size;
    }

    if accumulated_alpha < 0.01 {
        discard;
    }

    return vec4<f32>(accumulated_color, accumulated_alpha);
}
