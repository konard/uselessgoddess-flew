// Volume raymarching shader for CT/DICOM data
// Supports rendering from both outside and inside the volume cube.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(3) @binding(0) var volume_texture: texture_3d<f32>;
@group(3) @binding(1) var volume_sampler: sampler;
@group(3) @binding(2) var<uniform> config: MaterialConfig;

struct MaterialConfig { threshold: f32 }

const MAX_STEPS: i32 = 400;

// Compute the ray-AABB intersection for an axis-aligned box spanning [-0.5, 0.5]^3.
// Returns (t_enter, t_exit); if t_enter > t_exit the ray misses.
fn ray_box_intersect(ray_origin: vec3<f32>, ray_dir_inv: vec3<f32>) -> vec2<f32> {
    let t0 = (-0.5 - ray_origin) * ray_dir_inv;
    let t1 = ( 0.5 - ray_origin) * ray_dir_inv;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    let t_enter = max(max(t_min.x, t_min.y), t_min.z);
    let t_exit  = min(min(t_max.x, t_max.y), t_max.z);
    return vec2<f32>(t_enter, t_exit);
}

// Central-difference gradient for surface normal estimation.
fn get_normal(uvw: vec3<f32>) -> vec3<f32> {
    let s = 1.0 / vec3<f32>(textureDimensions(volume_texture));
    let dx = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(s.x, 0.0, 0.0), 0.0).r
           - textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(s.x, 0.0, 0.0), 0.0).r;
    let dy = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(0.0, s.y, 0.0), 0.0).r
           - textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(0.0, s.y, 0.0), 0.0).r;
    let dz = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(0.0, 0.0, s.z), 0.0).r
           - textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(0.0, 0.0, s.z), 0.0).r;
    // Gradient points toward increasing density; negate so the normal faces outward.
    return -normalize(vec3<f32>(dx, dy, dz));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let camera_pos = view.world_position.xyz;
    let ray_dir = normalize(in.world_position.xyz - camera_pos);

    // Express ray in local cube space (cube spans [-0.5, 0.5]).
    // in.world_position is already on the cube surface in world space, but the
    // cube transform may include a rotation, so we use the mesh position directly.
    let local_origin = in.world_position.xyz;

    // Guard against division by zero in ray_box_intersect.
    let safe_dir = select(ray_dir, sign(ray_dir) * 1e-6, abs(ray_dir) < vec3(1e-6));
    let ray_dir_inv = 1.0 / safe_dir;

    let t_range = ray_box_intersect(local_origin, ray_dir_inv);

    // If camera is outside and ray misses the box, discard.
    if t_range.x > t_range.y {
        discard;
    }

    // When inside the box, t_enter may be negative — clamp to 0.
    let t_start = max(t_range.x, 0.0);
    let t_end   = t_range.y;

    let dim = vec3<f32>(textureDimensions(volume_texture));
    // Step size slightly smaller than one voxel diagonal to avoid missing surfaces.
    let step_size = 1.0 / length(dim) * 0.8;

    let color_bone = vec3<f32>(0.95, 0.90, 0.80);
    let light_dir  = normalize(vec3<f32>(1.0, 2.0, 1.0));

    var t = t_start;
    var step = 0;
    loop {
        if t >= t_end || step >= MAX_STEPS { break; }

        let current_pos = local_origin + ray_dir * t;
        // Map from cube space [-0.5, 0.5] to texture UV [0, 1].
        let uvw = current_pos + 0.5;

        let density = textureSample(volume_texture, volume_sampler, uvw).r;

        if density > config.threshold {
            let normal  = get_normal(uvw);
            let diffuse = max(dot(normal, light_dir), 0.0);
            // Blinn-Phong specular highlight.
            let view_dir  = -ray_dir;
            let half_vec  = normalize(light_dir + view_dir);
            let specular  = pow(max(dot(normal, half_vec), 0.0), 32.0) * 0.4;
            let ambient   = 0.2;

            // Depth-based ambient occlusion approximation.
            let depth_ao = 1.0 - (f32(step) / f32(MAX_STEPS));

            let final_color = color_bone * (diffuse * 0.8 + ambient) * depth_ao + specular;
            return vec4<f32>(final_color, 1.0);
        }

        t += step_size;
        step += 1;
    }

    discard;
}
