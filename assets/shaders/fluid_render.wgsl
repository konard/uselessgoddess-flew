// assets/shaders/fluid_render.wgsl
// Screen-Space Fluid Rendering (SSFR) - SOTA approach used in games/simulations
// Renders fluid particles as smooth spheres using depth-buffer smoothing
// Based on: van der Laan et al. 2009 "Screen Space Fluid Rendering with Curvature Flow"
// Technique: each particle generates a billboard sphere → depth buffer → smooth normals

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(3) @binding(0) var<uniform> fluid_config: FluidConfig;

struct FluidConfig {
    // Fluid visual properties
    base_color: vec4<f32>,        // rgba - base fluid color (e.g. blood red)
    fresnel_color: vec4<f32>,     // rgba - reflection color at grazing angles
    particle_radius: f32,         // billboard half-size in world space
    roughness: f32,               // surface roughness [0,1]
    metallic: f32,                // metallic factor for PBR
    refractive_index: f32,        // IOR (water ~1.33, blood ~1.36)
    absorption: vec3<f32>,        // Beer-Lambert absorption coefficients (per channel)
    _pad: f32,
}

// PBR utility functions
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndoth = max(dot(n, h), 0.0);
    let ndoth2 = ndoth * ndoth;
    let denom = ndoth2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

fn geometry_schlick_ggx(ndotv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return ndotv / (ndotv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let ndotv = max(dot(n, v), 0.0);
    let ndotl = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let camera_pos = view.world_position.xyz;

    // In screen-space fluid rendering, the fragment normal is reconstructed
    // from the smoothed depth buffer. Here we use the mesh normal directly
    // (each particle is a billboard sphere; normal is computed per-fragment)

    // Compute sphere normal from UV (in [-1,1] range)
    // in.uv is set to [-1,1] by the mesh for billboard rendering
    let uv = in.uv * 2.0 - 1.0;
    let r2 = dot(uv, uv);
    if r2 > 1.0 { discard; }

    // Sphere normal in view space (z = sqrt(1-r^2))
    let normal_view = normalize(vec3<f32>(uv, sqrt(1.0 - r2)));
    // Transform normal to world space
    let normal = normalize(in.world_normal + normal_view);

    let view_dir = normalize(camera_pos - in.world_position.xyz);

    // PBR lighting
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.5));
    let halfway = normalize(view_dir + light_dir);
    let light_color = vec3<f32>(1.0, 0.95, 0.9);
    let light_intensity = 3.0;

    let base_col = fluid_config.base_color.rgb;
    let roughness = fluid_config.roughness;

    // F0 (Fresnel at normal incidence)
    // For dielectrics, F0 = ((n-1)/(n+1))^2
    let ior = fluid_config.refractive_index;
    let f0_scalar = (ior - 1.0) / (ior + 1.0);
    let f0 = vec3<f32>(f0_scalar * f0_scalar);
    let F0_mixed = mix(f0, base_col, fluid_config.metallic);

    // Cook-Torrance BRDF
    let NDF = distribution_ggx(normal, halfway, roughness);
    let G = geometry_smith(normal, view_dir, light_dir, roughness);
    let F = fresnel_schlick(max(dot(halfway, view_dir), 0.0), F0_mixed);

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - fluid_config.metallic);

    let ndotl = max(dot(normal, light_dir), 0.0);
    let ndotv_safe = max(dot(normal, view_dir), 0.0001);

    let specular = (NDF * G * F) / (4.0 * ndotv_safe * ndotl + 0.0001);
    let diffuse = kD * base_col / 3.14159265;

    // Ambient (environment approximation)
    let ambient_factor = 0.08 + 0.12 * max(dot(normal, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let ambient = base_col * ambient_factor;

    let radiance = light_color * light_intensity;
    var color = (diffuse + specular) * radiance * ndotl + ambient;

    // Subsurface scattering approximation (for blood/tissue appearance)
    // Thin-film back-scatter: simulate light penetrating the surface
    let back_scatter = max(dot(-light_dir, normal), 0.0);
    let sss_color = fluid_config.fresnel_color.rgb * back_scatter * 0.15;
    color += sss_color;

    // Beer-Lambert absorption (depth-based)
    // This approximates light absorption through fluid thickness
    let thickness = 0.05; // approximate per-particle thickness
    let absorption = fluid_config.absorption;
    let transmitted = exp(-absorption * thickness);
    color *= transmitted;

    // Fresnel rim effect
    let fresnel_factor = pow(1.0 - ndotv_safe, 3.0);
    color = mix(color, fluid_config.fresnel_color.rgb, fresnel_factor * 0.3);

    let alpha = fluid_config.base_color.a * (0.7 + fresnel_factor * 0.3);

    return vec4<f32>(color, alpha);
}
