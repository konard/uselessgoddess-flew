use {
    bevy::{
        asset::RenderAssetUsages,
        image::ImageSampler,
        prelude::*,
        render::render_resource::{
            AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
        },
        shader::ShaderRef,
    },
    dicom::object::open_file,
    std::fs,
};

use bevy_flycam::prelude::*;

mod dicom_scan;
mod pbf;

const DICOM_FOLDER: &str = "assets/dicom";
const MIN_HU: f32 = -1000.0;
const MAX_HU: f32 = 3000.0;

// --- RESOLUTION SETTINGS ---
// 1 = full (512x512), 2 = half (256x256), 4 = quarter (128x128)
const DOWNSCALE_FACTOR: usize = 1;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MaterialPlugin::<VolumeMaterial>::default())
        .add_plugins(MaterialPlugin::<FluidMaterial>::default())
        .add_plugins(NoCameraPlayerPlugin)
        .add_plugins(pbf::PbfPlugin)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 12.0,
        })
        .insert_resource(KeyBindings {
            move_ascend: KeyCode::KeyE,
            move_descend: KeyCode::KeyQ,
            ..Default::default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, update_ui)
        .run();
}

struct VolumeInfo {
    data: Vec<u8>,
    width: u32,
    height: u32,
    depth: u32,
}

fn load_dicom_volume(folder_path: &str) -> VolumeInfo {
    println!("Reading DICOM folder: {}", folder_path);

    let paths = fs::read_dir(folder_path).expect("Folder not found!");
    let mut files: Vec<_> = paths
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "dcm"))
        .collect();

    files.sort_by(|a, b| {
        let get_instance = |path: &std::path::PathBuf| -> Option<i32> {
            open_file(path)
                .ok()
                .and_then(|obj| obj.element_by_name("InstanceNumber").cloned().ok())
                .and_then(|e| e.to_int::<i32>().ok())
        };
        match (get_instance(a), get_instance(b)) {
            (Some(va), Some(vb)) => va.cmp(&vb),
            _ => a.file_name().cmp(&b.file_name()),
        }
    });

    let filtered_files: Vec<_> = files.into_iter().step_by(DOWNSCALE_FACTOR).collect();
    let new_depth = filtered_files.len() as u32;

    let mut width = 0;
    let mut height = 0;
    let mut new_width = 0;
    let mut new_height = 0;

    let mut final_voxels: Vec<f32> = vec![];

    for (i, path) in filtered_files.iter().enumerate() {
        if i % 10 == 0 {
            println!("Loading downscaled slice {}/{}", i, new_depth);
        }

        let obj = open_file(path).unwrap();

        if width == 0 {
            width = obj
                .element_by_name("Columns")
                .ok()
                .and_then(|e| e.to_int::<u32>().ok())
                .unwrap_or(512);
            height = obj
                .element_by_name("Rows")
                .ok()
                .and_then(|e| e.to_int::<u32>().ok())
                .unwrap_or(512);

            new_width = width / (DOWNSCALE_FACTOR as u32);
            new_height = height / (DOWNSCALE_FACTOR as u32);
            final_voxels.reserve_exact((new_width * new_height * new_depth) as usize);
        }

        let intercept: f32 = obj
            .element_by_name("RescaleIntercept")
            .ok()
            .and_then(|e| e.to_str().ok())
            .and_then(|s: std::borrow::Cow<str>| s.trim().parse::<f32>().ok())
            .unwrap_or(0.0);
        let slope: f32 = obj
            .element_by_name("RescaleSlope")
            .ok()
            .and_then(|e| e.to_str().ok())
            .and_then(|s: std::borrow::Cow<str>| s.trim().parse::<f32>().ok())
            .unwrap_or(1.0);

        let pixel_bytes = obj.element_by_name("PixelData").unwrap().to_bytes().unwrap();

        for r in 0..new_height {
            for c in 0..new_width {
                let orig_y = r * (DOWNSCALE_FACTOR as u32);
                let orig_x = c * (DOWNSCALE_FACTOR as u32);

                let idx = (orig_y * width + orig_x) as usize;
                let chunk_start = idx * 2;

                let raw_val_u16 = u16::from_le_bytes([
                    pixel_bytes[chunk_start],
                    pixel_bytes[chunk_start + 1],
                ]);
                let hu = (raw_val_u16 as f32) * slope + intercept;

                let norm = ((hu - MIN_HU) / (MAX_HU - MIN_HU)).clamp(0.0, 1.0);
                final_voxels.push(norm);
            }
        }
    }

    println!(
        "Volume Loaded! New Scaled Size: {}x{}x{}",
        new_width, new_height, new_depth
    );

    VolumeInfo {
        data: bytemuck::cast_slice(&final_voxels).to_vec(),
        width: new_width,
        height: new_height,
        depth: new_depth,
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut volume_materials: ResMut<Assets<VolumeMaterial>>,
    mut fluid_materials: ResMut<Assets<FluidMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut pbf_state: ResMut<pbf::PbfState>,
) {
    let volume = load_dicom_volume(DICOM_FOLDER);

    // Upload volume as linear 3D texture (R32Float)
    let mut image = Image::new(
        Extent3d {
            width: volume.width,
            height: volume.height,
            depth_or_array_layers: volume.depth,
        },
        TextureDimension::D3,
        volume.data,
        TextureFormat::R32Float,
        RenderAssetUsages::all(),
    );
    // Use linear (trilinear) sampling for smooth rendering
    image.sampler = ImageSampler::linear();
    let image_handle = images.add(image);

    // --- Volume (CT) rendering ---
    // Fix orientation: CT head scans typically have Z = superior-inferior (depth slices),
    // so we rotate the cuboid 90° around X to make the head face forward (+Z direction)
    // instead of downward (-Y direction).
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_length(1.0))),
        MeshMaterial3d(volume_materials.add(VolumeMaterial {
            volume_texture: image_handle,
            config: MaterialConfig {
                threshold: 0.25,
                window_center: 0.4,
                window_width: 0.8,
                opacity_scale: 1.0,
            },
        })),
        // Rotate 90° around X axis: head faces forward (horizontal), not downward
        Transform::from_xyz(0.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // --- Camera ---
    // Position slightly in front of the volume, looking at origin
    commands.spawn((
        FlyCam,
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.3, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // --- Lighting ---
    // Key light (warm, upper-right)
    commands.spawn((
        PointLight {
            intensity: 8000.0,
            color: Color::srgb(1.0, 0.95, 0.9),
            shadows_enabled: false,
            range: 20.0,
            ..default()
        },
        Transform::from_xyz(3.0, 6.0, 3.0),
    ));
    // Fill light (cool, left)
    commands.spawn((
        PointLight {
            intensity: 3000.0,
            color: Color::srgb(0.7, 0.8, 1.0),
            shadows_enabled: false,
            range: 20.0,
            ..default()
        },
        Transform::from_xyz(-4.0, 2.0, -2.0),
    ));

    // --- PBF Fluid Particles ---
    // Initialize simulation with particles distributed inside the volume
    pbf_state.initialize(1024); // start with 1k particles (can scale up)

    // Spawn particle mesh (instanced billboard spheres)
    let particle_mesh = meshes.add(Sphere::new(pbf_state.particle_radius).mesh().uv(8, 6));
    let fluid_mat = fluid_materials.add(FluidMaterial {
        config: FluidConfig {
            base_color: Vec4::new(0.6, 0.05, 0.05, 0.85), // blood red
            fresnel_color: Vec4::new(1.0, 0.4, 0.4, 1.0),  // bright red rim
            particle_radius: pbf_state.particle_radius,
            roughness: 0.15,
            metallic: 0.0,
            refractive_index: 1.36, // blood IOR
            absorption: Vec3::new(0.8, 2.5, 2.5), // red absorbs green/blue
            _pad: 0.0,
        },
    });

    // Spawn individual particle entities (for now; later use GPU instancing)
    for i in 0..pbf_state.particles.len().min(512) {
        let pos = pbf_state.particles[i].position;
        commands.spawn((
            Mesh3d(particle_mesh.clone()),
            MeshMaterial3d(fluid_mat.clone()),
            Transform::from_translation(pos),
            pbf::ParticleEntity { index: i as u32 },
        ));
    }

    // --- UI text overlay ---
    commands.spawn((
        Text::new("Flew - Medical Fluid Simulation\nWASD + Mouse: Fly | E/Q: Up/Down\nSpace: Reset fluid"),
        TextFont { font_size: 16.0, ..default() },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        SimUiText,
    ));
}

#[derive(Component)]
struct SimUiText;

fn update_ui(
    pbf_state: Res<pbf::PbfState>,
    mut texts: Query<&mut Text, With<SimUiText>>,
) {
    for mut text in texts.iter_mut() {
        *text = Text::new(format!(
            "Flew - Medical Fluid Simulation\n\
             WASD + Mouse: Fly | E/Q: Up/Down\n\
             Particles: {} | Viscosity: {:.2} | Pressure: {:.2}",
            pbf_state.particles.len(),
            pbf_state.viscosity,
            pbf_state.pressure_scale,
        ));
    }
}

// ---- Volume Material ----
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct VolumeMaterial {
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub volume_texture: Handle<Image>,

    #[uniform(2)]
    pub config: MaterialConfig,
}

#[derive(Clone, Default, ShaderType, Debug)]
pub struct MaterialConfig {
    pub threshold: f32,
    pub window_center: f32,
    pub window_width: f32,
    pub opacity_scale: f32,
}

impl Material for VolumeMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/volume.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

// ---- Fluid Material ----
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct FluidMaterial {
    #[uniform(0)]
    pub config: FluidConfig,
}

#[derive(Clone, Default, ShaderType, Debug)]
pub struct FluidConfig {
    pub base_color: Vec4,
    pub fresnel_color: Vec4,
    pub particle_radius: f32,
    pub roughness: f32,
    pub metallic: f32,
    pub refractive_index: f32,
    pub absorption: Vec3,
    pub _pad: f32,
}

impl Material for FluidMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/fluid_render.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}
