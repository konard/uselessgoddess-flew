use {
  bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    pbr::MaterialPipelineKey,
    prelude::*,
    render::{
      mesh::MeshVertexBufferLayoutRef,
      render_resource::{
        AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType,
        SpecializedMeshPipelineError, TextureDimension, TextureFormat,
      },
    },
    shader::ShaderRef,
  },
  dicom::object::open_file,
  std::fs,
};

use bevy_flycam::prelude::*;

const DICOM_FOLDER: &str = "assets/dicom";
const MIN_HU: f32 = -1000.0;
const MAX_HU: f32 = 3000.0;

// --- НАСТРОЙКА РАЗРЕШЕНИЯ ---
// 1 = 512x512, 2 = 256x256, 4 = 128x128
const DOWNSCALE_FACTOR: usize = 1;

fn main() {
  App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(MaterialPlugin::<VolumeMaterial>::default())
    .add_plugins(NoCameraPlayerPlugin)
    .insert_resource(MovementSettings {
      sensitivity: 0.00015,
      speed: 12.0, // Можете уменьшить скорость, если модель 1.0 метр
    })
    .insert_resource(KeyBindings {
      move_ascend: KeyCode::KeyE,
      move_descend: KeyCode::KeyQ,
      ..Default::default()
    })
    .add_systems(Startup, setup)
    // .add_systems(Update, rotate_cube) // Отключите вращение, чтобы летать самим
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
    let get_instance = |path: &std::path::PathBuf| {
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

  // --- ДАУНСКЕЙЛ (уменьшение количества файлов/слоев) ---
  let filtered_files: Vec<_> =
    files.into_iter().step_by(DOWNSCALE_FACTOR).collect();
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
        .and_then(|e| e.to_int().ok())
        .unwrap_or(512);
      height = obj
        .element_by_name("Rows")
        .ok()
        .and_then(|e| e.to_int().ok())
        .unwrap_or(512);

      // Вычисляем новые размеры
      new_width = width / (DOWNSCALE_FACTOR as u32);
      new_height = height / (DOWNSCALE_FACTOR as u32);
      // Резервируем память под финальный массив
      final_voxels.reserve_exact((new_width * new_height * new_depth) as usize);
    }

    let intercept = obj
      .element_by_name("RescaleIntercept")
      .ok()
      .and_then(|e| e.to_str().ok())
      .and_then(|s| s.trim().parse::<f32>().ok())
      .unwrap_or(0.0);
    let slope = obj
      .element_by_name("RescaleSlope")
      .ok()
      .and_then(|e| e.to_str().ok())
      .and_then(|s| s.trim().parse::<f32>().ok())
      .unwrap_or(1.0);

    let pixel_bytes =
      obj.element_by_name("PixelData").unwrap().to_bytes().unwrap();

    // --- ДАУНСКЕЙЛ (пропуск пикселей по X и Y) ---
    for r in 0..new_height {
      for c in 0..new_width {
        // Исходные координаты в файле 512x512
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
  mut materials: ResMut<Assets<VolumeMaterial>>,
  mut images: ResMut<Assets<Image>>,
) {
  let volume = load_dicom_volume(DICOM_FOLDER);

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
  // Use linear filtering for smooth interpolation between voxels
  image.sampler = ImageSampler::linear();
  let image_handle = images.add(image);

  commands.spawn((
    Mesh3d(meshes.add(Cuboid::from_length(1.0))),
    MeshMaterial3d(materials.add(VolumeMaterial {
      volume_texture: image_handle,
      config: MaterialConfig::default(),
    })),
    Transform::from_xyz(0.0, 0.0, 0.0),
  ));

  commands.spawn((
    FlyCam,
    Camera3d::default(),
    Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
  ));

  commands.spawn((
    PointLight { intensity: 1500.0, ..default() },
    Transform::from_xyz(4.0, 8.0, 4.0),
  ));
}

/// Material for volume raymarching rendering of CT/DICOM data.
/// Supports flying inside the volume with proper culling disabled.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct VolumeMaterial {
  #[texture(0, dimension = "3d")]
  #[sampler(1)]
  pub volume_texture: Handle<Image>,

  #[uniform(2)]
  pub config: MaterialConfig,
}

/// Configuration parameters for volume rendering.
/// These can be adjusted in real-time for interactive exploration.
#[derive(Clone, ShaderType, Debug)]
pub struct MaterialConfig {
  /// Density threshold below which voxels are considered transparent.
  /// Higher values show only dense structures (bones).
  /// Range: 0.0 - 1.0, default: 0.25
  pub threshold: f32,

  /// Number of raymarching steps. More steps = better quality, slower.
  /// Range: 32 - 512, default: 128
  pub step_count: f32,

  /// Density scale factor for opacity accumulation.
  /// Higher values = more opaque rendering.
  /// Range: 1.0 - 50.0, default: 15.0
  pub density_scale: f32,

  /// Jitter strength for reducing banding artifacts.
  /// Range: 0.0 - 1.0, default: 0.5
  pub jitter_strength: f32,
}

impl Default for MaterialConfig {
  fn default() -> Self {
    Self {
      threshold: 0.25,
      step_count: 128.0,
      density_scale: 15.0,
      jitter_strength: 0.5,
    }
  }
}

impl Material for VolumeMaterial {
  fn fragment_shader() -> ShaderRef {
    "shaders/volume.wgsl".into()
  }

  fn alpha_mode(&self) -> AlphaMode {
    AlphaMode::Blend
  }

  /// Disable backface culling to allow flying inside the volume.
  /// This is critical for medical visualization where the camera
  /// needs to explore inside anatomical structures.
  fn specialize(
    _pipeline: &bevy::pbr::MaterialPipeline<Self>,
    descriptor: &mut RenderPipelineDescriptor,
    _layout: &MeshVertexBufferLayoutRef,
    _key: MaterialPipelineKey<Self>,
  ) -> Result<(), SpecializedMeshPipelineError> {
    // Disable culling - render both front and back faces
    // This allows the camera to fly inside the volume cube
    descriptor.primitive.cull_mode = None;
    Ok(())
  }
}
