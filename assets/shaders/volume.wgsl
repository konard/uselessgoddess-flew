// assets/shaders/volume.wgsl

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(3) @binding(0) var volume_texture: texture_3d<f32>;
@group(3) @binding(1) var volume_sampler: sampler;
@group(3) @binding(2) var<uniform> config: MaterialConfig; 

struct MaterialConfig { threshold: f32 }

const MAX_STEPS: i32 = 400;

// Функция для расчета нормали (остается без изменений)
// С NEAREST сэмплером она будет возвращать "кубические" нормали
fn get_normal(uvw: vec3<f32>) -> vec3<f32> {
    // Получаем размеры текстуры
    let dim = vec3<f32>(textureDimensions(volume_texture));
    // Шаг градиента равен размеру одного вокселя
    let s = 1.0 / dim;

    let dx = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(s.x,0.0,0.0), 0.0).r - 
             textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(s.x,0.0,0.0), 0.0).r;
    let dy = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(0.0,s.y,0.0), 0.0).r - 
             textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(0.0,s.y,0.0), 0.0).r;
    let dz = textureSampleLevel(volume_texture, volume_sampler, uvw + vec3(0.0,0.0,s.z), 0.0).r - 
             textureSampleLevel(volume_texture, volume_sampler, uvw - vec3(0.0,0.0,s.z), 0.0).r;
             
    // Используем -normalize, т.к. градиент указывает в сторону увеличения плотности (внутрь),
    // а нормаль должна указывать наружу.
    return -normalize(vec3<f32>(dx, dy, dz));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let camera_pos = view.world_position.xyz;
    let ray_dir = normalize(in.world_position.xyz - camera_pos);

    // Начальная позиция внутри куба (диапазон -0.5 .. 0.5)
    var current_pos = in.world_position.xyz;

    // Определяем размер шага, чтобы он был чуть меньше диагонали вокселя,
    // это гарантирует, что мы не пропустим воксель.
    let dim = vec3<f32>(textureDimensions(volume_texture));
    let step_size = 1.0 / length(dim) * 0.9; // 0.9 - коеффициент безопасности

    // Цвет и свет
    let color_bone = vec3<f32>(0.95, 0.90, 0.80);
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.0));

    for (var i = 0; i < MAX_STEPS; i++) {
        // Конвертируем мировую позицию (-0.5 .. 0.5) в UV (0.0 .. 1.0)
        let uvw = current_pos + 0.5;

        // Оптимизированная проверка выхода за границы
        if (any(uvw < vec3(0.0)) || any(uvw > vec3(1.0))) {
            break; // Если вышли из куба, прекращаем цикл
        }

        let density = textureSample(volume_texture, volume_sampler, uvw).r;

        if (density > config.threshold) {
            let normal = get_normal(uvw);
            let diffuse = max(dot(normal, light_dir), 0.0);
            let ambient = 0.2; // Немного увеличим фоновый свет
            
            let depth_ao = 1.0 - (f32(i) / f32(MAX_STEPS));
            
            let final_color = color_bone * (diffuse * 0.8 + ambient) * depth_ao;
            return vec4<f32>(final_color, 1.0);
        }

        current_pos += ray_dir * step_size;
    }

    discard;
}