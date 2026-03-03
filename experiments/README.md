# Volume Rendering Experiments

This directory contains experimental scripts and tools for testing and validating the volume raymarching renderer.

## Testing the Renderer

### Prerequisites
- DICOM data folder in `assets/dicom/` with `.dcm` files
- Rust toolchain (stable)
- System libraries: wayland-client, libudev, libasound (Linux)

### Running the Application

```bash
# Build and run with default settings
cargo run

# Build in release mode for better performance
cargo run --release
```

### Controls

- **WASD** - Move forward/left/backward/right
- **E/Q** - Move up/down
- **Mouse** - Look around
- **ESC** - Unlock cursor

### Adjusting Parameters

Edit `MaterialConfig::default()` in `src/main.rs` to change rendering parameters:

```rust
MaterialConfig {
    threshold: 0.25,       // Bone visibility threshold (0.0-1.0)
    step_count: 128.0,     // Raymarching quality (32-512)
    density_scale: 15.0,   // Opacity multiplier (1.0-50.0)
    jitter_strength: 0.5,  // Anti-banding (0.0-1.0)
}
```

## Implemented SOTA Features

The current shader (`assets/shaders/volume.wgsl`) implements:

1. **Gradient-based shading**: Central difference gradients for depth perception and surface lighting
2. **Adaptive step refinement**: Smaller steps near density transitions for better surface quality
3. **Early ray termination**: Exit when accumulated alpha > 0.98 for performance
4. **Proper coordinate transforms**: Handles mesh translation, rotation, and scale correctly
5. **Camera inside volume**: Works correctly when flying inside anatomical structures
6. **Anti-banding jitter**: Pseudo-random ray offset to eliminate stepping artifacts

## Experiment Ideas

1. **Transfer Function Tuning**: Modify `transfer_function()` in `volume.wgsl` to highlight different tissue types (e.g., blood vessels, tumors)
2. **Step Count vs FPS**: Test different `step_count` values (64, 128, 256, 512) to find the performance/quality sweet spot for your GPU
3. **Empty Space Skipping**: Add an occupancy grid prepass to skip large empty regions
4. **Advanced Lighting**: Implement ambient occlusion or global illumination sampling
5. **Fluid Integration**: Add particle rendering layer on top of volume for fluid simulation
