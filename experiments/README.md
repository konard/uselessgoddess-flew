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

## Experiment Ideas

1. **Transfer Function Tuning**: Modify `transfer_function()` in `volume.wgsl` to highlight different tissue types
2. **Step Count vs FPS**: Test different `step_count` values to find the performance/quality sweet spot
3. **Adaptive Sampling**: Implement larger steps in empty regions for better performance
4. **Lighting**: Add gradient-based lighting for better depth perception
