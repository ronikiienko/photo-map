# photo-map

A viewer that lays your photos out as a map: similar photos sit close together, and you search them with words or another photo.
Rust, wgpu and egui. The photos are embedded on your own machine with ONNX Runtime (CUDA and TensorRT when there is an NVIDIA card).

- `src/` the viewer and the map builder (scan, copy, embed, layout, regions, names, bake)
- `tests/` fixtures for the tests
- `vocab/` the label vocabulary used to name regions
- `tools/` Python for preparing models (ONNX export and conversion), benchmarks and dataset fetchers

Not in git, made or downloaded here: `maps/` (your maps), `runtime/` (ONNX Runtime, CUDA, TensorRT), `.cache/` (models, engines, vocabulary vectors).

Build and run: `cargo run --release`. The bundled JPEG decoder (libjpeg-turbo) is compiled from source and needs `cmake` (and `nasm` on x86 machines, not on Apple Silicon): install them with your package manager (`pacman -S cmake nasm`, `apt install cmake nasm`, `brew install cmake nasm`).

To keep them inside this folder instead, put them in `.venv/bin` and `.tools/nasm/usr/bin` and add a local, git-ignored `.cargo/config.toml`:

```
[env]
CMAKE = { value = ".venv/bin/cmake", relative = true }
ASM_NASM = { value = ".tools/nasm/usr/bin/nasm", relative = true }
```

To keep cargo's package cache inside the folder too: `env CARGO_HOME=.cargo-home cargo run --release`.
