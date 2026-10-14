# photo-map

A viewer that lays your photos out as a map: similar photos sit close together, and you search them with words or another photo.
Rust, wgpu and egui. The photos are embedded on your own machine with ONNX Runtime (CUDA and TensorRT when there is an NVIDIA card).

- `src/` the viewer and the map builder (scan, copy, embed, layout, regions, names, bake)
- `tests/` fixtures for the tests
- `vocab/` the label vocabulary used to name regions
- `tools/` Python for preparing models (ONNX export and conversion), benchmarks and dataset fetchers

Not in git, made or downloaded here: `maps/` (your maps), `runtime/` (ONNX Runtime, CUDA, TensorRT), `.cache/` (models, engines, vocabulary vectors).

Build and run: `./cargo.sh run --release`. It is cargo with the package cache (`.cargo-home/`) and the build tools kept inside this folder. The bundled JPEG decoder (libjpeg-turbo) needs `cmake` and `nasm` to compile; the script finds them in `.venv/bin` and `.tools/nasm/usr/bin`, or use your own if they are installed system-wide.
