# photo-map

A viewer that lays your photos out as a map: similar photos sit close together, and you search them with words or another photo.
Rust, wgpu and egui. The photos are embedded on your own machine with ONNX Runtime (CUDA and TensorRT when there is an NVIDIA card).

- `src/` the viewer and the map builder (scan, copy, embed, layout, regions, names, bake)
- `tests/` fixtures for the tests
- `vocab/` the label vocabulary used to name regions (built into the program)
- `assets/` the icon and `places.zip`, the towns that name where a photo was taken (built into the program)
- `tools/` Python for preparing models (ONNX export and conversion), benchmarks and dataset fetchers

Not in git, made or downloaded by the app: `maps/` (your maps), `runtime/` (ONNX Runtime, CUDA, TensorRT), `.cache/` (models, engines, vocabulary vectors). They go in this folder when the program is run from here (`cargo run`), else in the system's place for an app's data (`~/.local/share/photo-map`, `~/Library/Application Support/photo-map`, `%APPDATA%\photo-map\data`); `PHOTO_MAP_HOME` names another.

Build and run: `cargo run --release`. It is plain Rust: no C compiler, cmake or nasm is needed. To keep cargo's package cache inside this folder: `env CARGO_HOME=.cargo-home cargo run --release`.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.

The models the app runs (SigLIP 2) and the libraries it downloads (ONNX Runtime, NVIDIA CUDA, cuDNN, TensorRT) are not part of this repository and have their own licenses.

`assets/places.zip` is made from [GeoNames](https://www.geonames.org/) data (cities1000, admin1 codes, country info), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
