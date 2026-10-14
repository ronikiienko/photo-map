# Task for an agent on a Mac: make photo-map run well on Apple Silicon

You are working on `photo-map`, a Rust (wgpu + egui) desktop app that lays a folder of photos out as a zoomable map, where similar
photos sit together, and searches them with words or another photo. The app embeds the photos itself, in-process, with ONNX Runtime
(no Python in the app). Everything so far was developed and measured on one Linux machine with an NVIDIA card. Nobody has built or
run it on a Mac yet. This is your job.

Read `README.md` first, then `src/models.rs`, `src/runtime.rs`, `src/packs.rs`, `src/manager.rs`, `src/embed.rs`, `src/siglip.rs`
and `src/build.rs` (the map building pipeline).

## Goals, in order

1. **Build and run on macOS (Apple Silicon).** Get `cargo run --release` working, with the app starting on the Maps page.
2. **Make the models run on the Mac, and fast.** Measure the options (below) and report numbers.
3. **Make the downloads work on macOS**: the runtime libraries (ONNX Runtime) and the model files, from the app, with the user
   confirming. Nothing is installed or downloaded without the user pressing a button.
4. **Evaluate MLX** as a backend for the models and say whether it is worth it.

## What the app does when it makes a map

Stages: scan, copy, embed, layout, regions, names, bake, finish (resumable, state in `job.json`). **Embed is the slow one**:
every photo goes through a model that turns it into a vector (768 numbers). That is "map creation time" to optimise.

Models (`src/models.rs`, one is chosen per map and recorded in its manifest):

| Model | Photo input | Files (in `.cache/onnx/<folder>/`) |
|---|---|---|
| EmbeddingGemma 2 | up to about 1260 patches, padded to 1264 | about 870 MB |
| SigLIP 2 Base, 512 px | 512 x 512 squeezed square | about 750 MB |
| SigLIP 2 Base, 256 px | 256 x 256 squeezed square | about 750 MB |

SigLIP has separate photo and text encoders (two ONNX files, fp16 inside, float32 at the edges). Gemma has one multimodal model
whose photo encoder is split in two for TensorRT (`vision_encoder_fp16_trt.onnx` and `vision_pool.onnx` are TensorRT-specific; the
CUDA-only and CPU path uses `vision_encoder_fp16.onnx` / `..._fused.onnx`). The conversion scripts are in `tools/` (they need
Python and the project's `.venv`; they are for development only, the app never runs them).

## Reference numbers from Linux (RTX 3050 Ti Laptop, 4 GB, TensorRT; varies by about 30% between sessions)

- Whole map-making embed stage: Gemma about 20 to 25 photos/s, SigLIP 512 about 55 to 75, SigLIP 256 about 200 to 300.
- The processor side (decoding, shrinking, preparing the photo) is no longer the limit: about 260 photos/s (512 px) and 740
  photos/s (256 px) on 12 cores, with large camera JPEGs. Test: `cargo test --release siglip_cpu_side_speed -- --ignored --nocapture`.
- Speed tests you can reuse (they need `maps/rare-species.photomap/originals` and `maps/wikivoyage.photomap/originals`, which are
  not in the repo; any folder of real photos works if you change the paths): `siglip_embed_speed`, `siglip_cpu_side_speed`, and
  `another_model_makes_its_own_vectors` in `src/build.rs`.
- Accuracy check against PyTorch: `siglip_matches_pytorch` (needs `.cache/onnx/<folder>/reference.json`, made by
  `tools/siglip_export.py`). Vectors must keep a cosine of at least 0.999 to the reference.

## Where the project stands on platforms

- Compiles for Windows (checked with a cross compile); never built for macOS.
- The detection and install code is Linux-only: library names (`.so`), search folders, the NVIDIA driver probe, pack
  definitions (`src/runtime.rs`, `src/packs.rs`). `Setup` and the pack list need per-OS tables.
- Other Linux-only bits: `xdg-open` in `src/home.rs` (macOS: `open`), the restart in `src/main.rs` uses `sh`,
  `DISPLAY` / `WAYLAND_DISPLAY` handling in `main()`, and the app's data folder is the source tree (`CARGO_MANIFEST_DIR`), which
  will not do for an installed app (decide where `maps/`, `runtime/` and `.cache/` live on a Mac and tell the owner).
- ONNX Runtime 1.31.0 for macOS exists only as `onnxruntime-1.31.0-cp312-cp312-macosx_14_0_arm64.whl` on PyPI (Apple Silicon,
  macOS 14 or newer; there is no Intel Mac wheel). Inside the wheel the library is `onnxruntime/capi/libonnxruntime.1.31.0.dylib`
  and the CoreML execution provider is built into it. Packs are zip files read with HTTP range requests; see `src/packs.rs`
  (`print_pins` and `installs_a_real_pack` show how to add and pin one).
- The `ort` crate is `=2.0.0-rc.14` with `load-dynamic`; its `coreml` feature is not enabled yet.

## Questions to answer with measurements

Report a table per model of photos/s and, if you can read it, memory use, for each backend, using the same photos:

1. ONNX Runtime on the processor only.
2. ONNX Runtime with the CoreML execution provider: try compute units CPU only, CPU and GPU, and all (Neural Engine). Try fp16
   and fp32 model files. CoreML copes badly with dynamic shapes: SigLIP has fixed input sizes, so should be easy; check what
   Gemma's photo encoder does (its padded input is a fixed size, but the model may have dynamic axes inside). Check the vectors
   still match PyTorch (cosine at least 0.999) with CoreML, since it may run in lower precision.
3. **MLX** (Apple's array framework). Questions: does it beat ONNX Runtime with CoreML for these models? Is there a usable port
   of SigLIP 2 and of EmbeddingGemma 2's photo encoder (community conversions exist for several models; check what is current)?
   Can it be called from Rust without Python (the C interface `mlx-c`, or an unofficial Rust binding; check maturity)? What
   would it cost to maintain a second backend that only exists on Macs? MLX does not read ONNX, so each model would be
   reimplemented, with its own weights, tokenizer and preprocessing. Do this question after question 2, and only seriously if
   CoreML turns out slow or incorrect for a model. Give a recommendation, not a survey.
4. How long does a map of about 1,000 and about 10,000 photos take end to end on the Mac, for each model, with the best backend?

## Downloads on macOS

- **ONNX Runtime:** add a macOS arm64 pack (processor and CoreML in one library) to `src/packs.rs`, pin it, and make the Runtime
  window and the New map offer work there (`src/manager.rs`). There is no NVIDIA anything on a Mac: the offers must not mention
  CUDA, and must not offer Linux files on a Mac.
- **Model files:** an open problem. The ONNX files the app runs are the owner's own conversions (the scripts in `tools/`), and the
  owner does **not** want to host them. Find out what upstream publishes at stable addresses (for example the `onnx-community`
  repositories on Hugging Face) for SigLIP 2 and EmbeddingGemma, whether those files run unchanged in this app (inputs, outputs,
  float32 edges, fp16), and what the app would have to adapt locally. Report; do not build a hosting solution.

## Rules (the owner's, please keep them)

- No Python inside the app. Python is only for the scripts in `tools/`.
- Everything installed or downloaded goes inside the project folder (including the cargo cache: `CARGO_HOME=.cargo-home`).
  Use `.cache/test/` for test downloads. Never write to the user's home or system folders.
- Nothing downloads or installs without the user's confirmation in the app. Packs are checked against pinned fingerprints.
- The app only ever removes what it installed itself (a pack has a `pack.json`); it never touches other runtime files.
- No speed claims in the interface; show measured speed after the fact.
- Keep Linux working: `cargo test --release -- --test-threads=1` must still pass there. Run long commands in the background and
  report progress. Don't run GPU or speed tests while the app is open.
- Don't commit maps, models, `runtime/`, `.cache/`, `target/` or any user photos (they are git-ignored; keep it that way).
- Work on a branch named `macos` and push it; do not merge into `main`. Small commits, with messages that say what and why.

## What to hand back

1. A branch `macos` where the app builds and runs on the Mac, with the platform code made per-OS, not Linux-only.
2. A short report in `docs/macos-report.md`: the measurements above, what works, what does not, and your recommendation on
   CoreML versus MLX, with the reasons.
3. A list of anything the owner has to decide, each as a question with your recommended answer.
