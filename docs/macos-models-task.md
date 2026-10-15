# Task for the agent on the Mac: models and runtime after the Linux changes

Read docs/models.md first. Since the macos branch was merged, Linux changed how models and the runtime are handled; this is the
part only a Mac can check. Report what you find in docs/models.md (the tables) and keep the code working on both systems.

## What changed

- **Models are really downloaded** (src/modelstore.rs): the onnx-community files, pinned by SHA-256, into `.cache/onnx/<model>/`
  under their repository paths (`onnx/vision_model_fp16.onnx`, `onnx/text_model_fp16.onnx`, ...). The simulated fetch from
  `.cache/upstream/` is gone; delete `.cache/upstream/` and the old `.cache/onnx/*` folders and press Get in the Models window.
- **Patches are made in the app** (src/patch.rs) when a model is loaded, not by Python. For CoreML there is one: SigLIP's
  `vision_model_fp16.coreml.onnx`, which only writes the first convolution's padding out (`pads = [0,0,0,0]`). This is smaller than
  your `patchify.py` (convolution as reshape, transpose, matrix product), and it is untested on CoreML.
- **fp16 for both SigLIP sizes** (512 used fp32 on the Mac before).
- **The processor gets fp32 copies** of what runs per photo (SigLIP's photo encoder; Gemma's photo encoder and text model). On a
  Mac this also applies to Gemma's text model while its photo encoder is on CoreML, since the text model runs on the processor.
- **Runtime**: one button downloads everything missing (on a Mac the one ONNX Runtime pack). "Run the models on" is one choice
  (radio buttons). `PHOTO_MAP_BACKEND` = cpu, apple-neural, apple-gpu or apple-all picks one for a run (`PHOTO_MAP_COREML` still
  works).

## To do

1. Pull, build, and run the tests: `cargo test --release -- --test-threads=1`. Fix whatever breaks on macOS.
2. In the app: Runtime window (one download button, radio buttons, Remove all), Models window (Get with progress, Stop and Get
   again carries on, Remove). Get all three models.
3. **Does CoreML take the small SigLIP patch?** Make a map with SigLIP 256 on the neural engine. If CoreML still says the padding
   is missing, or something else in the file stops it, port your patchify rewrite into `siglip_coreml` in src/patch.rs (Rust,
   `onnx-protobuf`; see `gemma_fused` for how nodes and weights are edited) and say why the small one was not enough.
4. **Do the vectors agree?** With a folder of photos (any 24 or more):
   `PHOTO_MAP_PHOTOS=<folder> PHOTO_MAP_BACKEND=cpu cargo test --release backends_agree -- --ignored --test-threads=1 --nocapture`
   then the same with `apple-neural`, `apple-gpu`, `apple-all`. It compares each with the processor (lowest cosine should be
   above 0.99) and prints photos/s (the first run of a backend includes CoreML's preparing).
5. **Speeds**, for the table in docs/models.md: SigLIP 256 and 512 (now fp16) on each Apple option and on the processor (fp32
   copy), and Gemma on the processor and on the best Apple option. `cargo test --release processor_speed -- --ignored --nocapture`
   gives SigLIP 256's photo encoder fp16 against fp32 on the processor alone.
6. **Gemma on CoreML** was 0.1 photos/s before. If it still is, it is better to keep Gemma's photo encoder off CoreML (on the
   processor, fp32) when an Apple option is chosen: measure both and decide, and say so in docs/models.md.
7. Keep Linux working: nothing Mac-only outside `cfg` blocks, and say in the commit messages what was measured.
