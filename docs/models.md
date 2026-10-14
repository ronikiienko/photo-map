# Models: where they come from, what is done to them, and what we want

Written during the macOS work. It collects the open questions about the model files and what the owner wants. The measurements
behind it are from an M1 Mac (Python ONNX Runtime 1.29 probes and Rust tests); vectors are compared with the processor's fp32 output
because the PyTorch reference vectors were not available on the Mac.

## What the owner wants

1. **No hosting.** The project does not host model files or converted copies.
2. **Fetch from the public sources** (for example the `onnx-community` repositories on Hugging Face), with the address and a
   hash of each file pinned in the app, so a changed upstream file is refused, as packs do for the runtime libraries.
3. **Patch on the user's machine** where a model needs it, after the download, so no patched file is ever hosted.
4. **Models are chosen in the app**: a Models window lists them, "Get" fetches one, "Remove" deletes the files and everything made
   for them (prepared networks, vocabulary vectors). The app starts with no models; a map can only be made with an installed one.
   (Getting is simulated for now: it links the files in `.cache/upstream/`. The real fetch is still to do.)
5. **Preparing a model for a runtime happens when a map is made** (status line in "Understand"), not when it is downloaded, so
   changing runtime later does not leave a model without a prepared network.

## What is done to the files today

| Model | Source on the Mac | Changes |
|---|---|---|
| SigLIP 2 Base 256 / 512 | `onnx-community/siglip2-base-patch16-{256,512}-ONNX` | The first layer, a convolution that cuts the photo into 16 x 16 patches, is replaced by reshape, transpose, matrix multiply and add for the one fixed image size (script not in the repo: `/tmp/pm-scratch/patchify.py` on the Mac). Everything else is as published. The 512 files in use are fp32, the 256 ones fp16. |
| EmbeddingGemma 2 | `onnx-community/embeddinggemma-2-ONNX` | None on the Mac. |
| SigLIP on Linux | the project's own PyTorch export (`tools/siglip_export.py`) | outputs `image_features` / `text_features`; the app accepts upstream's `pooler_output` too |
| Gemma on Linux | origin not recorded in the repo; file names and edges match the onnx-community files | `tools/onnx_fuse_attention.py` (CUDA) and `tools/onnx_trt_ready.py` (TensorRT: standard operators only) |

### Why the SigLIP patch is needed

ONNX Runtime's CoreML provider (ML Program format) refuses a Conv without explicit pads ("Required param 'pad' is missing"). The
published files are valid ONNX (no pads means zero padding), so this is a gap in the provider, not in the files. Precision does not
matter: the layer is the same in the fp16 and fp32 files. The patched files compute the same thing and use only standard operators,
so any provider can run them. They are not needed on CUDA or TensorRT.

## Measurements (photos/s, M1; cosine to the processor's fp32 vectors)

| Model | Processor | CoreML neural engine | CoreML all parts | MLX plugin |
|---|---|---|---|---|
| SigLIP 256 | 5.7 | **61.6** (fp16, cos 0.9999) | 60.7 | 27 (fp32); fp16 gives wrong vectors (cos 0.48) |
| SigLIP 512 | 1.0 | 7.9 | **8.8** (cos 0.9999) | 4.5 (fp32) |
| Gemma | 0.4 (fp32), 0.2 (fp16) | vision part 0.1 | not finished | not finished |

- fp16 on CoreML is about 2.6 times faster than fp32 and matches it; batch size makes no difference to speed.
- The first CoreML run compiles the network for the chip: 2 to 4 minutes, sometimes much longer; the result (about 1 GB per
  model) is cached. ONNX Runtime and Apple give no progress while it runs.
- Gemma is a heavy model (0.4 photos/s on the processor) and its inputs have open sizes (batch, patch count, and an output length
  that depends on the data), which CoreML needs fixed. Its vision graph also has 113 `SimplifiedLayerNormalization` and 16
  `MultiHeadAttention` nodes (ONNX Runtime's own operators), so on CoreML it would be cut into many pieces. A properly measured
  Gemma on CoreML has not been done; the "standard operators" file from `tools/onnx_trt_ready.py` with fixed shapes is the candidate.
- MLX (GPU only, no neural engine) works through the `onnxruntime-ep-mlx` plugin without patches, but is slower than CoreML on
  SigLIP and has the fp16 problem, so it is not offered. Using MLX directly (`mlx-rs`, `mlx-embeddings`) is untested.

## Open questions (with recommended answers)

1. **Where do the patch steps live?** Recommended: a small recipe table in the app (URL, hash, steps per backend) and the steps
   written in Rust (the SigLIP convolution patch is small; the Gemma TensorRT rewrite is large and can stay Python-made for now).
   The Python scripts in `tools/` stay as the reference.
2. **Which protobuf crate for editing ONNX files in Rust**, without a C++ build dependency? Not checked yet.
3. **Run the patch only for CoreML?** Recommended: yes. The unpatched public file is used on the processor, CUDA and TensorRT.
4. **Where did the Gemma ONNX files on Linux come from?** Check with the Linux machine's history; then document it here. The reference
   vectors in `tests/` can show that the Mac's Gemma files are the same model (not run yet).
5. **SigLIP 512 uses fp32 files on the Mac** and the 256 ones fp16: use fp16 for both once the patch step exists.
6. **Report the Conv/pads problem to ONNX Runtime**, and keep the patch as a stopgap.
7. **Upstream file changes**: refuse on a hash mismatch and say so in the Models window; fall back to the processor.
8. **Where do `maps/`, `runtime/` and `.cache/` live in an installed Mac app?** Recommended: `~/Library/Application Support/photo-map`
   for maps and models, `~/Library/Caches/photo-map` for what can be made again (prepared networks).
