# Models: where they come from and what is done to them

One workflow on every computer: the app downloads the publisher's files, and makes what a runtime needs from them on the
computer itself. Nothing changed is ever hosted.

## Download (src/modelstore.rs, list in src/models.rs)

| Model | Repository (Hugging Face) | Revision | Files | Size |
|---|---|---|---|---|
| SigLIP 2 Base 512 | `onnx-community/siglip2-base-patch16-512-ONNX` | `323fe16b` | `tokenizer.json`, `onnx/text_model_fp16.onnx`, `onnx/vision_model_fp16.onnx` | 0.79 GB |
| SigLIP 2 Base 256 | `onnx-community/siglip2-base-patch16-256-ONNX` | `d1114256` | the same | 0.79 GB |

- Every file is pinned by size and SHA-256 and refused if it differs. Files are fetched in 16 MB pieces with retries; a stopped
  download carries on where it was. They are kept in `.cache/onnx/<model>/` under the paths they have in the repository.
- fp16 everywhere: half the download, and the publisher's fp16 conversion is used as is (going down from fp32 needs care that they
  have taken; going up is exact).
- SigLIP's tokenizer is the same file as Google's own (same SHA-256). The app pads search words to 64 tokens itself.
- Removing a model removes its folder (downloaded and patched files), its CoreML and TensorRT networks and its vocabulary vectors.

## Patches (src/patch.rs), made when the model is loaded for a map or a search

| Runtime | SigLIP |
|---|---|
| Processor | photo encoder in fp32 (`vision_model_fp16.processor.onnx`) |
| CUDA | as published |
| TensorRT | layer norms in one piece (`vision_model_fp16.tensorrt.onnx`) |
| CoreML | first convolution's padding written out, every `MatMul` by a weight a `Gemm` (`vision_model_fp16.coreml.onnx`) |

- **Processor, fp32**: ONNX Runtime's fp16 arithmetic on a processor is slower. SigLIP 256's photo encoder: 6.7 photos/s fp16,
  8.3 fp32 (Ryzen 5 5600H), identical vectors. Every fp16 number is an fp32 number,
  so the copy is the same model. Search words alone stay fp16 (SigLIP's text encoder would be 1.1 GB in fp32).
- **SigLIP, TensorRT**: the publisher's files are opset 14, so every layer norm is spelled out (mean, subtract, square, ...).
  ONNX Runtime fuses that on the processor and CUDA, but TensorRT squares in fp16 and overflows: the vectors came out wrong
  (cosine 0.48 to the right ones). Each norm becomes one `LayerNormalization` (opset 17), its input divided by 16 and epsilon by
  256 first (the same result).
- **SigLIP, CoreML**: ONNX Runtime's CoreML provider refuses a convolution without explicit padding ("Required param 'pad' is
  missing"); the file is valid (no padding means zeros), and it says `auto_pad = VALID`, which the patch replaces by
  `pads = [0, 0, 0, 0]`. That was enough for CoreML to take it (checked on an M-series Mac).
- **SigLIP, CoreML, `Gemm` for `MatMul`** (`weights_as_gemm`): the provider writes a `MatMul`'s weight into CoreML's program as
  text (a transposed copy, every number as a hex float), so SigLIP 256's `model.mil` was 1.1 GB, which CoreML parses and
  serializes again at every load: **48 s** with the compiled network already cached, 2 to 3 minutes the first time (the same with
  ONNX Runtime 1.29 from Python). A `Gemm` with `transB` keeps its weight in the binary weights file: `model.mil` 0.3 MB,
  **1.8 s** a load (0.2 s of it CoreML), **16 s** the first time. The speed is the same (70.4 photos/s against 70.9 without it). The vectors are the same (cosine 0.99997 to the processor's).
  SigLIP 256 on the neural engine: 71 photos/s, graphics 6, all parts 65, processor (fp32) 4, on 24 photos. SigLIP 512 on the neural
  engine: 8.5 photos/s, 85 s the first load, 2 s after.
- SigLIP's weights are inside its files, so its patched copies are full copies (190 MB, or 370 MB in fp32).
- ONNX files are read and written with the `onnx-protobuf` crate (pure Rust; it needs `protobuf` pinned to 3.4.0).

## Checks (on Linux, RTX 3050 Ti; `models_match_the_maps` in src/models.rs)

The public files, patched for each runtime, against vectors that existing maps hold (made with the earlier files: our own SigLIP
export). Lowest cosine over 24 photos per map:

| Model, map | Processor | CUDA | TensorRT |
|---|---|---|---|
| SigLIP 256, rare-species 2 | 0.99996 | 0.99996 | 0.99998 |
| SigLIP 256, nasa | 0.99988 | 0.99988 | 0.99995 |

So maps made before keep working. On a computer without these maps, `backends_agree` compares every backend with the processor.

## Open

- The Mac: the rest of the speeds.
- `tools/onnx_fuse_attention.py`, `tools/onnx_trt_ready.py` and `tools/siglip_export.py` are no longer used by anything.
- Report to ONNX Runtime: the Conv with `auto_pad = VALID` that its CoreML provider refuses, and that a `MatMul`'s weight is written
  into the CoreML program as text (1.1 GB for SigLIP 256, parsed at every load) where a `Gemm`'s goes to the binary weights file.
- Ready-made Core ML models exist, from the community and not from Google or Apple (found by search, not opened or measured):
  SigLIP 2 base 256 as separate image and text `.mlpackage` encoders, 8-bit palettized, macOS 15 or later, text padded to 64
  tokens without a mask (Hugging Face: asamkhya, karthikramesh, karaiman, kuluruvineeth/siglip2-base-coreml; kanevry is a mirror),
  and the so400m image tower only (antonlnz/siglip2-so400m-image-coreml). Their weights are quantized, so the vectors
  would differ from ours, and they are third-party files without the publisher's hashes; the patched publisher files already run at
  70 photos/s (SigLIP 256). Worth a benchmark only if a faster or smaller route is needed.
- In an installed Mac app, `maps/`, `runtime/` and `.cache/` should live in `~/Library/Application Support/photo-map` (and the
  remakeable part in `~/Library/Caches/photo-map`).

EmbeddingGemma 2 was offered too until 2026-10-29: on rare-species it found photos by name worse than SigLIP 256 (top-1 51% against
59%) at a tenth of the speed (20 to 25 photos/s against about 230), and took most of the patching above.
