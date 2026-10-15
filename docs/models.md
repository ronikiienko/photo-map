# Models: where they come from and what is done to them

One workflow on every computer: the app downloads the publisher's files, and makes what a runtime needs from them on the
computer itself. Nothing changed is ever hosted.

## Download (src/modelstore.rs, list in src/models.rs)

| Model | Repository (Hugging Face) | Revision | Files | Size |
|---|---|---|---|---|
| EmbeddingGemma 2 | `onnx-community/embeddinggemma-2-ONNX` | `daa72c51` | `tokenizer.json`, `onnx/model_fp16.onnx` (+ `_data`), `onnx/vision_encoder_fp16.onnx` (+ `_data`) | 0.91 GB |
| SigLIP 2 Base 512 | `onnx-community/siglip2-base-patch16-512-ONNX` | `323fe16b` | `tokenizer.json`, `onnx/text_model_fp16.onnx`, `onnx/vision_model_fp16.onnx` | 0.79 GB |
| SigLIP 2 Base 256 | `onnx-community/siglip2-base-patch16-256-ONNX` | `d1114256` | the same | 0.79 GB |

- Every file is pinned by size and SHA-256 and refused if it differs. Files are fetched in 16 MB pieces with retries; a stopped
  download carries on where it was. They are kept in `.cache/onnx/<model>/` under the paths they have in the repository.
- fp16 everywhere: half the download, and the publisher's fp16 conversion is used as is (going down from fp32 needs care that they
  have taken; going up is exact).
- SigLIP's tokenizer is the same file as Google's own (same SHA-256). The app pads search words to 64 tokens itself.
- Removing a model removes its folder (downloaded and patched files), its CoreML and TensorRT networks and its vocabulary vectors.

## Patches (src/patch.rs), made when the model is loaded for a map or a search

| Runtime | SigLIP | Gemma |
|---|---|---|
| Processor | photo encoder in fp32 (`vision_model_fp16.processor.onnx`) | photo encoder and text model in fp32 |
| CUDA | as published | fused attention (`vision_encoder_fp16.cuda.onnx`) |
| TensorRT | layer norms in one piece (`vision_model_fp16.tensorrt.onnx`) | standard operators + separate pooling (`.tensorrt.onnx`, `.pool.onnx`) |
| CoreML | first convolution's padding written out (`vision_model_fp16.coreml.onnx`) | as published |

- **Processor, fp32**: ONNX Runtime's fp16 arithmetic on a processor is slower. SigLIP 256's photo encoder: 6.7 photos/s fp16,
  8.3 fp32 (Ryzen 5 5600H), identical vectors; Gemma was about twice as fast in fp32 on an M1. Every fp16 number is an fp32 number,
  so the copy is the same model. Search words alone stay fp16 (SigLIP's text encoder would be 1.1 GB in fp32).
- **Gemma, CUDA**: the export widens attention heads from 64 to 68 to carry the padding mask, which keeps ONNX Runtime off its
  fused attention; the patch puts them back to 64 and passes the mask as attention bias.
- **Gemma, TensorRT**: TensorRT can't read ONNX Runtime's own `SimplifiedLayerNormalization` and `MultiHeadAttention`; they are
  written with standard operators (`RMSNormalization`, opset 23; matrix products and softmax). Each norm's input is divided by 16
  and epsilon by 256 first (the same result), because TensorRT squares in fp16 and overflows above 256. The pooling after the
  layers comes out wrong in TensorRT, so it is a second small network on CUDA.
- **SigLIP, TensorRT**: the publisher's files are opset 14, so every layer norm is spelled out (mean, subtract, square, ...).
  ONNX Runtime fuses that on the processor and CUDA, but TensorRT squares in fp16 and overflows: the vectors came out wrong
  (cosine 0.48 to the right ones). Each norm becomes one `LayerNormalization` (opset 17), with the same divide-by-16.
- **SigLIP, CoreML**: ONNX Runtime's CoreML provider refuses a convolution without explicit padding ("Required param 'pad' is
  missing"); the file is valid (no padding means zeros). The patch writes `pads = [0, 0, 0, 0]`. The Mac earlier used a bigger
  rewrite (the convolution as reshape, transpose and matrix product); whether the small patch is enough is to be checked on a Mac.
- Patched networks point at the publisher's weights where those are in a separate file (Gemma), so they take little room; SigLIP's
  weights are inside its files, so its patched copies are full copies (190 MB, or 370 MB in fp32).
- ONNX files are read and written with the `onnx-protobuf` crate (pure Rust; it needs `protobuf` pinned to 3.4.0).

## Checks (on Linux, RTX 3050 Ti; `models_match_the_maps` in src/models.rs)

The public files, patched for each runtime, against vectors that existing maps hold (made with the earlier files: our own SigLIP
export, and Gemma's Python-patched files). Lowest cosine over 24 photos per map:

| Model, map | Processor | CUDA | TensorRT |
|---|---|---|---|
| SigLIP 256, rare-species 2 | 0.99996 | 0.99996 | 0.99998 |
| SigLIP 256, nasa | 0.99988 | 0.99988 | 0.99995 |
| Gemma, rare-species | 0.99879 | 0.99880 | 0.99851 |
| Gemma, everything from usb | 0.99986 | 0.99983 | 1.00000 |

So maps made before keep working ("everything from usb" was made on TensorRT: the Rust patch gives exactly what the Python one did). On a computer without these maps, `backends_agree` compares every backend with the processor.

## Open

- The Mac: CoreML with the small SigLIP patch, Gemma on CoreML, speeds (see docs/macos-models-task.md).
- `tools/onnx_fuse_attention.py`, `tools/onnx_trt_ready.py` and `tools/siglip_export.py` are no longer used by anything.
- Report the Conv/pads problem to ONNX Runtime.
- In an installed Mac app, `maps/`, `runtime/` and `.cache/` should live in `~/Library/Application Support/photo-map` (and the
  remakeable part in `~/Library/Caches/photo-map`).
