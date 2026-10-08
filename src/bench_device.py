"""Time embedding chunks on GPU vs CPU (does not touch the index).
CPU is timed on a sample and scaled up: all chunks are the same size, so time is linear."""

import json
import time

import config
import torch
from sentence_transformers import SentenceTransformer

texts = [c["text"] for c in json.loads((config.INDEX_DIR / "chunks.json").read_text())]
SAMPLE = 32

for device, dtype, n in [("cuda", torch.bfloat16, len(texts)), ("cpu", torch.float32, SAMPLE), ("cpu", torch.bfloat16, SAMPLE)]:
    model = SentenceTransformer(
        config.MODEL_ID,
        device=device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.encode(texts[:2], prompt_name="Document")  # warm-up
    start = time.time()
    for i in range(0, n, 8):
        model.encode(texts[i : i + 8], prompt_name="Document")
        print(f"  {device} {dtype}: {min(i + 8, n)}/{n} chunks, {time.time() - start:.0f}s", flush=True)
    seconds = time.time() - start
    full = seconds * len(texts) / n
    print(f"RESULT {device} {str(dtype).removeprefix('torch.')}: {n / seconds:.2f} chunks/s -> all {len(texts)} chunks ~{full:.0f}s", flush=True)
    del model
