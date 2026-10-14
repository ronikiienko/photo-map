"""Export SigLIP 2 for the viewer: the photo encoder and the text encoder as ONNX (float16 inside, float32 in and out), the tokenizer,
and a reference file the Rust tests compare against.

    .venv/bin/python src/siglip_export.py [google/siglip2-base-patch16-512]
    -> .cache/onnx/<model>/vision_fp16.onnx   pixel_values (batch, 3, size, size) float32, scaled to -1..1 -> image_features (batch, dim) float32
       .cache/onnx/<model>/text_fp16.onnx     input_ids (batch, 64) int64 -> text_features (batch, dim) float32
       .cache/onnx/<model>/tokenizer.json     (copied; it already pads to 64 and ends a text with <eos>)
       .cache/onnx/<model>/reference.json     a few texts and photos with the vectors PyTorch makes for them

The vectors are not normalised here; the viewer does that.
"""

import json
import os
import shutil
import sys

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface"))

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config

name = sys.argv[1] if len(sys.argv) > 1 else "google/siglip2-base-patch16-512"
out_dir = config.ROOT / ".cache" / "onnx" / name.split("/")[1]
out_dir.mkdir(parents=True, exist_ok=True)


def vector(x):
    return x if torch.is_tensor(x) else x.pooler_output


class Photo(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, pixel_values):
        return vector(self.tower(pixel_values=pixel_values.half())).float()


class Text(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return vector(self.model.get_text_features(input_ids=input_ids)).float()


model = AutoModel.from_pretrained(name, dtype=torch.float16).cuda().eval()
processor = AutoProcessor.from_pretrained(name)
size = model.config.vision_config.image_size
photo, text = Photo(model.vision_model), Text(model)
x = torch.randn(2, 3, size, size, device="cuda")
ids = torch.randint(3, 1000, (2, 64), device="cuda")
with torch.no_grad():
    torch.onnx.export(photo, (x,), str(out_dir / "vision_fp16.onnx"), input_names=["pixel_values"], output_names=["image_features"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}}, opset_version=18, dynamo=False)
    torch.onnx.export(text, (ids,), str(out_dir / "text_fp16.onnx"), input_names=["input_ids"], output_names=["text_features"],
                      dynamic_axes={"input_ids": {0: "batch"}, "text_features": {0: "batch"}}, opset_version=18, dynamo=False)
snapshot = snapshot_download(name, allow_patterns=["tokenizer.json"])
shutil.copy(os.path.join(snapshot, "tokenizer.json"), out_dir / "tokenizer.json")

# the reference: what PyTorch makes of some words and photos (the processor's own preprocessing)
texts = ["a dog on a beach", "Old Town Square, Prague", "x " * 100, "a person playing guitar on a stage with colourful lights"]
photos = sorted((config.ROOT / "maps/wikivoyage.photomap/originals").glob("*.jpg"))[:3]
with torch.no_grad():
    tok = processor.tokenizer([t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    text_vectors = vector(model.get_text_features(input_ids=tok["input_ids"].cuda())).float().cpu().numpy()
    pixels = processor(images=[Image.open(p).convert("RGB") for p in photos], return_tensors="pt")["pixel_values"].cuda().half()
    photo_vectors = vector(model.get_image_features(pixel_values=pixels)).float().cpu().numpy()
(out_dir / "reference.json").write_text(json.dumps({
    "texts": texts, "ids": tok["input_ids"].tolist(), "text_vectors": text_vectors.tolist(),
    "photos": [str(p.relative_to(config.ROOT)) for p in photos], "photo_vectors": photo_vectors.tolist(),
}))
print("exported", out_dir, "size", size, "dim", photo_vectors.shape[1])
