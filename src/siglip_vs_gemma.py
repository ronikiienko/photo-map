"""SigLIP 2 against EmbeddingGemma 2 on Flickr30k: how good is caption -> photo search, and how fast do the photos get embedded?

    .venv/bin/python src/siglip_vs_gemma.py [siglip model ...] [--gallery N]
    # models: google/siglip2-base-patch16-256 (default), google/siglip2-large-patch16-256, google/siglip2-so400m-patch14-384
    # --gallery: photos searched (default 1000, the usual Flickr30k test split; more = harder, taken from the rest in file order)

Gemma's side reuses the stored photo vectors (index/flickr30k, 140 soft tokens) and embeds the captions with the query prompt.
Quality: for each caption, the rank of its own photo among the gallery (recall@1/5/10), and photo -> captions the other way.
Speed: SigLIP's photo encoder on the GPU in float16 on photos already decoded and resized (the model alone), batch 32.
"""

import csv
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface"))

import numpy as np
import torch
from PIL import Image

import config

args = sys.argv[1:]
gallery_size = int(args[args.index("--gallery") + 1]) if "--gallery" in args else 1000
models = [a for a in args if a.startswith("google/")] or ["google/siglip2-base-patch16-256"]
cache = config.ROOT / ".cache" / "siglip_eval"
cache.mkdir(parents=True, exist_ok=True)

photos = json.loads((config.ROOT / "maps/flickr30k.photomap/photos.json").read_text())
rows = list(csv.DictReader(open(config.ROOT / "data/flickr30k/flickr_annotations_30k.csv")))
by_file = {r["filename"]: r for r in rows}
split_of = [by_file[os.path.basename(p["source"])]["split"] for p in photos]
test = [i for i, s in enumerate(split_of) if s == "test"]
rest = [i for i, s in enumerate(split_of) if s != "test"]
gallery = (test + rest)[:gallery_size]
captions = [json.loads(by_file[os.path.basename(photos[i]["source"])]["raw"])[:5] for i in gallery]
assert all(len(c) == 5 for c in captions)
flat = [c for cs in captions for c in cs]  # caption j belongs to photo j // 5
truth = np.arange(len(flat)) // 5
print(f"gallery {len(gallery)} photos, {len(flat)} captions", flush=True)


def report(name, photo_vecs, caption_vecs):
    sims = caption_vecs @ photo_vecs.T  # captions x photos
    order = np.argsort(-sims, axis=1)
    rank = np.argmax(order == truth[:, None], axis=1)
    t2i = {k: float((rank < k).mean() * 100) for k in (1, 5, 10)}
    # photo -> captions: the best-ranked of its five captions
    sims_t = sims.T
    top = np.argmax(sims_t, axis=1) // 5
    i2t1 = float((top == np.arange(len(gallery))).mean() * 100)
    print(f"RESULT {name:42s} caption->photo R@1 {t2i[1]:5.1f}  R@5 {t2i[5]:5.1f}  R@10 {t2i[10]:5.1f}   photo->caption R@1 {i2t1:5.1f}", flush=True)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- Gemma: stored photo vectors, captions embedded now
stored = np.load(config.ROOT / "index/flickr30k/image_embeddings.npy", mmap_mode="r")
gemma_photos = unit(np.asarray(stored[gallery], dtype=np.float32))
text_file = cache / f"gemma_captions_{gallery_size}.npy"
if text_file.exists():
    gemma_text = np.load(text_file)
else:
    from sentence_transformers import SentenceTransformer

    gemma = SentenceTransformer(config.MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"vision_config": None, "audio_config": None})
    gemma_text = gemma.encode(flat, prompt=config.QUERY_PROMPT, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(text_file, gemma_text)
    del gemma
    torch.cuda.empty_cache()
report("EmbeddingGemma 2 (140 tokens, stored)", gemma_photos, unit(gemma_text))

# --- SigLIP 2
from transformers import AutoModel, AutoProcessor

images = None
for name in models:
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(name)
    if images is None:
        t0 = time.time()
        images = [Image.open(photos[i]["source"]).convert("RGB") for i in gallery]
        print(f"decoded {len(images)} photos in {time.time() - t0:.0f} s", flush=True)
    # preprocessing on the CPU (resize to the model's square), timed apart from the model
    t0 = time.time()
    pixels = [proc(images=images[i : i + 32], return_tensors="pt")["pixel_values"] for i in range(0, len(images), 32)]
    prep = time.time() - t0
    with torch.no_grad():
        model.get_image_features(pixel_values=pixels[0].cuda().half())  # warm up
        torch.cuda.synchronize()
        t0 = time.time()
        feats = []
        for p in pixels:
            f = model.get_image_features(pixel_values=p.cuda().half())
            feats.append((f if torch.is_tensor(f) else f.pooler_output).float().cpu())
        torch.cuda.synchronize()
        enc = time.time() - t0
        texts = []
        for i in range(0, len(flat), 128):
            tok = proc(text=[t.lower() for t in flat[i : i + 128]], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
            f = model.get_text_features(input_ids=tok["input_ids"])
            texts.append((f if torch.is_tensor(f) else f.pooler_output).float().cpu())
    photo_vecs, text_vecs = unit(torch.cat(feats).numpy()), unit(torch.cat(texts).numpy())
    short = name.split("/")[1]
    print(f"SPEED {short}: model {len(images) / enc:.0f} photos/s on the GPU (float16, batch 32); CPU resize {len(images) / prep:.0f} photos/s; "
          f"{sum(p.numel() for p in model.vision_model.parameters()) / 1e6:.0f}M photo-encoder parameters; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    report(short, photo_vecs, text_vecs)
    np.save(cache / f"{short}_photos_{gallery_size}.npy", photo_vecs)
    del model
    torch.cuda.empty_cache()
