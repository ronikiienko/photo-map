"""A test map for sounds: maps/esc50.photomap, from ESC-50 (2,000 five-second environmental clips, 50 classes, data/esc50/).
    python src/map_esc50.py
    python src/map_esc50.py --captions       only (re)make the captions and names of the existing map
    python src/map_esc50.py --planes         maps/esc50-planes.photomap: the clips and the Wikivoyage photos whose captions are about airplanes or helicopters
The audio goes through the model's audio encoder (same vector space as the text and photos). Each clip's picture is its
spectrogram; its class (dog, rain, ...) is its caption (meta "caption", vectors in captions.bin, as for the animals).
The clips are copied to the bundle's audio/ (meta "audio" = audio/NNNNNN.wav) for the viewer to play.
"""

import csv
import json
import re
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram

import config
import name_regions
import photomap

ROOT = config.ROOT / "data/esc50/ESC-50-master"
PICTURES = config.ROOT / "data/esc50/spectrograms"
BUNDLE = config.ROOT / "maps/esc50.photomap"
PLANES = re.compile(r"\b(airplanes?|aeroplanes?|aircraft|airliners?|jets?|boeing|airbus|helicopters?|helipad|chopper)\b", re.I)
BATCH = 16
COLORS = np.array([[0, 0, 4], [60, 15, 110], [140, 40, 125], [220, 80, 90], [252, 160, 60], [252, 253, 191]], np.float32)  # dark ... light


def load(name):
    """The clip as mono 16 kHz (what the model wants) and as its original samples' rate."""
    rate, x = wavfile.read(ROOT / "audio" / name)
    x = x.astype(np.float32) / 32768
    if x.ndim > 1:
        x = x.mean(1)
    return resample_poly(x, 160, 441).astype(np.float32) if rate == 44100 else x


def picture(x, path):
    """Spectrogram of 16 kHz samples as a 320 x 200 colour image (low pitch at the bottom)."""
    _, _, s = spectrogram(x, 16000, nperseg=512, noverlap=384)
    db = 10 * np.log10(s + 1e-10)
    t = np.clip((db - (db.max() - 70)) / 70, 0, 1)[::-1]
    pos = t * (len(COLORS) - 1)
    lo = np.minimum(pos.astype(int), len(COLORS) - 2)
    rgb = COLORS[lo] + (COLORS[lo + 1] - COLORS[lo]) * (pos - lo)[..., None]
    Image.fromarray(rgb.astype(np.uint8)).resize((320, 200), Image.BILINEAR).save(path)


def embed(clips):
    cache = config.ROOT / "data/esc50/audio_embeddings.npy"
    if cache.exists():
        return np.load(cache)
    model = config.load_model(audio=True)
    vectors, start = [], time.time()
    for s in range(0, len(clips), 200):
        part = [{"audio": c} for c in clips[s : s + 200]]
        batch = BATCH
        while True:
            try:
                vectors.append(model.encode(part, batch_size=batch, normalize_embeddings=True))
                break
            except torch.OutOfMemoryError:
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"  out of GPU memory, retrying with batch {batch}", flush=True)
        done = min(s + 200, len(clips))
        print(f"  {done:,} / {len(clips):,} clips ({done / (time.time() - start):.0f}/s)", flush=True)
    vectors = np.concatenate(vectors)
    np.save(cache, vectors)
    return vectors


def planes():
    """The Wikivoyage photos about airplanes and helicopters: photos (title, source, meta), files, vectors (already made, same model)."""
    chunks = json.loads((config.ROOT / "index/wikivoyage/image_chunks.json").read_text())
    vectors = np.load(config.ROOT / "index/wikivoyage/image_embeddings.npy")
    keep = [i for i, c in enumerate(chunks) if PLANES.search(c["text"])]
    photos = [{"title": chunks[i]["text"], "source": chunks[i]["image"], "meta": {"caption": chunks[i]["text"], "article": ", ".join(chunks[i]["articles"]), "url": chunks[i]["url"]}} for i in keep]
    return photos, [config.ROOT / chunks[i]["image"] for i in keep], vectors[keep]


def add_captions(bundle):
    """meta "caption" = the item's own (a photo's), else its class for a clip; captions.bin = the vectors of those texts (embedded once each)."""
    photos = json.loads((bundle / "photos.json").read_text())
    texts = [p["meta"].get("caption") or p["meta"]["category"] for p in photos]
    words = sorted(set(texts))
    vectors = dict(zip(words, config.load_model().encode(words, normalize_embeddings=True, show_progress_bar=False)))
    np.stack([vectors[t] for t in texts]).astype(np.float16).tofile(bundle / "captions.bin")
    for p, t in zip(photos, texts):
        p["meta"]["caption"] = t
    (bundle / "photos.json").write_text(json.dumps(photos, ensure_ascii=False))
    photomap.step(f"{len(photos):,} captions ({len(words)} different)")


if __name__ == "__main__":
    if "--planes" in sys.argv:
        BUNDLE = config.ROOT / "maps/esc50-planes.photomap"
    if "--captions" in sys.argv:
        add_captions(BUNDLE)
        name_regions.name(BUNDLE)
        sys.exit()
    rows = list(csv.DictReader(open(ROOT / "meta/esc50.csv")))
    clips = [load(r["filename"]) for r in rows]
    photomap.step(f"{len(clips):,} clips loaded")
    vectors = embed(clips)
    PICTURES.mkdir(parents=True, exist_ok=True)
    files = []
    for r, x in zip(rows, clips):
        files.append(PICTURES / (r["filename"].removesuffix(".wav") + ".png"))
        picture(x, files[-1])
    photomap.step("spectrograms drawn")
    photos = [{"title": r["category"].replace("_", " "), "source": f"esc50/{r['filename']}",
               "meta": {"category": r["category"].replace("_", " "), "audio": f"audio/{i:06d}.wav"}} for i, r in enumerate(rows)]
    if "--planes" in sys.argv:  # the photos go after the clips (the clips' numbers stay the same)
        more, more_files, more_vectors = planes()
        photomap.step(f"{len(more):,} photos about airplanes and helicopters added")
        photos, files, vectors = photos + more, files + more_files, np.concatenate([vectors, more_vectors])
    photomap.make(BUNDLE, photos, files, vectors)
    (BUNDLE / "audio").mkdir()
    for i, r in enumerate(rows):
        shutil.copy2(ROOT / "audio" / r["filename"], BUNDLE / "audio" / f"{i:06d}.wav")
    add_captions(BUNDLE)
    name_regions.name(BUNDLE)
