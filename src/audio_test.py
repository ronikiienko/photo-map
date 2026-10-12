"""A quick look at EmbeddingGemma 2's audio side on ESC-50 (2,000 five-second sound clips, 50 classes: dog, rain, siren, ...;
data/esc50, from github.com/karolpiczak/ESC-50). Not part of the viewer.
    python src/audio_test.py                 embed the clips (once), print the numbers, write logs/audio_results.html
    python src/audio_test.py "dog barking"   top clips for words (the paths play with: ffplay -nodisp -autoexit PATH)
    python src/audio_test.py some_clip.wav   top clips for a sound
The HTML page has players, so the results can be judged by ear: open logs/audio_results.html in a browser.
"""

import base64
import csv
import html
import io
import sys
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import config

ESC = config.ROOT / "data/esc50/ESC-50-master"
CACHE = config.ROOT / "index/esc50/audio_embeddings.npy"
RATE = 16000  # the model's audio rate
PROMPT = "the sound of {}"
QUERIES = ["dog barking", "a baby crying", "rain falling", "a siren", "someone typing on a keyboard", "waves on a beach",
           "birds singing", "a car horn", "people laughing", "a chainsaw", "church bells", "a helicopter", "fire crackling",
           "a door knock", "a crowd cheering", "music"]  # the last two are not among the 50 classes


def read(path):
    """A wav file as mono float32 at 16 kHz (any rate: ffmpeg is not needed for plain PCM wavs)."""
    with wave.open(str(path)) as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), {1: np.uint8, 2: np.int16, 4: np.int32}[width])
    x = raw.astype(np.float32).reshape(-1, channels).mean(1)
    x = (x - 128) / 128 if width == 1 else x / float(2 ** (8 * width - 1))
    g = gcd(rate, RATE)
    return resample_poly(x, RATE // g, rate // g).astype(np.float32)


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"vision_config": None})


def embed_clips(model, rows):
    if CACHE.exists():
        return np.load(CACHE)
    out, start = [], time.time()
    for s in range(0, len(rows), 32):
        clips = [{"audio": read(ESC / "audio" / r["filename"])} for r in rows[s : s + 32]]
        out.append(model.encode(clips, batch_size=32, normalize_embeddings=True))
        print(f"  {s + len(clips):,} / {len(rows):,} clips ({(s + len(clips)) / (time.time() - start):.0f}/s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, np.concatenate(out))
    return np.load(CACHE)


def words(model, texts):
    return model.encode(texts, prompt=config.QUERY_PROMPT, normalize_embeddings=True)


def top(rows, vectors, q, k=5):
    s = vectors @ q
    return [(rows[i], s[i]) for i in np.argsort(-s)[:k]]


def tag(r):
    """A player with the clip inside the page (16 kHz, 160 kB): the page then plays wherever it is opened, also from a sandboxed browser."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes((read(ESC / "audio" / r["filename"]).clip(-1, 1) * 32767).astype("<i2").tobytes())
    return f'<audio controls preload="none" src="data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"></audio>'


if __name__ == "__main__":
    rows = list(csv.DictReader((ESC / "meta/esc50.csv").open()))
    model = load_model()
    vectors = embed_clips(model, rows)
    cats = sorted({r["category"] for r in rows})
    label = np.array([cats.index(r["category"]) for r in rows])

    if len(sys.argv) > 1:
        arg = " ".join(sys.argv[1:])
        q = (model.encode([{"audio": read(arg)}], normalize_embeddings=True)[0] if Path(arg).is_file() else words(model, [arg])[0])
        for r, s in top(rows, vectors, q, 8):
            print(f"{s:.3f}  {r['category']:<16} {ESC / 'audio' / r['filename']}")
        sys.exit()

    # 1. words -> sound: each class name as a query, the class with the best cosine wins
    names = [c.replace("_", " ") for c in cats]
    for form in ("{}", PROMPT):
        w = words(model, [form.format(n) for n in names])
        acc = ((vectors @ w.T).argmax(1) == label).mean()
        print(f"words -> sound, class name as '{form}': {acc:.1%} of clips land on their own class (chance 2%)")
    # 2. sound -> sound: leave-one-out nearest clip, but only clips from other source recordings (a freesound clip is cut into several)
    src = np.array([r["src_file"] for r in rows])
    s = vectors @ vectors.T
    s[src[:, None] == src[None, :]] = -1
    nn = s.argmax(1)
    print(f"sound -> sound: nearest clip (from another recording) is the same class for {(label[nn] == label).mean():.1%}")
    # 3. recall of the words: for each class name the share of its 40 clips in the top 40
    w = words(model, [PROMPT.format(n) for n in names])
    sc = vectors @ w.T
    rec = [np.isin(np.where(label == i)[0], np.argsort(-sc[:, i])[:40]).mean() for i in range(len(cats))]
    order = np.argsort(rec)
    print("words -> sound, share of a class's 40 clips in the 40 best: mean %.1f%%" % (100 * np.mean(rec)))
    print("  worst:", ", ".join(f"{names[i]} {rec[i]:.0%}" for i in order[:8]))
    print("  best: ", ", ".join(f"{names[i]} {rec[i]:.0%}" for i in order[-8:]))

    # the page to judge by ear
    parts = ["<meta charset=utf-8><title>Gemma audio</title><style>body{font:15px sans-serif;margin:2em;background:#222;color:#eee}"
             "td{padding:4px 12px}h2{margin-top:1.5em}</style><h1>EmbeddingGemma 2 audio: ESC-50</h1>"
             "<p>Words in, the 6 closest clips out (category in brackets; the best-scoring first). The last two queries are not classes.</p>"]
    for t in QUERIES:
        parts.append(f"<h2>{html.escape(t)}</h2><table>")
        for r, sc_ in top(rows, vectors, words(model, [t])[0], 6):
            parts.append(f"<tr><td>{sc_:.2f}</td><td>[{r['category']}]</td><td>{tag(r)}</td></tr>")
        parts.append("</table>")
    parts.append("<h1>Sound to sound</h1><p>A random clip, then the clips closest to it.</p>")
    rng = np.random.default_rng(1)
    for i in rng.choice(len(rows), 8, replace=False):
        parts.append(f"<h2>[{rows[i]['category']}] {tag(rows[i])}</h2><table>")
        sims = s[i]
        for j in np.argsort(-sims)[:4]:
            parts.append(f"<tr><td>{sims[j]:.2f}</td><td>[{rows[j]['category']}]</td><td>{tag(rows[j])}</td></tr>")
        parts.append("</table>")
    (config.ROOT / "logs/audio_results.html").write_text("\n".join(parts))
    print("wrote logs/audio_results.html")
