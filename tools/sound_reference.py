"""What the publisher's own code makes of some sound and words, for listen.rs's tests to compare with (.cache/test/sound/).

EmbeddingGemma 2 through sentence-transformers on the processor in float32 (the reference: no float16 or bfloat16 in it):
- wave.f32, features.f32: 10 s of made-up sound and its features (Gemma4AudioFeatureExtractor, not cut at 30 s);
- queries.json: a few searches' vectors (the SearchQuery prompt);
- windows.json, windows.f32: the vectors of the first WINDOWS 10 s windows of a recording (no overlap), if one is given.

    python tools/sound_reference.py [recording]
"""

import json
import subprocess
import sys

import numpy as np

import config

OUT = config.ROOT / ".cache" / "test" / "sound"
WINDOWS = 16
RATE, WIN = 16_000, 160_000
QUERIES = ["the doctor makes a choice", "мафия делает свой выбор", "a dog barking", "investors and funding"]

from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor as fx  # noqa: E402

cut = fx.__call__
fx.__call__ = lambda self, *a, **k: cut(self, *a, **{**k, "truncation": False})

OUT.mkdir(parents=True, exist_ok=True)
st = SentenceTransformer("google/embeddinggemma-2", device="cpu", config_kwargs={"vision_config": None})

t = np.arange(WIN) / RATE
wave = (0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.05 * np.random.default_rng(0).standard_normal(WIN)).astype(np.float32)
f = st.preprocess([{"array": wave, "sampling_rate": RATE}])
features = f["input_features"][0].numpy().astype(np.float32)
assert f["input_features_mask"].all() and features.shape == (999, 128), features.shape
wave.tofile(OUT / "wave.f32")
features.tofile(OUT / "features.f32")
print(f"features {features.shape} -> {OUT}", flush=True)

vectors = st.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
(OUT / "queries.json").write_text(json.dumps([{"text": q, "vector": v.tolist()} for q, v in zip(QUERIES, vectors)], ensure_ascii=False))
print(f"{len(QUERIES)} queries", flush=True)

if len(sys.argv) > 1:
    source = sys.argv[1]
    a = np.frombuffer(subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", source, "-t", str(WINDOWS * 10), "-vn", "-ac", "1",
                                      "-ar", str(RATE), "-f", "f32le", "pipe:1"], check=True, capture_output=True).stdout, np.float32)
    wins = [{"array": a[i * WIN:(i + 1) * WIN], "sampling_rate": RATE} for i in range(len(a) // WIN)]
    v = st.encode(wins, batch_size=4, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    v.tofile(OUT / "windows.f32")
    (OUT / "windows.json").write_text(json.dumps({"source": source, "windows": len(wins)}))
    print(f"{len(wins)} windows of {source}", flush=True)
