"""Write the vectors as readable text to index/embeddings_dump.txt."""

import json

import config
import numpy as np

vectors = np.load(config.INDEX_DIR / "embeddings.npy")
chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text())

with open(config.INDEX_DIR / "embeddings_dump.txt", "w") as f:
    for i, (v, c) in enumerate(zip(vectors, chunks)):
        f.write(f"chunk {i} [{c['title']}]\n")
        f.write(f"text: {c['text'][:200]!r}\n")
        for j in range(0, len(v), 64):  # 64 numbers per line
            f.write(" ".join(f"{x:+.4f}" for x in v[j : j + 64]) + "\n")
        f.write("\n")
