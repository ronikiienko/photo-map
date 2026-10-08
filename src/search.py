"""Search the book.
    python src/search.py "your query"   # one search
    python src/search.py                # keep asking; empty line quits
"""

import json
import readline  # noqa: F401 — enables arrow keys and history in input()
import sys

import config
import numpy as np

vectors = np.load(config.INDEX_DIR / "embeddings.npy")
chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text())

# Optional shortening (Matryoshka): keep the first DIM numbers, rescale to length 1.
vectors = vectors[:, : config.DIM]
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

model = config.load_model()


def search(query):
    q = model.encode(query, prompt=config.QUERY_PROMPT, truncate_dim=config.DIM, normalize_embeddings=True)
    scores = vectors @ q  # cosine similarity with every chunk
    for i in np.argsort(-scores)[: config.TOP_K]:
        c = chunks[i]
        print(f"\n{scores[i]:.3f}  [{c['title']}]")
        print(f"  {config.BOOK}:{c['line']}")  # open the book at the chunk's line
        print(c["text"][:400].replace("\n", " "))


if len(sys.argv) > 1:
    search(" ".join(sys.argv[1:]))
else:
    while query := input("\nquery> "):
        search(query)
