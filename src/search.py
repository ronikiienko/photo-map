"""Search the corpus chosen in config.CORPUS.
    python src/search.py "your query"   # one search
    python src/search.py                # keep asking; empty line quits
    python src/search.py --photos ...   # photos only (skips the text index)
Drag an image file onto the terminal (or type its path) to search with the picture instead of text.
"""

import json
import readline  # noqa: F401 — enables arrow keys and history in input()
import sys
import urllib.parse
from pathlib import Path

import config
import numpy as np

PHOTOS_ONLY = "--photos" in sys.argv
if PHOTOS_ONLY:
    sys.argv.remove("--photos")
    vectors, chunks = np.zeros((0, 768), dtype=np.float32), []
else:
    vectors = np.load(config.INDEX_DIR / "embeddings.npy")
    chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text())
n_wiki = len(chunks)  # rows after this are custom articles

# Hand-written articles from add_custom.py, if any: append them to the index.
if not PHOTOS_ONLY and (config.INDEX_DIR / "custom_embeddings.npy").exists():
    vectors = np.concatenate([vectors, np.load(config.INDEX_DIR / "custom_embeddings.npy")])
    chunks += json.loads((config.INDEX_DIR / "custom_chunks.json").read_text())

# Photos from embed_images.py, if any: searched as a separate list, because image vectors and text vectors
# score on different scales (an image query likes every image a bit, a text query likes text a bit more).
photos, photo_vectors = [], np.zeros((0, vectors.shape[1]), dtype=np.float32)
if (config.INDEX_DIR / "image_embeddings.npy").exists():
    photo_vectors = np.load(config.INDEX_DIR / "image_embeddings.npy")
    photos = json.loads((config.INDEX_DIR / "image_chunks.json").read_text())


def shorten(v):
    """Optional shortening (Matryoshka): keep the first DIM numbers, rescale to length 1."""
    v = v[:, : config.DIM]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)  # broken images are all-zero rows


vectors, photo_vectors = shorten(vectors), shorten(photo_vectors)

model = config.load_model(images=True)
model[0].processor.image_processor.max_soft_tokens = 140  # same image budget as embed_images.py

PHOTOS_K = 10 if PHOTOS_ONLY else 3  # photo results shown under the text results
IMAGE_TYPES ={".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def as_image(query):
    """A file dropped on the terminal arrives as its path: maybe quoted, with escaped spaces, or as file://...
    Return the path if it is an image file, else None."""
    text = query.strip().strip("'\"").replace("\\ ", " ")
    if text.startswith("file://"):
        text = urllib.parse.unquote(text.removeprefix("file://"))
    path = Path(text).expanduser()
    return path if path.suffix.lower() in IMAGE_TYPES and path.is_file() else None


def search(query):
    image = as_image(query)
    if image:
        print(f"(searching with image {image.name})")
        # images get no prompt: the task prefixes are for text only
        q = model.encode({"image": str(image)}, truncate_dim=config.DIM, normalize_embeddings=True)
    else:
        q = model.encode(query, prompt=config.QUERY_PROMPT, truncate_dim=config.DIM, normalize_embeddings=True)
    scores = vectors @ q  # cosine similarity with every chunk
    for i in np.argsort(-scores)[: config.TOP_K]:
        c = chunks[i]
        print(f"\n{scores[i]:.3f}  [{c['title']}]")
        if "url" in c:
            print(f"  {c['url']}")
        else:
            print(f"  {config.BOOK}:{c['line']}")  # open the book at the chunk's line
        print(c["text"][:400].replace("\n", " "))
    # custom articles that didn't make the top: show where they ranked
    rank = np.argsort(np.argsort(-scores))
    for i in range(n_wiki, len(chunks)):
        if rank[i] >= config.TOP_K:
            print(f"\n  custom [{chunks[i]['title']}] {chunks[i]['url']}: rank {rank[i] + 1}, score {scores[i]:.3f}")
    if photos:
        if not PHOTOS_ONLY:
            print("\n--- photos ---")
        scores = photo_vectors @ q
        for i in np.argsort(-scores)[:PHOTOS_K]:
            p = photos[i]
            more = len(p.get("articles", [])) - 1  # e.g. a flag shown on many pages
            also = f" (+{more} more articles)" if more > 0 else ""
            # plain absolute path and URL: Ctrl-click in the terminal opens the photo in the editor, the article in the browser
            url = urllib.parse.quote(p["url"], safe=":/")  # brackets and accents would cut the terminal's link short
            print(f"\n{scores[i]:.3f}  [{p['title']}]{also} {p['text'][:100]}\n  {config.ROOT / p['image']}\n  {url}")


if len(sys.argv) > 1:
    search(" ".join(sys.argv[1:]))
else:
    while query := input("\nquery> "):
        search(query)
