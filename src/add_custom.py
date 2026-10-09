"""Embed hand-written articles from data/custom/articles.json and save them next to the main index.
search.py adds them to the results, so new articles don't need a rebuild of the whole index.

articles.json: [{"title": ..., "text": ..., "image": "data/custom/photo.jpg" or null, "label": shown in results}, ...]
An article with an image is embedded as text + image together, in one vector.
"""

import json

import config
import numpy as np

articles = json.loads((config.ROOT / "data" / "custom" / "articles.json").read_text())
model = config.load_model(images=True)

vectors, chunks = [], []
for a in articles:
    text = config.DOC_PROMPT.format(title=a["title"]) + a["text"]
    if a.get("image"):
        # <|image|> marks where the picture goes in the input; the model reads text and image as one sequence
        inp = {"text": text + " <|image|>", "image": str(config.ROOT / a["image"])}
    else:
        inp = text
    vectors.append(model.encode(inp, normalize_embeddings=True))
    chunks.append({"title": a["title"], "text": a["text"], "url": f"(custom: {a.get('label', 'article')})"})
    print(f"embedded: {a['title']}{' + image' if a.get('image') else ''}")

np.save(config.INDEX_DIR / "custom_embeddings.npy", np.array(vectors))
(config.INDEX_DIR / "custom_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
