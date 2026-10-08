"""Split the book into stories, chunk them, embed the chunks, save to index/."""

import json
import re

import config
import numpy as np

from chunking import chunk_uniform

# Story headings look like "VIII. THE ADVENTURE OF THE SPECKLED BAND" at the start of a line.
# Table-of-contents lines are indented, so they don't match.
HEADING = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\. (.+)$")


def read_stories():
    """Return [(title, text, first_line)]; first_line = line number in the book where the text starts."""
    lines = config.BOOK.read_text(encoding="utf-8").replace("\r", "").split("\n")
    stories = []
    for n, line in enumerate(lines, start=1):
        if line.startswith("*** END OF"):  # Gutenberg footer starts here
            break
        m = HEADING.match(line)
        if m:
            stories.append((m.group(2).title(), [], n + 1))
        elif stories:
            stories[-1][1].append(line)
    return [(title, "\n".join(text), first_line) for title, text, first_line in stories]


model = config.load_model()
chunks, vectors = [], []

for title, text, first_line in read_stories():
    doc_chunks = chunk_uniform(text, model.tokenizer, config.CHUNK_TOKENS, config.OVERLAP_TOKENS)
    texts = [c for _, c in doc_chunks]
    # batch_size=8: the default 32 x 512-token chunks doesn't fit in 4 GB of GPU memory
    vectors.append(model.encode(texts, prompt=config.DOC_PROMPT.format(title=title), normalize_embeddings=True, batch_size=8))
    for start, c in doc_chunks:
        line = first_line + text.count("\n", 0, start)  # line in the book where the chunk starts
        chunks.append({"title": title, "text": c, "line": line})

config.INDEX_DIR.mkdir(exist_ok=True)
np.save(config.INDEX_DIR / "embeddings.npy", np.concatenate(vectors))
(config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
print(f"saved {len(chunks)} chunks")
