"""Chunk Simple English Wikipedia and embed it, saving progress so it can be stopped and rerun.
    step 1: chunks.json        all chunks, fewest tokens first (similar lengths batch together = faster)
    step 2: shards/NNNN.npy    vectors for 10k chunks each; finished shards are skipped on rerun
    step 3: embeddings.npy     all shards joined
"""

import json
import time

import config
import numpy as np
import pyarrow.parquet as pq

from chunking import chunk_uniform

SHARD = 10_000
TOKENS_PER_BATCH = 8 * 512  # 8 full-size chunks fit in 4 GB of GPU memory; short chunks get bigger batches

model = config.load_model()
chunks_file = config.INDEX_DIR / "chunks.json"
shard_dir = config.INDEX_DIR / "shards"
shard_dir.mkdir(parents=True, exist_ok=True)


def prompt(c):
    # The prompt has the article title in it, so it's different for every chunk: put it in front of the text ourselves.
    return config.DOC_PROMPT.format(title=c["title"]) + c["text"]


# Step 1: chunk
if chunks_file.exists():
    chunks = json.loads(chunks_file.read_text())
else:
    articles = pq.read_table(config.WIKI).to_pylist()
    chunks = []
    for n, a in enumerate(articles, start=1):
        for _, c in chunk_uniform(a["text"], model.tokenizer, config.CHUNK_TOKENS, config.OVERLAP_TOKENS):
            chunks.append({"title": a["title"], "text": c, "url": a["url"]})
        if n % 20_000 == 0 or n == len(articles):
            print(f"chunking: {n}/{len(articles)} articles -> {len(chunks)} chunks", flush=True)
    del articles  # free the raw text
    # sort by length in tokens (title prompt included), so every batch holds chunks of almost the same size
    # (10k at a time: tokenizing all of them at once needs more than 6 GB of RAM)
    lengths = []
    for i in range(0, len(chunks), SHARD):
        lengths += [len(ids) for ids in model.tokenizer([prompt(c) for c in chunks[i : i + SHARD]]).input_ids]
    chunks = [chunks[i] for i in np.argsort(lengths, kind="stable")]
    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"sorted and saved {len(chunks)} chunks, longest {max(lengths)} tokens", flush=True)

# Step 2: embed, one shard at a time
texts = [prompt(c) for c in chunks]
total_chars = sum(map(len, texts))
done_chars, start = 0, time.time()
n_shards = (len(texts) + SHARD - 1) // SHARD
print(f"embedding {len(texts)} chunks in {n_shards} shards", flush=True)

for s in range(n_shards):
    part = texts[s * SHARD : (s + 1) * SHARD]
    path = shard_dir / f"{s:04d}.npy"
    if not path.exists():
        longest = len(model.tokenizer(part[-1]).input_ids)  # sorted by tokens: last is longest
        batch = max(8, min(128, TOKENS_PER_BATCH // longest))
        vectors = model.encode(part, normalize_embeddings=True, batch_size=batch)
        np.save(shard_dir / "tmp.npy", vectors)
        (shard_dir / "tmp.npy").rename(path)  # only finished shards get their real name
        done_chars += sum(map(len, part))
        rate = done_chars / (time.time() - start)
        eta = (total_chars - sum(map(len, texts[: (s + 1) * SHARD]))) / rate
        print(f"shard {s + 1}/{n_shards}: batch {batch}, {time.time() - start:.0f}s elapsed, ~{eta / 60:.0f} min left", flush=True)

# Step 3: join
vectors = np.concatenate([np.load(shard_dir / f"{s:04d}.npy") for s in range(n_shards)])
np.save(config.INDEX_DIR / "embeddings.npy", vectors)
print(f"DONE: {vectors.shape} vectors, {vectors.nbytes / 1e9:.2f} GB", flush=True)
