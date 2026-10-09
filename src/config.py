import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")  # keep model downloads inside the project
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the local copy, don't check huggingface.co (set to 0 to re-download)

MODEL_ID = "google/embeddinggemma-2"
CORPUS = "simplewiki"  # "sherlock" (build_index.py) or "simplewiki" (build_simplewiki.py)
BOOK = ROOT / "data" / "sherlock.txt"
WIKI = ROOT / "data" / "simplewiki" / "20231101.simple" / "train-00000-of-00001.parquet"
INDEX_DIR = ROOT / "index" / CORPUS

CHUNK_TOKENS = 512    # chunk size in tokens
OVERLAP_TOKENS = 64   # tokens shared between neighbouring chunks
DIM = 768             # vector size used for search: 768 / 512 / 256 / 128
TOP_K = 5             # number of search results

# Text the model sees in front of each input (from the model's prompt table).
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: {title} | text: "   # model default is "title: none | text: "; we fill in the story title


def load_model(images=False):
    import torch
    from sentence_transformers import SentenceTransformer

    # bfloat16 on GPU (fast); float32 on CPU (CPU bfloat16 is ~4x slower). Never float16: it can produce NaN vectors.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # text-only = 270M; images=True also loads the 170M vision encoder
    parts = {"audio_config": None} if images else {"vision_config": None, "audio_config": None}
    return SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": dtype}, config_kwargs=parts)
