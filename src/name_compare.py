"""Compare small language models at naming a region from its captions, on the same sample of Wikivoyage regions.
    python src/name_compare.py Qwen3.5-0.8B-Q4_K_M Qwen3.5-2B-Q4_K_M Qwen3.5-4B-Q4_K_M [per_level]
Each model is a GGUF in .cache/llama.cpp/models/ (hard-link the file from LM Studio's models folder). One llama-server runs
at a time (started here, stopped when done), so the GPU holds one model. Prints, per region, each model's name next to the
name regions.json has, and each model's mean time per region. A region's prompt is short: the 25 commonest captions, 70 characters each.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter

import numpy as np

import config
import photomap

LLAMA = config.ROOT / ".cache/llama.cpp"
BIN = next((LLAMA / "bin").glob("llama-b*")) / "llama-server"
LIBS = ":".join([str(BIN.parent), *map(str, (LLAMA / "bin").glob("cudart-llama-*"))])
PORT = 8099
LEVELS = (1, 3, 5)  # 0-based: levels 2, 4, 6
CAPTIONS, WIDTH = 25, 70
BAD = {"travel", "photo", "photos", "photography", "picture", "pictures", "image", "images", "caption", "captions", "map"}
PROMPT = """What is the main subject that most of these picture captions are about? Answer with one or two plain words (a noun such as a kind of building, place or object), in lowercase, and nothing else. If there is no main subject, answer: none

Captions:
{captions}"""


def prompt_for(texts, idx):
    count = Counter(t.strip() for t in (texts[i] for i in idx) if t)
    return PROMPT.format(captions="\n".join(c[:WIDTH] for c, _ in count.most_common(CAPTIONS)))


class Server:
    """llama-server for one GGUF, started on entering and always stopped on leaving."""

    def __init__(self, gguf):
        self.gguf = gguf

    def __enter__(self):
        env = {**os.environ, "LD_LIBRARY_PATH": LIBS}
        self.proc = subprocess.Popen([str(BIN), "-m", str(LLAMA / "models" / f"{self.gguf}.gguf"), "-ngl", "99", "-c", "4096", "--host", "127.0.0.1",
                                      "--port", str(PORT), "--jinja"], env=env, stdout=open(config.ROOT / "logs/llama_server.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(120):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health")).get("status") == "ok":
                    return self
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited: see logs/llama_server.log")
            time.sleep(1)
        raise RuntimeError("llama-server did not start")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def ask(self, prompt):
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})))
        answer = r["choices"][0]["message"]["content"].strip().lower().strip(".")
        return "none" if set(answer.split()) & BAD else answer


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    per = int(next((a for a in sys.argv[1:] if a.isdigit()), 8))
    bundle = config.ROOT / "maps/wikivoyage.photomap"
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    texts = [p["meta"].get("caption", "") for p in json.loads((bundle / "photos.json").read_text())]
    levels = photomap.cluster(xy)
    named = json.loads((bundle / "regions.json").read_text())["levels"]
    rng = np.random.default_rng(1)
    sample = []  # (level, region, photos, prompt, current name)
    for lv in LEVELS:
        lab = levels[lv]
        members = photomap.members(lab, lab.max() + 1)
        current = {int(lab[int(np.argmin(((xy - [r["x"], r["y"]]) ** 2).sum(1)))]): r["name"] for r in named[lv]}
        ok = [r for r, m in enumerate(members) if sum(bool(texts[i]) for i in m) >= 10]
        for r in sorted(rng.choice(ok, min(per, len(ok)), replace=False)):
            sample.append((lv + 1, len(members[r]), prompt_for(texts, members[r]), current.get(r, "-"), texts[members[r][0]]))
    answers, times = {}, {}
    for gguf in args:
        with Server(gguf) as server:
            server.ask(sample[0][2])  # warm-up
            t = time.time()
            answers[gguf] = [server.ask(s[2]) for s in sample]
            times[gguf] = (time.time() - t) / len(sample) * 1000
    short = [g.replace("Qwen3.5-", "").replace("-Q4_K_M", "") for g in args]
    print(f"{'lvl':>3} {'photos':>6}  " + "  ".join(f"{s:<18}" for s in short) + "  counting-method name        e.g. caption")
    for i, (lv, size, _, cur, example) in enumerate(sample):
        print(f"{lv:>3} {size:>6}  " + "  ".join(f"{answers[g][i]:<18}" for g in args) + f"  {cur:<24}  {example[:50]}")
    print("mean ms per region: " + ", ".join(f"{s} {times[g]:.0f}" for s, g in zip(short, args)))
