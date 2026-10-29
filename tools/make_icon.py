"""Draw the app's icon: a dark rounded square with three clusters of small photo tiles, like a map zoomed out.

    .venv/bin/python tools/make_icon.py    -> assets/icon.png (256 px)
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
SCALE = 4  # drawn 4 times bigger, then shrunk: smooth edges
OUT = Path(__file__).resolve().parent.parent / "assets" / "icon.png"

# each cluster: centre (0..1), radius (0..1), its colours (a region of alike photos)
CLUSTERS = [
    ((0.34, 0.36), 0.20, [(236, 166, 72), (226, 128, 60), (245, 200, 110)]),  # warm: sunsets, deserts
    ((0.68, 0.42), 0.17, [(70, 150, 220), (52, 112, 190), (120, 190, 235)]),  # blue: sea, sky
    ((0.47, 0.70), 0.18, [(96, 176, 98), (64, 140, 80), (150, 205, 120)]),  # green: forests, fields
]


def main():
    s = SIZE * SCALE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 8 * SCALE
    d.rounded_rectangle((pad, pad, s - pad, s - pad), radius=48 * SCALE, fill=(30, 34, 42, 255))
    rng = random.Random(7)
    tile = 15 * SCALE
    gap = 4 * SCALE
    step = tile + gap
    # a grid of tiles; a tile is drawn when it falls in a cluster, in that cluster's colours (denser at the centre)
    for gy in range(pad + gap * 3, s - pad - tile - gap * 2, step):
        for gx in range(pad + gap * 3, s - pad - tile - gap * 2, step):
            cx, cy = (gx + tile / 2) / s, (gy + tile / 2) / s
            for (ox, oy), r, colours in CLUSTERS:
                dist = math.hypot(cx - ox, cy - oy) / r
                if dist < 1.0 and rng.random() < 1.15 - dist * 0.7:
                    c = rng.choice(colours)
                    d.rounded_rectangle((gx, gy, gx + tile, gy + tile), radius=3 * SCALE, fill=c + (255,))
                    break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
