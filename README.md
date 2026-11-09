# photo-map

> **A personal exploration project.** Made to explore what image embedding models can do for a photo
> collection. It works, but it is rough and half-finished in places, and tested on Linux and macOS only.

A desktop app that lays out a photo and video collection as a zoomable map: pictures that look alike sit together, regions
of the map get names, and search works by words ("dog on a beach") or by example photos. The search is like Google Photos'
search, but local, and on top of it the whole collection is one big map to wander around.

Tested on Linux with an NVIDIA graphics card (an RTX 3050 Ti laptop, 4 GB). It also runs on the processor alone, slower.
Apple silicon Macs work too. Windows is not supported.

## What it does

- Makes a map from folders of photos and videos, locally: each picture is embedded with SigLIP 2 (through ONNX
  Runtime, on CUDA or TensorRT when there is an NVIDIA card), laid out so that similar ones are close, and the regions named.
  Photos are read where they are, never copied or uploaded.
- Videos are searched by what is in them: frames are taken through each video (one every couple of seconds) and embedded
  like photos; a video matches by its best frame, which it then shows, and playing starts there.
- Labels: the map is split into regions at several zoom levels, coarse ones holding finer ones, and each region is named
  automatically from about 4,500 phrases: the phrase whose vector is nearest the region's average (less the whole map's
  average, so a region is named by what sets it apart). Zoomed out, a few broad names; zooming in, finer ones fade in.
- Search by words or by one or more photos; the matches light up on the map. Filters by name, camera, date and place.
- Zoom from the whole collection down to a single photo; videos play in place, with sound.
- Paths: right-click a photo, then another, and the map lights up a chain of photos leading from one to the other, each like
  the one before it. The chain is the one whose biggest jump is the smallest of all (found along the map's minimum spanning
  tree, made when the map is built), so it morphs gradually, e.g. from a beach to a city street.

Some effort went into speed. The models are patched to run on TensorRT in fp16 (about 1.5x faster than CUDA alone), and
the map builder is a pipeline: photos are decoded on the processor while the graphics card embeds the ones before, each
photo is read once, and only videos' keyframes are decoded. On a laptop RTX 3050 Ti a map is made at about 230 photos a
second (big 20+ MP JPEGs are limited by decoding, about 66 a second).

## Build and run

    cargo run --release

On first start the app offers to download what it needs: ONNX Runtime (with CUDA, cuDNN and TensorRT on an NVIDIA card),
the models, and ffmpeg for videos if there is none on the system.

The app keeps its maps, downloads and settings in `~/.local/share/photo-map` (on a Mac `~/Library/Application Support/photo-map`),
or wherever `PHOTO_MAP_HOME` points.

## In this repository

- `src/` the app: the viewer and the map builder (scan, embed, layout, regions, names, bake)
- `vocab/` the words used to name regions (built into the program)
- `assets/` the icon, ideas for searches, and `places.zip`, the towns that say where a photo was taken (built into the program)
- `tools/make_icon.py` draws the icon
- `docs/` notes, including an experiment that was dropped: searching videos by their sound (`docs/audio.md`)

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.

The models the app runs (SigLIP 2) and what it downloads (ONNX Runtime, NVIDIA CUDA, cuDNN, TensorRT, ffmpeg) are not part
of this repository and have their own licenses.

`assets/places.zip` is made from [GeoNames](https://www.geonames.org/) data (cities1000, admin1 codes, country info), licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
