# Videos: why they were slow, what was changed, what it gave, what is left

Written 2026-10-15. Measured on the RTX 3050 Ti laptop (12 logical cores), model SigLIP 2 Base 256. Test videos:

- **Screen recording**: 16 minutes (993 s), H.264 Main, 1600x720, 683 MB, a key frame about every second (0.99, 1.99, 3.05 ... s),
  no B-frames. One picture every 2 seconds = 497 pictures. A copy sits in /tmp/pm-bench-media/clip.mp4.
- **Phone clip**: 32 s, H.264 High, 1920x1080, 60 fps (on the Samsung phone drive).

## Result

The whole map of the screen recording (Videos stage, embedding, finishing), start to finish, no pause, two runs each
(`video_map_speed`):

| | Time |
|---|---|
| Before | 28.5 s, 29.0 s |
| Pictures cut in parallel, model fed from its own thread, faster resize | 18.8 s, 18.9 s |
| Plus sample times moved onto key frames | 5.7 s, 5.7 s |
| Plus scene pictures made only for the chosen frames | 5.0 s, 5.2 s |
| Plus one picture decoded per sample | 4.8 s, 4.6 s |

About 6 times faster: roughly 105 embedded pictures per second end to end (a lower bound: the 4.7 s also holds the model start,
the scene grouping and the writing of the map). Decoding alone (`video_where_time_goes`, every 2 s): 31 pictures/s at the start,
251/s now; the 1080p phone clip 7/s at the start, 74/s after the second round (2.5 s -> 0.23 s for 17 pictures).

## What was found

1. **The card was never the limit.** GPU use swung between 12 % and 67 % because it waited for the processor. Measured on its own
   (blank pictures through `Model::embed_prepared`): about 270 pictures/s. The TensorRT engine is built for 1 to 4 pictures (`BATCH` in
   src/siglip.rs) so bigger lots are split into 4s anyway; a bigger batch was not tried and nothing was changed on the model side.
2. **The picture work after decoding ran on one thread, one picture at a time**, and the same thread then waited for the card.
   Per picture, old code: resize to the model size 9-11 ms, prepare 0.3-0.8 ms, resize to 1024 px 24 ms, JPEG of that 12-17 ms,
   thumbnail 2-3 ms; about 50 ms, so at most 20 pictures/s, with the card idle meanwhile and the processor idle while the card worked.
3. **Most of the decoding was thrown away.** A decoder can only start at a key frame. The sample times (0, 2, 4 ... s) fall anywhere
   between key frames, so the decoder ran from the key frame up to the wanted picture: 28 pictures fed per picture kept (13959 for 497),
   about 350 ms of processor per kept picture. Opening the file (0.13 s for 993 s of video) and reading the packets (under 1 s in all)
   were not the problem. Making an OpenH264 decoder costs 0.2 ms.
4. **A key frame is expensive: about 16 ms at 1600x720 and 40 ms at 1080p**, one decoder, one core (`video_h264_compare`, now removed).
   The 5 ms per picture measured at first was an average over mostly predicted pictures of a static screen recording.

## What was changed

- **src/video.rs, `frames()` / `plan()` / `run()`**
  - A wanted time within half the interval of a key frame is moved onto that key frame (`snap`). A key frame decodes by itself. The picture keeps
    its real time, so scene titles stay true. Times with no key frame near (long gaps between key frames) are decoded exactly as before.
  - `frames()` was split into `group_starts`, `group_shows`, `plan` (which groups hold the wanted times) and `run` (decode them side by
    side in the rayon pool, hand them over in order), so `pictures_at` shares them.
  - `Job::in_order`: a group whose pictures show in the order they are coded (no B-frames) reads and feeds only the packets up to the last
    wanted one and then asks the decoder for the picture it still holds (OpenH264 holds the last one back). Other groups are read whole as before.
  - `pictures_at(path, times, each)`: decodes just the pictures showing at the given times (the exact `VideoFrame::seconds` values).
- **src/build.rs, `understand_videos`**
  - Frames are collected `CUT` (16) at a time and, side by side with rayon, resized to the model's size, prepared and thumbnailed, then sent through
    a channel (2 lots deep) to a thread of its own that calls the model, collects the vectors and updates progress and the recent pictures.
    Decoding goes on in the pool meanwhile.
  - The per-frame 1024 px picture and JPEG are gone. After the scenes are chosen from the vectors, `pictures_at` decodes only the chosen
    frames again and each gets the same resize (`SCENE_PICTURE` 1024 px) and JPEG (quality 85) as before. If a chosen picture cannot be read
    again, that video is left out and the map says so. `nearest_picture` and its test are gone: a scene's picture is found by its exact time.
- **src/photo.rs, `fit`**: resizes with `fast_image_resize` (bilinear convolution, widened when shrinking = the triangle filter) instead of
  `image::imageops::resize`. `fit` is also used by bake.rs, gallery.rs and stream.rs.
- **Tests, all `#[ignore]`, run on purpose** (plain `cargo test --release <name> -- --ignored --nocapture`): `video_map_speed` (build.rs; the whole
  map of the folder in PHOTO_MAP_MEDIA, no pause), `video_cpu_steps` (build.rs; time of each per-picture step, still has the old 1024 px and JPEG
  steps in it), `video_where_time_goes` (video.rs; opening and decoding at several intervals, PHOTO_MAP_VIDEO), `fit_matches_the_old_resize`
  (photo.rs; PHOTO_MAP_PICTURE). `builds_a_map_with_a_video` pauses and resumes in the middle, which restarts the video, so its time is not
  a speed measure (it is the correctness test).

## Quality

- Resizing: the new `fit` differs from the old by at most 1 of 255 per value (mean 0.02-0.09), on two video frames and a photo, at 256 and 1024 px.
- Scene pictures: the same frames, resize and JPEG quality as before, made later. By construction identical; the files were not compared
  byte by byte. The scenes of the end-to-end test (every 4 s, 40 scenes) are identical title for title before and after the "one picture
  per sample" change.
- Moving samples onto key frames is the one change that alters results. On the screen recording at one picture every 4 s both ways give 40
  scenes (the cap); the first nine are identical and the rest agree to within about one sample, but the cap joins different neighbours (for
  example 1:59 to 3:48 as one scene where exact sampling gives four). Which is better was not judged by eye. To get exact times back at the
  old speed make `snap` return its argument; a tighter limit than half the interval is in between.
- Embedding, the photo path (`load_for`, `prepare`) and the JPEG quality are untouched.

## Videos against photos (CPU side)

| | One core | All cores |
|---|---|---|
| Photos, Base 256, 381 KB originals, read + prepare (`siglip_cpu_side_speed`) | 117/s | 617/s |
| Photos, Base 512 | 60/s | 210/s |
| Video pictures, Base 256 | about 55-60/s (estimate from the step timings) | decode only 251/s; whole map about 105/s |

Photos are still ahead, about 2x per core. Why: a JPEG is decoded already shrunk by the decoder (libjpeg-turbo-rs, scaled decode, in `photo::load_for`,
which also keeps twice the size the model needs and rounds up to a multiple of 1/8) and each photo stands alone; a video picture is decoded at
full size (H.264 cannot be decoded small: every block is predicted from its full-size neighbours), converted to RGBA at full size, resized to the
model size and then squeezed into the model's square by `prepare` (two resizes). The 512 model is slower than 256 because it asks for 4x the pixels
(bigger scaled read, 4x the floats to make, and worse scaling over cores, probably memory traffic - a guess).

## Assembly and `nasm`

- **OpenH264 is built without its assembly.** The build log (`target/release/build/openh264-sys2-*/output`) says "Failed to compile NASM files, not using
  any assembly"; `nasm` (and `yasm`) is not installed. So H.264 decoding runs plain C. The audit below guesses 2-3x faster decoding with it.
  A fix is a `nasm` binary kept inside the project and put on PATH for the build (the project must stay self-contained), then a rebuild of `openh264-sys2`.
  (An older setup had one in `.tools/nasm`; it is gone.) Another is loading Cisco's prebuilt OpenH264 library at runtime (the `openh264` crate's
  `libloading` feature), which needs a download step with a hash check like packs.rs does.
- **JPEG and resizing do not need `nasm`.** `libjpeg-turbo-rs` 0.8.0 is a pure-Rust port (its only dependency is `thiserror`) with its own SIMD in
  Rust, and `fast_image_resize` has AVX2 paths chosen at run time. The old `turbojpeg-sys` build outputs in `target/` that mention nasm are leftovers.
- **`rusty_h264` 0.9.1 was tried** (pure Rust, `default-features = false`; its own SIMD feature also needs `nasm`), against OpenH264 on one core, one key
  frame at a time: 1600x720 16.2 ms (OpenH264) against 17.8 ms; 1920x1080 39.8 ms against 40.9 ms. The luma planes were identical (largest difference 0 over
  64 key frames). It is a working drop-in with no C build, but not faster. Not adopted (Cargo.toml and Cargo.lock were put back). Its README claims about
  109 -> 145 Mpx/s with `-C target-cpu=x86-64-v3`; that would apply to the whole build and stop the binary on older CPUs, and was not measured here.
  `rusty_h264::Decoder::decode` takes one Annex-B access unit and returns the picture at once (no hold-back), which would fit the one-picture-per-sample idea.

## Audit by a second model (read-only, from the code; not yet acted on)

Checked by me: item 1 (the build log says exactly that). The rest is the reviewer's reading of the code and is not verified.

1. OpenH264 without assembly (above): biggest, output unchanged.
2. `open()` is called three times per video (`info`, `frames`, `pictures_at`) and reads the MP4 index with an unbuffered `File` (re_mp4 does one small read per table
   entry, hundreds of thousands of system calls per video): wrap in a `BufReader` and open once; about 0.3-0.4 s here, more on a slow disk. Output unchanged.
3. The 40 or so scene pictures are resized and JPEG-saved one at a time inside the `pictures_at` callback (about 0.5 s): do them in parallel. Output unchanged.
   (`libjpeg-turbo-rs` has an encoder that may be faster than `image`'s; that would change the JPEG bytes slightly.)
4. The decode queue in `run` can grow without limit while one slow group holds up the order (33 MB per 4K frame): only start a job while `launched < next + ahead`.
   A memory fix.
5. `hand_over` (parallel resize, then send) runs inside the callback that receives frames, so nothing new is started while it runs and workers go idle; it also
   holds 16 full-size frames. Do the resize, prepare and thumbnail inside each decode job. Guess 0.2-0.4 s. Output unchanged.
6. A portrait video is rotated (`turn`) at full size before it is shrunk: shrink first. Might change values by 1.
7. Groups with B-frames read the whole group from disk even when only the key frame is wanted; read packets one by one inside the decode loop.
8. H.265 (untested): colour conversion per pixel with floats (slow); pictures skipped by the decoder at an open group count in `times`, which would give
   later frames wrong time labels; a monochrome or 4:2:2 stream could panic in the colour conversion; H.264 is always converted as BT.601 and H.265 as
   BT.709 (a BT.709 H.264 video comes out slightly off), 10-bit is shifted down to 8.
   Guesses to measure first: per-group decoder memory (page faults, compare user and sys time), the scene merging loops in scenes.rs are O(R^2 d) (slow
   for multi-hour videos), and a pause throws away all vectors of the video in progress.
   Not significant: copies in `annex_b`, batch flattening in siglip.rs, the zeroed RGBA buffer, `fs::copy` in `finish_videos`, thumbnails made for every
   frame when only a few are shown.

## Photos and videos are chosen separately

The New map window has a "Put the photos on the map" box next to the existing "Put the videos on the map" one (shown when the folder has both, with
readable videos). Unchecking the photos makes a videos-only map. `Job::with_photos` (default true, so maps started before this resume as they were)
makes `scan` skip the photo files; `build::create` takes a last argument `with_photos` and refuses a map with neither. The Create button is enabled when
a chosen kind has something in it; without readable videos the photos are always in, so the choice cannot be left unfixable. Videos are still cut into
scenes (at most 40 per video, a run of pictures that look alike is one scene); a choice of how videos appear (every sampled frame, scenes, one per video)
was raised but not built.

## Not measured, not done

- H.265 speed and correctness (all tests used H.264 clips).
- Reading videos from the external drive at scale; a run on the 15 GB / 73-minute 1080p60 video there did not finish (the check was stopped).
- Resizing once instead of twice before the model (about 5 ms per picture) and shrinking the YUV planes before the colour conversion: both change the
  vectors slightly, so they were not done without asking.
- Hardware decoding (NVDEC) would take decoding off the processor but needs FFmpeg or a CUDA video library.
- Builds: use plain `cargo` as the owner does (no `CARGO_HOME` override, which makes their next build recompile everything); run builds only when asked.
