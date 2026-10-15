# Videos: why they were slow, what was changed, what it gave

Measured on 2026-10-15 on the RTX 3050 Ti laptop with 12 cores. Test video: a 16-minute screen recording (H.264 Main, 1600x720,
683 MB, a key frame about every second, no B-frames), one picture every 2 seconds (497 pictures), model SigLIP 2 Base 256.
A second video was a 32-second phone clip (1080p, 60 fps, High profile).

## Result

The whole map of that one video (the Videos stage, the embedding and the finishing), start to finish, no pause, two runs each:

| | Time |
|---|---|
| Before | 28.5 s, 29.0 s |
| Cutting in parallel, model on its own thread | 18.8 s, 18.9 s |
| Plus sample times moved onto key frames | 5.7 s, 5.7 s |
| Plus scene pictures made only for the chosen frames | 5.0 s, 5.2 s |
| Plus one picture decoded per sample | 4.8 s, 4.6 s |

That is about 5 times faster, roughly 87 embedded pictures per second end to end (a lower bound: the 5.7 s also holds the start,
the scene grouping and the writing of the map). Decoding alone: 168 pictures/s on the screen recording (was 31), 74/s on the 1080p
phone clip (was 7). The card alone does about 270 pictures/s, so it was never the limit.

## What was found

1. **The card was not the bottleneck.** GPU use swung between 12 % and 67 % because it waited for the processor. Measured on its
   own (blank pictures, `Model::embed_prepared`): 274 pictures/s in lots of 4. The TensorRT engine is built for 1 to 4 pictures
   (`BATCH` in src/siglip.rs), so bigger lots are split into 4s anyway; nothing was changed there.
2. **The picture work after decoding ran on one thread, one picture at a time**, and the same thread then waited for the card. Per
   picture, with the old code: resize to the model size 9-11 ms, prepare 0.3-0.8 ms, resize to 1024 px 24 ms, JPEG of that 12-17 ms,
   thumbnail 2-3 ms. About 50 ms, so at most 20 pictures/s, and the card sat idle meanwhile (and the processor idle while the card
   worked).
3. **Most of the decoding was thrown away.** A decoder can only start at a key frame. The sample times (0, 2, 4 ... s) fall anywhere
   between key frames (here every ~1 s, drifting: 0.99, 1.99, 3.05, 4.05 ...), so the decoder had to run from the key frame up to the
   wanted picture. Counted: 28 pictures fed to the decoder for every picture kept (13959 for 497). A single picture decodes in about
   5 ms and making a decoder costs 0.2 ms, so that was the whole cost: about 350 ms of processor per kept picture. Opening the file
   (0.13 s for 993 s of video) and reading the packets (under 1 s in all) are not an issue.
4. **Why photos are 300+/s and videos were not.** A JPEG is decoded already shrunk (libjpeg-turbo scaled decode) and each photo stands
   alone. A video picture is decoded at full size, converted to RGBA at full size, and (in the video stage only) also resized to
   1024 px and written as a JPEG, because any picture may become a scene's picture.

## What was changed

- `src/video.rs`, `frames()`: a wanted time within half the interval of a key frame is moved onto that key frame (`snap`). A key
  frame is decoded by itself, so a sample now costs 2 pictures fed (the key frame, plus the one OpenH264 holds back) instead of 28.
  Other times (no key frame near, long key-frame intervals) are decoded exactly as before. The picture keeps its real time, so
  scene titles stay true.
- `src/build.rs`, `understand_videos`: pictures are collected `CUT` (16) at a time, cut side by side with rayon (resize, prepare, the
  1024 px JPEG, thumbnail) and sent through a channel (2 lots deep) to a thread of its own that calls the model, collects the vectors
  and updates the progress. Decoding keeps going in the rayon pool meanwhile. Pictures are named by their order among those asked
  for, so a picture that failed leaves a gap (`nearest_picture` already copes with gaps).
- `src/photo.rs`, `fit`: resizes with `fast_image_resize` (bilinear convolution, widened when shrinking, which is the triangle filter)
  instead of `image::imageops::resize`. Checked against the old function on two video frames and a photo, at 256 and 1024 px: the
  largest difference in any value is 1 of 255, the mean 0.02-0.09. `fit` is also used by bake.rs, gallery.rs and stream.rs.
- Tests (all `#[ignore]`, run on purpose): `video_cpu_steps` (build.rs, time of each per-picture step), `video_map_speed` (build.rs,
  the whole map of the folder in PHOTO_MAP_MEDIA, no pause), `video_where_time_goes` (video.rs, opening and decoding at several
  intervals), `fit_matches_the_old_resize` (photo.rs, PHOTO_MAP_PICTURE).
  Note `builds_a_map_with_a_video` pauses and resumes in the middle, which restarts the video, so its time is not a speed measure.

## Quality

- Resizing: same as before to within 1 of 255 (above).
- Moving samples onto key frames: not identical. On the 16-minute video at one picture every 4 s, both ways give 40 scenes (the
  cap); the first nine are identical and the rest agree to within about one sample, but the cap joins different neighbours (for
  example 1:59 to 3:48 as one scene, where exact sampling gives four). Which is better was not judged by eye. To get exact times
  back at the old speed, make `snap` return its argument; a tighter limit than half the interval is in between.
- Embedding, the photo path (`load_for`, `prepare`) and the JPEG quality (85) are untouched.

## Scene pictures made only for the chosen frames (second round)

Before, every sampled picture was resized to 1024 px and saved as a JPEG (about 24 ms of the ~33 ms of cutting work per picture),
and only the 40 or so scene pictures per video were kept. Now the scenes are chosen first, from the vectors alone; then
`video::pictures_at` decodes just the chosen pictures again (they are the same pictures: the exact times the vectors were made at,
usually key frames, so about 2 decoded pictures each) and each gets the same resize and JPEG (quality 85) as before. The result is the
same pictures by construction (same decoder, same frame, same resize function and quality); the files were not compared byte by byte.
`frames()` was split into `plan` (which groups hold the wanted times) and `run` (decode them side by side) so both calls share them.
`nearest_picture` and its test are gone (a scene's picture is found by its exact time now). If a chosen picture cannot be read again,
the video is left out and the map says so, as for any other unreadable video.

Timing on the same 16-minute video: 5.0 s and 5.2 s, from 5.7 s. Less than the cutting numbers suggest because decoding alone is
about 3 s of it, and the model start, scene grouping and finishing do not change.

## One picture decoded per sample (third round)

OpenH264 holds back the last picture until the next arrives, so a sample on a key frame used to decode the key frame and the one
after it. A group whose pictures show in the order they are coded (no B-frames; `Job::in_order`) now reads and feeds only the packets
up to the last wanted one, then asks the decoder for what it still holds. Groups with reordering are read and decoded as before.

Decoding alone on the 16-minute video: every 2 s 2.95 s -> 1.98 s, every 1 s 5.67 s -> 3.75 s. The whole map: 4.8 s and 4.6 s (from
5.0 s and 5.2 s). The scenes of the end-to-end test (every 4 s, 40 scenes) are identical to the run before, title for title.

## Left to do

- The picture is resized twice on its way to the model (to the size the model needs, then squeezed into its square by `prepare`).
  Going in one step from the full frame, as photos do from their scaled JPEG, saves about 5 ms per picture but changes the vectors
  slightly (so it was not done without asking).
- The decoder is OpenH264 (one thread per decoder); H.265 uses a pure-Rust decoder that was not measured.
