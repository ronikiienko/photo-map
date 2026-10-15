# Video (src/video.rs, src/scenes.rs)

Not yet part of a map: these are the two parts a map of videos is made from. What is missing is in "Open".

## Reading frames (`video::frames`)

- MP4 and MOV files, H.264 (`openh264`, C++ built by the crate) or H.265/HEVC (`rusty_h265`, pure Rust; Main and Main 10, 4:2:0), the
  container read by `re_mp4`. Any other codec or container is refused by name ("this video's codec (vp09) can't be read yet").
- One frame every N seconds, handed to a callback. Phone rotation is applied.
- A video is cut at its key frames into groups that can be decoded alone. A group with no wanted frame in it is skipped, and the
  groups that are wanted are decoded side by side (rayon), a few at a time. The cost is what lies between a key frame and the frame
  wanted, so it depends on how far apart the key frames are: a camera clip (key frame every second or two) is nearly free, a screen
  recording with a key frame every 30 s is the worst case.
- Checked (M1): colours and timing against ffmpeg (PSNR 45 to 46 dB on H.264, the saved JPEG being the difference; an HEVC frame
  looks the same). meow.mp4, 169 s 1080p H.264, every 5 s: 34 frames in 0.5 s. HEVC 1080p, 59 s, every 5 s: 12 frames in 2.4 s (the pure Rust
  decoder is slower). Lecture 8.mp4, 55 minutes 1920x1200 with key frames 30 s apart, every 30 s: 149 s (419 s before the groups were decoded in parallel).
- Not done: other codecs (VP9, AV1 in WebM or MP4), other containers (MKV, AVI), hardware decoding, 10-bit H.264, HEVC range extensions.

## Scenes (`scenes::split`)

From the vectors of the frames: a scene lasts until a frame's cosine to the scene's average falls under a threshold; scenes shorter
than 3 s are joined to the neighbour they resemble, scenes that a blink had split are joined again, and a cap on the count joins the
most alike neighbours. A scene has an average vector (unit length), its time span, and the frame nearest the average to stand for it.

The threshold depends on the model and the footage. SigLIP 256, a frame every 2 s: an animated film (Big Buck Bunny, 59 s) is one
scene at 0.85, 3 at 0.9, 8 at 0.95; a continuous game recording is one scene at 0.85 and 0.9. Default 0.9; it needs a setting.
`cargo test --release real_video_scenes -- --ignored --nocapture` (PHOTO_MAP_VIDEO, PHOTO_MAP_EVERY, PHOTO_MAP_THRESHOLD) prints the scenes
and saves each one's picture in .cache/test/scenes/.

## Open

- Putting scenes in a map: the build steps (Find, Copy, Understand, ...) know photos only. A scene needs an entity that refers to a
  video and a time span, a picture for the map's tiles, a search result that opens the video at the time, and re-use of vectors when
  a map is made again.
- Settings: the frame interval, the threshold, the most scenes per video (and an estimate of the time from the model's speed).
- Audio: not looked at (see the notes on EmbeddingGemma 2 in docs/models.md).
