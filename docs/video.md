# Video (src/video.rs, src/scenes.rs, the Videos stage in src/build.rs)

Videos in a folder become scenes on the map. **Read the assumptions first: each is a choice that can be changed, and each is said to
the user in the New map form.**

## Assumptions (what a map with videos takes for granted)

1. **Only what the pictures show is used. The sound is ignored.** (Audio is not looked at at all; see "Open".)
2. **Which videos**: MP4, MOV and M4V files holding H.264 or H.265. Anything else (MKV, AVI, WebM, VP9, AV1, ProRes...) is left out,
   counted in the form ("3 videos can't be read") and listed in the map's details and in `videos.json`.
3. **A picture is taken every 2 s** (1, 2, 5 or 10 in the form) and understood by the map's model like a photo, so each costs as much
   as a photo. The setting is fixed for the map.
4. **A scene is a run of pictures that look alike**: a picture starts a new scene when its vector's cosine to the scene so far is
   under 0.5 once the video's own average is taken away (see "Scenes"); scenes under 3 s are joined to the neighbour they resemble; a
   video makes **at most 40 scenes** (the most alike neighbours are joined). The numbers are guesses; a long steady shot should be one scene.
5. **A scene is one entity of the map, like a photo**: its picture is the frame nearest the scene's average (JPEG, 1024 px on the long
   side, in `originals/`), its vector is the scene's average vector, its title is "▶ name, 0:08 to 0:20". It is laid out, put in
   regions, named and searched exactly as photos are. Scenes come after the photos in `photos.json`.
6. **No playback**: the map shows the picture, the video's name and the time. The video's path and times are in the photo's `meta`
   (`video`, `from`, `to`, `pictures`).
7. **Videos are not copied or linked into the map.** Move or delete one and its scenes stay on the map but point at nothing.
8. **Videos are never reused from earlier maps** (photos are, by file): a bigger map of the same videos reads them again.
9. **A pause in the middle of a video loses that video's progress** (the finished videos are kept); the video is read again.
10. `1 in N` ("How many") applies to photos only.

## How a build does it

- Scan counts photos only; the form counts both (`home::count`: videos by their headers, so it also knows their length and which it
  can't read). A folder of only videos is a map.
- **Videos stage** (between Copy and Embed; shown in the list of steps only for maps with videos): each video is read by
  `video::frames`, its pictures kept as small JPEGs in `videos/<k>/pictures/`, understood four at a time, split with
  `scenes::split`; the scene pictures and vectors are kept in `videos/<k>/` (`done.json` marks a video finished, or says why it was
  left out). When all are done the scenes are appended to `photos.json` (the first N entries, N = the photos Copy wrote, are kept, so
  the stage can run again), their pictures to `originals/`, and their vectors go to `scene_vectors.bin`.
- **Embed stage** treats the scenes as already embedded: their vectors come from `scene_vectors.bin`, no model run.
- Finish keeps `videos.json` (counts, videos left out with the reason) in the map, adds `videos` to the manifest, and removes the working
  files.
- Progress is in seconds of video; the card shows the pictures going by, as for photos.

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

From the vectors of the frames, **with the video's own average vector taken away from every frame first**. A model's vectors for
pictures all share a large part (two unrelated frames are 0.7 to 0.8 alike), and the average of many frames is mostly that part, so
compared as they are a new frame always looks like the scene so far: a film full of cuts was one scene at any threshold under 0.9
(SigLIP 256 and Gemma alike), and a threshold had to be found per model. After centring, a scene lasts until a frame's cosine to the
scene's average falls under 0.5; scenes shorter than 3 s are joined to the neighbour they resemble (a short blink between two alike runs
joins all three); a cap on the count joins the most alike neighbours. A scene has an average vector (of the frames as they were, unit
length), its time span, and the frame nearest the average to stand for it.

Looked at on real footage, a frame every 2 s (the 0.5 is not tuned beyond this):

| footage | model | scenes | how they look |
|---|---|---|---|
| Big Buck Bunny, 59 s, cuts every few seconds | SigLIP 256 / Gemma | 11 / 10 | distinct shots (the tree, the rabbit standing, close-ups...); a returning shot is two scenes |
| a 169 s game recording, scenery always changing | SigLIP 256 | 23 | each a different stretch of level; a fast section is 4 s scenes |
| a 55 minute lecture recording, a frame every 15 s | SigLIP 256 | 43 (the cap makes it 40) | one to two minutes each, where the screen changes |

At 0 the film is 5 scenes, at 0.8 it is 11: the threshold is a "how fine" knob, now with a meaning that does not depend on the model.
**Not checked**: a long steady single shot (centring makes small differences count for more; the lecture is the nearest thing), a few
hundred frames of footage with nothing in it, and footage the models see differently (night, screens, people).
`cargo test --release real_video_scenes -- --ignored --nocapture` (PHOTO_MAP_VIDEO, PHOTO_MAP_EVERY, PHOTO_MAP_THRESHOLD,
PHOTO_MAP_MODEL = s256, s512 or gemma) prints how alike neighbouring frames are and how many scenes each threshold gives, and saves each
scene's picture in .cache/test/scenes/. A job started before centring has 0.9 in its job.json, which is far stricter now.

## Open

- Playing a scene: in the map (needs a player: the decoder is there, a renderer and sound are not) or in the system's player at the
  time (opening a file with its default program does not seek).
- The threshold and the cap are not in the form (only the interval is); the threshold depends on the model and the footage.
- Time left for the Videos stage is a percentage, not an estimate from the model's speed.
- Audio: not looked at (see the notes on EmbeddingGemma 2 in docs/models.md).
- Reusing a video's pictures when a map is made again; resuming inside a video.
- Other codecs and containers; hardware decoding (needs unsafe bindings to the system's decoder).
- A scene's frames are not kept (only its typical one), so a scene can't be shown as a strip.
