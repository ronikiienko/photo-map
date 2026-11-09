# Searching videos by their sound: what was tried (not in the app)

A test of searching a recording by what is said and heard in it, with EmbeddingGemma 2's audio encoder. It was then built
into the app (2026-11-05, kept on branch `sound-search`) and taken out again: on a map of phone videos (takeout-tagged, 2,488
videos) it found little, not even piano music reliably. Checked against the publisher's own code (sentence-transformers, float32
on the processor) on 10 of those videos: windows of 10 s matched (cosine 0.998-1.002 after float16), so it was the model, not
the setup; clips under 10 s were padded with silence where the publisher's code takes them at their length (cosine 0.87-0.97).

## Choosing a model

- **EmbeddingGemma 2** (Google, Apache 2.0): text, images, video and sound in one 768-number space. The audio encoder is ~300M
  (a conformer, 12 layers, width 1024, likely Gemma 4's speech encoder); its 25 tokens a second then go through the 130M text
  backbone. 16 kHz mono in, up to ~327 s per pass (8,192 tokens). ONNX export: `onnx-community/embeddinggemma-2-ONNX` (fp32
  audio encoder 1.17 GB, int8 ~580 MB). The model card: run in bf16 or fp32, **fp16 gives NaN or quietly worse vectors**.
  On speech retrieval (MSEB, MRR@10, Google's numbers) its own audio embedding 69.5 against 57.6 for speech-to-text then
  EmbeddingGemma's text.
- **CLAP** (LAION, Microsoft; 2023): small (audio part ~30-150M), ONNX exports exist; good on everyday sounds (rain, dogs,
  engines), near random on speech (MAEB). The cheap choice if only sounds matter.
- **Meta PE-AV** (Dec 2025, Apache 2.0): sound, video frames and text together, strong reported retrieval; PyTorch only, no
  ONNX found.
- MAEB's leaders (LCO-Embedding-Omni-7B, Qwen2-Audio-7B) are 7B: too big for a 4 GB card.
- For pictures EmbeddingGemma 2 was dropped (slightly better than SigLIP 2 Base 256, a tenth of the speed: commit 875fbdb).
  For sound the reason to try it was speech, which the sound-only models can't follow.

## The test

A 2-hour recording: a talk, a break with chatter, then short spoken presentations, several about the same made-up
product name. PyTorch, bf16, on the RTX 3050 Ti; queries with the `SearchQuery` prompt, sound with none.

- **The feature extractor cuts sound at 30 s** (`max_length` 480,000 samples, `truncation=True` by default): longer windows
  are silently their first 30 s. Calling it with `truncation=False` gives the whole window.
- Speed (windows overlapping by half, so every second is embedded twice): 10 s windows 73 s for the 2 h (100x real time,
  1.4 GB of card memory), 30 s 81 s (90x, 1.5 GB), 2 min 110 s (67x), 5.3 min 167 s (44x, 2.3 GB): attention grows with length.
- Raw scores don't tell a real match from none (a made-up "a cat meowing" scored 0.70 like real hits); how far the best hit
  stands out from the rest of the recording (z) does: real topics 3-7, made-up ones (cat, dog, rock concert, football, pasta)
  1.8-3.5.

### Window size

Whisper large-v3-turbo's transcript was the reference (mentions timed by their place in the transcript's segments). Hits
among the top 10 within 15 s of where it is said (in brackets what chance would give):

| Query | Said | 10 s | 30 s | 2 min |
|---|---|---|---|---|
| the made-up product name | 14x | 10 (0.4) | 9 (0.5) | 7 (0.8) |
| a term of the talk | 4x | 7 (0.2) | 6 (0.3) | 5 (0.8) |
| a term said in passing | 2x | 7 (0.1) | 2 (0.1) | 2 (0.3) |
| the talk's main subject | 27x | 9 (1.3) | 8 (1.7) | 8 (2.8) |
| a theme of the presentations | 29x | 8 (1.0) | 9 (1.5) | 10 (2.5) |

- **10 s windows are best**: a short mention (said twice) is diluted in 30 s. Windows of 2 and 5.3 min blend the talk, the
  presentations and the chatter: at 5.3 min real topics no longer stand out from made-up ones (z 1.7-3.4 against 2.0-2.6).
  A hierarchy of short and long windows doesn't help.
- One 5.3 min embedding against the average of the 30 s ones inside it: cosine 0.95 (unrelated stretches 0.87), and they
  search about as badly. A whole-video vector, if ever wanted, can be the average.
- Paraphrases work: a term searched by its textbook name lands where it is only described in other words, and a description
  of the product finds the presentations without its name.
- Terms never said get hits anyway: mostly in the noisy break, the best of bad matches, at low z.

### Overlap

10 s windows starting every 10, 5 or 2.5 s (TensorRT vectors), counted as moments would be shown: the best window first, windows
overlapping one already picked dropped, top 10, nine queries:

| Windows start every | Hits within 15 s | Mentions reached | Words, median s after the window's start |
|---|---|---|---|
| 10 s (none) | 36 | 38 | 3.1 |
| 5 s (half) | 36 | 37 | 3.2 |
| 2.5 s (three quarters) | 34 | 38 | 3.4 |

- **No overlap is enough**, at half or a quarter of the work and storage. Playing from the window's start (or a second or two
  before) lands just before the words.
- More windows give made-up queries more chances to score high by luck: their best z 1.7-3.5 with none, 2.0-3.9 overlapped (real
  topics 2.7-6.9 either way).

## Speech to text, for comparison

Whisper large-v3-turbo (809M, fp16, greedy, 4 pieces of 30 s at a time; 8 with beam search ran out of the 4 GB): 2 h in 169 s
(43x real time), 1.7 GB of card memory, ~12,000 words.

- The sentences mostly survive; **names don't**: the made-up product name came out six different ways, mostly as real words
  that sound alike, and other brand names were mangled too. A text search for the name misses most of them; the audio search
  found all ten places.
- **On chatter and noise it invents text**: in the break, short made-up phrases repeated up to dozens of times. The usual fixes
  are a voice detector (Silero VAD) and not feeding each piece's text into the next. The embedding can't invent text; noise
  only shows as weak matches.

## Other languages, messy recordings

A few noisy phone recordings, mostly Russian with some Ukrainian: a conversation between two people, a party game with a host,
and a show with a crowd. 10 s windows, no prompt for the sound.

- **Language is not the barrier; clear speech is.** The game host's lines were found 10 of 10 in the top 10, asked in Russian,
  Ukrainian or English (chance 3). Where a crowd shouts far from the phone, hits are near chance.
- **It finds what a stretch is about, not an exact quote.** In the conversation, where every window is the same two people on
  the same subject, a 2-second phrase lands in the top 5 for 31% of phrases (chance 10%), English versions at chance. Phrases
  that carry the conversation's meaning ranked 1-3; filler was lost among the rest.
- **Single words do worse than sentences.** Swear words as queries land near swearing (5-6 of 10 against chance 2), but
  "swearing" or "profanity" as an idea does not (1-3, the same as a made-up "a cat meowing").
- **Whisper on these** (language found by itself) gets some clean lines right and much wrong: words that sound alike, whole
  minutes in one segment, and text from its training subtitles on noise ("Продолжение следует...", Japanese and Portuguese
  lines, one line repeated up to a dozen times). It writes few swear words, and none where the speech is unclear: subtitles
  are often cleaned.

## Speed: TensorRT

The published ONNX files (onnx-community/embeddinggemma-2-ONNX, fp32) on TensorRT through ONNX Runtime, against PyTorch bf16
through sentence-transformers; 10 s windows without overlap, batch 8, RTX 3050 Ti. Per window:

| | PyTorch bf16 | TensorRT |
|---|---|---|
| sound features (log-mel, processor) | 6.0 ms | hidden: 3 threads make the next batches while the card works |
| audio encoder (305M) | 27.0 ms | 9.1 ms (fp16, norms' input divided by 16) |
| text model (271M) | 16.4 ms | 8.4 ms (bf16) |
| whole | 51 ms, 195x real time | 18 ms, **545x** |

The vectors agree with PyTorch's (cosine 0.9996 on average, worst 0.9975; 9.5 of each window's 10 nearest windows the same).
An hour of sound takes ~6.6 s; 56 hours of video ~6 min. Engines are built once per input shape (~1.5-3.5 min; audio ~800 MB,
text ~520 MB).

- **The pieces.** The audio encoder file (`audio_encoder.onnx`) is the audio tower plus its projection into the text model
  (512 wide): 128 log-mel bands at 100 frames/s in, 25 tokens/s out, padding dropped. The text model (`model.onnx`) takes the
  token ids, the attention mask and the audio tokens (and image and video tokens, taken out of the graph for TensorRT, which
  can't take empty tensors), and gives the pooled 768-number vector.
- **What TensorRT can't read**, written with standard operators (opset 23): `SimplifiedLayerNormalization` (109 in the audio
  encoder, 98 in the text model) -> `RMSNormalization`; `SkipSimplifiedLayerNormalization` (48) -> Add + `RMSNormalization`;
  ONNX Runtime's `RotaryEmbedding` (48) -> rotate half, times cos and sin (sliding-window layers have 256-wide heads, the
  full-attention ones 512).
- **One float `Range` kept the whole text model in fp32.** The word positions are counted in float; under bf16 TensorRT can't
  make that fill (an error in the log, then fp32 for everything: 15.6 ms, a 1 GB engine, the same numbers as CUDA fp32).
  Counted in int64 and converted after: 8.4 ms, a 520 MB engine.
- **fp16 against bf16.** On this card both are equally fast in principle, but TensorRT has more fp16 kernels: the audio encoder
  in bf16 took 21 ms, in fp16 9.1 ms. Plain fp16 overflows in the norms (vectors wrong, cosine of the mean 0.70); each norm's
  input divided by 16 and epsilon by 256 (the same result) fixes it, as it did for Gemma's photos. The encoder clamps its values
  everywhere (216 `Clip`s, ±9 to ±35), so only the norms' squares could overflow. The text model in fp16, with the same fix,
  gives infinities (Gemma's running sums between layers are large): bf16 stays (fp16 would have been 6.8 ms).
- **Doesn't help:** batch 16 (551x against 545x). ONNX Runtime on plain CUDA is slower than PyTorch bf16 (fp32: audio 59 ms,
  text 21 ms).
- **Odds and ends:** the HF cache keeps files as symbolic links and ONNX refuses them (and hard links when it reads weights
  itself); ONNX Runtime finds TensorRT only with the venv's `tensorrt_libs`, `nvidia/cu13/lib` and `nvidia/cudnn/lib` on
  `LD_LIBRARY_PATH`, and otherwise falls back to CUDA with only a printed warning. A 10 s window is 999 feature frames, not 1000.
