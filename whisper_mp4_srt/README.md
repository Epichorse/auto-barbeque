# Whisper MP4 -> English SRT

This project reads MP4 files automatically and generates English SRT subtitles with `Whisper large-v3`.

Model weights are downloaded to a local directory and loaded from there on future runs.

This project is **MPS-only**:

- macOS
- Apple Silicon (arm64, e.g. M1 Pro)
- PyTorch MPS available

## 1. Setup

```bash
cd whisper_mp4_srt
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Install `ffmpeg` first:

- macOS (Homebrew): `brew install ffmpeg`

## 2. Download and load local Whisper large-v3 weights

```bash
python transcribe_mp4_to_srt.py --download-only
```

Environment self-check (MPS readiness):

```bash
python transcribe_mp4_to_srt.py --self-check
```

Default local weights directory:

`./models/whisper`

You can customize it:

```bash
python transcribe_mp4_to_srt.py --download-only --model-dir /your/local/model_dir
```

## 3. Generate English SRT from MP4

Single MP4:

```bash
python transcribe_mp4_to_srt.py /path/to/video.mp4 -o ./output
```

Directory batch (recursive):

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output
```

Directory batch (non-recursive):

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output --non-recursive
```

Overwrite existing SRT files:

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output --overwrite
```

Tune subtitle readability:

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output \
  --max-chars-per-line 52 \
  --max-lines 1 \
  --max-cue-duration 6.0 \
  --min-words-per-cue 3 \
  --pause-split 0.7
```

Tune decoding robustness (reduce repetitive wrong text):

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output \
  --temperature 0,0.2,0.4,0.6,0.8,1.0
```

For music-heavy intros / hallucination suppression:

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output \
  --no-speech-threshold 0.3 \
  --logprob-threshold -0.2 \
  --compression-ratio-threshold 2.0
```

Use VAD speech-pause points from audio to split cues:

```bash
python transcribe_mp4_to_srt.py /path/to/mp4_folder -o ./output \
  --pause-split 0.3 \
  --vad-aggressiveness 2 \
  --vad-frame-ms 20 \
  --vad-min-pause 0.12 \
  --vad-bridge-gap 0.08
```

## 4. Notes

- Language is forced to English (`language=en`).
- Model defaults to `large-v3`.
- Runtime device is fixed to `mps` (no CPU/CUDA fallback).
- For MPS compatibility, model is initialized on CPU then moved to MPS internally.
- SRT cues are auto re-segmented using punctuation, pauses, line length, and cue duration.
- Decoding defaults to a multi-temperature fallback and disables conditioning on previous text to reduce repetitive loops.
- Segmentation prefers punctuation/pauses/clause boundaries, and can inject boundaries from WebRTC VAD.
- On first run, model download is large (multi-GB), so it may take time.
