# auto-barbeque

English SRT -> contextual Chinese subtitle revision pipeline for automotive videos.

Core flow:

1. Parse SRT cues.
2. Extract `start/mid/end` frames per cue.
3. Call `codex exec` with strict JSON schema.
4. Merge `revised_zh` back into a new SRT while keeping original timestamps.

## Requirements

- Python 3.9+
- `codex` CLI available in `PATH`
- Frame extraction runtime:
  - recommended: system `ffmpeg`
  - fallback: `imageio-ffmpeg` from Python

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` only includes the fallback frame extractor dependency.
If you already have `ffmpeg`, you can still keep it installed.

## Pipeline Commands

### 1) Prepare tasks

```bash
python3 barbeque_pipeline.py prepare \
  --video "/path/to/video.mp4" \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --schema schema.json
```

Skip frame extraction when needed:

```bash
python3 barbeque_pipeline.py prepare \
  --video "/path/to/video.mp4" \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --schema schema.json \
  --skip-frames
```

### 2) Run Codex

```bash
python3 barbeque_pipeline.py run \
  --tasks-dir work/tasks \
  --schema schema.json \
  --model gpt-5.2 \
  --search
```

Offline mock mode:

```bash
python3 barbeque_pipeline.py run \
  --tasks-dir work/tasks \
  --schema schema.json \
  --mock
```

### 3) Merge revised subtitles

```bash
python3 barbeque_pipeline.py merge \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --out out_final/output.zh.srt \
  --min-confidence 0.6
```

### 4) One-shot (prepare + run + merge)

```bash
python3 barbeque_pipeline.py full \
  --video "/path/to/video.mp4" \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --schema schema.json \
  --out out_final/output.zh.srt \
  --model gpt-5.2 \
  --search
```

## Batch Runner

`run_work_batch_parallel.py` is a convenience runner for predefined items in `work/`.

```bash
python3 run_work_batch_parallel.py \
  --root . \
  --model gpt-5.2 \
  --cue-workers 8
```

If you need proxy for Codex requests:

```bash
python3 run_work_batch_parallel.py --proxy "http://127.0.0.1:7897"
```

## Output Layout

```text
work/
  tasks/
    manifest.json
    000001/
      start.png
      mid.png
      end.png
      meta.json
      prompt.txt
      result.json
      codex.stderr.log
```

`merge` keeps cue timings unchanged and only replaces subtitle text from `result.json.revised_zh` when available.

## Repository Hygiene

Large local artifacts are ignored by default (`work/`, `out_final/`, model weights, videos, zip files, venvs).
See `.gitignore` and `OPEN_SOURCE_CHECKLIST.md` before pushing.

## Related Tooling

`whisper_mp4_srt/` contains a separate MP4->English SRT tool powered by Whisper.

## License

MIT (see `LICENSE`).
