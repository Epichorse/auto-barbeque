# Contributing

## Scope

This repository focuses on subtitle-processing pipeline code.
Large media/model/output files are intentionally excluded from source control.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Before opening a PR

Run basic checks:

```bash
python3 -m py_compile barbeque_pipeline.py run_work_batch_parallel.py
python3 barbeque_pipeline.py --help >/dev/null
```

If you touched `whisper_mp4_srt/`, run:

```bash
python3 -m py_compile whisper_mp4_srt/transcribe_mp4_to_srt.py
```

## Pull request rules

- Keep changes focused and minimal.
- Add/adjust README usage examples when behavior changes.
- Do not commit raw videos, generated tasks, or model weights.
- Include a short validation note in PR description (what was run and expected output).
