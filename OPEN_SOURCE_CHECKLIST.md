# Open Source Checklist

## 1) Verify ignore rules

```bash
git status --short
```

Expect **no** large runtime files under:

- `work/`
- `out_final/`
- `models/`
- `whisper_mp4_srt/models/`

## 2) Quick large-file scan (safety)

```bash
find . -type f -size +50M | sort
```

Anything in the output should usually be ignored before commit.

## 3) Validate scripts

```bash
python3 -m py_compile barbeque_pipeline.py run_work_batch_parallel.py
python3 barbeque_pipeline.py --help
```

## 4) Initialize git (if needed)

```bash
git init
git add .
git status
```

## 5) First commit

```bash
git commit -m "chore: prepare repository for open-source release"
```

## 6) Push to GitHub

```bash
git branch -M main
git remote add origin git@github.com:<your-org-or-user>/<repo>.git
git push -u origin main
```

## 7) GitHub repo settings (recommended)

- Add repository description/topics.
- Enable Issues and Discussions as needed.
- Add branch protection for `main`.
- Add a short "How to contribute" section in repo homepage.
