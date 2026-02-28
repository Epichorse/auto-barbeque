#!/usr/bin/env python3
"""Parallel batch runner for work/ videos with non-empty English SRT files.

Workflow per item:
1) prepare tasks and extract frames
2) audit missing/invalid result.json
3) run pending cue ranges in parallel (non-overlapping ranges only)
4) retry audit/run for a few rounds
5) merge revised_zh into output SRT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BatchItem:
    slug: str
    video: str
    srt: str


BATCH_ITEMS: tuple[BatchItem, ...] = (
    BatchItem(
        slug="jzx100_500hp",
        video="500 HP Toyota JZX100 2JZ - FIRST DRIVE.mp4",
        srt="[English] 500 HP Toyota JZX100 2JZ - FIRST DRIVE [DownSub.com].srt",
    ),
    BatchItem(
        slug="bugi_350z",
        video="BUGI's New Drift Car - NISSAN 350Z _ Nightride.mp4",
        srt="BUGI's_New_Drift_Car_-_NISSAN_350Z_Nightride_En.srt",
    ),
    BatchItem(
        slug="bmw_e30_project",
        video="Building a Secret Project BMW E30 _ NIGHTRIDE.mp4",
        srt="Building_a_Secret_Project_BMW_E30_NIGHTRIDE_En.srt",
    ),
    BatchItem(
        slug="drift_corvette_roadtest",
        video="DRIFT CORVETTE  First Road Test _ NIGHTRIDE.mp4",
        srt="DRIFT_CORVETTE_First_Road_Test_NIGHTRIDE_En.srt",
    ),
    BatchItem(
        slug="fixing_c5",
        video="Fixing My New C5 Corvette _ NIGHTRIDE.mp4",
        srt="Fixing_My_New_C5_Corvette_NIGHTRIDE_En.srt",
    ),
    BatchItem(
        slug="survive_winter",
        video="Trying To Survive winter... _ NIGHTRIDE.mp4",
        srt="Trying_To_Survive_winter..._NIGHTRIDE_En.srt",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel runner for subtitle-enabled work videos.")
    parser.add_argument("--root", default=".", help="Project root path.")
    parser.add_argument("--model", default="gpt-5.2", help="Model name for codex exec.")
    parser.add_argument("--cue-workers", type=int, default=8, help="Parallel workers per attempt.")
    parser.add_argument("--max-attempts", type=int, default=4, help="Retry rounds for missing/invalid cues.")
    parser.add_argument(
        "--proxy",
        default="",
        help="Optional proxy URL for codex calls (e.g. http://127.0.0.1:7897).",
    )
    parser.add_argument(
        "--report-dir",
        default="out_final/work_batch",
        help="Directory for run reports and merged SRT output.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional slug filter, e.g. --only jzx100_500hp bugi_350z",
    )
    parser.add_argument(
        "--python-bin",
        default=".venv/bin/python",
        help="Python executable used to call barbeque_pipeline.py",
    )
    return parser.parse_args()


def _print_lock() -> threading.Lock:
    return threading.Lock()


PRINT_LOCK = _print_lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    log(f"[exit] {proc.returncode}")
    return proc.returncode


def parse_result(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        left = text.find("{")
        right = text.rfind("}")
        if left == -1 or right == -1 or right <= left:
            raise
        data = json.loads(text[left : right + 1])

    if not isinstance(data, dict):
        raise ValueError(f"Not object: {path}")
    if not isinstance(data.get("revised_zh"), str):
        raise ValueError(f"Missing revised_zh: {path}")
    return data


def iter_task_dirs(tasks_dir: Path) -> list[Path]:
    dirs = [p for p in tasks_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    dirs.sort(key=lambda p: int(p.name))
    return dirs


def audit_tasks(tasks_dir: Path) -> dict[str, object]:
    task_dirs = iter_task_dirs(tasks_dir)
    missing: list[int] = []
    invalid: list[int] = []
    valid = 0

    for d in task_dirs:
        cue_id = int(d.name)
        result_file = d / "result.json"
        if not result_file.exists():
            missing.append(cue_id)
            continue
        try:
            parse_result(result_file)
            valid += 1
        except Exception:
            invalid.append(cue_id)

    return {
        "total": len(task_dirs),
        "valid": valid,
        "missing": missing,
        "invalid": invalid,
    }


def split_ids_to_ranges(ids: Iterable[int], workers: int) -> list[tuple[int, int]]:
    unique_ids = sorted(set(ids))
    if not unique_ids:
        return []
    if workers <= 1:
        return [(unique_ids[0], unique_ids[-1])]

    chunk_size = max(1, math.ceil(len(unique_ids) / workers))
    ranges: list[tuple[int, int]] = []
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i : i + chunk_size]
        ranges.append((chunk[0], chunk[-1]))
    return ranges


def run_pending_parallel(
    *,
    python_bin: Path,
    pipeline: Path,
    tasks_dir: Path,
    schema: Path,
    model: str,
    ranges: list[tuple[int, int]],
    cwd: Path,
    env: dict[str, str],
    cue_workers: int,
) -> list[dict[str, int]]:
    max_workers = min(max(1, cue_workers), len(ranges))
    results: list[dict[str, int]] = []

    def _worker(start_id: int, end_id: int) -> dict[str, int]:
        rc = run_cmd(
            [
                str(python_bin),
                str(pipeline),
                "run",
                "--tasks-dir",
                str(tasks_dir),
                "--schema",
                str(schema),
                "--model",
                model,
                "--start-id",
                str(start_id),
                "--end-id",
                str(end_id),
            ],
            cwd=cwd,
            env=env,
        )
        return {"start_id": start_id, "end_id": end_id, "rc": rc}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, start_id, end_id) for start_id, end_id in ranges]
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda x: (x["start_id"], x["end_id"]))
    return results


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    work_dir = root / "work"
    pipeline = root / "barbeque_pipeline.py"
    schema = root / "schema.json"
    # Keep the venv path as-is (do not resolve symlink to system interpreter),
    # otherwise site-packages from the venv can be lost.
    python_bin = (root / args.python_bin).expanduser()
    report_dir = (root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not python_bin.exists():
        raise SystemExit(f"Python bin not found: {python_bin}")
    if not pipeline.exists():
        raise SystemExit(f"Pipeline not found: {pipeline}")
    if not schema.exists():
        raise SystemExit(f"Schema not found: {schema}")

    selected_items = list(BATCH_ITEMS)
    if args.only:
        wanted = set(args.only)
        selected_items = [it for it in selected_items if it.slug in wanted]

    proxy = args.proxy.strip()
    env = os.environ.copy()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env[key] = proxy

    report: dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "cue_workers": args.cue_workers,
        "max_attempts": args.max_attempts,
        "items": [],
    }

    for item in selected_items:
        log("=" * 88)
        log(f"[{item.slug}] START {datetime.now().isoformat(timespec='seconds')}")

        video = work_dir / item.video
        srt = work_dir / item.srt
        tasks_dir = work_dir / "_processed_zh" / item.slug / "tasks"
        out_srt = report_dir / f"{item.slug}.zh.srt"

        item_report: dict[str, object] = {
            "slug": item.slug,
            "video": str(video),
            "srt": str(srt),
            "tasks_dir": str(tasks_dir),
            "out_srt": str(out_srt),
        }

        if not video.exists():
            item_report["status"] = "skipped_missing_video"
            report["items"].append(item_report)
            log(f"[{item.slug}] skip: missing video")
            continue
        if not srt.exists():
            item_report["status"] = "skipped_missing_srt"
            report["items"].append(item_report)
            log(f"[{item.slug}] skip: missing srt")
            continue
        if srt.stat().st_size == 0:
            item_report["status"] = "skipped_empty_srt"
            report["items"].append(item_report)
            log(f"[{item.slug}] skip: empty srt")
            continue

        tasks_dir.mkdir(parents=True, exist_ok=True)

        rc_prepare = run_cmd(
            [
                str(python_bin),
                str(pipeline),
                "prepare",
                "--video",
                str(video),
                "--srt",
                str(srt),
                "--tasks-dir",
                str(tasks_dir),
                "--schema",
                str(schema),
            ],
            cwd=root,
            env=env,
        )
        item_report["prepare_rc"] = rc_prepare
        if rc_prepare != 0:
            item_report["status"] = "prepare_failed"
            report["items"].append(item_report)
            log(f"[{item.slug}] stop: prepare failed")
            continue

        attempts: list[dict[str, object]] = []
        done = False
        for attempt in range(1, args.max_attempts + 1):
            before = audit_tasks(tasks_dir)
            pending_ids = sorted(set(before["missing"]) | set(before["invalid"]))
            log(
                f"[{item.slug}] audit before attempt {attempt}: "
                f"total={before['total']} valid={before['valid']} "
                f"missing={len(before['missing'])} invalid={len(before['invalid'])}"
            )

            record: dict[str, object] = {
                "attempt": attempt,
                "before": {
                    "total": before["total"],
                    "valid": before["valid"],
                    "missing": len(before["missing"]),
                    "invalid": len(before["invalid"]),
                },
                "ranges": [],
                "run_results": [],
                "after": None,
            }

            if not pending_ids:
                done = True
                record["after"] = record["before"]
                attempts.append(record)
                break

            ranges = split_ids_to_ranges(pending_ids, args.cue_workers)
            record["ranges"] = [{"start_id": s, "end_id": e} for s, e in ranges]
            log(f"[{item.slug}] attempt {attempt} ranges: {len(ranges)}")

            run_results = run_pending_parallel(
                python_bin=python_bin,
                pipeline=pipeline,
                tasks_dir=tasks_dir,
                schema=schema,
                model=args.model,
                ranges=ranges,
                cwd=root,
                env=env,
                cue_workers=args.cue_workers,
            )
            record["run_results"] = run_results

            after = audit_tasks(tasks_dir)
            record["after"] = {
                "total": after["total"],
                "valid": after["valid"],
                "missing": len(after["missing"]),
                "invalid": len(after["invalid"]),
            }
            attempts.append(record)

            log(
                f"[{item.slug}] audit after attempt {attempt}: "
                f"total={after['total']} valid={after['valid']} "
                f"missing={len(after['missing'])} invalid={len(after['invalid'])}"
            )

            if len(after["missing"]) == 0 and len(after["invalid"]) == 0:
                done = True
                break

        final = audit_tasks(tasks_dir)
        item_report["attempts"] = attempts
        item_report["final"] = {
            "total": final["total"],
            "valid": final["valid"],
            "missing": len(final["missing"]),
            "invalid": len(final["invalid"]),
        }

        rc_merge = run_cmd(
            [
                str(python_bin),
                str(pipeline),
                "merge",
                "--srt",
                str(srt),
                "--tasks-dir",
                str(tasks_dir),
                "--out",
                str(out_srt),
                "--min-confidence",
                "0.7",
            ],
            cwd=root,
            env=env,
        )
        item_report["merge_rc"] = rc_merge
        item_report["status"] = (
            "ok"
            if done
            and rc_merge == 0
            and item_report["final"]["missing"] == 0
            and item_report["final"]["invalid"] == 0
            else "partial"
        )
        report["items"].append(item_report)
        log(f"[{item.slug}] DONE status={item_report['status']}")

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report_path = report_dir / f"run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("=" * 88)
    log(f"Report written: {report_path}")
    compact = {
        "report": str(report_path),
        "items": [
            {
                "slug": it["slug"],
                "status": it.get("status"),
                "final": it.get("final"),
                "out_srt": it.get("out_srt"),
            }
            for it in report["items"]
        ],
    }
    log(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
