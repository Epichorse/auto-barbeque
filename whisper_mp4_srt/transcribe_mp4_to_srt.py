#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

MPS_DEVICE = "mps"
VAD_SAMPLE_RATE = 16_000
CLAUSE_STARTERS = {
    "and",
    "but",
    "because",
    "so",
    "then",
    "when",
    "while",
    "although",
    "however",
    "therefore",
    "which",
    "who",
    "that",
}
SPLIT_AVOID_END_WORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "and",
    "or",
    "but",
    "as",
    "if",
    "we",
    "i",
    "you",
    "it",
    "he",
    "she",
    "they",
    "this",
    "that",
    "these",
    "those",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read MP4 files and generate English SRT subtitles with Whisper large-v3 "
            "(MPS-only for Apple Silicon)."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input MP4 file or directory containing MP4 files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to write SRT files. Default: ./output",
    )
    parser.add_argument(
        "--model-name",
        default="large-v3",
        help="Whisper model name. Default: large-v3",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models/whisper"),
        help="Local cache directory for Whisper model weights.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="If input is a directory, only scan top-level files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SRT files.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/load model weights locally, then exit.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Print MPS environment diagnostics and exit.",
    )
    parser.add_argument(
        "--max-chars-per-line",
        type=int,
        default=52,
        help="Maximum characters per subtitle line. Default: 52",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1,
        help="Maximum lines per subtitle cue. Default: 1",
    )
    parser.add_argument(
        "--max-cue-duration",
        type=float,
        default=6.0,
        help="Maximum duration (seconds) per subtitle cue. Default: 6.0",
    )
    parser.add_argument(
        "--min-words-per-cue",
        type=int,
        default=3,
        help="Minimum words per subtitle cue whenever possible. Default: 3",
    )
    parser.add_argument(
        "--pause-split",
        type=float,
        default=0.7,
        help="Split cue when silence between words exceeds this value (seconds). Default: 0.7",
    )
    parser.add_argument(
        "--disable-vad-split",
        action="store_true",
        help="Disable WebRTC VAD-based split points.",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        help="WebRTC VAD aggressiveness (0-3). Higher means stricter speech detection. Default: 2",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=20,
        help="WebRTC VAD frame size in milliseconds. Must be 10, 20, or 30. Default: 20",
    )
    parser.add_argument(
        "--vad-min-pause",
        type=float,
        default=0.12,
        help="Minimum non-speech duration (seconds) to create a VAD split point. Default: 0.12",
    )
    parser.add_argument(
        "--vad-bridge-gap",
        type=float,
        default=0.08,
        help="Fill tiny non-speech gaps (seconds) between speech frames before splitting. Default: 0.08",
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Whisper temperature schedule. Use comma-separated values. Default: 0,0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Enable conditioning on previous text. Disabled by default to reduce repetition loops.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Whisper no_speech_threshold. Lower values suppress likely non-speech hallucinations more aggressively. Default: 0.6",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Whisper logprob_threshold used with no_speech_threshold. Default: -1.0",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Whisper compression_ratio_threshold to suppress repetitive output. Default: 2.4",
    )
    return parser.parse_args()


def ensure_mps_available() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "This project is MPS-only. It must run on macOS with Apple Silicon."
        )

    if platform.machine().lower() != "arm64":
        raise RuntimeError(
            "This project is MPS-only. Please use arm64 Python on Apple Silicon."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'torch'. Run: pip install -r requirements.txt"
        ) from exc

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())
    if not mps_built or not mps_available:
        raise RuntimeError(
            "MPS is not available in current PyTorch runtime. "
            f"is_built={mps_built}, is_available={mps_available}."
        )

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def run_self_check() -> int:
    print(f"[check] platform: {sys.platform}")
    print(f"[check] machine: {platform.machine()}")
    print(f"[check] python: {sys.version.split()[0]}")
    print(f"[check] ffmpeg_in_path: {shutil.which('ffmpeg') is not None}")

    try:
        import torch
    except ImportError:
        print("[check] torch_installed: False")
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        mps_built = bool(mps_backend and mps_backend.is_built())
        mps_available = bool(mps_backend and mps_backend.is_available())
        print("[check] torch_installed: True")
        print(f"[check] torch_version: {getattr(torch, '__version__', 'unknown')}")
        print(f"[check] mps_is_built: {mps_built}")
        print(f"[check] mps_is_available: {mps_available}")

    try:
        import whisper
    except ImportError:
        print("[check] whisper_installed: False")
    else:
        print("[check] whisper_installed: True")
        print(f"[check] whisper_version: {getattr(whisper, '__version__', 'unknown')}")

    try:
        ensure_mps_available()
    except Exception as exc:
        print(f"[check] mps_ready: False ({exc})")
        return 1

    print("[check] mps_ready: True")
    return 0


def collect_mp4_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mp4":
            raise ValueError(f"Input file is not MP4: {input_path}")
        return [input_path.resolve()]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    iterator: Iterable[Path]
    if recursive:
        iterator = input_path.rglob("*")
    else:
        iterator = input_path.glob("*")

    files = [p.resolve() for p in iterator if p.is_file() and p.suffix.lower() == ".mp4"]
    return sorted(files)


def output_path_for(mp4_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / f"{mp4_path.stem}.srt"

    relative = mp4_path.relative_to(input_root)
    return output_dir / relative.with_suffix(".srt")


def srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = srt_timestamp(float(seg["start"]))
            end = srt_timestamp(float(seg["end"]))
            text = str(seg["text"]).strip()
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+'([A-Za-z])", r"'\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wrap_text(text: str, max_chars_per_line: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def extract_timed_words(segments: list[dict]) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    for seg_idx, seg in enumerate(segments):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_words = seg.get("words") or []
        valid_count = 0
        for word in seg_words:
            try:
                token = str(word.get("word", ""))
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not token.strip() or end <= start:
                continue
            if not token.startswith(" "):
                token = f" {token}"
            words.append({"text": token, "start": start, "end": end, "seg_id": seg_idx})
            valid_count += 1
        if valid_count == 0 and seg_end > seg_start:
            fallback_text = normalize_text(str(seg.get("text", "")))
            if fallback_text:
                token_list = fallback_text.split()
                if not token_list:
                    continue
                duration = seg_end - seg_start
                total_weight = sum(max(1, len(tok)) for tok in token_list)
                elapsed = seg_start
                consumed_weight = 0
                for idx, token in enumerate(token_list):
                    weight = max(1, len(token))
                    consumed_weight += weight
                    next_elapsed = seg_start + duration * (consumed_weight / total_weight)
                    if idx == len(token_list) - 1:
                        next_elapsed = seg_end
                    text_value = f" {token}"
                    words.append(
                        {
                            "text": text_value,
                            "start": elapsed,
                            "end": next_elapsed,
                            "seg_id": seg_idx,
                        }
                    )
                    elapsed = next_elapsed
    return words


def split_for_srt(
    segments: list[dict],
    max_chars_per_line: int,
    max_lines: int,
    max_cue_duration: float,
    min_words_per_cue: int,
    pause_split: float,
    silence_boundaries: list[float] | None = None,
) -> list[dict[str, Any]]:
    words = extract_timed_words(segments)
    if not words:
        return []
    silence_points = sorted(silence_boundaries or [])

    cues: list[dict[str, Any]] = []

    def cue_text(items: list[dict[str, float | str]]) -> str:
        raw = "".join(str(it["text"]) for it in items)
        return normalize_text(raw)

    def cue_words(items: list[dict[str, float | str]]) -> int:
        return max(0, len(items))

    def choose_split_index(
        items: list[dict[str, float | str]],
        target_chars: int,
        min_words: int = 3,
    ) -> int:
        if len(items) <= 1:
            return 1
        if target_chars <= 0:
            return max(1, len(items) // 2)

        cumulative = 0
        best_idx = max(1, len(items) // 2)
        best_score = -10**9
        total = len(items)

        for i in range(1, total):
            prev_token = str(items[i - 1]["text"]).strip()
            next_token = str(items[i]["text"]).strip().lower()
            cumulative += len(prev_token) + 1

            left_words = i
            right_words = total - i
            if total >= min_words * 2 and (left_words < min_words or right_words < min_words):
                continue

            gap = float(items[i]["start"]) - float(items[i - 1]["end"])
            score = -abs(cumulative - target_chars)
            if has_boundary_between(
                float(items[i - 1]["end"]),
                float(items[i]["start"]),
                silence_boundaries,
            ):
                score += 240

            if prev_token.endswith((".", "!", "?")):
                score += 180
            elif prev_token.endswith((";", ":")):
                score += 130
            elif prev_token.endswith(","):
                score += 80

            prev_lower = prev_token.lower().strip(".,;:!?")
            if prev_lower in SPLIT_AVOID_END_WORDS:
                score -= 120

            if next_token in SPLIT_AVOID_END_WORDS:
                score -= 115

            if next_token in CLAUSE_STARTERS:
                score += 95

            if gap > pause_split:
                score += 140
            elif gap > pause_split * 0.6:
                score += 60

            if items[i - 1].get("seg_id") != items[i].get("seg_id"):
                score += 170

            if left_words <= 2 or right_words <= 2:
                score -= 35

            if score > best_score:
                best_score = score
                best_idx = i

        return max(1, min(total - 1, best_idx))

    def push(items: list[dict[str, float | str]]) -> None:
        text = cue_text(items)
        if not text:
            return
        lines = wrap_text(text, max_chars_per_line)
        if len(lines) > max_lines and len(items) > 1:
            target_chars = max_chars_per_line * max_lines
            split_idx = choose_split_index(
                items,
                target_chars=target_chars,
                min_words=min_words_per_cue,
            )
            push(items[:split_idx])
            push(items[split_idx:])
            return

        start = float(items[0]["start"])
        end = float(items[-1]["end"])
        if end <= start:
            end = start + 0.2
        cues.append({"start": start, "end": end, "text": "\n".join(lines)})

    current: list[dict[str, float | str]] = []
    target_chars = max_chars_per_line * max_lines
    hard_limit_chars = int(target_chars * 1.6)
    silence_idx = 0

    for word in words:
        if not current:
            current = [word]
            continue

        if current[-1].get("seg_id") != word.get("seg_id"):
            current_text = cue_text(current)
            current_duration = float(current[-1]["end"]) - float(current[0]["start"])
            if len(current_text) >= 18 or current_duration >= 1.0:
                push(current)
                current = [word]
                continue

        gap = float(word["start"]) - float(current[-1]["end"])
        candidate = current + [word]
        candidate_text = cue_text(candidate)
        candidate_lines = wrap_text(candidate_text, max_chars_per_line)
        candidate_duration = float(candidate[-1]["end"]) - float(candidate[0]["start"])
        candidate_chars = len(candidate_text)

        while silence_idx < len(silence_points) and silence_points[silence_idx] <= float(current[0]["start"]) + 1e-4:
            silence_idx += 1
        split_by_silence = (
            silence_idx < len(silence_points)
            and silence_points[silence_idx] <= float(word["end"]) + 1e-4
        )
        should_split = (
            split_by_silence
            or gap >= pause_split
            or candidate_duration > max_cue_duration
            or candidate_chars > hard_limit_chars
        )
        if should_split and len(current) < min_words_per_cue:
            can_delay_split = (
                candidate_duration <= max(max_cue_duration * 3.5, 6.0)
                and candidate_chars <= hard_limit_chars
                and gap <= max(0.9, pause_split * 5.0)
            )
            if can_delay_split:
                should_split = False

        if should_split:
            split_idx = 0
            if len(candidate) >= 4:
                split_idx = choose_split_index(
                    candidate,
                    target_chars=target_chars,
                    min_words=min_words_per_cue,
                )

            if 0 < split_idx < len(candidate):
                left = candidate[:split_idx]
                right = candidate[split_idx:]
                left_text = cue_text(left)
                left_duration = float(left[-1]["end"]) - float(left[0]["start"])
                left_words = cue_words(left)
                right_words = cue_words(right)
                if (
                    (left_words < min_words_per_cue or right_words < max(2, min_words_per_cue - 1))
                    and candidate_duration <= max(max_cue_duration * 2.5, 6.0)
                    and candidate_chars <= hard_limit_chars
                    and gap <= max(0.9, pause_split * 4.0)
                ):
                    current = candidate
                    continue
                if left_text and left_duration <= max(max_cue_duration * 2.0, 6.0):
                    push(left)
                    current = right
                else:
                    push(current)
                    current = [word]
            else:
                push(current)
                current = [word]

            if split_by_silence:
                current_end = float(current[-1]["end"]) if current else float(word["end"])
                while (
                    silence_idx < len(silence_points)
                    and silence_points[silence_idx] <= current_end + 1e-4
                ):
                    silence_idx += 1
        else:
            if len(candidate_lines) > max_lines:
                split_idx = choose_split_index(
                    candidate,
                    target_chars=target_chars,
                    min_words=min_words_per_cue,
                )
                if 0 < split_idx < len(candidate):
                    push(candidate[:split_idx])
                    current = candidate[split_idx:]
                else:
                    push(current)
                    current = [word]
            else:
                current = candidate

        token = str(word["text"]).strip()
        if current and token.endswith((".", "!", "?")):
            current_duration = float(current[-1]["end"]) - float(current[0]["start"])
            if current_duration >= 0.8:
                push(current)
                current = []

    if current:
        push(current)

    return merge_short_cues(
        cues,
        target_chars=target_chars,
        max_cue_duration=max_cue_duration,
        min_words_per_cue=min_words_per_cue,
        pause_split=pause_split,
        silence_boundaries=silence_boundaries,
    )


def merge_short_cues(
    cues: list[dict[str, Any]],
    target_chars: int,
    max_cue_duration: float,
    min_words_per_cue: int,
    pause_split: float,
    silence_boundaries: list[float] | None = None,
) -> list[dict[str, Any]]:
    if not cues:
        return []

    def edge_word(text: str, from_end: bool) -> str:
        parts = re.findall(r"[A-Za-z']+", text.lower())
        if not parts:
            return ""
        return parts[-1] if from_end else parts[0]

    merged: list[dict[str, Any]] = []
    i = 0
    soft_limit = int(target_chars * 1.2)
    hard_char_cap = target_chars + max(6, target_chars // 10)
    max_merge_duration = max_cue_duration * 1.25
    merge_gap_limit = min(0.5, max(0.2, pause_split * 0.75))

    while i < len(cues):
        cur = {
            "start": float(cues[i]["start"]),
            "end": float(cues[i]["end"]),
            "text": normalize_text(str(cues[i]["text"])),
        }

        while i + 1 < len(cues):
            nxt = {
                "start": float(cues[i + 1]["start"]),
                "end": float(cues[i + 1]["end"]),
                "text": normalize_text(str(cues[i + 1]["text"])),
            }

            cur_text = normalize_text(str(cur["text"]))
            nxt_text = normalize_text(str(nxt["text"]))
            if not nxt_text:
                i += 1
                continue

            cur_words = len(cur_text.split())
            nxt_words = len(nxt_text.split())
            cur_duration = float(cur["end"]) - float(cur["start"])
            gap = float(nxt["start"]) - float(cur["end"])
            combined_text = normalize_text(f"{cur_text} {nxt_text}")
            combined_duration = float(nxt["end"]) - float(cur["start"])
            cur_tail = edge_word(cur_text, from_end=True)
            nxt_head = edge_word(nxt_text, from_end=False)
            incomplete_tail = cur_tail in SPLIT_AVOID_END_WORDS
            weak_head = nxt_head in SPLIT_AVOID_END_WORDS

            should_try_merge = (
                cur_words < min_words_per_cue
                or nxt_words < max(2, min_words_per_cue - 1)
                or cur_words <= 4
                or cur_duration < 1.0
                or len(cur_text) < 20
                or (not cur_text.endswith((".", "!", "?", ";", ":")) and gap <= 0.35)
                or incomplete_tail
                or weak_head
            )

            if not should_try_merge:
                break

            merge_gap_cap = merge_gap_limit
            duration_cap = max_merge_duration
            char_cap = soft_limit
            if incomplete_tail or weak_head:
                merge_gap_cap = max(merge_gap_limit, 0.9, pause_split * 4.0)
                duration_cap = max(max_merge_duration, max_cue_duration * 1.9)
                char_cap = max(soft_limit, int(target_chars * 1.35))
            if cur_words < min_words_per_cue:
                merge_gap_cap = max(merge_gap_cap, 1.1)
                duration_cap = max(duration_cap, max_cue_duration * 4.0)
                char_cap = max(char_cap, int(target_chars * 1.8))
            char_cap = min(char_cap, hard_char_cap)

            boundary_block = has_boundary_between(
                float(cur["end"]),
                float(nxt["start"]),
                silence_boundaries,
            )
            if cur_words < min_words_per_cue and gap <= 0.35:
                boundary_block = False
            if (weak_head or incomplete_tail) and gap <= 0.35:
                boundary_block = False

            can_merge = (
                len(combined_text) <= char_cap
                and combined_duration <= duration_cap
                and gap <= merge_gap_cap
                and not boundary_block
            )
            if not can_merge:
                break

            cur = {
                "start": float(cur["start"]),
                "end": float(nxt["end"]),
                "text": combined_text,
            }
            i += 1

        merged.append(cur)
        i += 1

    if len(merged) <= 1:
        return merged

    compacted: list[dict[str, Any]] = []
    i = 0
    while i < len(merged):
        cur = {
            "start": float(merged[i]["start"]),
            "end": float(merged[i]["end"]),
            "text": normalize_text(str(merged[i]["text"])),
        }
        cur_words = len(cur["text"].split())
        if i + 1 < len(merged) and cur_words < min_words_per_cue:
            nxt = {
                "start": float(merged[i + 1]["start"]),
                "end": float(merged[i + 1]["end"]),
                "text": normalize_text(str(merged[i + 1]["text"])),
            }
            gap = float(nxt["start"]) - float(cur["end"])
            combined_text = normalize_text(f"{cur['text']} {nxt['text']}")
            combined_duration = float(nxt["end"]) - float(cur["start"])
            if (
                gap <= 0.45
                and len(combined_text) <= hard_char_cap
                and combined_duration <= max(max_merge_duration, max_cue_duration * 4.0)
            ):
                compacted.append(
                    {
                        "start": float(cur["start"]),
                        "end": float(nxt["end"]),
                        "text": combined_text,
                    }
                )
                i += 2
                continue

        compacted.append(cur)
        i += 1

    if not compacted:
        return compacted

    cleaned: list[dict[str, Any]] = []
    for idx, cue in enumerate(compacted):
        text = normalize_text(str(cue["text"]))
        words = re.findall(r"[A-Za-z']+", text)
        word_count = len(words)
        duration = float(cue["end"]) - float(cue["start"])
        prev_gap = (
            float(cue["start"]) - float(compacted[idx - 1]["end"])
            if idx > 0
            else 999.0
        )
        next_gap = (
            float(compacted[idx + 1]["start"]) - float(cue["end"])
            if idx + 1 < len(compacted)
            else 999.0
        )
        isolated_short = (
            word_count <= 2
            and duration <= 2.2
            and prev_gap >= 1.0
            and next_gap >= 1.0
        )
        sparse_short = word_count <= 2 and duration >= 4.0
        if isolated_short or sparse_short:
            continue
        cleaned.append(cue)

    return cleaned


def parse_temperature(value: str) -> tuple[float, ...]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("temperature schedule is empty")
    temps: list[float] = []
    for p in parts:
        t = float(p)
        if t < 0:
            raise ValueError("temperature must be >= 0")
        temps.append(t)
    return tuple(temps)


def has_boundary_between(
    start: float,
    end: float,
    boundaries: list[float] | None,
    tolerance: float = 1e-4,
) -> bool:
    if not boundaries:
        return False
    lo = min(start, end) - tolerance
    hi = max(start, end) + tolerance
    for b in boundaries:
        if lo < b <= hi:
            return True
    return False


def _read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def detect_vad_boundaries(
    media_path: Path,
    aggressiveness: int,
    frame_ms: int,
    min_pause: float,
    bridge_gap: float,
) -> list[float]:
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'webrtcvad'. Run: pip install -r requirements.txt"
        ) from exc

    if frame_ms not in (10, 20, 30):
        raise ValueError("vad_frame_ms must be one of 10, 20, 30")
    if aggressiveness < 0 or aggressiveness > 3:
        raise ValueError("vad_aggressiveness must be between 0 and 3")

    samples_per_frame = VAD_SAMPLE_RATE * frame_ms // 1000
    bytes_per_frame = samples_per_frame * 2
    frame_sec = frame_ms / 1000.0
    min_pause_frames = max(1, int(round(min_pause / frame_sec)))
    bridge_frames = max(0, int(round(bridge_gap / frame_sec)))
    vad = webrtcvad.Vad(aggressiveness)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(VAD_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "-",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("failed to launch ffmpeg for VAD")

    flags: list[bool] = []
    try:
        while True:
            frame = _read_exact(proc.stdout, bytes_per_frame)
            if len(frame) < bytes_per_frame:
                break
            try:
                flags.append(bool(vad.is_speech(frame, VAD_SAMPLE_RATE)))
            except Exception:
                flags.append(False)
    finally:
        proc.stdout.close()

    stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = proc.wait()
    if return_code != 0:
        detail = stderr_text if stderr_text else f"exit code {return_code}"
        raise RuntimeError(f"ffmpeg audio decode failed for VAD: {detail}")

    if not flags or not any(flags):
        return []

    if bridge_frames > 0:
        smoothed = flags[:]
        i = 0
        while i < len(smoothed):
            if smoothed[i]:
                i += 1
                continue
            j = i
            while j < len(smoothed) and not smoothed[j]:
                j += 1
            run_len = j - i
            if i > 0 and j < len(smoothed) and smoothed[i - 1] and smoothed[j] and run_len <= bridge_frames:
                for k in range(i, j):
                    smoothed[k] = True
            i = j
        flags = smoothed

    first_voiced = next((i for i, v in enumerate(flags) if v), None)
    if first_voiced is None:
        return []
    last_voiced = len(flags) - 1 - next((i for i, v in enumerate(reversed(flags)) if v), 0)

    boundaries: list[float] = []
    i = first_voiced
    while i <= last_voiced:
        if flags[i]:
            i += 1
            continue
        j = i
        while j <= last_voiced and not flags[j]:
            j += 1
        if (j - i) >= min_pause_frames:
            start_t = i * frame_sec
            end_t = j * frame_sec
            boundaries.append((start_t + end_t) / 2.0)
        i = j

    return sorted(set(round(x, 3) for x in boundaries))


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg first, then rerun."
        )


def load_whisper_model(model_name: str, model_dir: Path):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'openai-whisper'. Run: pip install -r requirements.txt"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] loading {model_name} on {MPS_DEVICE}")
    print(f"[model] local weights dir: {model_dir}")
    # Download to local cache if needed, then move to MPS.
    # Some PyTorch+MPS builds cannot move sparse buffers directly.
    model = whisper.load_model(model_name, device="cpu", download_root=str(model_dir))
    alignment_heads = getattr(model, "alignment_heads", None)
    if alignment_heads is not None and bool(getattr(alignment_heads, "is_sparse", False)):
        model.register_buffer(
            "alignment_heads",
            alignment_heads.to_dense(),
            persistent=False,
        )
    return model.to(MPS_DEVICE)


def main() -> int:
    args = parse_args()

    if args.self_check:
        return run_self_check()

    if args.max_chars_per_line < 8:
        print("[error] --max-chars-per-line must be >= 8", file=sys.stderr)
        return 2
    if args.max_lines < 1:
        print("[error] --max-lines must be >= 1", file=sys.stderr)
        return 2
    if args.max_cue_duration <= 0:
        print("[error] --max-cue-duration must be > 0", file=sys.stderr)
        return 2
    if args.min_words_per_cue < 1:
        print("[error] --min-words-per-cue must be >= 1", file=sys.stderr)
        return 2
    if args.pause_split < 0:
        print("[error] --pause-split must be >= 0", file=sys.stderr)
        return 2
    if not (0.0 <= args.no_speech_threshold <= 1.0):
        print("[error] --no-speech-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.compression_ratio_threshold <= 0:
        print("[error] --compression-ratio-threshold must be > 0", file=sys.stderr)
        return 2
    if args.vad_aggressiveness < 0 or args.vad_aggressiveness > 3:
        print("[error] --vad-aggressiveness must be between 0 and 3", file=sys.stderr)
        return 2
    if args.vad_frame_ms not in (10, 20, 30):
        print("[error] --vad-frame-ms must be one of 10, 20, 30", file=sys.stderr)
        return 2
    if args.vad_min_pause < 0:
        print("[error] --vad-min-pause must be >= 0", file=sys.stderr)
        return 2
    if args.vad_bridge_gap < 0:
        print("[error] --vad-bridge-gap must be >= 0", file=sys.stderr)
        return 2
    try:
        temperatures = parse_temperature(args.temperature)
    except Exception as exc:
        print(f"[error] invalid --temperature: {exc}", file=sys.stderr)
        return 2

    if not args.download_only and args.input is None:
        print("error: input is required unless --download-only/--self-check is used", file=sys.stderr)
        return 2

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_path = args.input.expanduser().resolve() if args.input else None

    try:
        ensure_mps_available()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        model = load_whisper_model(args.model_name, model_dir)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.download_only:
        print("[ok] model is downloaded and loaded from local weights.")
        return 0

    try:
        ensure_ffmpeg()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    assert input_path is not None
    try:
        mp4_files = collect_mp4_files(input_path, recursive=not args.non_recursive)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if not mp4_files:
        print(f"[warn] no MP4 files found under: {input_path}")
        return 0

    print(f"[run] found {len(mp4_files)} MP4 file(s)")
    print(f"[run] output dir: {output_dir}")
    failures = 0

    for idx, mp4_path in enumerate(mp4_files, start=1):
        out_path = output_path_for(mp4_path, input_path, output_dir)
        if out_path.exists() and not args.overwrite:
            print(f"[skip] ({idx}/{len(mp4_files)}) {out_path} already exists")
            continue

        print(f"[work] ({idx}/{len(mp4_files)}) transcribing: {mp4_path}")
        try:
            vad_boundaries: list[float] = []
            if not args.disable_vad_split:
                vad_boundaries = detect_vad_boundaries(
                    mp4_path,
                    aggressiveness=args.vad_aggressiveness,
                    frame_ms=args.vad_frame_ms,
                    min_pause=args.vad_min_pause,
                    bridge_gap=args.vad_bridge_gap,
                )
                preview = ", ".join(f"{b:.2f}" for b in vad_boundaries[:5])
                print(
                    f"[vad] detected {len(vad_boundaries)} split point(s) "
                    f"(aggr={args.vad_aggressiveness}, frame={args.vad_frame_ms}ms, "
                    f"min_pause={args.vad_min_pause}, bridge={args.vad_bridge_gap})"
                    + (f": {preview}" if preview else "")
                )

            result = model.transcribe(
                str(mp4_path),
                task="transcribe",
                language="en",
                fp16=False,
                verbose=False,
                temperature=temperatures,
                condition_on_previous_text=args.condition_on_previous_text,
                no_speech_threshold=args.no_speech_threshold,
                logprob_threshold=args.logprob_threshold,
                compression_ratio_threshold=args.compression_ratio_threshold,
            )
            raw_segments = result.get("segments", [])
            cues = split_for_srt(
                raw_segments,
                max_chars_per_line=args.max_chars_per_line,
                max_lines=args.max_lines,
                max_cue_duration=args.max_cue_duration,
                min_words_per_cue=args.min_words_per_cue,
                pause_split=args.pause_split,
                silence_boundaries=vad_boundaries,
            )
            write_srt(cues, out_path)
            print(f"[done] wrote: {out_path}")
        except Exception as exc:
            failures += 1
            print(f"[fail] {mp4_path}: {exc}", file=sys.stderr)

    if failures:
        print(f"[summary] completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("[summary] completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
