#!/usr/bin/env python3
"""
WhisperX + guide-rule subtitle segmentation (single-line, DP global optimization).

Pipeline:
1) WhisperX transcription + alignment to get word-level timestamps
2) Optional punctuation restoration on aligned words
3) Dynamic-programming segmentation with hard constraints:
   - duration in [min_duration, max_duration]
   - single line
   - max chars per cue
   - max CPS
4) Export SRT + JSON + validation report
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import torch
except Exception:  # pragma: no cover - runtime-dependent
    torch = None


MIN_DURATION_DEFAULT = 5.0 / 6.0
MAX_DURATION_DEFAULT = 7.0
MAX_CHARS_DEFAULT = 42
MAX_CPS_DEFAULT = 20.0

SENT_END_RE = re.compile(r"[.!?]+$")
CLAUSE_PUNCT_RE = re.compile(r"[,;:]+$")
TOKEN_STRIP_RE = re.compile(r"^[\s\"'`([{<]+|[\s\"'`)\]}>.,!?;:]+$")
NON_WORD_EDGE_RE = re.compile(r"^[^\w']+|[^\w']+$")

CONJUNCTIONS = {
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "because",
    "although",
    "though",
    "while",
    "if",
    "when",
    "unless",
    "until",
    "since",
    "then",
}

PREPOSITIONS = {
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "to",
    "into",
    "onto",
    "with",
    "without",
    "about",
    "over",
    "under",
    "before",
    "after",
    "between",
    "through",
    "during",
    "across",
    "behind",
    "inside",
    "outside",
}

DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "some",
    "any",
    "each",
    "every",
    "either",
    "neither",
    "another",
}

SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
NEGATIONS = {"not", "n't", "never", "no"}
AUXILIARIES = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
}
PHRASAL_PARTICLES = {"up", "down", "out", "off", "in", "on", "away", "back", "over"}

COMMON_VERBS = {
    "be",
    "have",
    "do",
    "say",
    "go",
    "get",
    "make",
    "know",
    "think",
    "take",
    "see",
    "come",
    "want",
    "look",
    "give",
    "use",
    "find",
    "tell",
    "work",
    "call",
    "try",
    "ask",
    "need",
    "feel",
    "become",
    "leave",
    "put",
    "mean",
}

FILLER_SINGLE = {
    "um",
    "uh",
    "hmm",
    "ah",
    "er",
    "eh",
    "like",
    "basically",
    "actually",
    "literally",
    "honestly",
    "well",
}

PAUSE_WORDS = {
    "well",
    "so",
    "now",
    "okay",
    "ok",
    "anyway",
    "right",
    "listen",
    "look",
    "alright",
    "actually",
    "basically",
    "honestly",
    "then",
}


@dataclass
class SegmentCandidate:
    start_idx: int
    end_idx: int
    start: float
    end: float
    duration: float
    text: str
    chars: int
    cps: float
    removed_tokens: int
    cost: float


def srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_lex(token: str) -> str:
    stripped = NON_WORD_EDGE_RE.sub("", token.strip())
    return stripped.lower()


def is_verb_like(token: str) -> bool:
    if not token:
        return False
    if token in COMMON_VERBS or token in AUXILIARIES:
        return True
    return token.endswith(("ing", "ed", "en", "ify", "ise", "ize"))


def is_adj_like(token: str) -> bool:
    if not token:
        return False
    return token.endswith(("ous", "ful", "ive", "able", "ible", "al", "ic", "less", "ary", "ory", "ish"))


def is_noun_like(token: str) -> bool:
    if not token:
        return False
    if token in CONJUNCTIONS or token in PREPOSITIONS or token in AUXILIARIES or token in NEGATIONS:
        return False
    if token.endswith(("tion", "ment", "ness", "ity", "ship", "age", "ism", "ist", "er", "or")):
        return True
    return len(token) >= 4 and token not in SUBJECT_PRONOUNS


def clean_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+'([A-Za-z])", r"'\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def render_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(t.strip() for t in tokens if t.strip())
    return clean_text(text)


def write_srt(segments: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(float(seg['start']))} --> {srt_timestamp(float(seg['end']))}\n")
            f.write(f"{seg['text']}\n\n")


def extract_words_from_aligned(aligned_data: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(aligned_data.get("segments", [])):
        for word_idx, w in enumerate(seg.get("words", [])):
            start = w.get("start")
            end = w.get("end")
            raw_word = str(w.get("word", "")).strip()
            if start is None or end is None or not raw_word:
                continue
            words.append(
                {
                    "word": raw_word,
                    "token": raw_word,
                    "start": float(start),
                    "end": float(end),
                    "seg_idx": seg_idx,
                    "word_idx": word_idx,
                }
            )
    words.sort(key=lambda x: (x["start"], x["end"]))
    return words


def clamp_word_durations(words: list[dict[str, Any]], max_word_duration: float) -> int:
    if not words:
        return 0

    adjusted = 0
    max_word_duration = max(0.1, float(max_word_duration))

    for i, w in enumerate(words):
        start = float(w["start"])
        end = float(w["end"])
        next_start = float(words[i + 1]["start"]) if i + 1 < len(words) else None

        if end <= start:
            fallback_end = start + 0.05
            if next_start is not None:
                fallback_end = min(fallback_end, next_start - 0.001)
            if fallback_end > start:
                w["end"] = round(fallback_end, 3)
                adjusted += 1
            continue

        upper_end = start + max_word_duration
        if next_start is not None:
            upper_end = min(upper_end, next_start - 0.001)

        if upper_end > start and end > upper_end:
            w["end"] = round(upper_end, 3)
            adjusted += 1

    return adjusted


def load_punctuation_model(model_name: str) -> Any:
    try:
        from deepmultilingualpunctuation import PunctuationModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'deepmultilingualpunctuation'. Install it in conda env 'whisper'."
        ) from exc

    try:
        try:
            return PunctuationModel(model=model_name)
        except TypeError:
            return PunctuationModel(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load punctuation model '{model_name}': {exc}") from exc


def restore_word_punctuation(
    words: list[dict[str, Any]],
    model: Any,
    block_words: int = 180,
) -> tuple[int, int]:
    restore_fn = getattr(model, "restore_punctuation", None)
    if restore_fn is None:
        raise RuntimeError("Loaded punctuation model does not provide restore_punctuation().")

    if not words:
        return 0, 0

    updated_blocks = 0
    total_blocks = 0
    block_words = max(1, int(block_words))

    for block_start in range(0, len(words), block_words):
        block = words[block_start : block_start + block_words]
        base_tokens = []
        for w in block:
            src = str(w["token"]).strip()
            lexical = NON_WORD_EDGE_RE.sub("", src)
            base_tokens.append(lexical if lexical else src)

        if not base_tokens:
            continue

        total_blocks += 1
        restored_text = clean_text(str(restore_fn(" ".join(base_tokens))))
        restored_tokens = restored_text.split()
        if len(restored_tokens) != len(base_tokens):
            continue

        for w, tok in zip(block, restored_tokens):
            w["token"] = tok
        updated_blocks += 1

    return updated_blocks, total_blocks


def boundary_cost(words: list[dict[str, Any]], end_idx: int) -> float:
    """Cost adjustment for boundary AFTER end_idx."""
    if end_idx >= len(words) - 1:
        last_token = str(words[end_idx]["token"])
        if SENT_END_RE.search(last_token):
            return -1.0
        return 0.2

    prev_token = str(words[end_idx]["token"]).strip()
    next_token = str(words[end_idx + 1]["token"]).strip()
    prev_lex = normalize_lex(prev_token)
    next_lex = normalize_lex(next_token)
    gap = float(words[end_idx + 1]["start"]) - float(words[end_idx]["end"])

    cost = 0.0

    # Pause-gap signal (weakened): keep as soft hint only.
    if gap >= 0.8:
        cost -= 0.9
    elif gap >= 0.3:
        cost -= 0.35
    elif gap <= 0.08:
        cost += 0.25

    # Pause/discourse words (strengthened)
    if prev_lex in PAUSE_WORDS or prev_lex in FILLER_SINGLE:
        cost -= 1.3
    if next_lex in PAUSE_WORDS or next_lex in FILLER_SINGLE:
        cost -= 1.9

    # Syntax and punctuation cues (strengthened)
    if SENT_END_RE.search(prev_token):
        cost -= 3.4
    elif CLAUSE_PUNCT_RE.search(prev_token):
        cost -= 1.3
    if next_lex in CONJUNCTIONS:
        cost -= 1.45
    if next_lex in PREPOSITIONS:
        cost -= 1.0

    # Penalties for splitting tight structures
    if prev_lex in DETERMINERS and is_noun_like(next_lex):
        cost += 5.6
    elif prev_lex in DETERMINERS:
        cost += 4.6
    if prev_lex in SUBJECT_PRONOUNS and is_verb_like(next_lex):
        cost += 5.0
    if prev_lex in AUXILIARIES and is_verb_like(next_lex):
        cost += 4.5
    if prev_lex in NEGATIONS and is_verb_like(next_lex):
        cost += 4.2
    if prev_lex == "to" and is_verb_like(next_lex):
        cost += 4.0
    if prev_lex in PREPOSITIONS and is_noun_like(next_lex):
        cost += 4.2
    if is_adj_like(prev_lex) and is_noun_like(next_lex):
        cost += 3.4
    if prev_lex in CONJUNCTIONS and next_lex and not SENT_END_RE.search(prev_token):
        cost += 2.7
    if prev_lex in COMMON_VERBS and next_lex in PHRASAL_PARTICLES:
        cost += 4.2

    if prev_token[:1].isupper() and next_token[:1].isupper():
        cost += 2.8

    return cost


def passes_text_constraints(text: str, max_chars: int) -> bool:
    if "\n" in text:
        return False
    if len(text) > max_chars:
        return False
    if "  " in text:
        return False
    if "\u2013" in text or "\u2014" in text:
        return False
    return True


def candidate_padding_capacity(words: list[dict[str, Any]], start_idx: int, end_idx: int, min_duration: float) -> float:
    start = float(words[start_idx]["start"])
    end = float(words[end_idx]["end"])
    left_bound = float(words[start_idx - 1]["end"]) if start_idx > 0 else 0.0
    right_bound = float(words[end_idx + 1]["start"]) if end_idx < len(words) - 1 else end + min_duration
    left_slack = max(0.0, start - left_bound)
    right_slack = max(0.0, right_bound - end)
    # Keep timing as a soft boundary by permitting a small virtual cushion.
    return left_slack + right_slack + 0.18


def required_duration(text: str, min_duration: float, max_cps: float) -> float:
    cps_duration = len(text) / max(1e-6, max_cps)
    return max(min_duration, cps_duration)


def condense_tokens_to_duration(
    tokens: list[str],
    available_duration: float,
    max_chars: int,
    max_cps: float,
) -> tuple[str, int]:
    return render_tokens(tokens), 0


def make_candidate(
    words: list[dict[str, Any]],
    start_idx: int,
    end_idx: int,
    min_duration: float,
    max_duration: float,
    max_chars: int,
    max_cps: float,
) -> SegmentCandidate | None:
    start = float(words[start_idx]["start"])
    end = float(words[end_idx]["end"])
    raw_duration = end - start
    if raw_duration <= 0 or raw_duration > max_duration:
        return None

    capacity = candidate_padding_capacity(words, start_idx, end_idx, min_duration=min_duration)
    max_display_duration = min(max_duration, raw_duration + capacity)

    tokens = [str(words[k]["token"]) for k in range(start_idx, end_idx + 1)]
    text = render_tokens(tokens)
    text = clean_text(text)

    removed_tokens = 0
    if not passes_text_constraints(text, max_chars=max_chars):
        return None
    # Use only cps for DP feasibility; min_duration is enforced by padding after DP.
    cps_need_duration = len(text) / max(1e-6, max_cps)
    if cps_need_duration > max_display_duration + 1e-6:
        return None
    chosen_text = text

    effective_duration = max(raw_duration, cps_need_duration, min_duration)
    if effective_duration > max_display_duration + 1e-6:
        # Allow if the timing is simply too tight; padding will handle it later.
        effective_duration = max(raw_duration, cps_need_duration)
    if not passes_text_constraints(chosen_text, max_chars=max_chars):
        return None

    chars = len(chosen_text)
    cps = chars / effective_duration if effective_duration > 0 else float("inf")
    if cps > max_cps + 1e-6:
        return None

    cost = 1.0
    cost += boundary_cost(words, end_idx)

    cps_soft_limit = max_cps * 0.88
    if cps > cps_soft_limit:
        cost += (cps - cps_soft_limit) * 1.4

    if chars > max_chars * 0.9:
        cost += (chars - max_chars * 0.9) * 0.08

    if effective_duration < (min_duration + 0.18):
        cost += (min_duration + 0.18 - effective_duration) * 0.9

    return SegmentCandidate(
        start_idx=start_idx,
        end_idx=end_idx,
        start=start,
        end=end,
        duration=effective_duration,
        text=chosen_text,
        chars=chars,
        cps=cps,
        removed_tokens=removed_tokens,
        cost=cost,
    )


def dp_segment(
    words: list[dict[str, Any]],
    min_duration: float,
    max_duration: float,
    max_chars: int,
    max_cps: float,
) -> list[SegmentCandidate] | None:
    n = len(words)
    if n == 0:
        return []

    inf = float("inf")
    dp = [inf] * (n + 1)
    prev: list[tuple[int, SegmentCandidate] | None] = [None] * (n + 1)
    dp[0] = 0.0

    for i in range(n):
        if not math.isfinite(dp[i]):
            continue
        for j in range(i, n):
            duration = float(words[j]["end"]) - float(words[i]["start"])
            if duration > max_duration + 1e-6:
                break

            cand = make_candidate(
                words=words,
                start_idx=i,
                end_idx=j,
                min_duration=min_duration,
                max_duration=max_duration,
                max_chars=max_chars,
                max_cps=max_cps,
            )
            if cand is None:
                continue

            nxt = j + 1
            new_cost = dp[i] + cand.cost
            if new_cost < dp[nxt]:
                dp[nxt] = new_cost
                prev[nxt] = (i, cand)

    if not math.isfinite(dp[n]):
        return None

    out: list[SegmentCandidate] = []
    cursor = n
    while cursor > 0:
        node = prev[cursor]
        if node is None:
            return None
        i, cand = node
        out.append(cand)
        cursor = i
    out.reverse()
    return out


def apply_min_duration_padding(segments: list[dict[str, Any]], min_duration: float) -> None:
    if not segments:
        return

    for _ in range(4):
        changed = False
        for i, seg in enumerate(segments):
            start = float(seg["start"])
            end = float(seg["end"])
            duration = end - start
            target_duration = max(min_duration, float(seg.get("planned_duration", min_duration)))
            if duration >= target_duration - 1e-6:
                continue

            need = target_duration - duration
            prev_end = float(segments[i - 1]["end"]) if i > 0 else 0.0
            next_start = (
                float(segments[i + 1]["start"])
                if i + 1 < len(segments)
                else end + max(need, 0.5)
            )

            left_room = max(0.0, start - prev_end)
            right_room = max(0.0, next_start - end)

            take_left = min(need * 0.5, left_room)
            take_right = min(need - take_left, right_room)
            remaining = need - (take_left + take_right)

            if remaining > 1e-6 and left_room > take_left:
                extra = min(remaining, left_room - take_left)
                take_left += extra
                remaining -= extra
            if remaining > 1e-6 and right_room > take_right:
                extra = min(remaining, right_room - take_right)
                take_right += extra

            if take_left > 0.0 or take_right > 0.0:
                seg["start"] = max(0.0, start - take_left)
                seg["end"] = end + take_right
                changed = True

        if not changed:
            break

    for seg in segments:
        duration = max(0.0, float(seg["end"]) - float(seg["start"]))
        seg["duration"] = duration
        seg["cps"] = (len(str(seg["text"])) / duration) if duration > 0 else float("inf")


def repair_segments_text(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    min_duration: float,
    max_chars: int,
    max_cps: float,
) -> None:
    for seg in segments:
        start_idx = int(seg["word_start_idx"])
        end_idx = int(seg["word_end_idx"])
        if start_idx < 0 or end_idx >= len(words) or start_idx > end_idx:
            continue
        duration = max(min_duration, float(seg["end"]) - float(seg["start"]))
        tokens = [str(words[i]["token"]) for i in range(start_idx, end_idx + 1)]
        text, removed = condense_tokens_to_duration(
            tokens=tokens,
            available_duration=duration,
            max_chars=max_chars,
            max_cps=max_cps,
        )
        if not text:
            continue

        prev_removed = int(seg.get("removed_tokens", 0))
        seg["text"] = text
        seg["chars"] = len(text)
        seg["removed_tokens"] = prev_removed + removed
        actual_duration = max(1e-6, float(seg["end"]) - float(seg["start"]))
        seg["cps"] = round(len(text) / actual_duration, 3)


def merge_problem_segments(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    min_duration: float,
    max_duration: float,
    max_chars: int,
    max_cps: float,
) -> None:
    if len(segments) < 2:
        return

    dur_eps = 1e-3
    cps_eps = 1e-3

    def needs_fix(seg: dict[str, Any]) -> bool:
        duration = float(seg["end"]) - float(seg["start"])
        if duration <= 0:
            return True
        cps = len(str(seg["text"])) / duration
        if duration < (min_duration - dur_eps):
            return True
        if cps > (max_cps + cps_eps):
            return True
        if len(str(seg["text"])) > max_chars:
            return True
        return False

    def build_merged(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
        start = float(left["start"])
        end = float(right["end"])
        duration = end - start
        if duration <= 0 or duration > max_duration + 1e-6:
            return None

        ws = int(left["word_start_idx"])
        we = int(right["word_end_idx"])
        if ws < 0 or we >= len(words) or ws > we:
            return None
        tokens = [str(words[i]["token"]) for i in range(ws, we + 1)]
        text, removed = condense_tokens_to_duration(
            tokens=tokens,
            available_duration=max(min_duration, duration),
            max_chars=max_chars,
            max_cps=max_cps,
        )
        if not text:
            return None

        cps = len(text) / max(duration, 1e-6)
        if cps > max_cps + 1e-3:
            return None
        if len(text) > max_chars:
            return None

        return {
            "index": int(left["index"]),
            "start": start,
            "end": end,
            "duration": duration,
            "planned_duration": max(float(left.get("planned_duration", min_duration)), float(right.get("planned_duration", min_duration))),
            "chars": len(text),
            "cps": cps,
            "text": text,
            "word_start_idx": ws,
            "word_end_idx": we,
            "removed_tokens": int(left.get("removed_tokens", 0)) + int(right.get("removed_tokens", 0)) + removed,
            "cost": float(left.get("cost", 0.0)) + float(right.get("cost", 0.0)),
        }

    i = 0
    while i < len(segments):
        seg = segments[i]
        if not needs_fix(seg):
            i += 1
            continue

        merged = False
        if i + 1 < len(segments):
            cand = build_merged(segments[i], segments[i + 1])
            if cand is not None:
                segments[i] = cand
                del segments[i + 1]
                merged = True

        if not merged and i > 0:
            cand = build_merged(segments[i - 1], segments[i])
            if cand is not None:
                segments[i - 1] = cand
                del segments[i]
                i = max(0, i - 1)
                merged = True

        if not merged:
            i += 1

    for idx, seg in enumerate(segments, start=1):
        seg["index"] = idx


def validate_segments(
    segments: list[dict[str, Any]],
    min_duration: float,
    max_duration: float,
    max_chars: int,
    max_cps: float,
) -> list[str]:
    issues: list[str] = []
    dur_eps = 1e-3
    cps_eps = 1e-3
    prev_end = -1.0
    for idx, seg in enumerate(segments, start=1):
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg["text"])
        duration = end - start
        cps = len(text) / duration if duration > 0 else float("inf")

        if duration < (min_duration - dur_eps) or duration > (max_duration + dur_eps):
            issues.append(f"#{idx}: duration {duration:.3f}s out of [{min_duration:.3f}, {max_duration:.3f}]")
        if "\n" in text:
            issues.append(f"#{idx}: contains newline (must be single-line)")
        if len(text) > max_chars:
            issues.append(f"#{idx}: chars {len(text)} > {max_chars}")
        if cps > (max_cps + cps_eps):
            issues.append(f"#{idx}: cps {cps:.2f} > {max_cps:.2f}")
        if "  " in text:
            issues.append(f"#{idx}: contains double spaces")
        if "\u2013" in text or "\u2014" in text:
            issues.append(f"#{idx}: contains en/em dash")
        if start < prev_end - 1e-6:
            issues.append(f"#{idx}: overlaps previous cue")
        prev_end = end
    return issues


def transcribe_align_with_whisperx(
    input_path: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    batch_size: int,
) -> dict[str, Any]:
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'whisperx' in current environment.") from exc

    print(f"[whisperx] load model={model_name} device={device} compute={compute_type}")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(input_path))

    transcribe_kwargs: dict[str, Any] = {"batch_size": max(1, int(batch_size))}
    if language:
        transcribe_kwargs["language"] = language
    print("[whisperx] transcribing...")
    result = model.transcribe(audio, **transcribe_kwargs)
    detected_lang = str(result.get("language") or language or "")
    print(f"[whisperx] raw segments: {len(result.get('segments', []))}, language={detected_lang or 'unknown'}")

    del model
    gc.collect()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    align_lang = language or detected_lang
    if not align_lang:
        raise RuntimeError("Failed to determine language for WhisperX alignment.")

    print(f"[whisperx] loading align model for language={align_lang}")
    align_model, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    del align_model
    gc.collect()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    aligned["language"] = align_lang
    return aligned


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="WhisperX word-level transcript + guide-rule DP subtitle segmentation"
    )
    ap.add_argument("input", help="Audio/video path, or aligned JSON path when using --from-json")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("output"), help="Output root directory")
    ap.add_argument("--slug", default="", help="Subdirectory name under output-dir (default: derived from input filename)")
    ap.add_argument("--from-json", action="store_true", help="Read aligned JSON directly instead of running WhisperX")

    ap.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    ap.add_argument("--language", default="en", help="Language code for WhisperX (default: en)")
    ap.add_argument("--device", default="cuda", help="Inference device (default: cuda)")
    ap.add_argument("--compute-type", default="int8", help="WhisperX compute type (default: int8)")
    ap.add_argument("--batch-size", type=int, default=16, help="WhisperX transcribe batch size (default: 16)")

    ap.add_argument("--min-duration", type=float, default=MIN_DURATION_DEFAULT, help="Min cue duration seconds")
    ap.add_argument("--max-duration", type=float, default=MAX_DURATION_DEFAULT, help="Max cue duration seconds")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS_DEFAULT, help="Max chars per cue (single line)")
    ap.add_argument("--max-cps", type=float, default=MAX_CPS_DEFAULT, help="Max characters per second")

    ap.add_argument("--restore-punctuation", action="store_true", help="Use punctuation restoration model")
    ap.add_argument("--no-restore-punctuation", action="store_false", dest="restore_punctuation")
    ap.set_defaults(restore_punctuation=True)
    ap.add_argument(
        "--punctuation-model-name",
        default="oliverguhr/fullstop-punctuation-multilang-large",
        help="Punctuation model name",
    )
    ap.add_argument(
        "--punctuation-block-words",
        type=int,
        default=180,
        help="Words per punctuation-restoration block",
    )

    ap.add_argument("--save-aligned-json", action="store_true", help="Always save aligned WhisperX JSON")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any validation issue exists")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.min_duration <= 0 or args.max_duration <= 0 or args.max_duration < args.min_duration:
        print("[error] invalid duration constraints", file=sys.stderr)
        return 2
    if args.max_chars <= 0 or args.max_cps <= 0:
        print("[error] invalid max-chars/max-cps constraints", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    slug = args.slug.strip() if args.slug else ""

    if args.from_json:
        if not input_path.exists():
            print(f"[error] json input not found: {input_path}", file=sys.stderr)
            return 2
        print(f"[input] loading aligned json: {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            aligned = json.load(f)
        if not slug:
            slug = input_path.stem
    else:
        if not input_path.exists():
            print(f"[error] media input not found: {input_path}", file=sys.stderr)
            return 2
        if not slug:
            slug = input_path.stem
        aligned = transcribe_align_with_whisperx(
            input_path=input_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
        )

    output_dir: Path = args.output_dir / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.from_json and args.save_aligned_json:
        aligned_json_path = output_dir / "aligned.json"
        aligned_json_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] aligned json -> {aligned_json_path}")

    words = extract_words_from_aligned(aligned)
    if not words:
        print("[error] no aligned words found", file=sys.stderr)
        return 1
    clamped = clamp_word_durations(words, max_word_duration=max(0.5, args.max_duration - 0.05))
    if clamped:
        print(f"[words] clamped anomalous word durations: {clamped}")
    print(f"[words] aligned words: {len(words)}")

    punctuation_ok = False
    if args.restore_punctuation:
        try:
            punc_model = load_punctuation_model(args.punctuation_model_name)
            updated_blocks, total_blocks = restore_word_punctuation(
                words=words,
                model=punc_model,
                block_words=args.punctuation_block_words,
            )
            punctuation_ok = updated_blocks > 0
            print(
                f"[punc] restored blocks: {updated_blocks}/{total_blocks} "
                f"(model={args.punctuation_model_name})"
            )
        except Exception as exc:
            print(f"[warn] punctuation restoration skipped: {exc}", file=sys.stderr)

    print("[segment] running global DP...")
    segments = dp_segment(
        words=words,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_chars=args.max_chars,
        max_cps=args.max_cps,
    )
    if segments is None:
        print("[error] no feasible segmentation path under current constraints", file=sys.stderr)
        print(
            "[hint] relax one constraint (e.g., --max-chars or --max-cps) "
            "or disable strict min duration for this file",
            file=sys.stderr,
        )
        return 1

    out_segments: list[dict[str, Any]] = []
    for i, seg in enumerate(segments, start=1):
        raw_duration = seg.end - seg.start
        out_segments.append(
            {
                "index": i,
                "start": seg.start,
                "end": seg.end,
                "duration": raw_duration,
                "planned_duration": seg.duration,
                "chars": seg.chars,
                "cps": seg.cps,
                "text": seg.text,
                "word_start_idx": seg.start_idx,
                "word_end_idx": seg.end_idx,
                "removed_tokens": seg.removed_tokens,
                "cost": round(seg.cost, 4),
            }
        )

    apply_min_duration_padding(out_segments, min_duration=args.min_duration)
    repair_segments_text(
        segments=out_segments,
        words=words,
        min_duration=args.min_duration,
        max_chars=args.max_chars,
        max_cps=args.max_cps,
    )
    merge_problem_segments(
        segments=out_segments,
        words=words,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_chars=args.max_chars,
        max_cps=args.max_cps,
    )
    apply_min_duration_padding(out_segments, min_duration=args.min_duration)
    repair_segments_text(
        segments=out_segments,
        words=words,
        min_duration=args.min_duration,
        max_chars=args.max_chars,
        max_cps=args.max_cps,
    )

    issues = validate_segments(
        segments=out_segments,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_chars=args.max_chars,
        max_cps=args.max_cps,
    )

    srt_path = output_dir / "en.guide.srt"
    json_path = output_dir / "en.guide.json"
    txt_path = output_dir / "en.guide.txt"

    write_srt(out_segments, srt_path)
    txt_path.write_text("\n".join(seg["text"] for seg in out_segments) + "\n", encoding="utf-8")

    payload = {
        "input": str(input_path),
        "language": str(aligned.get("language", args.language)),
        "model": args.model,
        "constraints": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "max_chars": args.max_chars,
            "max_cps": args.max_cps,
            "single_line": True,
            "no_double_spaces": True,
            "no_en_em_dash": True,
        },
        "options": {
            "restore_punctuation": args.restore_punctuation,
            "punctuation_applied": punctuation_ok,
        },
        "stats": {
            "word_count": len(words),
            "segment_count": len(out_segments),
            "total_duration": round(float(words[-1]["end"]) - float(words[0]["start"]), 3),
        },
        "validation": {
            "passed": len(issues) == 0,
            "issues": issues,
        },
        "segments": out_segments,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] srt: {srt_path}")
    print(f"[done] txt: {txt_path}")
    print(f"[done] json: {json_path}")
    print(f"[summary] segments={len(out_segments)} validation_issues={len(issues)}")
    if issues:
        print("[validation] issues:")
        for item in issues:
            print(f"  - {item}")

    if args.strict and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
