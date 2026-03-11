#!/usr/bin/env python3
"""
Filter extracted barbeque task frames using YOLO-World.

Rules:
- DISCARD if total person bounding-box area > PERSON_AREA_THRESHOLD of image area
- KEEP everything else (even if only persons detected — model may miss other objects)

Usage:
    python filter_frames_yolo.py <frames_dir> [options]
    python filter_frames_yolo.py work/tasks --ext png
    python filter_frames_yolo.py yolo_test_frames --ext jpg --preview
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PERSON_AREA_THRESHOLD = 0.70   # discard if persons cover >70% of frame
PERSON_ONLY_CONF = 0.50        # confidence threshold for detection
MODEL_NAME = "yolov8x-worldv2.pt"

# Classes to detect — "person" triggers discard logic; others are "content"
DETECT_CLASSES = [
    "person", "car", "truck", "motorcycle", "bicycle",
    "wheel", "tire", "engine", "road", "garage",
    "tool", "wrench", "hood", "dashboard", "steering wheel",
]


def load_model(model_name: str):
    from ultralytics import YOLOWorld
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} ...")
    model = YOLOWorld(model_name)
    model.set_classes(DETECT_CLASSES)
    return model, device


def analyze_frame(model, image_path: Path) -> dict:
    results = model.predict(str(image_path), conf=PERSON_ONLY_CONF, verbose=False)
    result = results[0]

    img_h, img_w = result.orig_shape
    img_area = img_h * img_w

    names = result.names  # {0: 'person', 1: 'car', ...}
    boxes = result.boxes

    person_area = 0.0
    non_person_count = 0
    person_count = 0
    detections = []

    for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        area = (x2 - x1) * (y2 - y1)
        ratio = area / img_area
        cls_name = names[int(cls_id)]
        detections.append({
            "class": cls_name,
            "conf": round(float(conf), 3),
            "area_ratio": round(ratio, 4),
        })
        if cls_name == "person":
            person_area += area
            person_count += 1
        else:
            non_person_count += 1

    person_area_ratio = person_area / img_area

    # Discard condition: person area exceeds threshold
    # (only_persons rule removed — model may miss other objects at lower conf)
    person_too_large = person_area_ratio > PERSON_AREA_THRESHOLD

    discard = person_too_large

    reason = ""
    if person_too_large:
        reason = f"person_area_too_large ({person_area_ratio:.1%} > {PERSON_AREA_THRESHOLD:.0%})"

    return {
        "path": str(image_path),
        "discard": discard,
        "reason": reason,
        "person_count": person_count,
        "person_area_ratio": round(person_area_ratio, 4),
        "non_person_count": non_person_count,
        "detections": detections,
    }


def find_frames(frames_dir: Path, ext: str) -> list[Path]:
    """Find frame images — supports both flat dir and barbeque task subfolders.
    Excludes 'discarded_preview' subdirectory to avoid double-processing."""
    frames = [
        p for p in frames_dir.rglob(f"*.{ext}")
        if "discarded_preview" not in p.parts
    ]
    frames.sort()
    return frames


def main() -> int:
    global PERSON_AREA_THRESHOLD, PERSON_ONLY_CONF
    parser = argparse.ArgumentParser(description="Filter frames by YOLO person detection")
    parser.add_argument("frames_dir", help="Directory containing frames (or barbeque tasks dir)")
    parser.add_argument("--ext", default="jpg", help="Image extension (default: jpg)")
    parser.add_argument("--model", default=MODEL_NAME, help=f"YOLO-World model (default: {MODEL_NAME})")
    parser.add_argument("--person-threshold", type=float, default=PERSON_AREA_THRESHOLD,
                        help=f"Person area ratio discard threshold (default: {PERSON_AREA_THRESHOLD})")
    parser.add_argument("--conf", type=float, default=PERSON_ONLY_CONF,
                        help=f"Detection confidence threshold (default: {PERSON_ONLY_CONF})")
    parser.add_argument("--preview", action="store_true",
                        help="Copy discarded frames to a 'discarded_preview' folder")
    parser.add_argument("--report", default="yolo_filter_report.json",
                        help="Output JSON report path (default: yolo_filter_report.json)")
    args = parser.parse_args()

    PERSON_AREA_THRESHOLD = args.person_threshold
    PERSON_ONLY_CONF = args.conf

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"Error: {frames_dir} does not exist")
        return 1

    frames = find_frames(frames_dir, args.ext)
    if not frames:
        print(f"No .{args.ext} files found in {frames_dir}")
        return 1

    print(f"Found {len(frames)} frames to analyze.")
    model, device = load_model(args.model)

    results = []
    kept = 0
    discarded = 0
    preview_dir = frames_dir / "discarded_preview"

    for i, frame_path in enumerate(frames, 1):
        info = analyze_frame(model, frame_path)
        results.append(info)
        status = "DISCARD" if info["discard"] else "keep  "
        det_summary = ", ".join(f"{d['class']}({d['conf']:.2f})" for d in info["detections"][:5])
        print(f"  [{i:3d}/{len(frames)}] {status} | {frame_path.name:<30} | {det_summary or 'nothing'}"
              + (f" | {info['reason']}" if info["discard"] else ""))
        if info["discard"]:
            discarded += 1
            if args.preview:
                preview_dir.mkdir(exist_ok=True)
                shutil.copy2(frame_path, preview_dir / frame_path.name)
        else:
            kept += 1

    # Save report
    report_path = Path(args.report)
    report_path.write_text(json.dumps({
        "total": len(frames),
        "kept": kept,
        "discarded": discarded,
        "person_threshold": PERSON_AREA_THRESHOLD,
        "conf": PERSON_ONLY_CONF,
        "frames": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Total: {len(frames)}  |  Kept: {kept}  |  Discarded: {discarded}")
    print(f"Report saved to: {report_path}")
    if args.preview and discarded:
        print(f"Discarded frames copied to: {preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
