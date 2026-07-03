"""Convert class-folder video datasets into unified binary pose sequences.

Many surveillance fight datasets ship as a directory of clips split into
class-named subfolders — e.g. seymanurakti/fight-detection-surv-dataset
(fight/ + noFight/) or RWF-2000 (Fight/ + NonFight/). This converter walks
those folders, derives an aggressive/neutral label from the folder name, and
pose-extracts each clip with the shared YOLO pipeline into unified .npz.

It complements the lab datasets with *real* surveillance footage — the domain
gap that made the model over-fire on CCTV.

Usage:
    python -m src.preprocessing.labeled_folder --root data/fight_surv --output outputs/fightsurv_poses --device 0
    python -m src.preprocessing.labeled_folder --root data/rwf2000/train --output outputs/rwf_poses --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Checked in order: a negative marker wins over "fight"/"violence" it contains
# ("noFight" -> neutral, not aggressive).
NEGATIVE_KEYWORDS = (
    "nofight",
    "no_fight",
    "nonfight",
    "non-fight",
    "nonviolence",
    "non_violence",
    "nonviolent",
    "normal",
    "neutral",
    "peace",
)
POSITIVE_KEYWORDS = ("fight", "violence", "violent", "punch", "kick", "assault", "abuse", "aggress")


def folder_label(name: str) -> int | None:
    """Folder name -> 1 (aggressive), 0 (neutral), or None if unrecognised."""
    s = name.lower().replace(" ", "").replace("-", "")
    if any(k.replace("-", "") in s for k in NEGATIVE_KEYWORDS):
        return 0
    if any(k in s for k in POSITIVE_KEYWORDS):
        return 1
    return None


def find_labeled_videos(root: Path) -> list[tuple[Path, int]]:
    """(video_path, label) for every clip under a recognised class subfolder."""
    labeled = []
    for video in sorted(root.rglob("*")):
        if video.suffix.lower() not in VIDEO_EXTS:
            continue
        label = None
        for part in reversed(video.parent.parts):  # nearest labelled ancestor wins
            label = folder_label(part)
            if label is not None:
                break
        if label is not None:
            labeled.append((video, label))
    return labeled


def convert_video(video_path, label, out_dir, model, max_persons=8, device=None) -> Path:
    from src.preprocessing.pose_extraction import extract_clip

    keypoints, scores = extract_clip(model, video_path, max_persons=max_persons, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.npz"
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        scores=scores,
        source=str(video_path),
        label=int(label),
        label_name="fight" if label else "normal",
        aggressive=bool(label),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="dataset root (class subfolders)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-persons", type=int, default=8)
    parser.add_argument("--device", default=None, help='e.g. "0" for GPU')
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    videos = find_labeled_videos(args.root)
    if not videos:
        raise SystemExit(
            f"no class-folder videos under {args.root} (expected fight/ noFight/ etc.)"
        )
    if args.limit:
        videos = videos[: args.limit]
    n_agg = sum(lbl for _, lbl in videos)
    print(f"{len(videos)} clips ({n_agg} fight, {len(videos) - n_agg} normal)")

    from ultralytics import YOLO

    model = YOLO("yolov8m-pose.pt")
    for i, (vid, label) in enumerate(videos):
        out = convert_video(vid, label, args.output, model, args.max_persons, args.device)
        print(
            f"[{i + 1}/{len(videos)}] {vid.name} ({'fight' if label else 'normal'}) -> {out.name}"
        )
    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
