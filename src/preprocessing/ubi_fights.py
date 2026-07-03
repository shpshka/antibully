"""Convert UBI-Fights videos + frame-level labels into unified pose sequences.

UBI-Fights (Degardin & Proenca, socia-lab.di.ubi.pt/EventDetection): 1,000
real-world surveillance-style videos (216 fight / 784 normal, 640x360 @30fps,
~80 hours) annotated 0/1 per frame. This is the in-domain data the pooled
classifier lacks: real CCTV framing, small people, and — crucially — hundreds
of hours of *normal* CCTV motion as hard negatives.

Each video is segmented by its frame-level annotation: contiguous fight spans
become aggressive clips, and non-fight spans are sampled into neutral clips
(capped per video so 80 hours of normal footage doesn't swamp the pool).
Segments are pose-extracted with the same YOLO-Pose pipeline as every other
RGB source and written as unified .npz with an explicit aggressive flag.

Expected layout after unzipping UBI_FIGHTS.zip (adjust --videos/--annotations
if the release differs):
    UBI_FIGHTS/videos/**.mp4           F_xxx (fight) / N_xxx (normal) names
    UBI_FIGHTS/annotation/**.csv       one 0/1 per line, one line per frame

Usage:
    python -m src.preprocessing.ubi_fights --root data/ubi_fights/UBI_FIGHTS --output outputs/ubi_poses --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MIN_SEGMENT_FRAMES = 48  # skip spans too short for the 64-frame window to see


def load_frame_labels(path: Path) -> np.ndarray:
    """Frame-level 0/1 labels from a UBI-Fights annotation csv (one value per line)."""
    text = path.read_text().replace(",", "\n")
    values = [line.strip() for line in text.splitlines() if line.strip()]
    return np.array([int(float(v)) for v in values], dtype=np.int8)


def label_segments(labels: np.ndarray, min_frames: int = MIN_SEGMENT_FRAMES):
    """Contiguous (start, end, is_fight) spans of a 0/1 frame-label array."""
    segments = []
    if len(labels) == 0:
        return segments
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if i - start >= min_frames:
                segments.append((start, i, bool(labels[start])))
            start = i
    return segments


def sample_neutral_segments(segments, max_per_video=4, segment_frames=150, rng=None):
    """Cap and chop non-fight spans so normal footage doesn't swamp the pool.

    Long normal spans are cut into ``segment_frames`` chunks; at most
    ``max_per_video`` neutral chunks are kept (spread across the video).
    """
    rng = rng or np.random.default_rng(42)
    chunks = []
    for start, end, is_fight in segments:
        if is_fight:
            continue
        for s in range(start, end - MIN_SEGMENT_FRAMES, segment_frames):
            chunks.append((s, min(s + segment_frames, end)))
    if len(chunks) > max_per_video:
        idx = np.linspace(0, len(chunks) - 1, max_per_video).astype(int)
        chunks = [chunks[i] for i in idx]
    return chunks


def convert_video(video_path, labels, out_dir, model, max_persons=8, device=None, rng=None):
    """Pose-extract one video and write per-segment unified .npz files."""
    from src.preprocessing.pose_extraction import extract_clip

    keypoints, scores = extract_clip(model, video_path, max_persons=max_persons, device=device)
    n = min(len(keypoints), len(labels))
    labels = labels[:n]

    out_dir.mkdir(parents=True, exist_ok=True)
    segments = label_segments(labels)
    written = 0
    for start, end, is_fight in segments:
        spans = [(start, end)] if is_fight else []
        for s, e in spans:
            _write_segment(out_dir, video_path, keypoints, scores, s, e, True)
            written += 1
    for s, e in sample_neutral_segments(segments, rng=rng):
        _write_segment(out_dir, video_path, keypoints, scores, s, e, False)
        written += 1
    return written


def _write_segment(out_dir, video_path, keypoints, scores, start, end, aggressive):
    stem = Path(video_path).stem
    out_path = out_dir / f"{stem}_f{start:06d}.npz"
    np.savez_compressed(
        out_path,
        keypoints=keypoints[start:end],
        scores=scores[start:end],
        source=f"{video_path}#{start}-{end}",
        label=int(aggressive),
        label_name="fight" if aggressive else "normal",
        aggressive=bool(aggressive),
    )


def find_annotation(video_path: Path, annotation_dir: Path) -> Path | None:
    for ext in (".csv", ".txt"):
        cand = annotation_dir / f"{video_path.stem}{ext}"
        if cand.exists():
            return cand
    matches = list(annotation_dir.rglob(f"{video_path.stem}.*"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="unzipped UBI_FIGHTS folder")
    parser.add_argument("--videos", default="videos", help="video subfolder name")
    parser.add_argument("--annotations", default="annotation", help="annotation subfolder name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-persons", type=int, default=8)
    parser.add_argument("--device", default=None, help='e.g. "0" for GPU')
    parser.add_argument("--limit", type=int, default=None, help="process only the first N videos")
    parser.add_argument(
        "--fights-first", action="store_true", help="process fight (F_*) videos before normal"
    )
    args = parser.parse_args()

    video_dir = args.root / args.videos
    ann_dir = args.root / args.annotations
    videos = sorted(video_dir.rglob("*.mp4")) + sorted(video_dir.rglob("*.avi"))
    if args.fights_first:
        videos = sorted(videos, key=lambda p: (not p.stem.startswith("F"), p.stem))
    if not videos:
        raise SystemExit(f"no videos under {video_dir}")
    if args.limit:
        videos = videos[: args.limit]

    from ultralytics import YOLO

    model = YOLO("yolov8m-pose.pt")
    rng = np.random.default_rng(42)
    total = 0
    for i, vid in enumerate(videos):
        ann = find_annotation(vid, ann_dir)
        if ann is None:
            print(f"[skip] no annotation for {vid.name}")
            continue
        labels = load_frame_labels(ann)
        written = convert_video(vid, labels, args.output, model, args.max_persons, args.device, rng)
        total += written
        print(f"[{i + 1}/{len(videos)}] {vid.name}: {written} segments (total {total})")
    print(f"done: {total} segments -> {args.output}")


if __name__ == "__main__":
    main()
