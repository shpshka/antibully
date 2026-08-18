"""Cross-dataset smoke validation on Fight Detection Surveillance samples."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime.violence import ViolenceClassifier


def read_clip(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"cannot decode {path}")
    return [frames[i] for i in np.linspace(0, len(frames) - 1, 16).astype(int)]


def main():
    classifier = ViolenceClassifier(lambda *args: None)
    deadline = time.time() + 120
    while classifier.status == "loading" and time.time() < deadline:
        time.sleep(0.25)
    if classifier.status != "ready":
        raise RuntimeError(classifier.error)
    rows = []
    for label in ("fight", "noFight"):
        for path in sorted((Path("data/validation") / label).glob("*.mp4")):
            probability = classifier.predict_clip(read_clip(path))
            rows.append((label, path.name, probability))
    for row in rows:
        print(f"{row[0]:7} {row[1]:12} violence={row[2]:.3f}")
    fight_mean = float(np.mean([score for label, _, score in rows if label == "fight"]))
    safe_mean = float(np.mean([score for label, _, score in rows if label == "noFight"]))
    print(
        {
            "fight_mean": round(fight_mean, 3),
            "nonfight_mean": round(safe_mean, 3),
            "separation": round(fight_mean - safe_mean, 3),
        }
    )
    if fight_mean <= safe_mean:
        raise SystemExit("cross-dataset ranking failed")


if __name__ == "__main__":
    main()
