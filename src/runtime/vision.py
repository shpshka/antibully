from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisionSignal:
    kind: str
    severity: str
    confidence: float
    message: str
    metadata: dict


class PoseAnalyzer:
    """Fast explainable pose signals; alerts are candidates for human review."""

    def __init__(self, crowd_threshold=5, crowd_persist_seconds=2.0, crowd_release_seconds=3.0):
        self.crowd_threshold = crowd_threshold
        self.crowd_persist_seconds = crowd_persist_seconds
        self.crowd_release_seconds = crowd_release_seconds
        self.history: dict[int, deque] = {}
        self.last_seen: dict[int, float] = {}
        self._crowd_candidate_since = None
        self._crowd_release_since = None
        self._crowd_active = False

    def reset(self):
        self.history.clear()
        self.last_seen.clear()
        self._crowd_candidate_since = None
        self._crowd_release_since = None
        self._crowd_active = False

    @staticmethod
    def _point(kp, idx):
        x, y, conf = kp[idx]
        return np.array([x, y]), float(conf)

    def analyze(self, result, now=None) -> tuple[list[VisionSignal], list[dict]]:
        now = now or time.monotonic()
        signals: list[VisionSignal] = []
        people: list[dict] = []
        if result.boxes is not None and result.keypoints is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids_tensor = result.boxes.id
            ids = (
                ids_tensor.int().cpu().tolist()
                if ids_tensor is not None
                else list(range(len(boxes)))
            )
            xy = result.keypoints.data.cpu().numpy()
            for track_id, box, kp in zip(ids, boxes, xy):
                x1, y1, x2, y2 = map(float, box)
                height = max(y2 - y1, 1.0)
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                left_wrist, lwc = self._point(kp, 9)
                right_wrist, rwc = self._point(kp, 10)
                left_shoulder, lsc = self._point(kp, 5)
                right_shoulder, rsc = self._point(kp, 6)
                raised = (
                    lwc > 0.35 and lsc > 0.35 and left_wrist[1] < left_shoulder[1] - 0.08 * height
                ) or (
                    rwc > 0.35 and rsc > 0.35 and right_wrist[1] < right_shoulder[1] - 0.08 * height
                )
                wrists = np.stack([left_wrist, right_wrist])
                hist = self.history.setdefault(track_id, deque(maxlen=8))
                speed = 0.0
                if hist:
                    dt = max(now - hist[-1][0], 0.02)
                    speed = float(
                        np.max(np.linalg.norm(wrists - hist[-1][1], axis=1)) / height / dt
                    )
                hist.append((now, wrists))
                self.last_seen[track_id] = now
                people.append(
                    {
                        "id": track_id,
                        "box": box,
                        "center": center,
                        "height": height,
                        "raised": bool(raised),
                        "wrist_speed": speed,
                    }
                )

        if len(people) >= self.crowd_threshold:
            self._crowd_release_since = None
            if self._crowd_candidate_since is None:
                self._crowd_candidate_since = now
            if (
                not self._crowd_active
                and now - self._crowd_candidate_since >= self.crowd_persist_seconds
            ):
                self._crowd_active = True
                conf = min(1.0, 0.55 + 0.08 * (len(people) - self.crowd_threshold))
                signals.append(
                    VisionSignal(
                        "crowd",
                        "medium",
                        conf,
                        f"Устойчивое скопление людей: {len(people)}",
                        {
                            "people": len(people),
                            "persist_seconds": self.crowd_persist_seconds,
                        },
                    )
                )
        else:
            self._crowd_candidate_since = None
            if self._crowd_active:
                if self._crowd_release_since is None:
                    self._crowd_release_since = now
                elif now - self._crowd_release_since >= self.crowd_release_seconds:
                    self._crowd_active = False
                    self._crowd_release_since = None

        # Hand motion is deliberately not an alert. It is visual context only;
        # actual violence is classified over a multi-second RGB clip by X3D.

        stale = [k for k, seen in self.last_seen.items() if now - seen > 3]
        for key in stale:
            self.history.pop(key, None)
            self.last_seen.pop(key, None)
        return signals, people


def draw_overlay(frame, people, state):
    colors = {"ok": (55, 190, 90), "alert": (40, 40, 230)}
    alert = state.get("alert", False)
    for person in people:
        x1, y1, x2, y2 = map(int, person["box"])
        color = colors["ok"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {person['id']}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (18, 22, 30), -1)
    people_count = state.get("people_count", len(people))
    label = f"PEOPLE {people_count} | FPS {state.get('fps', 0):.1f}"
    if alert:
        label += " | REVIEW ALERT"
    cv2.putText(
        frame,
        label,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colors["alert"] if alert else (240, 240, 240),
        2,
    )
    return frame
