from __future__ import annotations

import queue
import threading
import time
from collections import deque

import cv2
import numpy as np


class ViolenceClassifier:
    """Temporal X3D classifier fine-tuned on RWF-2000 CCTV fight clips."""

    REPO = "visionlab-ai/school-violence-detection-models"
    FILE = "final/final_x3d_realtime.pt"

    def __init__(self, on_alert, threshold=0.68, confirm_windows=2, interval=1.0):
        self.on_alert = on_alert
        self.threshold = threshold
        self.confirm_windows = confirm_windows
        self.interval = interval
        self.status = "loading"
        self.error = None
        self.probability = 0.0
        self.smoothed_probability = 0.0
        self.positive_windows = 0
        self.last_inference_ms = 0.0
        self.processed_windows = 0
        self.dropped_windows = 0
        self._frames = deque(maxlen=64)
        self._queue = queue.Queue(maxsize=1)
        self._running = True
        self._last_submit = 0.0
        self._generation = 0
        threading.Thread(target=self._worker, daemon=True, name="violence-x3d").start()

    def reset_stream(self):
        self._frames.clear()
        self._generation += 1
        self._last_submit = time.monotonic()
        self.probability = 0.0
        self.smoothed_probability = 0.0
        self.positive_windows = 0

    def add_frame(self, frame):
        # Store compact frames: a multi-second temporal window without retaining
        # full-resolution camera images in RAM.
        small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        self._frames.append(small)
        now = time.monotonic()
        if len(self._frames) >= 24 and now - self._last_submit >= self.interval:
            indexes = np.linspace(0, len(self._frames) - 1, 16).astype(int)
            clip = [self._frames[i].copy() for i in indexes]
            self._last_submit = now
            try:
                self._queue.put_nowait((clip, None, self._generation))
            except queue.Full:
                self.dropped_windows += 1

    def predict_clip(self, clip, timeout=60.0) -> float:
        response = queue.Queue(maxsize=1)
        self._queue.put((clip, response, None), timeout=timeout)
        return float(response.get(timeout=timeout))

    @staticmethod
    def _tensor(clip, torch):
        frames = []
        for frame in clip:
            crop = frame[16:240, 16:240]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        arr = np.stack(frames).astype(np.float32) / 255.0
        arr = (arr - 0.45) / 0.225
        return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)

    def _worker(self):
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from pytorchvideo.models.hub import x3d_m

            torch.set_num_threads(max(1, min(4, (__import__("os").cpu_count() or 2))))
            checkpoint_path = hf_hub_download(repo_id=self.REPO, filename=self.FILE)
            model = x3d_m(pretrained=False)
            # The published checkpoint contains numpy scalar metadata in
            # addition to tensors. Allow only that explicit safe numpy type;
            # keep weights_only=True so arbitrary pickle code cannot execute.
            safe_numpy_types = [np._core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
            with torch.serialization.safe_globals(safe_numpy_types):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
            dropout = float(checkpoint.get("hyper_params", {}).get("dropout", 0.3))
            model.blocks[5].proj = torch.nn.Sequential(
                torch.nn.Dropout(dropout), torch.nn.Linear(2048, 2)
            )
            # The training wrapper stored the hub model under `backbone`.
            state = {key.removeprefix("backbone."): value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
            model.eval()
            self.status = "ready"
            with torch.inference_mode():
                while self._running:
                    try:
                        item = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if isinstance(item, tuple) and len(item) == 3:
                        clip, response, generation = item
                    else:
                        clip, response, generation = item, None, self._generation
                    inference_started = time.perf_counter()
                    probability = float(
                        torch.softmax(model(self._tensor(clip, torch)), dim=1)[0, 1]
                    )
                    self.last_inference_ms = (time.perf_counter() - inference_started) * 1000
                    self.processed_windows += 1
                    if response is not None:
                        response.put_nowait(probability)
                        continue
                    if generation != self._generation:
                        continue
                    self.probability = probability
                    alpha = 0.45
                    self.smoothed_probability = (
                        alpha * probability + (1 - alpha) * self.smoothed_probability
                    )
                    if (
                        probability >= self.threshold
                        and self.smoothed_probability >= self.threshold - 0.08
                    ):
                        self.positive_windows += 1
                    else:
                        self.positive_windows = max(0, self.positive_windows - 1)
                    if self.positive_windows >= self.confirm_windows:
                        self.on_alert(
                            self.smoothed_probability,
                            {
                                "raw_probability": round(probability, 3),
                                "confirmed_windows": self.positive_windows,
                                "model": "X3D-M/RWF-2000",
                            },
                        )
                        self.positive_windows = 0
        except Exception as exc:  # noqa: BLE001 - worker reports all backend failures in status
            self.error = str(exc)
            self.status = "error"

    def state(self):
        return {
            "status": self.status,
            "error": self.error,
            "probability": round(self.probability, 3),
            "smoothed_probability": round(self.smoothed_probability, 3),
            "threshold": self.threshold,
            "confirmed_windows": self.positive_windows,
            "last_inference_ms": round(self.last_inference_ms, 1),
            "processed_windows": self.processed_windows,
            "dropped_windows": self.dropped_windows,
        }
