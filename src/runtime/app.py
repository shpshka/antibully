from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import AudioMonitor
from .event_log import EventLog
from .media import VIDEO_EVIDENCE_KINDS, select_pretrigger_frames, write_h264_clip
from .violence import ViolenceClassifier
from .vision import PoseAnalyzer, draw_overlay

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime_data"
EVENTS = EventLog(RUNTIME)
CLIPS = RUNTIME / "clips"
CLIPS.mkdir(exist_ok=True)
AUDIO = RUNTIME / "audio"
AUDIO.mkdir(exist_ok=True)
SESSION_STARTED_AT = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
SETTINGS_FILE = RUNTIME / "settings.json"


class Monitor:
    def __init__(self):
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            saved = {}
        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.model_name = os.getenv("POSE_MODEL", "yolo11n-pose.pt")
        self.imgsz = int(os.getenv("IMAGE_SIZE", "416"))
        self.frame_stride = max(1, int(os.getenv("FRAME_STRIDE", "2")))
        self.conf = float(os.getenv("DETECTION_CONFIDENCE", ".35"))
        self.stream_width = int(os.getenv("STREAM_WIDTH", "640"))
        self.stream_height = int(os.getenv("STREAM_HEIGHT", "360"))
        self.stream_fps = max(1, int(os.getenv("STREAM_FPS", "10")))
        self.jpeg_quality = min(95, max(40, int(os.getenv("JPEG_QUALITY", "72"))))
        self.evidence_fps = max(1, int(os.getenv("EVIDENCE_FPS", "10")))
        self.evidence_pretrigger_seconds = max(
            1.0, float(os.getenv("VIDEO_PRETRIGGER_SECONDS", "5"))
        )
        self.cooldown = float(
            saved.get("event_cooldown_seconds", os.getenv("EVENT_COOLDOWN_SECONDS", "8"))
        )
        self.analyzer = PoseAnalyzer(
            int(saved.get("crowd_threshold", os.getenv("CROWD_THRESHOLD", "5")))
        )
        self.running = False
        self.status = "stopped"
        self.error = None
        self.jpeg = None
        self.jpeg_version = 0
        self._jpeg_captured_at = 0.0
        self._capture_frame = None
        self._capture_version = 0
        self.capture_fps = 0.0
        self._capture_error = None
        self.people = 0
        self._people_overlay = []
        self.fps = 0.0
        self.last_event = {}
        self._last_frame = None
        self._evidence_frames = deque(
            maxlen=int(self.evidence_fps * (self.evidence_pretrigger_seconds + 1))
        )
        self._recent_signal_at = {}
        self._recent_evidence = {}
        self._last_logged = {}
        self._lock = threading.Lock()
        self._event_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._camera_thread = None
        audio_device = os.getenv("AUDIO_DEVICE", "").strip()
        if audio_device.isdigit():
            audio_device = int(audio_device)
        elif not audio_device:
            audio_device = None
        self.audio = AudioMonitor(
            self._audio_event,
            model_size=os.getenv("WHISPER_MODEL", "small"),
            loud_threshold=float(saved.get("loud_threshold", os.getenv("LOUD_THRESHOLD", ".06"))),
            evidence_dir=AUDIO,
            input_device=audio_device,
        )
        self.violence = ViolenceClassifier(
            self._violence_alert,
            threshold=float(
                saved.get("violence_threshold", os.getenv("VIOLENCE_THRESHOLD", ".68"))
            ),
            confirm_windows=int(
                saved.get("violence_confirm_windows", os.getenv("VIOLENCE_CONFIRM_WINDOWS", "2"))
            ),
        )

    def _audio_event(self, kind, severity, confidence, message, metadata):
        self.log_event(kind, severity, confidence, message, metadata=metadata)

    def _violence_alert(self, confidence, metadata):
        with self._lock:
            frame = self._last_frame.copy() if self._last_frame is not None else None
        self.log_event(
            "fight", "high", confidence, "Видео-модель подтвердила возможную драку", frame, metadata
        )

    def log_event(self, kind, severity, confidence, message, frame=None, metadata=None):
        with self._event_lock:
            return self._log_event_locked(kind, severity, confidence, message, frame, metadata)

    def _log_event_locked(self, kind, severity, confidence, message, frame=None, metadata=None):
        now = time.monotonic()
        if now - self._last_logged.get(kind, 0) < self.cooldown:
            return None
        self._last_logged[kind] = now
        self._recent_signal_at[kind] = now
        metadata = dict(metadata or {})
        snapshot = None
        if frame is not None:
            name = f"{int(time.time() * 1000)}_{kind}.jpg"
            cv2.imwrite(str(EVENTS.snapshots / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            snapshot = f"/snapshots/{name}"
        if kind in VIDEO_EVIDENCE_KINDS and self._evidence_frames:
            clip_name = f"{int(time.time() * 1000)}_{kind}.mp4"
            with self._lock:
                evidence = select_pretrigger_frames(
                    self._evidence_frames,
                    now,
                    self.evidence_pretrigger_seconds,
                )
            frames = [evidence_frame for _, evidence_frame in evidence]
            metadata["clip"] = f"/clips/{clip_name}"
            metadata["video_pretrigger_seconds"] = round(
                evidence[-1][0] - evidence[0][0] if len(evidence) > 1 else 0.0,
                1,
            )
            threading.Thread(
                target=self._write_clip,
                args=(CLIPS / clip_name, frames, self.evidence_fps),
                daemon=True,
                name="evidence-writer",
            ).start()
        self.last_event = EVENTS.add(kind, severity, confidence, message, snapshot, metadata)
        self._recent_evidence[kind] = {
            "at": now,
            "snapshot": snapshot,
            "metadata": dict(metadata),
        }
        video_recent = now - self._recent_signal_at.get("fight", -1e9) <= 12
        speech_recent = any(
            now - self._recent_signal_at.get(key, -1e9) <= 12
            for key in ("speech_toxicity", "profanity", "verbal_abuse", "loud_voice")
        )
        if (
            kind != "multimodal_risk"
            and video_recent
            and speech_recent
            and now - self._last_logged.get("multimodal_risk", 0) >= self.cooldown
        ):
            self._last_logged["multimodal_risk"] = now
            fight_evidence = self._recent_evidence.get("fight", {})
            speech_evidence = max(
                (
                    self._recent_evidence.get(key, {})
                    for key in ("speech_toxicity", "profanity", "verbal_abuse", "loud_voice")
                ),
                key=lambda item: item.get("at", -1e9),
                default={},
            )
            fight_metadata = fight_evidence.get("metadata", {})
            speech_metadata = speech_evidence.get("metadata", {})
            fusion_metadata = {
                "fusion_window_seconds": 12,
                "trigger": kind,
            }
            if fight_metadata.get("clip"):
                fusion_metadata["clip"] = fight_metadata["clip"]
            if speech_metadata.get("audio"):
                fusion_metadata["audio"] = speech_metadata["audio"]
            if speech_metadata.get("text"):
                fusion_metadata["text"] = speech_metadata["text"]
            fusion_snapshot = fight_evidence.get("snapshot") or snapshot
            self.last_event = EVENTS.add(
                "multimodal_risk",
                "critical",
                max(0.85, float(confidence)),
                "Видео и речь одновременно указывают на риск",
                fusion_snapshot,
                fusion_metadata,
            )
        return self.last_event

    @staticmethod
    def _write_clip(path, frames, fps=10):
        write_h264_clip(path, frames, fps)

    def start(self):
        with self._lifecycle_lock:
            if self.running:
                return
            if self._camera_thread is not None and self._camera_thread.is_alive():
                self._camera_thread.join(timeout=3)
                if self._camera_thread.is_alive():
                    self.status = "stopping"
                    return
            with self._lock:
                self._last_frame = None
                self._capture_frame = None
                self._capture_version = 0
                self._evidence_frames.clear()
                self.jpeg = None
                self.jpeg_version += 1
                self._jpeg_captured_at = 0.0
                self.people = 0
                self._people_overlay = []
                self.fps = 0.0
                self.capture_fps = 0.0
                self._capture_error = None
                self.error = None
            with self._event_lock:
                self._last_logged.clear()
                self._recent_signal_at.clear()
                self._recent_evidence.clear()
            self.analyzer.reset()
            self.violence.reset_stream()
            self.running = True
            self.status = "starting"
            self._camera_thread = threading.Thread(
                target=self._camera_loop, daemon=True, name="camera-monitor"
            )
            self._camera_thread.start()
            if os.getenv("AUDIO_ENABLED", "true").lower() in {"1", "true", "yes"}:
                self.audio.start()

    def stop(self):
        with self._lifecycle_lock:
            self.running = False
            self.status = "stopping"
            self.audio.stop()

    def _capture_latest_frames(self, cap, stop_event):
        last_stream_frame = 0.0
        last_evidence_frame = 0.0
        measured = 0
        measured_at = time.monotonic()
        failed_reads = 0
        while self.running and not stop_event.is_set():
            ok, frame = cap.read()
            captured_at = time.monotonic()
            if not ok:
                failed_reads += 1
                if failed_reads >= 5:
                    self._capture_error = "Камера перестала отдавать кадры"
                    return
                time.sleep(0.02)
                continue
            failed_reads = 0
            measured += 1
            elapsed = captured_at - measured_at
            if elapsed >= 1:
                self.capture_fps = measured / elapsed
                measured = 0
                measured_at = captured_at
            with self._lock:
                self._capture_frame = frame
                self._last_frame = frame
                self._capture_version += 1
                people_count = self.people
                people_overlay = list(self._people_overlay)
                analysis_fps = self.fps
            if captured_at - last_evidence_frame >= 1 / self.evidence_fps:
                evidence = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._evidence_frames.append((captured_at, evidence))
                last_evidence_frame = captured_at
            if captured_at - last_stream_frame >= 1 / self.stream_fps:
                overlay_frame = draw_overlay(
                    frame.copy(),
                    people_overlay,
                    {
                        "fps": analysis_fps,
                        "people_count": people_count,
                        "alert": False,
                    },
                )
                preview = cv2.resize(
                    overlay_frame,
                    (self.stream_width, self.stream_height),
                    interpolation=cv2.INTER_AREA,
                )
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if encoded_ok:
                    with self._lock:
                        self.jpeg = encoded.tobytes()
                        self.jpeg_version += 1
                        self._jpeg_captured_at = captured_at
                last_stream_frame = captured_at

    def _camera_loop(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error = f"Не удалось открыть камеру {self.camera_index}"
            self.status = "error"
            self.running = False
            return
        capture_stop = threading.Event()
        capture_thread = threading.Thread(
            target=self._capture_latest_frames,
            args=(cap, capture_stop),
            daemon=True,
            name="camera-capture",
        )
        capture_thread.start()
        try:
            from ultralytics import YOLO

            model = YOLO(self.model_name)
            self.status = "running"
            count, measured, started = 0, 0, time.monotonic()
            last_capture_version = -1
            while self.running:
                with self._lock:
                    frame = self._capture_frame
                    capture_version = self._capture_version
                if frame is None or capture_version == last_capture_version:
                    if not capture_thread.is_alive() and self._capture_error:
                        raise RuntimeError(self._capture_error)
                    time.sleep(0.005)
                    continue
                last_capture_version = capture_version
                self.violence.add_frame(frame)
                count += 1
                if count % self.frame_stride == 0:
                    results = model.track(
                        frame,
                        persist=True,
                        verbose=False,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        device="cpu",
                        tracker="bytetrack.yaml",
                    )
                    signals, people = self.analyzer.analyze(results[0])
                    with self._lock:
                        self.people = len(people)
                        self._people_overlay = people
                    for signal in signals:
                        self.log_event(
                            signal.kind,
                            signal.severity,
                            signal.confidence,
                            signal.message,
                            frame,
                            signal.metadata,
                        )
                    measured += 1
                    elapsed = time.monotonic() - started
                    if elapsed >= 1:
                        self.fps = measured / elapsed
                        measured = 0
                        started = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - isolate camera/model failures from the API
            self.error = str(exc)
            self.status = "error"
            self.log_event(
                "camera_error", "low", 1, "Ошибка видеоаналитики", metadata={"error": str(exc)}
            )
        finally:
            capture_stop.set()
            capture_thread.join(timeout=1)
            cap.release()
            capture_thread.join(timeout=1)
            self.running = False
            if self.status != "error":
                self.status = "stopped"

    def state(self):
        with self._lock:
            stream_age_ms = (
                max(0.0, (time.monotonic() - self._jpeg_captured_at) * 1000)
                if self._jpeg_captured_at
                else None
            )
        return {
            "status": self.status,
            "running": self.running,
            "error": self.error,
            "people": self.people,
            "analysis_fps": round(self.fps, 2),
            "model": self.model_name,
            "stream": {
                "width": self.stream_width,
                "height": self.stream_height,
                "fps": self.stream_fps,
                "jpeg_quality": self.jpeg_quality,
                "capture_fps": round(self.capture_fps, 1),
                "age_ms": round(stream_age_ms, 1) if stream_age_ms is not None else None,
            },
            "audio": self.audio.state(),
            "violence": self.violence.state(),
            "session_started_at": SESSION_STARTED_AT,
            "last_event": self.last_event,
        }

    def configuration(self):
        return {
            "violence_threshold": self.violence.threshold,
            "violence_confirm_windows": self.violence.confirm_windows,
            "crowd_threshold": self.analyzer.crowd_threshold,
            "loud_threshold": self.audio.loud_threshold,
            "event_cooldown_seconds": self.cooldown,
        }

    def update_configuration(self, values):
        self.violence.threshold = values.violence_threshold
        self.violence.confirm_windows = values.violence_confirm_windows
        self.analyzer.crowd_threshold = values.crowd_threshold
        self.audio.loud_threshold = values.loud_threshold
        self.cooldown = values.event_cooldown_seconds
        SETTINGS_FILE.write_text(json.dumps(self.configuration(), indent=2), encoding="utf-8")
        return self.configuration()

    def clear_events(self):
        with self._event_lock:
            removed = EVENTS.clear()
            self.last_event = {}
            self._last_logged.clear()
            self._recent_signal_at.clear()
            self._recent_evidence.clear()
        return removed


MONITOR = Monitor()


class RuntimeSettings(BaseModel):
    violence_threshold: float = Field(0.68, ge=0.4, le=0.95)
    violence_confirm_windows: int = Field(2, ge=1, le=5)
    crowd_threshold: int = Field(5, ge=2, le=30)
    loud_threshold: float = Field(0.06, ge=0.01, le=0.5)
    event_cooldown_seconds: float = Field(8, ge=2, le=300)


@asynccontextmanager
async def lifespan(app):
    MONITOR.start()
    yield
    MONITOR.stop()


app = FastAPI(title="School-Safe Monitor", lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=EVENTS.snapshots), name="snapshots")
app.mount("/clips", StaticFiles(directory=CLIPS), name="clips")
app.mount("/audio", StaticFiles(directory=AUDIO), name="audio")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return MONITOR.state()


@app.get("/api/events")
def events(limit: int = 100, include_history: bool = False):
    return EVENTS.recent(min(max(limit, 1), 500), None if include_history else SESSION_STARTED_AT)


@app.get("/api/events.csv")
def events_csv():
    return Response(
        EVENTS.csv_text(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )


@app.delete("/api/events")
def clear_events():
    return {"ok": True, "removed": MONITOR.clear_events()}


@app.get("/api/settings")
def get_settings():
    return MONITOR.configuration()


@app.put("/api/settings")
def update_settings(body: RuntimeSettings):
    return MONITOR.update_configuration(body)


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile, save_demo: bool = Form(True)):
    payload = await file.read(50 * 1024 * 1024 + 1)
    if len(payload) > 50 * 1024 * 1024:
        raise HTTPException(413, "Максимальный размер видео — 50 МБ")
    import tempfile

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(payload)
            path = Path(temporary.name)
        cap = cv2.VideoCapture(str(path))
        declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = []
        frames_read = 0
        try:
            if declared_frames > 0:
                frame_limit = min(declared_frames, 600)
                target_indexes = set(np.linspace(0, max(0, frame_limit - 1), 16).astype(int))
                for index in range(frame_limit):
                    ok = cap.grab()
                    if not ok:
                        break
                    frames_read += 1
                    if index in target_indexes:
                        ok, frame = cap.retrieve()
                        if ok:
                            samples.append(
                                cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                            )
            else:
                rng = np.random.default_rng(0)
                while frames_read < 600:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    compact = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                    frames_read += 1
                    if len(samples) < 16:
                        samples.append(compact)
                    else:
                        replacement = int(rng.integers(0, frames_read))
                        if replacement < 16:
                            samples[replacement] = compact
        finally:
            cap.release()
        if len(samples) < 4:
            raise HTTPException(400, "Видео не удалось декодировать")
        clip = [samples[i] for i in np.linspace(0, len(samples) - 1, 16).astype(int)]
        probability = MONITOR.violence.predict_clip(clip)
        verdict = "review" if probability >= MONITOR.violence.threshold else "normal"
        saved_clip = None
        if save_demo and verdict == "review":
            allowed_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            safe_suffix = suffix.lower() if suffix.lower() in allowed_suffixes else ".mp4"
            clip_name = f"{int(time.time() * 1000)}_uploaded_review{safe_suffix}"
            saved_path = CLIPS / clip_name
            with saved_path.open("wb") as output:
                output.write(payload)
            saved_clip = f"/clips/{clip_name}"
            MONITOR.last_event = EVENTS.add(
                "uploaded_fight",
                "high",
                probability,
                "Загруженное видео требует проверки",
                metadata={
                    "clip": saved_clip,
                    "original_filename": file.filename,
                    "frames_read": frames_read,
                    "source": "manual_upload",
                },
            )
        return {
            "filename": file.filename,
            "frames_read": frames_read,
            "violence_probability": round(probability, 3),
            "threshold": MONITOR.violence.threshold,
            "verdict": verdict,
            "saved_clip": saved_clip,
        }
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@app.post("/api/start")
def start():
    MONITOR.start()
    return {"ok": True}


@app.post("/api/stop")
def stop():
    MONITOR.stop()
    return {"ok": True}


@app.get("/video")
def video():
    def frames():
        last_version = -1
        while True:
            with MONITOR._lock:
                jpeg = MONITOR.jpeg
                version = MONITOR.jpeg_version
            if jpeg and version != last_version:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                last_version = version
            time.sleep(0.01)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
