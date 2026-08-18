from __future__ import annotations

import os
import queue
import re
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from .profanity import find_profanity, find_verbal_abuse
from .toxicity import RussianToxicityClassifier


class AudioMonitor:
    """Continuous Russian speech monitor with overlapping evidence windows."""

    def __init__(
        self,
        on_event,
        model_size="small",
        sample_rate=16000,
        window_seconds=6.0,
        overlap_seconds=1.5,
        pretrigger_seconds=5.0,
        loud_threshold=0.12,
        evidence_dir: Path | None = None,
        evidence_url="/audio",
        input_device=None,
    ):
        self.on_event = on_event
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.overlap_samples = int(sample_rate * min(overlap_seconds, window_seconds / 2))
        self.pretrigger_samples = int(sample_rate * pretrigger_seconds)
        self.capture_block_size = max(1, int(sample_rate * 0.25))
        self.loud_threshold = loud_threshold
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.evidence_url = evidence_url.rstrip("/")
        self.input_device = input_device
        self.device_name = "system default"
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.status = "stopped"
        self.last_text = ""
        self.level = 0.0
        self.noise_floor = 0.008
        self.toxicity = RussianToxicityClassifier()
        self._context = deque(maxlen=2)
        self._queue = queue.Queue(maxsize=128)
        self._thread = None
        self._last_transcript_key = ""
        self.dropped_blocks = 0
        self._model = None

    def start(self):
        if self.running:
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                return
        self._clear_queue()
        self._context.clear()
        self._last_transcript_key = ""
        self.running = True
        self.status = "loading speech model"
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-monitor")
        self._thread.start()

    def stop(self):
        self.running = False
        if not self.status.startswith("error"):
            self.status = "stopping"

    def _clear_queue(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self):
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel

            if self._model is None:
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=max(1, min(6, os.cpu_count() or 2)),
                )
            model = self._model
            self.toxicity.load()
            device_info = sd.query_devices(self.input_device, "input")
            self.device_name = str(device_info["name"])
            self.status = "listening"

            def callback(indata, frames, timing, status):
                if not self.running:
                    raise sd.CallbackStop
                audio = indata[:, 0].copy()
                self.level = float(np.sqrt(np.mean(np.square(audio))))
                try:
                    self._queue.put_nowait(audio)
                except queue.Full:
                    # Keep the newest live audio if inference temporarily falls behind.
                    self.dropped_blocks += 1
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(audio)
                    except queue.Empty:
                        pass

            buffered = np.empty(0, dtype=np.float32)
            history = np.empty(0, dtype=np.float32)
            advance_samples = self.window_samples - self.overlap_samples
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.capture_block_size,
                device=self.input_device,
                callback=callback,
            ):
                while self.running:
                    try:
                        block = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    buffered = np.concatenate((buffered, block))
                    while len(buffered) >= self.window_samples:
                        window = buffered[: self.window_samples].copy()
                        buffered = buffered[advance_samples:]
                        self._analyze_window(window, model, history.copy())
                        history = np.concatenate((history, window[:advance_samples]))[
                            -self.pretrigger_samples :
                        ]
        except Exception as exc:  # noqa: BLE001 - audio/model backends expose varied exceptions
            self.status = f"error: {exc}"
            self.on_event("audio_error", "low", 1.0, "Ошибка микрофона", {"error": str(exc)})
        finally:
            self.running = False
            if not self.status.startswith("error"):
                self.status = "stopped"

    def _analyze_window(self, audio: np.ndarray, model, pretrigger_audio: np.ndarray | None = None):
        frame_size = max(1, self.sample_rate // 50)
        usable = audio[: len(audio) - len(audio) % frame_size]
        if not len(usable):
            return
        frame_rms = np.sqrt(np.mean(np.square(usable.reshape(-1, frame_size)), axis=1))
        level = float(np.percentile(frame_rms, 90))
        ambient = float(np.percentile(frame_rms, 25))
        self.noise_floor = 0.97 * self.noise_floor + 0.03 * ambient
        dynamic_threshold = max(self.loud_threshold, self.noise_floor * 4)
        evidence_url = None
        if pretrigger_audio is None:
            pretrigger_audio = np.empty(0, dtype=np.float32)
        evidence_audio = np.concatenate((pretrigger_audio, audio))

        def emit(kind, severity, confidence, message, metadata):
            nonlocal evidence_url
            if evidence_url is None:
                evidence_url = self._save_audio(evidence_audio, kind)
            metadata = dict(metadata)
            if evidence_url:
                metadata["audio"] = evidence_url
                metadata["audio_duration_seconds"] = round(
                    len(evidence_audio) / self.sample_rate, 1
                )
                metadata["audio_pretrigger_seconds"] = round(
                    len(pretrigger_audio) / self.sample_rate, 1
                )
            self.on_event(kind, severity, confidence, message, metadata)

        if level >= dynamic_threshold:
            confidence = min(
                0.95, 0.55 + (level - dynamic_threshold) / max(dynamic_threshold, 0.01)
            )
            emit(
                "loud_voice",
                "medium",
                confidence,
                "Громкий голос/крик",
                {
                    "audio_level": round(level, 4),
                    "dynamic_threshold": round(dynamic_threshold, 4),
                    "noise_floor": round(self.noise_floor, 4),
                },
            )
        # Skip near-silence to keep CPU usage low and reduce hallucinations.
        if level < max(0.006, self.noise_floor * 1.35):
            return
        segments, _ = model.transcribe(
            audio,
            language="ru",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 300},
            condition_on_previous_text=False,
            initial_prompt="Русская разговорная речь. Точная дословная транскрипция.",
            hotwords="дурак идиот дебил тупой сука блять пиздец хуй угроза драка",
            hallucination_silence_threshold=1.0,
        )
        reliable_segments = [
            segment
            for segment in segments
            if segment.avg_logprob >= -0.9 and segment.no_speech_prob <= 0.7
        ]
        text = " ".join(segment.text.strip() for segment in reliable_segments).strip()
        if not text or not self._plausible_transcript(text):
            return
        self.last_text = text
        transcript_key = re.sub(r"[^а-яё0-9]+", "", text.lower())
        if transcript_key and transcript_key == self._last_transcript_key:
            return
        self._last_transcript_key = transcript_key
        self._context.append(text)
        context = " ".join(self._context)
        bad = find_profanity(text)
        if bad:
            emit(
                "profanity",
                "high",
                0.9,
                "Обнаружена ненормативная лексика",
                {"text": text, "matches": bad},
            )
        abuse = find_verbal_abuse(text)
        if abuse:
            emit(
                "verbal_abuse",
                "high",
                0.85,
                "Возможная угроза или оскорбление",
                {"text": text, "matches": abuse},
            )
        # Score the current utterance, not stale context, to avoid repeated false alarms.
        scores = self.toxicity.predict(text)
        if scores:
            dominant = max(("insult", "obscenity", "threat"), key=scores.get)
            thresholds = {"insult": 0.72, "obscenity": 0.65, "threat": 0.52}
            if scores[dominant] >= thresholds[dominant] and scores["toxicity"] >= 0.72:
                names = {
                    "insult": "Оскорбительная речь",
                    "obscenity": "Ненормативная лексика",
                    "threat": "Возможная угроза",
                }
                emit(
                    "speech_toxicity",
                    "high",
                    scores[dominant],
                    names[dominant],
                    {
                        "text": text,
                        "context": context,
                        "class": dominant,
                        "scores": scores,
                        "model": self.toxicity.MODEL,
                    },
                )

    def _save_audio(self, audio: np.ndarray, kind: str) -> str | None:
        if self.evidence_dir is None:
            return None
        safe_kind = re.sub(r"[^a-z0-9_-]", "_", kind.lower())
        name = f"{int(time.time() * 1000)}_{safe_kind}.wav"
        path = self.evidence_dir / name
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(pcm.tobytes())
        return f"{self.evidence_url}/{name}"

    @staticmethod
    def _plausible_transcript(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        prompt_fragments = (
            "точная дословная транскрипция",
            "русская разговорная речь",
            "сохраняй мат",
        )
        if any(fragment in normalized for fragment in prompt_fragments):
            return False
        letters = re.sub(r"[^а-яёa-z]", "", normalized)
        if not letters:
            return False
        longest_run = max(
            (len(match.group(0)) for match in re.finditer(r"(.)\1*", letters)), default=0
        )
        return longest_run < max(8, int(len(letters) * 0.4))

    def state(self):
        return {
            "status": self.status,
            "running": self.running,
            "level": round(self.level, 4),
            "last_text": self.last_text,
            "noise_floor": round(self.noise_floor, 4),
            "toxicity_model": self.toxicity.status,
            "speech_model": self.model_size,
            "device": self.device_name,
            "window_seconds": round(self.window_samples / self.sample_rate, 1),
            "pretrigger_seconds": round(self.pretrigger_samples / self.sample_rate, 1),
            "queued_audio_seconds": round(
                self._queue.qsize() * self.capture_block_size / self.sample_rate, 2
            ),
            "dropped_audio_blocks": self.dropped_blocks,
        }
