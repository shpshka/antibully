import wave
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np

from src.runtime.audio import AudioMonitor
from src.runtime.event_log import EventLog
from src.runtime.media import (
    VIDEO_EVIDENCE_KINDS,
    select_pretrigger_frames,
    write_h264_clip,
)
from src.runtime.profanity import find_profanity, find_verbal_abuse
from src.runtime.vision import PoseAnalyzer


def test_profanity_detects_inflections_and_obfuscation():
    assert find_profanity("ну это полный пиздец") == ["пиздец"]
    assert find_profanity("БЛЯТЬ, хватит") == ["блять"]
    assert find_profanity("какого хуя") == ["хуя"]
    assert not find_profanity("обычная спокойная речь")
    assert find_verbal_abuse("Я тебя убью") == ["я тебя убью"]
    assert not find_verbal_abuse("Я тебя понимаю")


def test_event_log_roundtrip(tmp_path: Path):
    log = EventLog(tmp_path)
    saved = log.add("crowd", "medium", 0.75, "test", metadata={"people": 6})
    recent = log.recent()
    assert recent[0]["id"] == saved["id"]
    assert recent[0]["metadata"] == {"people": 6}
    assert "crowd" in log.csv_text()


def test_event_log_clear_keeps_storage_usable(tmp_path: Path):
    log = EventLog(tmp_path)
    log.add("crowd", "medium", 0.75, "before clear")
    assert log.clear() == 1
    assert log.recent() == []
    assert "before clear" not in log.jsonl_path.read_text(encoding="utf-8")
    saved = log.add("fight", "high", 0.9, "after clear")
    assert log.recent()[0]["id"] == saved["id"]


def test_audio_evidence_is_playable_wav(tmp_path: Path):
    monitor = AudioMonitor(lambda *args: None, sample_rate=16000, evidence_dir=tmp_path)
    url = monitor._save_audio(np.zeros(16000, dtype=np.float32), "profanity")
    assert url is not None
    path = tmp_path / url.rsplit("/", 1)[-1]
    with wave.open(str(path), "rb") as recording:
        assert recording.getnchannels() == 1
        assert recording.getframerate() == 16000
        assert recording.getnframes() == 16000


def test_speech_evidence_includes_five_second_pretrigger(tmp_path: Path):
    events = []
    monitor = AudioMonitor(
        lambda *args: events.append(args),
        sample_rate=1000,
        window_seconds=6,
        pretrigger_seconds=5,
        evidence_dir=tmp_path,
    )
    monitor.toxicity.predict = lambda text: None
    segment = SimpleNamespace(
        text="Ты сука",
        avg_logprob=-0.1,
        no_speech_prob=0.1,
    )
    model = SimpleNamespace(transcribe=lambda *args, **kwargs: ([segment], None))

    monitor._analyze_window(
        np.full(6000, 0.02, dtype=np.float32),
        model,
        np.zeros(5000, dtype=np.float32),
    )

    profanity = next(event for event in events if event[0] == "profanity")
    metadata = profanity[4]
    assert metadata["audio_pretrigger_seconds"] == 5.0
    assert metadata["audio_duration_seconds"] == 11.0
    with wave.open(str(tmp_path / metadata["audio"].rsplit("/", 1)[-1]), "rb") as recording:
        assert recording.getnframes() == 11000


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def int(self):
        return _Tensor(self.value.astype(int))

    def tolist(self):
        return self.value.tolist()


class _Boxes:
    def __init__(self, people=1):
        self.xyxy = _Tensor(
            [[100 + 40 * index, 100, 300 + 40 * index, 500] for index in range(people)]
        )
        self.id = _Tensor(list(range(1, people + 1)))

    def __len__(self):
        return len(self.id.value)


class _Keypoints:
    def __init__(self, wrist_x, people=1):
        data = np.zeros((people, 17, 3), np.float32)
        data[..., 2] = 0.9
        for index in range(people):
            offset = 40 * index
            data[index, 5] = [160 + offset, 250, 0.9]
            data[index, 6] = [240 + offset, 250, 0.9]
            data[index, 9] = [wrist_x + offset, 120, 0.9]
            data[index, 10] = [250 + offset, 300, 0.9]
        self.data = _Tensor(data)


class _Result:
    def __init__(self, wrist_x, people=1):
        self.boxes = _Boxes(people)
        self.keypoints = _Keypoints(wrist_x, people)


def test_single_raised_hand_is_not_an_alert():
    analyzer = PoseAnalyzer(crowd_threshold=5)
    analyzer.analyze(_Result(120), now=1.0)
    signals, people = analyzer.analyze(_Result(280), now=1.1)
    assert people[0]["raised"] is True
    assert people[0]["wrist_speed"] > 1
    assert signals == []


def test_crowd_alert_requires_persistence_and_does_not_repeat():
    analyzer = PoseAnalyzer(
        crowd_threshold=2,
        crowd_persist_seconds=2,
        crowd_release_seconds=3,
    )
    assert analyzer.analyze(_Result(120, people=2), now=1)[0] == []
    signals, _ = analyzer.analyze(_Result(120, people=2), now=3.1)
    assert [signal.kind for signal in signals] == ["crowd"]
    assert analyzer.analyze(_Result(120, people=2), now=6)[0] == []

    analyzer.analyze(_Result(120, people=0), now=7)
    analyzer.analyze(_Result(120, people=0), now=10.1)
    assert analyzer.analyze(_Result(120, people=2), now=11)[0] == []
    signals, _ = analyzer.analyze(_Result(120, people=2), now=13.1)
    assert [signal.kind for signal in signals] == ["crowd"]

    analyzer.reset()
    assert analyzer._crowd_active is False
    assert analyzer.history == {}


def test_audio_rejects_prompt_echo_and_repeated_vowel_hallucinations():
    assert not AudioMonitor._plausible_transcript("Русская разговорная речь")
    assert not AudioMonitor._plausible_transcript("Ф" + "у" * 100)
    assert AudioMonitor._plausible_transcript("Я тебя сейчас побью")


def test_audio_stop_reports_stopping_state():
    monitor = AudioMonitor(lambda *args: None)
    monitor.running = True
    monitor.status = "listening"
    monitor.stop()
    assert monitor.running is False
    assert monitor.status == "stopping"


def test_evidence_video_is_browser_compatible_h264(tmp_path: Path):
    path = tmp_path / "evidence.mp4"
    frames = [np.zeros((90, 160, 3), dtype=np.uint8) for _ in range(12)]
    write_h264_clip(path, frames, fps=10)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.frames == 12


def test_profanity_keeps_five_seconds_of_video_context():
    assert "profanity" in VIDEO_EVIDENCE_KINDS
    items = [(second, f"frame-{second}") for second in range(1, 11)]
    selected = select_pretrigger_frames(items, now=10, seconds=5)
    assert [timestamp for timestamp, _ in selected] == [5, 6, 7, 8, 9, 10]
