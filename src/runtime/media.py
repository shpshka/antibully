from __future__ import annotations

from pathlib import Path

import av

VIDEO_EVIDENCE_KINDS = frozenset({"fight", "multimodal_risk", "profanity"})


def select_pretrigger_frames(items, now: float, seconds: float):
    selected = [(timestamp, frame) for timestamp, frame in items if timestamp >= now - seconds]
    if not selected and items:
        selected = list(items)[-1:]
    return selected


def write_h264_clip(path: Path, frames: list, fps: int = 10) -> None:
    """Atomically write BGR numpy frames as a browser-compatible H.264 MP4."""
    if not frames:
        return
    height, width = frames[0].shape[:2]
    temporary = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        with av.open(str(temporary), "w", options={"movflags": "faststart"}) as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "23"}
            for evidence_frame in frames:
                video_frame = av.VideoFrame.from_ndarray(evidence_frame, format="bgr24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
