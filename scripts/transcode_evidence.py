"""Convert existing evidence clips to browser-compatible H.264 in place."""

from __future__ import annotations

import argparse
from pathlib import Path

import av


def transcode(path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.h264.part{path.suffix}")
    try:
        with av.open(str(path)) as source:
            input_stream = source.streams.video[0]
            rate = input_stream.average_rate or 12
            with av.open(str(temporary), "w", options={"movflags": "faststart"}) as target:
                output_stream = target.add_stream("libx264", rate=rate)
                output_stream.width = input_stream.width
                output_stream.height = input_stream.height
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"preset": "veryfast", "crf": "23"}
                for frame in source.decode(input_stream):
                    for packet in output_stream.encode(frame):
                        target.mux(packet)
                for packet in output_stream.encode():
                    target.mux(packet)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for clip in sorted(args.directory.glob("*.mp4")):
        transcode(clip)
        print(f"H.264: {clip}")


if __name__ == "__main__":
    main()
