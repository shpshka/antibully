"""Transcribe a Russian audio file and run the same toxicity layers as runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faster_whisper import WhisperModel

from src.runtime.profanity import find_profanity, find_verbal_abuse
from src.runtime.toxicity import RussianToxicityClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    whisper = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(
        str(args.audio),
        language="ru",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        initial_prompt="Русская разговорная речь. Точная дословная транскрипция.",
        hotwords="дурак идиот дебил тупой сука блять пиздец хуй угроза драка",
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    toxicity = RussianToxicityClassifier()
    toxicity.load()
    result = {
        "text": text,
        "profanity": find_profanity(text),
        "verbal_abuse": find_verbal_abuse(text),
        "toxicity": toxicity.predict(text),
    }
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
