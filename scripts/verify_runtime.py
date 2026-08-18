"""Download, load and smoke-test the production runtime models."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.toxicity import RussianToxicityClassifier
from src.runtime.violence import ViolenceClassifier


def wait_for(classifier, timeout=240):
    deadline = time.time() + timeout
    while classifier.status == "loading" and time.time() < deadline:
        time.sleep(0.5)
    if classifier.status != "ready":
        raise RuntimeError(classifier.error or "model load timed out")


def main():
    violence = ViolenceClassifier(lambda *args: None)
    wait_for(violence)
    # Feed a static negative-control clip. This checks tensor shape and a full
    # forward pass; its exact score is model-dependent and is printed for QA.
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    for _ in range(30):
        violence.add_frame(frame)
    deadline = time.time() + 60
    while violence.probability == 0 and time.time() < deadline:
        time.sleep(0.5)

    toxicity = RussianToxicityClassifier()
    toxicity.load()
    if toxicity.status != "ready":
        raise RuntimeError(toxicity.error)
    safe = toxicity.predict("Привет, давай спокойно поговорим")
    threat = toxicity.predict("Я тебя сейчас побью, тупой идиот")
    if threat.get("toxicity", 0) <= safe.get("toxicity", 1):
        raise RuntimeError(f"toxicity ordering failed: safe={safe}, threat={threat}")
    print({"violence": violence.state(), "safe_text": safe, "threat_text": threat})


if __name__ == "__main__":
    main()
