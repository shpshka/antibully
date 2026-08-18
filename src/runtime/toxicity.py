from __future__ import annotations

import threading


class RussianToxicityClassifier:
    MODEL = "cointegrated/rubert-tiny-toxicity"
    LABELS = ("non_toxic", "insult", "obscenity", "threat", "dangerous")

    def __init__(self):
        self.status = "loading"
        self.error = None
        self._lock = threading.Lock()
        self.tokenizer = self.model = None

    def load(self):
        if self.model is not None:
            self.status = "ready"
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
            self.model.eval()
            self.status = "ready"
        except Exception as exc:  # noqa: BLE001 - optional model must fail closed
            self.error = str(exc)
            self.status = "error"

    def predict(self, text: str) -> dict:
        if self.model is None:
            return {}
        with self._lock, self.torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
            values = self.torch.sigmoid(self.model(**inputs).logits)[0].cpu().tolist()
        scores = dict(zip(self.LABELS, map(float, values)))
        scores["toxicity"] = 1 - scores["non_toxic"] * (1 - scores["dangerous"])
        return {key: round(value, 3) for key, value in scores.items()}
