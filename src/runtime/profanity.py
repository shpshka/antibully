from __future__ import annotations

import re

# Stems intentionally cover inflections and common obfuscation. Matches are
# never treated as proof of bullying; they only create a review event.
_STEMS = (
    "бля",
    "бляд",
    "блять",
    "сука",
    "суч",
    "хуй",
    "хуя",
    "хуи",
    "хуе",
    "хуё",
    "хуйн",
    "пизд",
    "ебан",
    "ёбан",
    "ебат",
    "ёбат",
    "ебл",
    "долбоеб",
    "долбоёб",
    "мудак",
    "мраз",
    "гандон",
    "пидор",
    "пидар",
    "шлюх",
    "твар",
    "урод",
    "нахуй",
    "похуй",
    "fuck",
    "shit",
    "bitch",
    "asshole",
)
_RE = re.compile(r"(?iu)(?<![\w])([\w*#@!ё-]{2,})(?![\w])")
_SUBS = str.maketrans({"@": "а", "3": "з", "6": "б", "0": "о", "1": "и", "*": "", "#": ""})
_ABUSE_PATTERNS = (
    r"\bя\s+тебя\s+(?:убью|ударю|побью)",
    r"\b(?:убью|зарежу|сломаю|побью)\b",
    r"\b(?:заткнись|сдохни|вали отсюда|пош[её]л вон)\b",
    r"\b(?:тупой|тупая|дебил|идиот|ничтожество|чмо)\b",
)
_ABUSE_RE = re.compile("|".join(_ABUSE_PATTERNS), re.IGNORECASE)


def find_profanity(text: str) -> list[str]:
    found: list[str] = []
    for token in _RE.findall(text.lower()):
        normalized = token.translate(_SUBS)
        if any(stem in normalized for stem in _STEMS):
            found.append(token)
    return list(dict.fromkeys(found))


def find_verbal_abuse(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0).lower() for match in _ABUSE_RE.finditer(text)))
