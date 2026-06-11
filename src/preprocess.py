"""
src/preprocess.py — Multilingual text cleaning for Hindi, English, and Hinglish.

Key design decisions vs original chatbot.py:
- Unicode NFC normalization preserves Devanagari correctly
- Lowercasing only applied to ASCII characters (Hindi is case-agnostic but normalization matters)
- No aggressive contraction expansion — Qwen's tokenizer handles subwords natively
- Script detection to identify language mix per sentence
"""

import re
import unicodedata
from typing import Literal

# ─── Unicode Ranges ──────────────────────────────────────────────────────────
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')   # Hindi script
LATIN_RANGE      = re.compile(r'[a-zA-Z]')           # English / Roman Hinglish

# ─── English Contractions (only applied when Latin chars detected) ────────────
_CONTRACTIONS = {
    "i'm": "i am", "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "what's": "what is", "where's": "where is",
    "who's": "who is", "how's": "how is", "won't": "will not",
    "can't": "cannot", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "n't": " not",
    "'ll": " will", "'ve": " have", "'re": " are", "'d": " would",
}


def detect_script(text: str) -> Literal["hindi", "english", "hinglish", "unknown"]:
    """
    Detects the dominant script of a given text.
    Returns 'hindi', 'english', 'hinglish', or 'unknown'.
    """
    has_devanagari = bool(DEVANAGARI_RANGE.search(text))
    has_latin      = bool(LATIN_RANGE.search(text))

    if has_devanagari and has_latin:
        return "hinglish"
    elif has_devanagari:
        return "hindi"
    elif has_latin:
        return "english"
    return "unknown"


def clean_text(text: str) -> str:
    """
    Cleans text for any of the three supported languages.

    Steps:
    1. Unicode NFC normalization — critical for Devanagari composed characters
    2. Strip leading/trailing whitespace
    3. Lowercase only ASCII characters (preserves Devanagari)
    4. Expand English contractions (only if Latin chars present)
    5. Remove control characters and non-printable chars
    6. Collapse multiple spaces

    Does NOT:
    - Strip punctuation aggressively (punctuation is useful for the model)
    - Apply whitespace tokenization (the model's BPE tokenizer handles this)
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Unicode normalization (NFC) — handles Devanagari compound chars
    text = unicodedata.normalize("NFC", text)

    # Step 2: Strip
    text = text.strip()

    # Step 3: Lowercase ASCII only
    text = "".join(c.lower() if c.isascii() and c.isalpha() else c for c in text)

    # Step 4: Expand contractions (English / Hinglish parts)
    if LATIN_RANGE.search(text):
        for contraction, expansion in _CONTRACTIONS.items():
            text = text.replace(contraction, expansion)

    # Step 5: Remove control characters / null bytes
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Step 6: Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def format_as_chat(
    user_msg: str,
    assistant_msg: str,
    system_prompt: str = "",
) -> list[dict]:
    """
    Formats a single Q-A pair into the OpenAI-style chat messages format
    that Qwen2.5 expects.

    Returns a list of message dicts ready for tokenizer.apply_chat_template().
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user",      "content": clean_text(user_msg)})
    messages.append({"role": "assistant", "content": clean_text(assistant_msg)})
    return messages


def make_hinglish(english_text: str, hindi_words: list[str] | None = None) -> str:
    """
    Synthetic Hinglish generator: randomly replaces common English filler
    words/phrases with their Hindi equivalents to create code-mixed text.

    This augments training data when real Hinglish data is scarce.
    """
    _en_to_hi_filler = {
        "okay": "theek hai",
        "ok": "theek hai",
        "yes": "haan",
        "no": "nahi",
        "please": "please",     # kept same in Hinglish
        "thank you": "shukriya",
        "thanks": "shukriya",
        "sorry": "maafi",
        "hello": "namaste",
        "bye": "alvida",
        "what": "kya",
        "how": "kaise",
        "why": "kyun",
        "where": "kahan",
        "when": "kab",
        "who": "kaun",
        "i": "main",
        "you": "aap",
        "we": "hum",
        "they": "woh",
        "is": "hai",
        "are": "hain",
        "going": "ja raha",
        "eating": "kha raha",
        "come": "aao",
        "go": "jao",
        "good": "accha",
        "bad": "bura",
        "friend": "dost",
        "work": "kaam",
        "home": "ghar",
        "food": "khana",
        "water": "paani",
        "time": "waqt",
        "today": "aaj",
        "tomorrow": "kal",
        "yesterday": "kal",
    }

    import random
    words = english_text.split()
    result = []
    for word in words:
        lower = word.lower().rstrip(".,!?")
        if lower in _en_to_hi_filler and random.random() < 0.35:
            result.append(_en_to_hi_filler[lower])
        else:
            result.append(word)
    return " ".join(result)
