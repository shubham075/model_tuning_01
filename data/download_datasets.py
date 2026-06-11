"""
data/download_datasets.py — Downloads and prepares multilingual training data.

Datasets used:
  English  : HuggingFaceH4/ultrachat_200k        (multi-turn English chat)
  Hindi    : cfilt/iitb-english-hindi             (1.6M Hindi-English pairs, IIT Bombay)
  Hinglish : Synthesized via preprocess.make_hinglish()

Output format (saved as JSONL in data/processed/):
  {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."},
                {"role":"assistant","content":"..."}]}

Run:
  python data/download_datasets.py
"""

import sys
import json
import random
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (
    PROCESSED_DIR,
    MAX_ENGLISH_SAMPLES,
    MAX_HINDI_SAMPLES,
    MAX_HINGLISH_SAMPLES,
    MAX_EVAL_SAMPLES,
    SYSTEM_PROMPT,
)
from preprocess import clean_text, format_as_chat, make_hinglish

random.seed(42)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(data):,} samples → {path}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ─── English: ultrachat_200k ──────────────────────────────────────────────────

def download_english(n: int = MAX_ENGLISH_SAMPLES) -> list[dict]:
    """
    Loads multi-turn English conversations from HuggingFaceH4/ultrachat_200k.
    Each example already contains a list of messages; we take the first two
    (user + assistant turn) to keep it simple and memory-efficient.
    """
    print("\n[1/3] Downloading English dataset (ultrachat_200k)...")
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,   # streaming avoids downloading the full 1GB
    )

    samples = []
    for row in tqdm(ds, total=n, desc="  English"):
        msgs = row.get("messages", [])
        # Find first user→assistant pair
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                user_msg = clean_text(msgs[i]["content"])
                asst_msg = clean_text(msgs[i + 1]["content"])
                if len(user_msg) > 10 and len(asst_msg) > 10:
                    samples.append(
                        {"messages": format_as_chat(user_msg, asst_msg, SYSTEM_PROMPT)}
                    )
                    break
        if len(samples) >= n:
            break

    print(f"  Collected {len(samples):,} English samples.")
    return samples


# ─── Hindi: IITB English-Hindi Parallel Corpus ───────────────────────────────

def download_hindi(n: int = MAX_HINDI_SAMPLES) -> list[dict]:
    """
    Uses the CFILT IITB English-Hindi parallel corpus.
    Format: user asks in English, assistant replies in Hindi.
    This teaches the model to understand both and reply in Hindi.

    We also create reverse pairs (user asks in Hindi, assistant answers in Hindi)
    by using the Hindi sentence as both context and response.
    """
    print("\n[2/3] Downloading Hindi dataset (cfilt/iitb-english-hindi)...")
    from datasets import load_dataset

    ds = load_dataset(
        "cfilt/iitb-english-hindi",
        split="train",
        streaming=True,
    )

    samples = []
    for row in tqdm(ds, total=n * 2, desc="  Hindi"):
        translation = row.get("translation", {})
        en = clean_text(translation.get("en", ""))
        hi = clean_text(translation.get("hi", ""))

        if len(en) < 5 or len(hi) < 5:
            continue

        # Pair 1: English question → Hindi answer
        # ("Translate to Hindi" style — teaches bilingual understanding)
        if len(samples) < n // 2:
            samples.append(
                {"messages": format_as_chat(
                    f"{en}",
                    hi,
                    SYSTEM_PROMPT,
                )}
            )

        # Pair 2: Hindi question → Hindi answer
        # (Creates a pure Hindi conversation pair from parallel data)
        if len(samples) < n:
            samples.append(
                {"messages": format_as_chat(hi, hi, SYSTEM_PROMPT)}
            )

        if len(samples) >= n:
            break

    print(f"  Collected {len(samples):,} Hindi samples.")
    return samples


# ─── Hinglish: Synthetic Generation ──────────────────────────────────────────

def generate_hinglish(english_samples: list[dict], n: int = MAX_HINGLISH_SAMPLES) -> list[dict]:
    """
    Creates synthetic Hinglish by taking English conversation pairs and
    applying word-level Hindi mixing via make_hinglish().

    Not as good as real Hinglish data, but gives the model exposure to
    code-mixed text patterns without needing a separate Hinglish corpus.
    """
    print("\n[3/3] Generating synthetic Hinglish samples...")
    pool = random.sample(english_samples, min(n * 2, len(english_samples)))
    samples = []

    for item in tqdm(pool, desc="  Hinglish"):
        msgs = item["messages"]
        user_idx = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
        asst_idx = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)

        if user_idx is None or asst_idx is None:
            continue

        hl_user = make_hinglish(msgs[user_idx]["content"])
        hl_asst = make_hinglish(msgs[asst_idx]["content"])

        samples.append(
            {"messages": format_as_chat(hl_user, hl_asst, SYSTEM_PROMPT)}
        )
        if len(samples) >= n:
            break

    print(f"  Generated {len(samples):,} Hinglish samples.")
    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Multilingual Dataset Downloader")
    print("  Languages: English | Hindi | Hinglish")
    print("=" * 60)

    # 1. English
    english = download_english(MAX_ENGLISH_SAMPLES)

    # 2. Hindi
    hindi = download_hindi(MAX_HINDI_SAMPLES)

    # 3. Hinglish (generated from English pool)
    hinglish = generate_hinglish(english, MAX_HINGLISH_SAMPLES)

    # ── Merge & Shuffle ───────────────────────────────────────────────────────
    all_data = english + hindi + hinglish
    random.shuffle(all_data)

    total = len(all_data)
    eval_n = min(MAX_EVAL_SAMPLES, int(total * 0.05))   # 5% eval, capped
    eval_data  = all_data[:eval_n]
    train_data = all_data[eval_n:]

    # ── Save ──────────────────────────────────────────────────────────────────
    save_jsonl(train_data, PROCESSED_DIR / "train.jsonl")
    save_jsonl(eval_data,  PROCESSED_DIR / "eval.jsonl")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Download complete!")
    print(f"  Train samples : {len(train_data):,}")
    print(f"  Eval  samples : {len(eval_data):,}")
    print(f"  Total         : {total:,}")
    print(f"  Saved to      : {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
