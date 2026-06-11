#!/usr/bin/env python3
"""
kaggle_train.py — Kaggle T4 Training Pipeline (Smart LoRA-only approach)
=========================================================================

KEY INSIGHT
-----------
Qwen/Qwen2.5-3B-Instruct-GGUF already exists on HuggingFace (uploaded by Qwen team).
That is the BASE model — it has NO fine-tuning on our Hindi/English/Hinglish dataset.

Our training produces LoRA adapter weights (~50–100 MB) that teach the model
our specific language patterns on top of the base.

SMART APPROACH: Only convert the tiny LoRA adapter to GGUF format (~50 MB)
instead of converting the entire merged model (~2 GB). At runtime, llama.cpp
loads the base GGUF + our LoRA adapter together.

Pipeline:
  1. Install dependencies
  2. Clone GitHub repo
  3. Download & preprocess multilingual training data
  4. Fine-tune Qwen2.5-3B with QLoRA (T4-optimized: batch=4, fp16)
  5. Convert LoRA adapter → GGUF LoRA format   ← ONLY ~50 MB, no full merge!
  6. Push LoRA GGUF to HuggingFace Hub

Local inference (no GPU):
  Base GGUF  : downloaded directly from Qwen/Qwen2.5-3B-Instruct-GGUF (~2 GB)
  LoRA GGUF  : downloaded from your HF repo (~50 MB)
  → Together : your fully fine-tuned model running on CPU

USAGE on Kaggle:
  1. Create a new Kaggle Notebook (GPU T4 x1)
  2. Settings → Internet → ON
  3. Run:
       !python kaggle_train.py \\
           --hf_token  "hf_YOUR_TOKEN_HERE" \\
           --hf_repo   "your-username/qwen25-multilingual-lora-gguf" \\
           --github_url "https://github.com/your-username/chatbot_multilingual"

  Or use Kaggle Secrets: HF_TOKEN, HF_REPO, GITHUB_URL

  After completion, run locally (no GPU):
       python chatbot_multilingual.py --mode cpu_chat \\
           --base_gguf  ./models/qwen2.5-3b-instruct-q4_k_m.gguf \\
           --lora_gguf  ./models/multilingual_lora.gguf
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ─── Paths (Kaggle filesystem) ────────────────────────────────────────────────
WORKING        = Path("/kaggle/working")
REPO_DIR       = WORKING / "chatbot_multilingual"
ADAPTER_DIR    = WORKING / "checkpoints" / "best_lora_adapter"
LORA_GGUF_PATH = WORKING / "multilingual_lora.gguf"
LLAMA_CPP_DIR  = Path("/tmp/llama.cpp")

# ─── Base GGUF (already on HuggingFace, no conversion needed) ────────────────
BASE_GGUF_REPO     = "Qwen/Qwen2.5-3B-Instruct-GGUF"
BASE_GGUF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"   # ~2 GB

# ─── T4-optimised training overrides ─────────────────────────────────────────
# config.py targets 4GB VRAM (GTX 1650 Ti). T4 has 16GB — we run 4x faster.
KAGGLE_ENV = {
    "KAGGLE_BATCH_SIZE": "4",           # was 1
    "KAGGLE_GRAD_ACCUM": "4",           # was 16 (effective batch still = 16)
    "KAGGLE_OPTIM":      "paged_adamw_8bit",
    "KAGGLE_FP16":       "true",
    "KAGGLE_EPOCHS":     "3",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: str | list, cwd=None, env_extra: dict | None = None):
    env = {**os.environ, **(env_extra or {})}
    if isinstance(cmd, str):
        print(f"\n$ {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    else:
        print(f"\n$ {' '.join(str(c) for c in cmd)}")
        result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def _pip(*packages: str):
    """Install packages without touching torch/torchvision."""
    _run([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade", "--upgrade-strategy", "only-if-needed",
        *packages
    ])


def _find_adapter() -> Path:
    """Returns the best LoRA adapter path from training output."""
    if ADAPTER_DIR.exists():
        return ADAPTER_DIR
    # Fall back to latest checkpoint
    ckpt_dir = WORKING / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        print(f"[Adapter] Using latest checkpoint: {checkpoints[-1]}")
        return checkpoints[-1]
    raise FileNotFoundError(
        "No LoRA adapters found in /kaggle/working/checkpoints/\n"
        "Make sure training completed successfully (Step 4)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Install Python dependencies
# ══════════════════════════════════════════════════════════════════════════════

def step1_install():
    print("\n" + "═"*60)
    print("  STEP 1 — Installing Python dependencies")
    print("═"*60)

    # ── Phase 1: co-resolve the tightly coupled core libs ────────────────────
    # All four packages have strict inter-version constraints that MUST be
    # resolved in a single pip call:
    #   • peft>=0.14.0       — removes BloomPreTrainedModel import (fixes torchvision crash)
    #   • transformers       — requires tokenizers >=0.22.0, <=0.23.0
    #   • tokenizers pinned  — Phase 2 alone would upgrade it to 0.23.1+, breaking transformers
    #   • trl                — depends on both peft and transformers
    _run([
        sys.executable, "-m", "pip", "install", "-q",
        "peft>=0.14.0",
        "transformers>=4.45.0",
        "tokenizers>=0.20.0,<=0.23.0",
        "trl>=0.11.0",
    ])

    # ── Phase 2: remaining training deps (do not touch torch/torchvision) ─────
    _pip(
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "datasets>=3.0.0",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.25.0",
        "scipy>=1.13.0",
        "tqdm>=4.66.0",
    )
    print("[Step 1] ✓ Dependencies installed")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clone the repo
# ══════════════════════════════════════════════════════════════════════════════

def step2_clone(github_url: str):
    print("\n" + "═"*60)
    print("  STEP 2 — Cloning repository")
    print("═"*60)
    if REPO_DIR.exists():
        print(f"[Step 2] Repo already at {REPO_DIR}, pulling latest...")
        _run("git pull", cwd=REPO_DIR)
    else:
        _run(f"git clone {github_url} {REPO_DIR}")
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / "src"))
    print("[Step 2] ✓ Repository ready")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Download & preprocess training data
# ══════════════════════════════════════════════════════════════════════════════

def step3_download_data():
    print("\n" + "═"*60)
    print("  STEP 3 — Downloading training datasets")
    print("═"*60)
    processed_train = REPO_DIR / "data" / "processed" / "train.jsonl"
    if processed_train.exists():
        print("[Step 3] ✓ Data already downloaded, skipping.")
        return
    _run(
        [sys.executable, "chatbot_multilingual.py", "--mode", "download"],
        cwd=REPO_DIR,
    )
    print("[Step 3] ✓ Datasets ready")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — QLoRA Fine-tuning (T4-optimized)
# ══════════════════════════════════════════════════════════════════════════════

def step4_train():
    print("\n" + "═"*60)
    print("  STEP 4 — QLoRA Fine-tuning on T4 GPU")
    print("  Config: batch=4, grad_accum=4, paged_adamw_8bit, fp16")
    print("  Expected time: ~60–90 minutes")
    print("═"*60)

    # Write a config patch that overrides settings for T4 before training starts
    patch_script = REPO_DIR / "_kaggle_config_patch.py"
    patch_script.write_text(
        """# Auto-generated by kaggle_train.py — patches config for Kaggle T4
import config, os
from pathlib import Path

config.BATCH_SIZE              = int(os.environ.get("KAGGLE_BATCH_SIZE", "4"))
config.GRAD_ACCUMULATION_STEPS = int(os.environ.get("KAGGLE_GRAD_ACCUM", "4"))
config.FP16_TRAINING           = os.environ.get("KAGGLE_FP16", "true") == "true"
config.NUM_EPOCHS              = int(os.environ.get("KAGGLE_EPOCHS", "3"))
config.CHECKPOINTS_DIR = Path("/kaggle/working/checkpoints")
config.LOGS_DIR        = Path("/kaggle/working/logs")
config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
print("[KagglePatch] ✓ T4-optimized config applied.")
""",
        encoding="utf-8",
    )

    launcher = REPO_DIR / "_kaggle_train_launcher.py"
    launcher.write_text(
        """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import _kaggle_config_patch  # noqa — applies T4 overrides before train imports config
from train import main
main()
""",
        encoding="utf-8",
    )

    _run(
        [sys.executable, "_kaggle_train_launcher.py"],
        cwd=REPO_DIR,
        env_extra=KAGGLE_ENV,
    )
    print("[Step 4] ✓ Training complete — LoRA adapters saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Convert LoRA adapter → GGUF LoRA format
#
# WHY THIS IS BETTER THAN MERGING:
#   • Full merge approach: Load 6GB FP16 model + LoRA → merge → convert 6GB →
#     quantize → 2GB GGUF. Takes ~30 min extra, needs ~12GB disk.
#   • LoRA-only approach: Convert ONLY the adapter weights → ~50MB GGUF LoRA.
#     Takes ~5 min. The base GGUF already exists on HuggingFace!
# ══════════════════════════════════════════════════════════════════════════════

def step5_convert_lora_to_gguf():
    print("\n" + "═"*60)
    print("  STEP 5 — Converting LoRA adapter → GGUF format")
    print("  (Only ~50 MB — base GGUF downloaded separately from HF)")
    print("═"*60)

    if LORA_GGUF_PATH.exists():
        print(f"[Step 5] ✓ LoRA GGUF already at {LORA_GGUF_PATH}")
        return

    adapter_path = _find_adapter()
    print(f"[Step 5] LoRA adapter: {adapter_path}")

    # ── 5a. Clone llama.cpp ───────────────────────────────────────────────────
    if not LLAMA_CPP_DIR.exists():
        print("[Step 5] Cloning llama.cpp (depth=1)...")
        _run(f"git clone --depth=1 https://github.com/ggerganov/llama.cpp {LLAMA_CPP_DIR}")

    req_file = LLAMA_CPP_DIR / "requirements.txt"
    if req_file.exists():
        _pip(f"-r {req_file}")

    # ── 5b. Convert PEFT LoRA → GGUF LoRA ────────────────────────────────────
    # llama.cpp's convert_lora_to_gguf.py converts HuggingFace PEFT adapter
    # format into a llama.cpp-compatible GGUF LoRA file.
    convert_lora_script = LLAMA_CPP_DIR / "convert_lora_to_gguf.py"

    if not convert_lora_script.exists():
        # Older llama.cpp: try alternative name
        convert_lora_script = LLAMA_CPP_DIR / "convert-lora-to-gguf.py"

    if not convert_lora_script.exists():
        raise FileNotFoundError(
            "convert_lora_to_gguf.py not found in llama.cpp!\n"
            "This script was added in llama.cpp ~March 2024.\n"
            "Try: git -C /tmp/llama.cpp pull to get the latest version."
        )

    print(f"[Step 5] Converting {adapter_path.name} → {LORA_GGUF_PATH.name} ...")
    _run(
        f"{sys.executable} {convert_lora_script} "
        f"{adapter_path} "
        f"--outfile {LORA_GGUF_PATH} "
        f"--base Qwen/Qwen2.5-3B-Instruct"
    )

    size_mb = LORA_GGUF_PATH.stat().st_size / 1e6
    print(f"[Step 5] ✓ LoRA GGUF ready: {LORA_GGUF_PATH.name} ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Push LoRA GGUF to HuggingFace Hub
# ══════════════════════════════════════════════════════════════════════════════

def step6_push_to_hub(hf_token: str, hf_repo: str):
    print("\n" + "═"*60)
    print(f"  STEP 6 — Pushing LoRA GGUF to HuggingFace Hub: {hf_repo}")
    print("═"*60)

    from huggingface_hub import HfApi, login

    login(token=hf_token)
    api = HfApi()

    try:
        api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[Step 6] Note: {e}")

    # Upload LoRA GGUF
    size_mb = LORA_GGUF_PATH.stat().st_size / 1e6
    print(f"[Step 6] Uploading {LORA_GGUF_PATH.name} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(LORA_GGUF_PATH),
        path_in_repo=LORA_GGUF_PATH.name,
        repo_id=hf_repo,
        repo_type="model",
    )

    # Write a clear README explaining the two-file setup
    readme = f"""---
language:
- en
- hi
tags:
- qwen2.5
- multilingual
- hindi
- hinglish
- gguf
- lora
- llama-cpp
base_model: Qwen/Qwen2.5-3B-Instruct
license: apache-2.0
---

# Qwen2.5-3B Multilingual LoRA — GGUF format

Fine-tuned LoRA adapter for `Qwen/Qwen2.5-3B-Instruct` on a multilingual
**Hindi · English · Hinglish** dataset (~16,000 samples).

This repo contains **only the LoRA adapter** in GGUF format (~50 MB).
You need to pair it with the base model GGUF from Qwen's official repo.

## How this differs from Qwen/Qwen2.5-3B-Instruct-GGUF

| | Qwen's official GGUF | This repo |
|---|---|---|
| What it is | Base model, no fine-tuning | Fine-tuning adapter only |
| Size | ~2 GB | ~50 MB |
| Hindi/Hinglish | Generic | Trained on curated dataset |
| Use together? | ✓ Base file | ✓ Add-on file |

## CPU Inference Setup (no GPU needed)

```bash
pip install llama-cpp-python huggingface-hub

# 1. Download base model from Qwen's official repo (~2 GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \\
    {BASE_GGUF_FILENAME} --local-dir ./models/

# 2. Download your fine-tuned LoRA adapter (~50 MB)
huggingface-cli download {hf_repo} \\
    {LORA_GGUF_PATH.name} --local-dir ./models/

# 3. Run the chatbot (loads both files together)
python chatbot_multilingual.py --mode cpu_chat \\
    --base_gguf ./models/{BASE_GGUF_FILENAME} \\
    --lora_gguf ./models/{LORA_GGUF_PATH.name}
```

## Training Details
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: QLoRA (4-bit NF4, LoRA rank=16, alpha=32)
- Hardware: Kaggle T4 GPU (16 GB VRAM)
- Dataset: ~16,000 samples (English: 8k, Hindi: 5k, Hinglish: 3k)
- Epochs: 3
"""
    readme_path = WORKING / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="model",
    )

    print(f"[Step 6] ✓ Pushed to https://huggingface.co/{hf_repo}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Kaggle T4 Training Pipeline — Smart LoRA-only GGUF approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace write token (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--hf_repo",
        default=os.environ.get("HF_REPO", ""),
        help="HF repo to push LoRA GGUF (e.g. username/qwen25-multilingual-lora)",
    )
    p.add_argument(
        "--github_url",
        default=os.environ.get("GITHUB_URL", ""),
        help="GitHub repo URL to clone",
    )
    p.add_argument(
        "--start_from", type=int, default=1,
        help="Resume from step N (1–6). Useful if session timed out.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.hf_token:
        print("[Error] --hf_token is required. Get one at: huggingface.co/settings/tokens")
        sys.exit(1)
    if not args.hf_repo:
        print("[Error] --hf_repo is required (e.g. your-username/qwen25-multilingual-lora)")
        sys.exit(1)
    if not args.github_url:
        print("[Error] --github_url is required (your GitHub repo URL)")
        sys.exit(1)

    t_start = time.time()
    print("\n" + "═"*60)
    print("  Kaggle T4 — Multilingual Chatbot Training Pipeline")
    print(f"  HF Repo       : {args.hf_repo}")
    print(f"  Starting from : Step {args.start_from}")
    print("═"*60)
    print()
    print("  APPROACH: Train → Convert LoRA only (~50 MB)")
    print("  Base GGUF (~2 GB) downloaded separately from:")
    print(f"  https://huggingface.co/{BASE_GGUF_REPO}")
    print("═"*60)

    steps = [
        (1, "Install deps",             step1_install),
        (2, "Clone repo",               lambda: step2_clone(args.github_url)),
        (3, "Download data",            step3_download_data),
        (4, "QLoRA training",           step4_train),
        (5, "Convert LoRA → GGUF",      step5_convert_lora_to_gguf),
        (6, "Push LoRA GGUF to HF Hub", lambda: step6_push_to_hub(args.hf_token, args.hf_repo)),
    ]

    for num, name, fn in steps:
        if num < args.start_from:
            print(f"\n[Step {num}] Skipping: {name}")
            continue
        fn()

    elapsed = (time.time() - t_start) / 60
    print("\n" + "═"*60)
    print(f"  ✓ Done in {elapsed:.1f} minutes!")
    print()
    print("  LOCAL SETUP (no GPU needed):")
    print()
    print("  # 1. Install CPU inference engine")
    print("  pip install llama-cpp-python huggingface-hub")
    print()
    print(f"  # 2. Download base model from Qwen's official repo (~2 GB)")
    print(f"  huggingface-cli download {BASE_GGUF_REPO} \\")
    print(f"      {BASE_GGUF_FILENAME} --local-dir ./models/")
    print()
    print(f"  # 3. Download your fine-tuned LoRA adapter (~50 MB)")
    print(f"  huggingface-cli download {args.hf_repo} \\")
    print(f"      {LORA_GGUF_PATH.name} --local-dir ./models/")
    print()
    print("  # 4. Chat!")
    print("  python chatbot_multilingual.py --mode cpu_chat \\")
    print(f"      --base_gguf ./models/{BASE_GGUF_FILENAME} \\")
    print(f"      --lora_gguf ./models/{LORA_GGUF_PATH.name}")
    print("═"*60)


if __name__ == "__main__":
    main()
