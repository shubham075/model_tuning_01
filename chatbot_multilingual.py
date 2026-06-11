"""
chatbot_multilingual.py — Main entry point for the multilingual chatbot.

Modes:
  --mode download   Download and prepare training datasets
  --mode train      Fine-tune the model with QLoRA (needs GPU)
  --mode chat       Interactive chat via GPU (LoRA or GPTQ)
  --mode cpu_chat   Interactive chat via CPU only (GGUF Q4_K_M) ← No GPU needed!

Examples:
  # Step 1: Download data
  python chatbot_multilingual.py --mode download

  # Step 2: Train on Kaggle (run kaggle_train.py there, then download GGUF)
  #   See: kaggle_train.py

  # Step 3: Chat on CPU (no GPU required)
  python chatbot_multilingual.py --mode cpu_chat --gguf ./models/qwen25_multilingual_q4km.gguf

  # Step 3b: Chat on GPU (with fine-tuned LoRA adapters)
  python chatbot_multilingual.py --mode chat --adapter checkpoints/best_lora_adapter
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multilingual Hindi/English/Hinglish Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["download", "train", "chat", "cpu_chat"],
        required=True,
        help="Which operation to run.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (for --mode chat after fine-tuning).",
    )
    parser.add_argument(
        "--base_gguf",
        type=str,
        default=None,
        help=(
            "Path to the base GGUF file (for --mode cpu_chat).\n"
            "Download from Qwen's official repo (~2 GB):\n"
            "  huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \\\n"
            "      qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models/"
        ),
    )
    parser.add_argument(
        "--lora_gguf",
        type=str,
        default=None,
        help=(
            "Path to your fine-tuned LoRA GGUF adapter (for --mode cpu_chat).\n"
            "Download from YOUR HF repo after Kaggle training (~50 MB):\n"
            "  huggingface-cli download your-username/qwen25-multilingual-lora \\\n"
            "      multilingual_lora.gguf --local-dir ./models/"
        ),
    )
    return parser.parse_args()


def mode_download():
    """Downloads and prepares training datasets."""
    print("[Mode: download]")
    # Import here so heavy libs aren't loaded for other modes
    from data.download_datasets import main as download_main
    download_main()


def mode_train():
    """Fine-tunes the model with QLoRA."""
    print("[Mode: train]")
    try:
        print("[Debug] Step 1: importing config...")
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        print("[Debug] Step 2: importing torch...")
        import torch
        print(f"[Debug] torch OK — CUDA: {torch.cuda.is_available()}")

        print("[Debug] Step 3: importing transformers Trainer...")
        from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
        print("[Debug] transformers OK")

        print("[Debug] Step 4: importing peft...")
        from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
        print("[Debug] peft OK")

        print("[Debug] Step 5: importing model & dataset modules...")
        from model import load_for_training
        from dataset import MultilingualChatDataset, get_collate_fn
        print("[Debug] model & dataset OK")

        print("[Debug] Step 6: importing train.main...")
        from train import main as train_main
        print("[Debug] All imports successful — starting training...")
        train_main()

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    except BaseException as e:
        import traceback
        print(f"\n[ERROR] Training failed at the step printed above.")
        print(f"Error type : {type(e).__name__}")
        print(f"Error msg  : {e}")
        print("\nFull traceback:")
        traceback.print_exc()



def mode_chat(adapter_path: str | None = None):
    """Starts interactive chat using GPU (LoRA / GPTQ)."""
    print("[Mode: chat]")
    from src.inference import run_chat
    run_chat(lora_adapter_path=adapter_path)


def mode_cpu_chat(base_gguf_path: str | None = None, lora_gguf_path: str | None = None):
    """Starts interactive chat entirely on CPU using GGUF Q4_K_M + optional LoRA."""
    print("[Mode: cpu_chat]")
    from src.inference_cpu import run_cpu_chat
    run_cpu_chat(base_gguf_path=base_gguf_path, lora_gguf_path=lora_gguf_path)


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Multilingual Chatbot — Hindi · English · Hinglish")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 60 + "\n")

    if args.mode == "download":
        mode_download()

    elif args.mode == "train":
        # Verify data exists before attempting training
        from config import PROCESSED_DIR
        train_file = PROCESSED_DIR / "train.jsonl"
        if not train_file.exists():
            print(
                "[Error] Training data not found!\n"
                "  Run first: python chatbot_multilingual.py --mode download"
            )
            sys.exit(1)
        mode_train()

    elif args.mode == "chat":
        mode_chat(adapter_path=args.adapter)

    elif args.mode == "cpu_chat":
        mode_cpu_chat(base_gguf_path=args.base_gguf, lora_gguf_path=args.lora_gguf)


if __name__ == "__main__":
    main()
