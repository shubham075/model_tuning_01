"""
src/train.py — QLoRA / LoRA fine-tuning loop for the multilingual chatbot.

Uses HuggingFace Trainer (not SFTTrainer) for full control over label masking.

Supported hardware (auto-detected):
  - Kaggle TPU v5e-8  : bf16, standard LoRA, adamw_torch, 8 cores via xmp.spawn
  - Kaggle T4 GPU     : fp16, QLoRA 4-bit, paged_adamw_8bit
  - GTX 1650 Ti (local): fp16, QLoRA 4-bit, gradient accumulation

Run (GPU/CPU locally):
  python src/train.py

Run (TPU via Kaggle launcher):
  Handled automatically by _kaggle_train_launcher.py via xmp.spawn
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    BATCH_SIZE, GRAD_ACCUMULATION_STEPS, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO, LR_SCHEDULER_TYPE, WEIGHT_DECAY,
    SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS, FP16_TRAINING, BF16_TRAINING, MAX_SEQ_LENGTH,
)
from model import load_for_training
from dataset import MultilingualChatDataset, get_collate_fn

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


# ─── Hardware Detection ─────────────────────────────────────────────────────────────

def _is_tpu() -> bool:
    """
    Returns True when running on a Kaggle TPU v5e-8 (PJRT backend).

    Intentionally does NOT call xm.xla_device() here — doing so at module
    import time triggers PJRT initialization before the env is set up,
    which can cause 'Expected 8 worker addresses, got 1' on v5e-8.

    Instead, we check:
      1. torch_xla is importable (pre-installed on Kaggle TPU notebooks)
      2. PJRT_DEVICE=TPU is set (we set this in _kaggle_train_launcher.py)
    """
    try:
        import torch_xla  # noqa — just check availability, don't init
        return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
    except ImportError:
        return False

ON_TPU: bool = _is_tpu()


def get_training_args() -> TrainingArguments:
    # paged_adamw_8bit requires bitsandbytes (CUDA-only) — not available on TPU
    optim = "adamw_torch" if ON_TPU else os.environ.get("KAGGLE_OPTIM", "adamw_torch")

    return TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=NUM_EPOCHS,

        # ── Batching ────────────────────────────────────────────────────────
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,

        # ── Precision & Memory ─────────────────────────────────────────────
        gradient_checkpointing=True,
        fp16=FP16_TRAINING and not ON_TPU,  # FP16 on CUDA GPU only
        bf16=BF16_TRAINING or ON_TPU,       # BF16 on TPU (native) or if explicitly set
        optim=optim,
        dataloader_pin_memory=False,        # Required for TPU; harmless on GPU

        # ── Learning rate ───────────────────────────────────────────────
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        # ── Logging & Saving ──────────────────────────────────────────
        logging_dir=str(LOGS_DIR),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── Misc ─────────────────────────────────────────────────────────
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


def compute_metrics(eval_pred):
    """Simple perplexity metric from eval loss."""
    import math
    # Trainer passes logits; we just use the loss from eval
    # (Trainer computes loss internally when labels are provided)
    return {}


def main():
    print("=" * 60)
    print("  LoRA Fine-tuning — Multilingual Chatbot")
    if ON_TPU:
        import torch_xla.core.xla_model as xm
        print(f"  Device: TPU ({xm.xla_device()}) — bf16, standard LoRA")
    elif torch.cuda.is_available():
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)}) — fp16, QLoRA")
    else:
        print("  Device: CPU (training not recommended)")
    print("=" * 60)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    model, tokenizer = load_for_training()

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_ds = MultilingualChatDataset(tokenizer, split="train")
    eval_ds  = MultilingualChatDataset(tokenizer, split="eval")

    collate_fn = get_collate_fn(tokenizer)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = get_training_args()

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,         # Stop if no improvement for 5 evals
                early_stopping_threshold=0.001,
            )
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[Train] Starting training...")
    trainer.train()

    # ── Save best LoRA adapters ───────────────────────────────────────────────
    best_adapter_path = CHECKPOINTS_DIR / "best_lora_adapter"
    model.save_pretrained(str(best_adapter_path))
    tokenizer.save_pretrained(str(best_adapter_path))

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best LoRA adapter saved → {best_adapter_path}")
    print("  Run inference with:")
    print("    python chatbot_multilingual.py --mode chat --adapter checkpoints/best_lora_adapter")
    print("=" * 60)


if __name__ == "__main__":
    main()
