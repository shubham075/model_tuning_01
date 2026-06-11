"""
src/train.py — QLoRA fine-tuning loop for the multilingual chatbot.

Uses HuggingFace Trainer (not SFTTrainer) for full control over label masking.
Supports GTX 1650 Ti (4GB VRAM) via:
  - 4-bit NF4 quantization (bitsandbytes)
  - LoRA adapters (only ~1% of params trainable)
  - Gradient checkpointing (trades compute for memory)
  - FP16 training
  - Gradient accumulation (effective batch = 16)

Run:
  python src/train.py
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
    SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS, FP16_TRAINING, MAX_SEQ_LENGTH,
)
from model import load_for_training
from dataset import MultilingualChatDataset, get_collate_fn

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=NUM_EPOCHS,

        # ── Batching ────────────────────────────────────────────────────────
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,

        # ── Memory savings ───────────────────────────────────────────────
        gradient_checkpointing=True,
        fp16=FP16_TRAINING,
        # adamw_torch works on all setups; switch to paged_adamw_8bit
        # after: pip install bitsandbytes (already installed)
        optim="adamw_torch",
        dataloader_pin_memory=False,

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
        eval_strategy="steps",          # transformers 5.x renamed evaluation_strategy
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
    print("  QLoRA Fine-tuning — Multilingual Chatbot")
    print(f"  Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
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
