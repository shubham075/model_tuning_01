"""
config.py — Central configuration for the multilingual chatbot.
All hyperparameters, paths, and model IDs live here.

Includes both:
  - GPU Training settings (QLoRA on GTX 1650 Ti / Kaggle T4)
  - CPU Inference settings (GGUF Q4_K_M via llama-cpp-python)
"""

import os
from pathlib import Path

# ─── Directory Layout ─────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data" / "raw"
PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
ONNX_DIR        = ROOT_DIR / "onnx"
LOGS_DIR        = ROOT_DIR / "logs"

for _d in [DATA_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, ONNX_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Model IDs ────────────────────────────────────────────────────────────────
# Used for QLoRA training (downloaded in FP16, quantized on-the-fly by BnB)
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Used for fast inference after training (GPTQ-Int4 fits in 4GB VRAM)
GPTQ_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"

# ─── LoRA Hyperparameters ────────────────────────────────────────────────────
LORA_R              = 16       # Rank — higher = more capacity, more VRAM
LORA_ALPHA          = 32       # Scaling factor (usually 2x rank)
LORA_DROPOUT        = 0.05
# Target all attention + FFN projections for best coverage
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ─── Training Hyperparameters ─────────────────────────────────────────────────
MAX_SEQ_LENGTH          = 512   # Keep short for 4GB VRAM
BATCH_SIZE              = 1     # 1 per step on 4GB VRAM
GRAD_ACCUMULATION_STEPS = 16    # Effective batch = 16
LEARNING_RATE           = 2e-4
NUM_EPOCHS              = 3
WARMUP_RATIO            = 0.05
LR_SCHEDULER_TYPE       = "cosine"
WEIGHT_DECAY            = 0.01
SAVE_STEPS              = 200
EVAL_STEPS              = 200
LOGGING_STEPS           = 25
FP16_TRAINING           = True  # GTX 1650 Ti supports FP16

# ─── Dataset Limits (adjust based on available time) ─────────────────────────
MAX_ENGLISH_SAMPLES  = 8_000   # From ultrachat_200k
MAX_HINDI_SAMPLES    = 5_000   # From IITB parallel corpus (formatted as chat)
MAX_HINGLISH_SAMPLES = 3_000   # Synthetic generation
MAX_EVAL_SAMPLES     = 400     # Total across all languages

# ─── Inference Parameters ────────────────────────────────────────────────────
MAX_NEW_TOKENS      = 256
TEMPERATURE         = 0.7
TOP_P               = 0.9
TOP_K               = 50
REPETITION_PENALTY  = 1.15
DO_SAMPLE           = True

# ─── Chat System Prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful and friendly AI assistant who speaks fluently in Hindi, "
    "English, and Hinglish (mixed Hindi-English). Always reply in the same "
    "language or mix that the user is using. Be natural, concise, and accurate."
)

# ─── CPU Inference (GGUF / llama-cpp-python) ──────────────────────────────────
#
# TWO-FILE SETUP (recommended after Kaggle training):
#
#   File 1 — Base GGUF (~2 GB): download from Qwen's official HF repo
#     huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
#         qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models/
#
#   File 2 — LoRA GGUF (~50 MB): download from YOUR HF repo after Kaggle training
#     huggingface-cli download your-username/qwen25-multilingual-lora \
#         multilingual_lora.gguf --local-dir ./models/
#
#   Run (both files together = your fully fine-tuned model on CPU):
#     python chatbot_multilingual.py --mode cpu_chat \
#         --base_gguf ./models/qwen2.5-3b-instruct-q4_k_m.gguf \
#         --lora_gguf ./models/multilingual_lora.gguf

# Default paths (override via --base_gguf and --lora_gguf flags)
DEFAULT_BASE_GGUF_PATH = str(ROOT_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")
DEFAULT_LORA_GGUF_PATH = str(ROOT_DIR / "models" / "multilingual_lora.gguf")

# Number of CPU threads for llama.cpp
# Rule of thumb: PHYSICAL cores only (not logical/hyperthreaded)
# Ryzen 6700H = 6 physical cores → CPU_N_THREADS = 6
# Check yours: python -c "import os; print(os.cpu_count() // 2)"
CPU_N_THREADS      = 6

# Context window in tokens. 2048 = good for chat (~100 MB extra RAM)
CPU_N_CTX          = 2048

# Generation settings for CPU mode
CPU_MAX_TOKENS     = 256
CPU_TEMPERATURE    = 0.7
CPU_TOP_P          = 0.9
CPU_TOP_K          = 50
CPU_REPEAT_PENALTY = 1.15
