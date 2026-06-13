"""
src/model.py — Model loading for training (QLoRA / LoRA) and inference.

Hardware auto-detection order:
  1. TPU v5e-8  → BF16 + standard LoRA (no quantization; bitsandbytes is CUDA-only)
  2. Modern GPU (Turing+, compute ≥7.0) → 4-bit NF4 QLoRA via bitsandbytes
  3. Pascal GPU (compute <7.0, e.g. P100) → FP16 standard LoRA

Inference strategy order:
  1. Base model 4-bit via bitsandbytes (~2GB VRAM)  ← confirmed working
  2. Base model FP16 split GPU+CPU                  (no quantization deps)
  3. Small 1.5B FP16 fully on GPU                   (last resort)
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    BASE_MODEL_ID, GPTQ_MODEL_ID,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)

SMALL_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# ─── Hardware Detection ───────────────────────────────────────────────────────────────────

def _detect_hardware() -> str:
    """
    Returns one of: 'tpu' | 'gpu_modern' | 'gpu_pascal' | 'cpu'
    - tpu        : Kaggle TPU v5e-8 (torch_xla available)
    - gpu_modern : Turing+ GPU (compute ≥7.0, e.g. T4, A100) → QLoRA ok
    - gpu_pascal : Pascal GPU  (compute <7.0, e.g. P100)     → FP16 LoRA only
    - cpu        : No accelerator found
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.xla_device()   # raises if no TPU present
        return 'tpu'
    except Exception:
        pass

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return 'gpu_modern' if major >= 7 else 'gpu_pascal'

    return 'cpu'


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str = BASE_MODEL_ID) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ─── BnB Config ───────────────────────────────────────────────────────────────

def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ─── Training ─────────────────────────────────────────────────────────────────

def load_for_training():
    """Returns (model, tokenizer) ready for fine-tuning.
    Auto-detects hardware and picks the right loading strategy:
      - TPU v5e-8   → bf16 standard LoRA (bitsandbytes not supported on TPU)
      - Modern GPU  → 4-bit NF4 QLoRA via bitsandbytes
      - Pascal GPU  → FP16 standard LoRA
    """
    hw = _detect_hardware()
    tokenizer = load_tokenizer(BASE_MODEL_ID)

    if hw == 'tpu':
        # ── TPU v5e-8: bf16 standard LoRA (bitsandbytes is CUDA-only) ─────────
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"[Model] TPU v5e-8 detected — loading {BASE_MODEL_ID} in bf16")
        print("        (No 4-bit quantization — bitsandbytes is CUDA-only)")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    elif hw == 'gpu_modern':
        # ── Turing+ GPU (T4, A100, RTX): 4-bit NF4 QLoRA ─────────────────────
        print(f"[Model] Modern GPU ({torch.cuda.get_device_name(0)}) — 4-bit QLoRA")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=_bnb_4bit_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    elif hw == 'gpu_pascal':
        # ── Pascal GPU (P100): FP16 LoRA, no quantization ─────────────────────
        print(f"[Model] Pascal GPU ({torch.cuda.get_device_name(0)}) — FP16 LoRA")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    else:
        raise RuntimeError(
            "[Error] No supported training device found (TPU / GPU).\n"
            "  CPU training is not supported — use Kaggle TPU v5e-8 or a GPU."
        )

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ─── Inference helpers ────────────────────────────────────────────────────────

def _try_load(label: str, fn):
    """
    Runs fn(). Prints ✓ or ✗ with the full error.
    Uses BaseException so sys.exit() (from broken auto-gptq) is also caught.
    """
    print(f"\n[Model] Trying: {label} ...")
    try:
        result = fn()
        print(f"[Model] ✓  {label}  — loaded!")
        return result
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        print(f"[Model] ✗  {label}  — {type(e).__name__}: {e}")
        return None


# ─── Inference ────────────────────────────────────────────────────────────────

def load_for_inference(lora_adapter_path: str | None = None):
    """
    Returns (model, tokenizer). Tries strategies in order with full output.
    Will never silently exit.
    """
    use_quantization = True
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:
            print(f"[Model] Detected Pascal GPU ({torch.cuda.get_device_name(0)}).")
            print("        Using FP16 for base + LoRA loading and skipping standalone 4-bit.")
            use_quantization = False

    # S0 — LoRA adapter over base (only after fine-tuning)
    if lora_adapter_path and Path(lora_adapter_path).exists():
        def _lora():
            tokenizer = load_tokenizer(BASE_MODEL_ID)
            if use_quantization:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    quantization_config=_bnb_4bit_config(),
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            model = PeftModel.from_pretrained(base, lora_adapter_path)
            model.eval()
            return model, tokenizer

        r = _try_load(f"Base + LoRA ({lora_adapter_path})", _lora)
        if r:
            return r

    # ── GPTQ BLOCK (disabled until gptqmodel is installed) ──────────────────
    # To enable: install VS Build Tools, then `pip install gptqmodel ninja`
    # Then uncomment this block:
    #
    # def _gptq():
    #     tokenizer = load_tokenizer(GPTQ_MODEL_ID)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         GPTQ_MODEL_ID, device_map="auto", trust_remote_code=True,
    #     )
    #     model.eval()
    #     return model, tokenizer
    # r = _try_load(f"GPTQ-Int4 ({GPTQ_MODEL_ID})", _gptq)
    # if r: return r
    # ─────────────────────────────────────────────────────────────────────────

    # S1 — Base model 4-bit via bitsandbytes (~2GB VRAM)  [confirmed working]
    if use_quantization:
        def _bnb4bit():
            tokenizer = load_tokenizer(BASE_MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=_bnb_4bit_config(),
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model.eval()
            return model, tokenizer

        r = _try_load(f"3B 4-bit BnB ({BASE_MODEL_ID})", _bnb4bit)
        if r:
            return r

    # S2 — Base model FP16 split GPU+CPU (no quantization lib needed)
    def _fp16_split():
        tokenizer = load_tokenizer(BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            max_memory={0: "3GiB", "cpu": "4GiB"},
        )
        model.eval()
        return model, tokenizer

    r = _try_load(f"3B FP16 GPU+CPU split ({BASE_MODEL_ID})", _fp16_split)
    if r:
        return r

    # S3 — Smaller 1.5B fully on GPU (~3.1GB VRAM, always fits GTX 1650 Ti)
    def _small():
        print("  [Info] Using 1.5B model — still supports Hindi/English/Hinglish.")
        tokenizer = load_tokenizer(SMALL_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            SMALL_MODEL_ID,
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        return model, tokenizer

    r = _try_load(f"1.5B FP16 GPU ({SMALL_MODEL_ID})", _small)
    if r:
        return r

    raise RuntimeError(
        "[Error] All model strategies failed.\n"
        "  Check: internet connection, disk space (~10GB needed), CUDA install."
    )
