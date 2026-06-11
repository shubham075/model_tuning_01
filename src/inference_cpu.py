"""
src/inference_cpu.py — CPU-only streaming chat using GGUF + llama-cpp-python.

Supports two modes:
  1. Base GGUF only  : Qwen/Qwen2.5-3B-Instruct-GGUF (no fine-tuning)
  2. Base GGUF + LoRA: Base model + your fine-tuned LoRA adapter  ← RECOMMENDED

The LoRA adapter is the result of Kaggle training (Steps 1–6 in kaggle_train.py).
It is applied ON TOP of the base GGUF at runtime — no merging needed.

No GPU, no CUDA, no bitsandbytes required.
Works on any CPU with AVX2 (Ryzen 6700H, Intel 10th gen+).

RAM requirements:
  Base GGUF (Q4_K_M) : ~2.0 GB
  + LoRA adapter      : ~0.1 GB
  Total               : ~2.5 GB  (fits easily in 16 GB RAM)

Speed on Ryzen 6700H : ~5–15 tokens/sec

Setup:
  pip install -r requirements_cpu.txt

Downloads:
  # Base model (from Qwen's official repo, ~2 GB)
  huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \\
      qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models/

  # Your fine-tuned LoRA adapter (~50 MB)
  huggingface-cli download your-username/qwen25-multilingual-lora \\
      multilingual_lora.gguf --local-dir ./models/

Run:
  python chatbot_multilingual.py --mode cpu_chat \\
      --base_gguf ./models/qwen2.5-3b-instruct-q4_k_m.gguf \\
      --lora_gguf ./models/multilingual_lora.gguf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    SYSTEM_PROMPT,
    CPU_MAX_TOKENS,
    CPU_TEMPERATURE,
    CPU_TOP_P,
    CPU_TOP_K,
    CPU_REPEAT_PENALTY,
    CPU_N_CTX,
    CPU_N_THREADS,
    DEFAULT_BASE_GGUF_PATH,
)

_BANNER_BASE = """
╔══════════════════════════════════════════════════════════════╗
║   Multilingual Chatbot — CPU Mode (Base GGUF only)          ║
║   Languages: Hindi · English · Hinglish                     ║
║   Type 'quit' or Ctrl+C to exit                             ║
╚══════════════════════════════════════════════════════════════╝
"""

_BANNER_LORA = """
╔══════════════════════════════════════════════════════════════╗
║   Multilingual Chatbot — CPU Mode (Base + Fine-tuned LoRA)  ║
║   Languages: Hindi · English · Hinglish                     ║
║   Type 'quit' or Ctrl+C to exit                             ║
╚══════════════════════════════════════════════════════════════╝
"""


def _check_llama_cpp():
    """Ensures llama-cpp-python is installed."""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        print(
            "\n[Error] llama-cpp-python is not installed.\n"
            "\n  Install it:\n"
            "    pip install -r requirements_cpu.txt\n"
            "\n  For best speed on Ryzen 6700H (AVX2):\n"
            "    pip install llama-cpp-python --prefer-binary   (Windows)\n"
        )
        sys.exit(1)


def _check_file(path: str, label: str) -> Path:
    """Checks a file exists and prints a helpful error if not."""
    p = Path(path)
    if not p.exists():
        print(
            f"\n[Error] {label} not found: {path}\n"
            "\n  Download it with:\n"
            "    pip install huggingface-hub\n"
            "    huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \\\n"
            "        qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models/\n"
        )
        sys.exit(1)
    return p


def load_model(base_gguf_path: str, lora_gguf_path: str | None = None):
    """
    Load the GGUF model, optionally with a LoRA adapter.

    Args:
        base_gguf_path  : Path to the base model GGUF file.
        lora_gguf_path  : Path to the LoRA GGUF file (optional).
                          If provided, fine-tuning is applied at runtime.
    Returns:
        llm (Llama instance)
    """
    Llama = _check_llama_cpp()
    base_path = _check_file(base_gguf_path, "Base GGUF")

    if lora_gguf_path:
        lora_path = _check_file(lora_gguf_path, "LoRA GGUF adapter")
        print(f"\n[CPU] Base model : {base_path.name}")
        print(f"[CPU] LoRA adapter: {lora_path.name}  ← your fine-tuning")
        print(f"[CPU] Threads     : {CPU_N_THREADS}  |  Context: {CPU_N_CTX} tokens")
        print("[CPU] Loading... (first load may take 15–30s)\n")

        llm = Llama(
            model_path=str(base_path),
            lora_path=str(lora_path),      # ← applies LoRA weights at runtime
            lora_scale=1.0,                # 1.0 = full LoRA strength
            n_ctx=CPU_N_CTX,
            n_threads=CPU_N_THREADS,
            n_gpu_layers=0,                # 0 = pure CPU, no GPU
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )
    else:
        print(f"\n[CPU] Base model : {base_path.name}  (no LoRA — base model only)")
        print(f"[CPU] Threads    : {CPU_N_THREADS}  |  Context: {CPU_N_CTX} tokens")
        print("[CPU] Loading... (first load may take 15–30s)\n")

        llm = Llama(
            model_path=str(base_path),
            n_ctx=CPU_N_CTX,
            n_threads=CPU_N_THREADS,
            n_gpu_layers=0,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )

    print("[CPU] ✓ Model loaded successfully!\n")
    return llm


def _build_prompt(messages: list[dict]) -> str:
    """
    Build a Qwen2.5 chat prompt from message history.
    Format: <|im_start|>role\\ncontent<|im_end|>\\n
    """
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"   # open assistant turn
    return prompt


def run_cpu_chat(base_gguf_path: str | None = None, lora_gguf_path: str | None = None):
    """
    Interactive streaming chat loop — runs entirely on CPU.

    Args:
        base_gguf_path : Path to base model GGUF. Falls back to config default.
        lora_gguf_path : Path to LoRA GGUF adapter. If None, runs base-only.
    """
    base_gguf_path = base_gguf_path or DEFAULT_BASE_GGUF_PATH

    banner = _BANNER_LORA if lora_gguf_path else _BANNER_BASE
    print(banner)

    if not lora_gguf_path:
        print(
            "  ⚠ No LoRA adapter specified — running base model only.\n"
            "  For fine-tuned responses, add: --lora_gguf ./models/multilingual_lora.gguf\n"
        )

    llm = load_model(base_gguf_path, lora_gguf_path)

    # Multi-turn conversation history
    history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        # ── User input ────────────────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Goodbye!]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("\n[Goodbye!]")
            break

        history.append({"role": "user", "content": user_input})

        # ── Stream response ───────────────────────────────────────────────────
        print("\nBot: ", end="", flush=True)
        full_response = ""

        try:
            stream = llm(
                _build_prompt(history),
                max_tokens=CPU_MAX_TOKENS,
                temperature=CPU_TEMPERATURE,
                top_p=CPU_TOP_P,
                top_k=CPU_TOP_K,
                repeat_penalty=CPU_REPEAT_PENALTY,
                stream=True,
                stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            )
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                print(token, end="", flush=True)
                full_response += token

        except KeyboardInterrupt:
            print("\n[Generation stopped]")

        print("\n")

        # Clean any leaked stop tokens
        full_response = full_response.replace("<|im_end|>", "").strip()
        history.append({"role": "assistant", "content": full_response})

        # ── Trim history to avoid context overflow ────────────────────────────
        # Keep system prompt + last 8 turns (4 exchanges)
        if len(history) > 9:
            history = [history[0]] + history[-8:]
