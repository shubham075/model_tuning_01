# Kaggle Training → CPU Deployment Guide

## Architecture Overview

```mermaid
flowchart LR
    A["Kaggle T4 GPU\n(16GB VRAM)"] -->|"kaggle_train.py"| B["QLoRA Fine-tune\nQwen2.5-3B"]
    B --> C["Merge LoRA\n→ FP16 model"]
    C --> D["Convert\n→ GGUF Q4_K_M\n~2 GB"]
    D --> E["Push to\nHuggingFace Hub"]
    E -->|"huggingface-cli download"| F["Local CPU\nRyzen 6700H\n16GB RAM"]
    F -->|"--mode cpu_chat"| G["Chat at\n~5–15 tok/s\nNo GPU needed"]
```

---

## PHASE 1 — Train on Kaggle T4 (one-time, ~2 hours)

### Step 1.1 — Set up Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Settings → **Accelerator** → `GPU T4 x1`
3. Settings → **Internet** → `ON` (required to download model + datasets)
4. In the first cell, add your secrets:

```python
# Cell 1 — Set secrets (use Kaggle Secrets panel instead for security)
import os
os.environ["HF_TOKEN"]    = "hf_YOUR_TOKEN_HERE"        # huggingface.co/settings/tokens
os.environ["HF_REPO"]     = "your-username/qwen25-multilingual-gguf"
os.environ["GITHUB_URL"]  = "https://github.com/your-username/chatbot_multilingual"
```

### Step 1.2 — Run the full pipeline

```python
# Cell 2 — Upload kaggle_train.py and run it
!python kaggle_train.py \
    --hf_token  "$HF_TOKEN" \
    --hf_repo   "$HF_REPO" \
    --github_url "$GITHUB_URL"
```

> [!NOTE]
> Expected timeline on T4:
> | Step | Time |
> |---|---|
> | Install deps | ~3 min |
> | Download datasets | ~5 min |
> | QLoRA Training (3 epochs) | ~60–90 min |
> | Merge LoRA | ~5 min |
> | Convert to GGUF | ~15 min (includes cmake build) |
> | Push to HF Hub | ~10 min (2GB upload) |
> | **Total** | **~2 hours** |

### Step 1.3 — Resume from a step (if session times out)

```python
# Resume from Step 4 (skip install/clone/data download)
!python kaggle_train.py \
    --hf_token "$HF_TOKEN" \
    --hf_repo  "$HF_REPO" \
    --start_from 4
```

---

## PHASE 2 — Local CPU Setup (Ryzen 6700H, 16GB RAM)

### Step 2.1 — Install CPU requirements (no GPU needed!)

```bash
# Only 3 packages needed for CPU inference
pip install -r requirements_cpu.txt

# For maximum speed with AVX2 (Ryzen 6700H has AVX2 ✓)
# Use this instead of the line above for ~30% faster inference:
pip install llama-cpp-python --no-cache-dir
# Windows users:
pip install llama-cpp-python --prefer-binary
```

> [!TIP]
> **Do NOT install** `torch`, `bitsandbytes`, `peft`, or `transformers` for CPU mode.
> `llama-cpp-python` is self-contained — it's a Python wrapper around llama.cpp written in C++.

### Step 2.2 — Download the GGUF from HuggingFace Hub

```bash
pip install huggingface-hub

# Download (~2 GB)
huggingface-cli download your-username/qwen25-multilingual-gguf \
    qwen25_multilingual_q4km.gguf \
    --local-dir ./models/
```

### Step 2.3 — Start chatting!

```bash
python chatbot_multilingual.py --mode cpu_chat --gguf ./models/qwen25_multilingual_q4km.gguf
```

**Example session:**
```
╔══════════════════════════════════════════════════════════════╗
║   Multilingual Chatbot — CPU Mode (GGUF Q4_K_M)            ║
╚══════════════════════════════════════════════════════════════╝

You: Aaj ka weather kaisa hai Delhi mein?
Bot: Delhi mein aaj ka weather generally warm aur humid rehta hai...

You: What is machine learning?
Bot: Machine learning is a subset of artificial intelligence where...

You: quit
[Goodbye!]
```

---

## Hardware Comparison

| | Kaggle T4 (Training) | Local Ryzen 6700H (Inference) |
|---|---|---|
| **Purpose** | Train + Convert | Chat |
| **VRAM/RAM needed** | 16 GB VRAM | ~2.5 GB RAM |
| **Speed** | ~4 tok/s (training) | ~5–15 tok/s (inference) |
| **Cost** | Free (30 hrs/week) | Free (your machine) |
| **File used** | `kaggle_train.py` | GGUF Q4_K_M (~2 GB) |

---

## File Summary

| File | Purpose |
|---|---|
| `kaggle_train.py` | Run on Kaggle — trains, merges, converts, pushes |
| `src/inference_cpu.py` | CPU chat engine (llama-cpp-python) |
| `requirements_cpu.txt` | CPU-only dependencies (3 packages) |
| `requirements.txt` | GPU training dependencies |
| `config.py` | All settings — CPU thread count, GGUF path, etc. |

---

## Tuning CPU Performance

Edit `config.py` to match your CPU:

```python
# For Ryzen 6700H (6 physical cores):
CPU_N_THREADS = 6   # ← physical cores only, not 12 logical

# Longer context = more RAM but better conversation memory:
CPU_N_CTX = 2048    # default (uses ~100 MB extra RAM)
CPU_N_CTX = 4096    # longer memory (uses ~200 MB extra RAM)
```

> [!WARNING]
> **Do NOT set `CPU_N_THREADS` to 12** (the logical/hyperthreaded count on Ryzen 6700H).
> Setting it to the physical core count (6) is faster for llama.cpp.
> Check with: `python -c "import os; print(os.cpu_count() // 2)"`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `llama-cpp-python` install fails on Windows | `pip install llama-cpp-python --prefer-binary` |
| GGUF file not found | Check `--gguf` path, run `huggingface-cli download` |
| Slow inference (<3 tok/s) | Set `CPU_N_THREADS` to physical cores (6 for Ryzen 6700H) |
| OOM on Kaggle during merge | Restart session, run with `--start_from 5` |
| GGUF push to HF fails | Check `--hf_token` has **write** permissions |


# Download kaggle_train.py from your GitHub repo
!wget https://raw.githubusercontent.com/shubham075/model_tuning_01/main/kaggle_train.py
!ls  # confirm it's there



import os
os.environ["HF_TOKEN"]   = "HF_TOKEN_REMOVED" 
os.environ["HF_REPO"]    = "shubham075/qwen25-multilingual-lora"  # your HF repo name
os.environ["GITHUB_URL"] = "https://github.com/shubham075/model_tuning_01"  # ← your actual repo



!python kaggle_train.py \
    --hf_token  "$HF_TOKEN" \
    --hf_repo   "$HF_REPO" \
    --github_url "$GITHUB_URL"
