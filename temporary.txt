cd d:\CODES\Codes\Codes\chatbot_multilingual

# 1. Install dependencies (CUDA version of torch first!)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download data (English + Hindi + synthetic Hinglish)
python chatbot_multilingual.py --mode download

# 3a. Chat NOW without fine-tuning (uses Qwen2.5-3B-Instruct-GPTQ-Int4)
python chatbot_multilingual.py --mode chat

# 3b. OR fine-tune first, then chat with your adapted model
python chatbot_multilingual.py --mode train
python chatbot_multilingual.py --mode chat --adapter checkpoints/best_lora_adapter


Qwen2.5-3B-Instruct

# Base model (Qwen's official, already exists, ~2 GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models/

# Your fine-tuned LoRA (~50 MB, from your Kaggle training)
huggingface-cli download your-username/qwen25-multilingual-lora \
    multilingual_lora.gguf --local-dir ./models/

# Run — llama.cpp loads both together at runtime
python chatbot_multilingual.py --mode cpu_chat \
    --base_gguf ./models/qwen2.5-3b-instruct-q4_k_m.gguf \
    --lora_gguf ./models/multilingual_lora.gguf


Viewed .gitignore:1-10
Edited .gitignore
Ran command: `git status`
Ran command: `cls`
Ran command: `git branch -M main`
Ran command: `git remote add origin https://github.com/shubham075/model_tuning_01.git`
Ran command: `git push -u origin main`

## Answer 1 — Kaggle Cell 1

**Yes, run it** — but replace the placeholder values with your real ones:

```python
import os
os.environ["HF_TOKEN"]   = "hf_xxxxxxxxxxxxxxx"          # your real HF token
os.environ["HF_REPO"]    = "shubham075/qwen25-multilingual-lora"  # your HF repo name
os.environ["GITHUB_URL"] = "https://github.com/shubham075/model_tuning_01"  # ← your actual repo
```
---

**Step 1 — Install:**
```powershell
pip install llama-cpp-python --prefer-binary
```

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Run this now. If it still fails, try this fallback with a specific version:

```powershell
pip install llama-cpp-python==0.3.2 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu



**Step 2 — Chat (base model, no fine-tuning yet):**
```powershell
python chatbot_multilingual.py --mode cpu_chat --base_gguf ./models/qwen2.5-3b-instruct-q4_k_m.gguf
```

It will work in Hindi, English, and Hinglish right now. The only difference from the fine-tuned version is that it hasn't been trained on your specific dataset yet. After Kaggle training, you'll add `--lora_gguf` to make it better.

> **Note:** First load takes ~15–30 seconds. Then you'll see the chat prompt.