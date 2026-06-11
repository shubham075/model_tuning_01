"""
src/inference.py — Multi-turn chat interface with streaming output.

Supports:
  - Multi-turn conversation history (remembers context)
  - Streaming token-by-token output (feels responsive)
  - Hindi, English, and Hinglish input
  - GPTQ-Int4 or base+LoRA model

Usage (via main entry point):
  python chatbot_multilingual.py --mode chat
  python chatbot_multilingual.py --mode chat --adapter checkpoints/best_lora_adapter
"""

import sys
import threading
from pathlib import Path
from typing import Generator

import torch
from transformers import TextIteratorStreamer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    SYSTEM_PROMPT, MAX_NEW_TOKENS, TEMPERATURE,
    TOP_P, TOP_K, REPETITION_PENALTY, DO_SAMPLE,
)
from model import load_for_inference
from preprocess import clean_text, detect_script


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    stream: bool = True,
) -> Generator[str, None, None] | str:
    """
    Generates a response given the full conversation history (messages).

    Args:
        model      : loaded HuggingFace model
        tokenizer  : corresponding tokenizer
        messages   : list of {"role": ..., "content": ...} dicts
        stream     : if True, yields tokens one-by-one (streaming)

    Returns:
        Generator (stream=True) or str (stream=False)
    """
    # Apply chat template — produces a single string with special tokens
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # Adds the assistant turn marker
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE if DO_SAMPLE else 1.0,
        top_p=TOP_P if DO_SAMPLE else 1.0,
        top_k=TOP_K if DO_SAMPLE else 0,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if stream:
        # ── Streaming: runs generation in a background thread ─────────────────
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,       # Don't re-emit the input
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=model.generate,
            kwargs=gen_kwargs,
        )
        thread.start()
        return streamer   # caller iterates over this

    else:
        # ── Non-streaming ─────────────────────────────────────────────────────
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)

        # Decode only new tokens (skip the input prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── Chat Session ─────────────────────────────────────────────────────────────

class ChatSession:
    """
    Manages a multi-turn conversation with the chatbot.
    Keeps history and handles the system prompt.
    """

    def __init__(self, model, tokenizer, system_prompt: str = SYSTEM_PROMPT):
        self.model  = model
        self.tokenizer = tokenizer
        self.history: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_input: str, stream: bool = True) -> str:
        """
        Sends a user message, streams/returns the bot response,
        and appends both to the conversation history.
        """
        user_input = clean_text(user_input)
        if not user_input:
            return "(empty input)"

        lang = detect_script(user_input)
        self.history.append({"role": "user", "content": user_input})

        print(f"\n[lang detected: {lang}]")
        print("Bot: ", end="", flush=True)

        full_response = ""

        if stream:
            streamer = generate_response(
                self.model, self.tokenizer, self.history, stream=True
            )
            for token in streamer:
                print(token, end="", flush=True)
                full_response += token
            print()  # newline after streaming ends
        else:
            full_response = generate_response(
                self.model, self.tokenizer, self.history, stream=False
            )
            print(full_response)

        self.history.append({"role": "assistant", "content": full_response.strip()})
        return full_response.strip()

    def reset(self):
        """Clears conversation history (keeps system prompt)."""
        self.history = [self.history[0]]
        print("[Chat reset]")


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def run_chat(lora_adapter_path: str | None = None):
    """
    Starts an interactive chat session in the terminal.
    Supports:
      /reset   — clears conversation history
      /quit    — exits
      /history — prints full conversation history
    """
    print("\n" + "=" * 60)
    print("  Multilingual Chatbot")
    print("  Supports: Hindi | English | Hinglish")
    print("  Commands: /reset  /history  /quit")
    print("=" * 60 + "\n")

    model, tokenizer = load_for_inference(lora_adapter_path=lora_adapter_path)
    session = ChatSession(model, tokenizer)

    print("Model loaded! Start chatting.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! / अलविदा!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "quit", "exit", "bye", "alvida"):
            print("Goodbye! / अलविदा!")
            break

        if user_input.lower() == "/reset":
            session.reset()
            continue

        if user_input.lower() == "/history":
            for i, msg in enumerate(session.history):
                print(f"  [{msg['role']}]: {msg['content'][:80]}...")
            continue

        # ── Normal message ────────────────────────────────────────────────────
        session.chat(user_input, stream=True)
        print()
