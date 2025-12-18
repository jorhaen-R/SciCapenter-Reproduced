#!/usr/bin/env python3
"""
Run local inference with a LoRA-adapted model to verify "continuous learning".

Loads:
  - Base model: Qwen/Qwen2.5-0.5B-Instruct
  - LoRA adapter: backend/local_model (from train_local.py)
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_prompt() -> str:
    system = (
        "You are a helpful scientific writing assistant. Generate two improved figure captions based on the context. "
        "Return plain text only (NOT JSON) using this format:\n"
        "Short: <one sentence>\n"
        "Long: <2-4 sentences>"
    )
    user = (
        'Original Caption: "Figure 5: A prompt for finding related variables/statements."\n'
        "Context Info:\n"
        '1. OCR Text: ""\n'
        "2. Mention Paragraphs:\n"
        "- With these variables and statements, we can use static analysis to confirm whether the vulnerability exists or not. "
        "An example of the prompt sent to GPT to ask for related variables or expressions is shown in Figure 5.\n\n"
        "Task: Provide Short and Long improved captions."
    )

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    device = _get_device()
    print(f"Device: {device}")

    backend_dir = Path(__file__).resolve().parent
    adapter_dir = backend_dir / "local_model"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir} (run train_local.py first)"
        )

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch_dtype)
    base_model.to(device)

    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.to(device)
    model.eval()

    prompt = _build_prompt()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print("\n=== Model Output ===")
    print(decoded)

    prompt_len = inputs["input_ids"].shape[-1]
    continuation = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    print("\n=== Generated Text (Continuation) ===")
    print(continuation)


if __name__ == "__main__":
    main()

