#!/usr/bin/env python3
"""
Local fine-tuning (LoRA) demo for "continuous learning" without cloud APIs.

Base model:
  - Qwen/Qwen2.5-0.5B-Instruct

Dataset:
  - backend/real_train.jsonl (JSONL with {"messages":[{role,content}, ...]} )

Output:
  - backend/local_model/ (LoRA adapter weights + tokenizer)

Dependencies (typical):
  pip install -U "transformers>=4.40" datasets peft accelerate torch
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _find_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _format_qwen_chat(messages: List[Dict[str, Any]]) -> str:
    """
    Format chat messages into Qwen ChatML-style prompt string.
    Expected roles: system, user, assistant.
    """
    system_text = ""
    user_text = ""
    assistant_text = ""

    for msg in messages or []:
        role = (msg or {}).get("role", "")
        content = (msg or {}).get("content", "")
        if role == "system" and not system_text:
            system_text = str(content)
        elif role == "user" and not user_text:
            user_text = str(content)
        elif role == "assistant" and not assistant_text:
            assistant_text = str(content)

    # ChatML template requested in the prompt.
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


def main() -> None:
    device = _get_device()
    print(f"Device: {device}")

    base_dir = _find_base_dir()
    train_path = base_dir / "backend" / "real_train.jsonl"
    output_dir = base_dir / "backend" / "local_model"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model: {BASE_MODEL}")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch_dtype)
    model.to(device)

    print(f"Loading dataset: {train_path}")
    dataset = load_dataset(
    "json",
    data_files=str(BASE_DIR / "backend" / "real_train.jsonl"),
    split="train",
    )


    def add_text(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example.get("messages", [])
        return {"text": _format_qwen_chat(messages)}

    dataset = dataset.map(add_text, remove_columns=dataset.column_names)

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        labels = encoded["input_ids"].copy()
        pad_id = tokenizer.pad_token_id
        labels = [(-100 if token_id == pad_id else token_id) for token_id in labels]
        encoded["labels"] = labels
        return encoded

    tokenized = dataset.map(tokenize, batched=False)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # TensorBoard is optional; disable if not installed to avoid runtime errors.
    try:
        import tensorboard  # noqa: F401
        report_to_opt = ["tensorboard"]
    except Exception:
        report_to_opt = []
        print("[WARN] tensorboard not installed; skipping TensorBoard logging.")

    training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=4,                  # 先跑 4 轮，视效果再增减
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,       # 有效 batch=2，梯度更稳
    learning_rate=1e-4,                  # 稳定一点
    logging_steps=10,
    save_steps=200,                      # 少写盘
    save_total_limit=2,
    fp16=(device.type == "cuda"),
    report_to=report_to_opt,
    logging_dir=str(output_dir / "logs"),
    remove_unused_columns=False,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    print("🚀 Starting Local Training... This updates actual weights!")
    trainer.train()

    print(f"Saving LoRA adapter + tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Training Complete. New weights saved to ./local_model")


if __name__ == "__main__":
    main()
