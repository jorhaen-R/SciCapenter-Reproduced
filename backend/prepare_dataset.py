#!/usr/bin/env python3
"""
Prepare fine-tuning datasets from raw user feedback logs.

Input:
  - feedback_data.jsonl (one JSON object per line)

Output:
  - train.jsonl / valid.jsonl in OpenAI/DeepSeek chat JSONL format:
      {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_MESSAGE = (
    "You are a helpful scientific writing assistant. Improve the figure caption based on the context."
)


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(item).strip()
            if text:
                items.append(text)
        return items
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _build_user_prompt(original_caption: str, ocr_text: str, context_paragraphs: List[str]) -> str:
    paragraphs_text = (
        "\n".join([f"- {p}" for p in context_paragraphs]) if context_paragraphs else "(none)"
    )
    return (
        f'Original Caption: "{original_caption}"\n'
        "Context Info:\n"
        f'1. OCR Text: "{ocr_text}"\n'
        f"2. Mention Paragraphs:\n{paragraphs_text}\n\n"
        "Task: Provide the improved caption."
    )


def _is_valid_example(final_caption: str, ocr_text: str, context_paragraphs: List[str]) -> bool:
    if not final_caption or len(final_caption.strip()) < 10:
        return False
    has_ocr = bool((ocr_text or "").strip())
    has_paras = any(p.strip() for p in (context_paragraphs or []))
    if not has_ocr and not has_paras:
        return False
    return True


def _to_chat_jsonl_record(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    original_caption = str(entry.get("original_caption", "") or "").strip()
    final_caption = str(entry.get("final_caption", "") or "").strip()
    ocr_text = str(entry.get("ocr_text", "") or "").strip()
    context_paragraphs = _as_string_list(entry.get("context_paragraphs"))

    if not _is_valid_example(final_caption, ocr_text, context_paragraphs):
        return None

    user_message = _build_user_prompt(original_caption, ocr_text, context_paragraphs)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_caption},
        ]
    }


def main() -> None:
    backend_dir = Path(__file__).resolve().parent
    input_path = backend_dir / "feedback_data.jsonl"
    train_path = backend_dir / "train.jsonl"
    valid_path = backend_dir / "valid.jsonl"

    total_raw = 0
    examples: List[Dict[str, Any]] = []

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_raw += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            record = _to_chat_jsonl_record(entry if isinstance(entry, dict) else {})
            if record is not None:
                examples.append(record)

    random.seed(42)
    random.shuffle(examples)

    split_idx = int(len(examples) * 0.8)
    train_samples = examples[:split_idx]
    valid_samples = examples[split_idx:]

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train_samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with valid_path.open("w", encoding="utf-8") as f:
        for ex in valid_samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Found {total_raw} raw logs.")
    print(f"Saved {len(train_samples)} training samples.")
    print(f"Saved {len(valid_samples)} validation samples.")


if __name__ == "__main__":
    main()

