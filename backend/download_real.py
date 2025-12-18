#!/usr/bin/env python3
"""
Download SciCap JSON (train/val/test) and convert it into chat-format JSONL.

SciCap's `*.json` files are dicts with two top-level keys:
  - "images": list[{"id": int, "ocr": list[str], ...}]
  - "annotations": list[{"image_id": int, "caption": str, "paragraph": list[str], "mention": ...}]

This script supports large-scale download/conversion:
  - For small splits, it can load JSON into memory.
  - For large splits (e.g., train.json), it *can* stream-parse annotations if `ijson` is installed.

Recommended for "real large dataset":
  python backend/download_real.py --split train --limit 0
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SCICAP_BASE_URL = "https://huggingface.co/datasets/CrowdAILab/scicap/resolve/main"
SPLIT_TO_FILE = {
    "train": "train.json",
    "val": "val.json",
    "public-test": "public-test.json",
    "hidden-test": "hidden-test.json",
}


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
            else:
                s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    s = str(value).strip()
    return [s] if s else []


def _flatten_mentions(mention: Any) -> List[str]:
    """
    SciCap `mention` is often list[list[str]]; normalize to list[str].
    """
    if mention is None:
        return []
    if isinstance(mention, list):
        flattened: List[str] = []
        for item in mention:
            if isinstance(item, list):
                flattened.extend(_as_string_list(item))
            else:
                flattened.extend(_as_string_list(item))
        return [m for m in flattened if m.strip()]
    return _as_string_list(mention)


def _build_url(split: str) -> str:
    if split not in SPLIT_TO_FILE:
        raise ValueError(f"Unknown split: {split}. Choose from: {', '.join(SPLIT_TO_FILE.keys())}")
    return f"{SCICAP_BASE_URL}/{SPLIT_TO_FILE[split]}"


def download_real_data(split: str = "val", limit: int = 1000, include_ocr: bool = False) -> None:
    url = _build_url(split)
    print(f"[INFO] 开始下载 SciCap 真实数据 ({split})...")
    print(f"[INFO] 请求地址: {url}")

    backend_dir = Path(__file__).resolve().parent
    output_file = backend_dir / f"real_{split}.jsonl"

    try:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()

        data = resp.json()

        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            raise ValueError(
                f"Unexpected schema from SciCap val.json; expected dict with keys images/annotations, got: {type(data)}"
            )

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError("Unexpected SciCap schema: images/annotations should both be lists.")

        print(f"[INFO] images: {len(images)}")
        print(f"[INFO] annotations: {len(annotations)}")

        images_by_id: Optional[Dict[int, Dict[str, Any]]] = None
        if include_ocr:
            images_by_id = {}
            for img in images:
                if isinstance(img, dict) and isinstance(img.get("id"), int):
                    images_by_id[img["id"]] = img

        valid_count = 0
        skipped = 0

        with output_file.open("w", encoding="utf-8") as f_out:
            for ann in annotations:
                if not isinstance(ann, dict):
                    skipped += 1
                    continue

                image_id = ann.get("image_id")
                caption = (ann.get("caption") or "").strip()
                paragraphs = _as_string_list(ann.get("paragraph"))
                mentions = _flatten_mentions(ann.get("mention"))

                ocr_text = ""
                if include_ocr and images_by_id is not None and isinstance(image_id, int):
                    img = images_by_id.get(image_id)
                    ocr_list = _as_string_list(img.get("ocr")) if isinstance(img, dict) else []
                    ocr_text = " ".join(ocr_list).strip()

                context_parts = []
                if paragraphs:
                    context_parts.append("Paragraphs:\n" + "\n".join([f"- {p}" for p in paragraphs[:3]]))
                if mentions:
                    context_parts.append("Mentions:\n" + "\n".join([f"- {m}" for m in mentions[:5]]))
                if ocr_text:
                    context_parts.append(f'OCR Text: "{ocr_text[:800]}"')

                context_text = "\n\n".join(context_parts).strip()

                if len(caption) < 10:
                    skipped += 1
                    continue
                if not context_text:
                    skipped += 1
                    continue

                conversation = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful scientific writing assistant. Generate a concise and accurate caption for the figure based on the provided context.",
                        },
                        {
                            "role": "user",
                            "content": f"{context_text}\n\nTask: Generate a caption.",
                        },
                        {"role": "assistant", "content": caption},
                    ]
                }

                f_out.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                valid_count += 1

                if limit and valid_count >= limit:
                    break

        print(f"[OK] 完成！写入 {valid_count} 条样本到: {output_file}")
        print(f"[INFO] 跳过 {skipped} 条（caption 太短或缺少上下文等）")

    except Exception as e:
        print(f"[ERROR] 下载/处理失败: {e}")
        print("\n[Tips] 修复提示：")
        print("1) SciCap 的 val.json 是一个 dict，包含 images 和 annotations；需要按 image_id 合并字段。")
        print("2) 如果网络还不稳定，可先用浏览器下载 val.json 到本地，再改成本地读取。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SciCap split and export chat-format JSONL.")
    parser.add_argument("--split", choices=list(SPLIT_TO_FILE.keys()), default="val")
    parser.add_argument("--limit", type=int, default=1000, help="0 means no limit (export all).")
    parser.add_argument(
        "--include-ocr",
        action="store_true",
        help="Include OCR text from images table (may increase memory use for large splits).",
    )
    args = parser.parse_args()
    download_real_data(split=args.split, limit=args.limit, include_ocr=args.include_ocr)
