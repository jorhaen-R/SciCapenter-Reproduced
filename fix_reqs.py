# fix_reqs.py
import os

# 纯净的依赖列表
deps = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "openai",
    "python-dotenv",
    "pdfplumber",
    "pydantic",
    "requests",
    "httpx",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors"
]

# 写入文件（强制 UTF-8 无 BOM，强制 \n 换行）
with open("requirements.txt", "w", encoding="utf-8", newline="\n") as f:
    for dep in deps:
        f.write(dep + "\n")

print(f"✅ requirements.txt 已重置！包含 {len(deps)} 个依赖。")
print("文件格式：UTF-8 (No BOM), Unix Line Endings (LF)")