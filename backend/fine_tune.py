#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from openai import APIError, BadRequestError, NotFoundError
except Exception:  # pragma: no cover
    APIError = Exception
    BadRequestError = Exception
    NotFoundError = Exception


def _find_and_load_env(base_dir: Path) -> None:
    backend_env = base_dir / "backend" / ".env"
    root_env = base_dir / ".env"

    if backend_env.exists():
        load_dotenv(dotenv_path=backend_env, override=True)
        print(f"Loaded .env from: {backend_env}")
        return

    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=True)
        print(f"Loaded .env from: {root_env}")
        return

    print("No .env found in backend/ or repo root; using system environment variables.")


BASE_DIR = Path(__file__).resolve().parent.parent
_find_and_load_env(BASE_DIR)


api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"

if not api_key:
    print("ERROR: Missing OPENAI_API_KEY in environment (or .env).")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)


train_path = BASE_DIR / "backend" / "train.jsonl"
valid_path = BASE_DIR / "backend" / "valid.jsonl"


def upload_file(path: Path) -> str:
    print(f"Uploading {path}...")
    with path.open("rb") as f:
        created = client.files.create(file=f, purpose="fine-tune")
    return created.id


def start_finetuning(training_file_id: str, validation_file_id: str) -> object:
    model = "deepseek-chat"

    try:
        return client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
            hyperparameters={"n_epochs": 2},
        )
    except TypeError:
        return client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
        )


def _print_api_support_message(err: Exception) -> None:
    print(
        "⚠️ API Error: The provider might not support Fine-tuning via API yet. "
        f"Details: {err}"
    )


def main() -> None:
    if not train_path.exists():
        print(f"ERROR: Training file not found: {train_path}")
        sys.exit(1)
    if not valid_path.exists():
        print(f"ERROR: Validation file not found: {valid_path}")
        sys.exit(1)

    try:
        train_id = upload_file(train_path)
        valid_id = upload_file(valid_path)
    except (APIError, BadRequestError, NotFoundError) as e:
        _print_api_support_message(e)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        sys.exit(1)

    try:
        job = start_finetuning(train_id, valid_id)
    except (APIError, BadRequestError, NotFoundError) as e:
        _print_api_support_message(e)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to start fine-tuning job: {e}")
        sys.exit(1)

    job_id = getattr(job, "id", None) or getattr(job, "job_id", None) or "<unknown>"
    print(f"🚀 Job Started! ID: {job_id}")


if __name__ == "__main__":
    main()
