import os
import threading
import shutil
import uuid
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Optional local model deps
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except Exception as _local_import_err:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    PeftModel = None
    print(f"[LocalModel] Optional deps not available: {_local_import_err}")

# ---------------- 1) Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
OUTPUT_DIR = BASE_DIR / "pdffigures2_output"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    api_key=api_key,
    base_url=base_url if base_url else "https://api.deepseek.com",
)

# Import pipeline with fallback path tweak
try:
    from pdf_pipeline import pipeline
except ImportError:
    import sys
    sys.path.append(str(BASE_DIR))
    from pdf_pipeline import pipeline

# Ensure data dirs exist
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- 2) Local model globals ----------------
LOCAL_BASE_MODEL_ID = os.getenv("LOCAL_BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
LOCAL_ADAPTER_DIR = Path(__file__).resolve().parent / "local_model"
LOCAL_MODEL = None
LOCAL_TOKENIZER = None
LOCAL_DEVICE = None


def _load_local_model_once() -> None:
    """Load the local model if available; safe to call multiple times."""
    global LOCAL_MODEL, LOCAL_TOKENIZER, LOCAL_DEVICE

    if LOCAL_MODEL is not None and LOCAL_TOKENIZER is not None:
        return

    try:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None or PeftModel is None:
            print("[LocalModel] Not loading: missing torch/transformers/peft.")
            return

        if not LOCAL_ADAPTER_DIR.exists():
            print(f"[LocalModel] Adapter not found at {LOCAL_ADAPTER_DIR} (train locally first).")
            return

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        LOCAL_DEVICE = device
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        print(f"[LocalModel] Loading tokenizer: {LOCAL_BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_MODEL_ID, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"[LocalModel] Loading base model: {LOCAL_BASE_MODEL_ID} on {device}")
        base_model = AutoModelForCausalLM.from_pretrained(LOCAL_BASE_MODEL_ID, torch_dtype=torch_dtype)
        base_model.to(device)

        print(f"[LocalModel] Loading LoRA adapter: {LOCAL_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, str(LOCAL_ADAPTER_DIR))
        model.to(device)
        model.eval()

        LOCAL_MODEL = model
        LOCAL_TOKENIZER = tokenizer
        print("[LocalModel] Loaded successfully.")
    except Exception as e:
        print(f"[LocalModel] Load failed: {e}")


app = FastAPI(title="SciCapenter Backend")


# ---------------- 3) Manual CORS middleware ----------------
@app.middleware("http")
async def cors_handler(request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(
            content="ok",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ---------------- 4) Models ----------------
class RatingDimensions(BaseModel):
    clarity: int = 0
    completeness: int = 0
    faithfulness: int = 0


class CaptionChecklist(BaseModel):
    helpfulness: bool = False
    ocr: bool = False
    relation: bool = False
    stats: bool = False
    takeaway: bool = False
    visual: bool = False


class RatingContext(BaseModel):
    original_caption: str = ""
    mention_paragraphs: List[str] = []
    image_text: str = ""


class RateCaptionRequest(BaseModel):
    caption: str
    context: Optional[RatingContext] = None


class RateCaptionResponse(BaseModel):
    score: int
    explanation: str
    dimensions: RatingDimensions
    checklist: CaptionChecklist
    model: str


class GenerateCaptionRequest(BaseModel):
    context: RatingContext


class GeneratedItem(BaseModel):
    text: str
    rating: int


class GenerateCaptionResponse(BaseModel):
    short_caption: GeneratedItem
    long_caption: GeneratedItem


class FeedbackRequest(BaseModel):
    figure_id: str
    original_caption: str
    context_paragraphs: List[str]
    ocr_text: str
    final_caption: str
    rating: int = 0


# ---------------- 5) Helpers ----------------
def clean_json_string(json_str: str) -> str:
    cleaned = re.sub(r"^```json\s*", "", json_str, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def heuristic_rate(caption: str) -> RateCaptionResponse:
    words = len(caption.split())
    has_stats = bool(re.search(r"\d", caption))
    checklist = CaptionChecklist(stats=has_stats)
    if words < 10:
        return RateCaptionResponse(
            score=2,
            explanation="[Rule] Caption is too short to be informative.",
            dimensions=RatingDimensions(clarity=2, completeness=1, faithfulness=3),
            checklist=checklist,
            model="heuristic",
        )
    if words > 100:
        return RateCaptionResponse(
            score=5,
            explanation="[Rule] Detailed caption, likely covers key points.",
            dimensions=RatingDimensions(clarity=4, completeness=5, faithfulness=4),
            checklist=checklist,
            model="heuristic",
        )
    return RateCaptionResponse(
        score=3,
        explanation="[Rule] Moderate length.",
        dimensions=RatingDimensions(clarity=3, completeness=3, faithfulness=3),
        checklist=checklist,
        model="heuristic",
    )


def heuristic_rate_with_model(caption: str, model_label: str) -> RateCaptionResponse:
    base = heuristic_rate(caption)
    return RateCaptionResponse(
        score=base.score,
        explanation=base.explanation,
        dimensions=base.dimensions,
        checklist=base.checklist,
        model=model_label,
    )


def analyze_with_llm(caption: str, context: Optional[RatingContext] = None) -> Optional[RateCaptionResponse]:
    try:
        ctx_obj = context or RatingContext()
        user_content = (
            f'Target Caption: "{caption}"\n\n'
            "Context Info:\n"
            f'1. Text detected inside the image (OCR): "{ctx_obj.image_text}"\n'
            f'2. Paragraphs mentioning the figure: "{" ".join(ctx_obj.mention_paragraphs)[:1000]}..."\n'
            f'3. Original caption from paper: "{ctx_obj.original_caption}"'
        )
        system_prompt = (
            "You are a strict scientific editor. Analyze the caption.\n"
            "Checklist Evaluation Rules (Based on Text Presence):\n"
            "Stats: Mark TRUE if the caption contains ANY numbers, percentages, or values.\n"
            "Visual: Mark TRUE if the caption mentions ANY visual attributes (color, shape, line, chart type).\n"
            "Relation: Mark TRUE if the caption describes ANY comparison or trend.\n"
            "OCR: Mark TRUE if the caption mentions specific entities/variables found in the provided 'OCR Text' OR 'Paragraph Context'.\n"
            "Helpfulness: Mark TRUE if the caption accurately summarizes the figure's topic (even if it is brief), AND is readable/grammatical.\n\n"
            "SCORING CRITERIA:\n"
            "- 5 Stars: Excellent. Accurate, insightful, and well-written.\n"
            "- 4 Stars: Good. Accurate but could be more specific.\n"
            "- 3 Stars: Average. Generic or slightly vague.\n"
            "- High Score for Short Captions: If the caption is short (Title-style) but accurately identifies the figure (e.g., 'Table 2: Dataset Statistics'), give it a Good Score (4/5). Do NOT penalize brevity if it is accurate.\n"
            "- Penalty: Only penalize if the caption is factually wrong or hallucinated (mentions things not in the image/text).\n"
            "- Faithfulness: If the caption matches the Original Caption provided in context, favor a high Faithfulness score even if OCR is partial.\n"
            "- Checklist flags reflect text presence; quality score reflects faithfulness/coverage.\n\n"
            "Checklist and Score:\n"
            "- Determine the boolean status (True/False) for each of the six aspects above.\n"
            "- Then, assign the overall Score (0-5) based on coverage and accuracy of those aspects.\n"
            "- Correlation Rule: If you give a High Score (4 or 5), the caption MUST usually satisfy at least one specific checklist item.\n"
            "- If 'Helpfulness' is False, the score MUST be low (0-2).\n"
            "- Do NOT hallucinate True values to boost the score.\n\n"
            "Output Format:\n"
            "{\n"
            "\"score\": int,\n"
            "\"explanation\": \"string\",\n"
            "\"dimensions\": {\"clarity\": int, \"completeness\": int, \"faithfulness\": int},\n"
            "\"checklist\": {\"helpfulness\": bool, \"ocr\": bool, \"relation\": bool, \"stats\": bool, \"takeaway\": bool, \"visual\": bool}\n"
            "}"
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=False,
            timeout=20,
        )
        content = response.choices[0].message.content
        cleaned = clean_json_string(content)
        data = json.loads(cleaned)
        dims_data = data.get("dimensions", {}) or {}
        checklist_data = data.get("checklist", {}) or {}

        return RateCaptionResponse(
            score=data.get("score", 0),
            explanation="[AI] " + data.get("explanation", ""),
            dimensions=RatingDimensions(**dims_data),
            checklist=CaptionChecklist(**checklist_data),
            model="deepseek-chat",
        )
    except Exception as e:
        print(f"[AI] Error: {e}")
        return None


def generate_captions_with_llm(context: RatingContext) -> Optional[GenerateCaptionResponse]:
    try:
        ctx_obj = context or RatingContext()
        system_prompt = (
            "You are a helpful scientific writing assistant. Based on the provided figure context (OCR and Paragraphs), "
            "generate two captions:\n"
            "Short Caption: A concise title-like description (1 sentence).\n"
            "Long Caption: A detailed explanation covering specific data trends, relationships, and visual features found in the context (2-4 sentences).\n"
            "Also rate your own captions (0-5) based on quality.\n"
            "Return STRICT JSON: {\"short_caption\": {\"text\": \"...\", \"rating\": int}, \"long_caption\": {\"text\": \"...\", \"rating\": int}}"
        )
        user_content = (
            f"Context Info:\n"
            f"1. Text detected inside the image (OCR): \"{ctx_obj.image_text}\"\n"
            f"2. Paragraphs mentioning the figure: \"{' '.join(ctx_obj.mention_paragraphs)[:1000]}...\"\n"
            f"3. Original caption from paper: \"{ctx_obj.original_caption}\""
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=False,
            timeout=20,
        )
        content = response.choices[0].message.content
        cleaned = clean_json_string(content)
        data = json.loads(cleaned)
        sc = data.get("short_caption", {}) or {}
        lc = data.get("long_caption", {}) or {}
        if isinstance(sc, str):
            sc = {"text": sc, "rating": 0}
        if isinstance(lc, str):
            lc = {"text": lc, "rating": 0}
        return GenerateCaptionResponse(
            short_caption=GeneratedItem(text=sc.get("text", ""), rating=int(sc.get("rating", 0) or 0)),
            long_caption=GeneratedItem(text=lc.get("text", ""), rating=int(lc.get("rating", 0) or 0)),
        )
    except Exception as e:
        print(f"[AI] Generation Error: {e}")
        return None


# ---------------- 6) Generation helpers ----------------
def _build_local_prompt(context: RatingContext) -> str:
    system = (
        "You are a helpful scientific writing assistant. Generate two improved figure captions based on the context. "
        "Return plain text only (NOT JSON) using this format:\n"
        "Short: <one sentence>\n"
        "Long: <2-4 sentences>"
    )
    ctx_obj = context or RatingContext()
    paragraphs = ctx_obj.mention_paragraphs or []
    paragraphs_text = "\n".join([f"- {p}" for p in paragraphs]) if paragraphs else "(none)"

    user = (
        f'Original Caption: "{ctx_obj.original_caption}"\n'
        "Context Info:\n"
        f'1. OCR Text: "{ctx_obj.image_text}"\n'
        f"2. Mention Paragraphs:\n{paragraphs_text}\n\n"
        "Task: Provide Short and Long improved captions."
    )

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _parse_short_long(text: str) -> Tuple[str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return ("", "")

    pattern = re.compile(
        r"(?:^|\n)\s*(?:Short(?:\s+Caption)?)\s*[:\-]\s*(?P<short>.*?)(?=\n\s*(?:Long(?:\s+Caption)?)\s*[:\-]|\Z)"
        r"(?:\n\s*(?:Long(?:\s+Caption)?)\s*[:\-]\s*(?P<long>.*))?",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(cleaned)
    if m:
        short = (m.group("short") or "").strip()
        long = (m.group("long") or "").strip()
        if short and long:
            return (short, long)
        if short and not long:
            return (short, short)

    long_caption = cleaned
    first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    short_caption = first_sentence if first_sentence else cleaned[:140].strip()
    return (short_caption, long_caption)


def generate_with_local(context: RatingContext) -> GenerateCaptionResponse:
    if LOCAL_MODEL is None or LOCAL_TOKENIZER is None or LOCAL_DEVICE is None:
        raise RuntimeError("Local model not loaded.")
    if torch is None:
        raise RuntimeError("torch is not available.")

    ctx_obj = context or RatingContext()
    base_context = f"Context Paragraphs: {str(ctx_obj.mention_paragraphs)[:800]}\nOCR: {ctx_obj.image_text}"

    def _run_inference(user_prompt: str, max_tokens: int = 200, min_tokens: int = 0) -> str:
        messages = [
            {"role": "system", "content": "You are a scientific assistant. Provide the requested caption."},
            {"role": "user", "content": user_prompt},
        ]
        text = LOCAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = LOCAL_TOKENIZER([text], return_tensors="pt")
        inputs = {k: v.to(LOCAL_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = LOCAL_MODEL.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                temperature=0.7,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=LOCAL_TOKENIZER.pad_token_id,
                eos_token_id=LOCAL_TOKENIZER.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        raw_output = LOCAL_TOKENIZER.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        cleaned = re.sub(r"^(Caption:|Output:)", "", raw_output, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^figure\s*\d+\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned or raw_output

    short_input = f"{base_context}\n\nTask: Generate a single short sentence (title-like) for this figure."
    short_text = _run_inference(short_input, max_tokens=60, min_tokens=0)

    long_input = f"{base_context}\n\nTask: Write a detailed descriptive caption (multiple sentences) explaining trends, relationships, and data."
    long_text = _run_inference(long_input, max_tokens=220, min_tokens=30)

    sentences = re.split(r"(?<=[.!?])\s+", short_text)
    if sentences and sentences[0].strip():
        short_text = sentences[0].strip()
    else:
        short_text = short_text[:200].strip()

    if len(short_text) > 300:
        short_text = short_text[:300].rstrip()
    if len(long_text) > 1200:
        long_text = long_text[:1200].rstrip()

    if short_text == long_text:
        long_text = f"Detailed view: {long_text}"

    short_eval = analyze_with_llm(short_text, context) if short_text else None
    short_rating = int(short_eval.score) if short_eval else (int(heuristic_rate(short_text).score) if short_text else 0)

    long_eval = analyze_with_llm(long_text, context) if long_text else None
    long_rating = int(long_eval.score) if long_eval else (int(heuristic_rate(long_text).score) if long_text else 0)

    return GenerateCaptionResponse(
        short_caption=GeneratedItem(text=short_text, rating=short_rating),
        long_caption=GeneratedItem(text=long_text, rating=long_rating),
    )


def generate_captions_heuristic(context: RatingContext) -> GenerateCaptionResponse:
    ctx_obj = context or RatingContext()

    base = (ctx_obj.original_caption or "").strip() or "Figure caption (draft)."
    short_text = re.split(r"(?<=[.!?])\s+", base, maxsplit=1)[0].strip() or base[:140].strip()

    snippets: List[str] = []
    if (ctx_obj.image_text or "").strip():
        snippets.append(f"OCR mentions: {ctx_obj.image_text.strip()[:250]}")
    if ctx_obj.mention_paragraphs:
        first_para = (ctx_obj.mention_paragraphs[0] or "").strip()
        if first_para:
            snippets.append(f"Context: {first_para[:400]}")

    long_text = base + ((" " + " ".join(snippets)) if snippets else "")

    return GenerateCaptionResponse(
        short_caption=GeneratedItem(text=short_text, rating=int(heuristic_rate(short_text).score)),
        long_caption=GeneratedItem(text=long_text, rating=int(heuristic_rate(long_text).score)),
    )


def get_model_priority(user_pref: str) -> List[str]:
    pref = (user_pref or "").strip().lower()
    if pref == "local":
        return ["local", "deepseek"]
    return ["deepseek", "local"]


def generate_captions_dispatch(context: RatingContext, model_pref: str = "deepseek") -> GenerateCaptionResponse:
    for choice in get_model_priority(model_pref):
        if choice == "deepseek":
            result = generate_captions_with_llm(context)
            if result:
                return result
        elif choice == "local":
            if LOCAL_MODEL is not None and LOCAL_TOKENIZER is not None:
                try:
                    return generate_with_local(context)
                except Exception as e:
                    print(f"[LocalModel] Generation error: {e}")
    return generate_captions_heuristic(context)


def rate_caption_dispatch(caption: str, context: Optional[RatingContext], model_pref: str = "deepseek") -> RateCaptionResponse:
    for choice in get_model_priority(model_pref):
        if choice == "deepseek":
            result = analyze_with_llm(caption, context)
            if result:
                return result
        elif choice == "local":
            if LOCAL_MODEL is not None and LOCAL_TOKENIZER is not None:
                return heuristic_rate_with_model(caption, "local-heuristic")

    return heuristic_rate(caption)


# ---------------- 7) Events ----------------
@app.on_event("startup")
async def startup_event():
    enable_local = os.getenv("ENABLE_LOCAL_MODEL", "").strip().lower() in ("1", "true", "yes")
    if not enable_local:
        print("[Startup] Local model loading disabled (set ENABLE_LOCAL_MODEL=1 to enable).")
        return
    thread = threading.Thread(target=_load_local_model_once, daemon=True)
    thread.start()
    print("[Startup] Local model loading started in background.")


# ---------------- 8) Endpoints ----------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    out_path = OUTPUT_DIR / doc_id

    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    figures: List[Dict[str, Any]] = []
    try:
        pipeline.run_pdffigures2(str(pdf_path), str(out_path))
        figures = pipeline.extract_from_pdf(str(pdf_path), out_path, "/pdffigures2_output")
    except Exception as e:
        print(f"[Pipeline Warning] Extraction incomplete: {e}")

    return {"doc_id": doc_id, "figures": figures}


@app.post("/rate_caption", response_model=RateCaptionResponse)
async def rate_caption(payload: RateCaptionRequest, model_pref: str = "deepseek"):
    return rate_caption_dispatch(payload.caption, payload.context, model_pref=model_pref)


@app.post("/generate_captions", response_model=GenerateCaptionResponse)
async def generate_captions(
    payload: GenerateCaptionRequest,
    response: Response,
    model_pref: str = "deepseek",
):
    # User explicitly requested Local, and it is available
    if model_pref == "local" and LOCAL_MODEL is not None:
        print(f"[Gen] User requested Local Qwen.")
        response.headers["X-Model-Used"] = "Local-Qwen"
        return generate_with_local(payload.context)

    # Default or Fallback to DeepSeek
    print(f"[Gen] User requested DeepSeek (or Local not avail).")
    response.headers["X-Model-Used"] = "DeepSeek-V3"
    return generate_captions_with_llm(payload.context)


@app.post("/submit_feedback")
async def submit_feedback(payload: FeedbackRequest):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "figure_id": payload.figure_id,
        "original_caption": payload.original_caption,
        "context_paragraphs": payload.context_paragraphs,
        "ocr_text": payload.ocr_text,
        "final_caption": payload.final_caption,
        "rating": payload.rating,
    }

    feedback_path = Path(__file__).resolve().parent / "feedback_data.jsonl"
    try:
        with feedback_path.open(mode="a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")

    return {"status": "success", "message": "Data saved for training"}


# ---------------- 9) Static file serving ----------------
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/pdffigures2_output", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
