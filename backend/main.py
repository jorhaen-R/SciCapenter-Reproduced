import os
import shutil
import uuid
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from textwrap import dedent

# ---------------- 1. 标准化配置加载 (Standard Config) ----------------

# 定义基准路径: 根目录 (SciCapenter-Reproduced)
# backend/main.py -> parent = backend -> parent = Root
BASE_DIR = Path(__file__).resolve().parent.parent 

# 定义 .env 文件的绝对路径 (标准位置：根目录)
ENV_PATH = BASE_DIR / ".env"

# 加载环境变量
print(f"DEBUG: Looking for config at: {ENV_PATH}")
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print("DEBUG: ✅ .env file found and loaded.")
else:
    print("DEBUG: ❌ .env file NOT found! Falling back to system env.")

# 获取 Key
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# 安全打印用于调试
safe_key = f"{api_key[:5]}...{api_key[-4:]}" if api_key else "None"
print(f"DEBUG: Final API Key: {safe_key}")
print(f"DEBUG: Final Base URL: {base_url}")

# 初始化 DeepSeek 客户端
if not api_key:
    print("WARNING: No API Key found. AI features will fail.")

client = OpenAI(
    api_key=api_key,
    base_url=base_url if base_url else "https://api.deepseek.com"
)

# ---------------- 2. 路径配置 ----------------
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
OUTPUT_DIR = BASE_DIR / "pdffigures2_output"

# 导入 Pipeline (处理模块路径问题)
try:
    from pdf_pipeline import pipeline
except ImportError:
    import sys
    sys.path.append(str(BASE_DIR))
    from pdf_pipeline import pipeline

# 确保文件夹存在
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SciCapenter Backend")

# ---------------- 3. 中间件 (CORS) ----------------
# 暴力 CORS 解决本地调试问题
@app.middleware("http")
async def cors_handler(request, call_next):
    if request.method == "OPTIONS":
        from fastapi.responses import JSONResponse
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

# ---------------- 4. 数据模型 ----------------
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
    # We send the same context as rating
    context: RatingContext

class GeneratedItem(BaseModel):
    text: str
    rating: int

class GenerateCaptionResponse(BaseModel):
    short_caption: GeneratedItem
    long_caption: GeneratedItem

# ---------------- 5. 核心逻辑函数 ----------------
def clean_json_string(json_str: str) -> str:
    """清洗 LLM 返回的 JSON 字符串"""
    cleaned = re.sub(r"^```json\s*", "", json_str, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()

def heuristic_rate(caption: str) -> RateCaptionResponse:
    """兜底规则评分"""
    words = len(caption.split())
    has_stats = bool(re.search(r"\d", caption))
    checklist = CaptionChecklist(stats=has_stats)
    if words < 10:
        return RateCaptionResponse(
            score=2, 
            explanation="[Rule] Caption is too short to be informative.", 
            dimensions=RatingDimensions(clarity=2, completeness=1, faithfulness=3),
            checklist=checklist,
            model="heuristic"
        )
    elif words > 100:
        return RateCaptionResponse(
            score=5, 
            explanation="[Rule] Detailed caption, likely covers key points.", 
            dimensions=RatingDimensions(clarity=4, completeness=5, faithfulness=4),
            checklist=checklist,
            model="heuristic"
        )
    else:
        return RateCaptionResponse(
            score=3, 
            explanation="[Rule] Moderate length.", 
            dimensions=RatingDimensions(clarity=3, completeness=3, faithfulness=3),
            checklist=checklist,
            model="heuristic"
        )

def analyze_with_llm(caption: str, context: Optional[RatingContext] = None) -> Optional[RateCaptionResponse]:
    """调用 DeepSeek AI"""
    try:
        print(f"[AI] Requesting DeepSeek for: {caption[:30]}...")
        ctx_obj = context or RatingContext()
        print(f"--------------------------------------------------")
        print(f"[DEBUG] OCR Text for AI: {context.image_text}")
        print(f"--------------------------------------------------")
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
            "Helpfulness: Mark TRUE if the caption is grammatically correct and informative.\n\n"
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
            "- If 'Helpfulness' is False, the score MUST be low (0-2).\n"
            "- Do NOT hallucinate True values to boost the score.\n\n"
            # --- 输出格式 ---
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
                {"role": "user", "content": user_content}
            ],
            stream=False,
            timeout=20
        )
        content = response.choices[0].message.content
        print(f"[AI] Raw Response: {content}")
        
        # 清洗并解析
        cleaned = clean_json_string(content)
        data = json.loads(cleaned)
        dims_data = data.get("dimensions", {}) or {}
        checklist_data = data.get("checklist", {}) or {}
        
        return RateCaptionResponse(
            score=data.get("score", 0),
            explanation="[AI] " + data.get("explanation", ""),
            dimensions=RatingDimensions(**dims_data),
            checklist=CaptionChecklist(**checklist_data),
            model="deepseek-chat"
        )
    except Exception as e:
        print(f"[AI] Error: {e}")
        return None

def generate_captions_with_llm(context: RatingContext) -> Optional[GenerateCaptionResponse]:
    """Generate short and long captions via LLM"""
    try:
        ctx_obj = context or RatingContext()
        system_prompt = (
            "You are a helpful scientific writing assistant. Based on the provided figure context (OCR and Paragraphs), "
            "generate two captions:\n"
            "Short Caption: A concise title-like description (1 sentence).\n"
            "Long Caption: A detailed explanation covering specific data trends, relationships, and visual features found "
            "in the context (2-4 sentences).\n"
            "Also rate your own captions (0-5) based on quality.\n"
            "Return STRICT JSON: {\"short_caption\": {\"text\": \"...\", \"rating\": int}, \"long_caption\": {\"text\": \"...\", \"rating\": int}}"
        )
        user_content = (
            f'Context Info:\n'
            f'1. Text detected inside the image (OCR): "{ctx_obj.image_text}"\n'
            f'2. Paragraphs mentioning the figure: "{" ".join(ctx_obj.mention_paragraphs)[:1000]}..."\n'
            f'3. Original caption from paper: "{ctx_obj.original_caption}"'
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            stream=False,
            timeout=20
        )
        content = response.choices[0].message.content
        print(f"[AI] Raw Generation Response: {content}")
        cleaned = clean_json_string(content)
        data = json.loads(cleaned)
        sc = data.get("short_caption", {}) or {}
        lc = data.get("long_caption", {}) or {}
        # support string fallbacks
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

# ---------------- 6. API 接口 ----------------

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
    
    figures = []
    try:
        # 允许 pdffigures2 失败而不崩掉整个接口
        pipeline.run_pdffigures2(str(pdf_path), str(out_path))
        figures = pipeline.extract_from_pdf(str(pdf_path), out_path, f"pdffigures2_output/{doc_id}")
    except Exception as e:
        print(f"[Pipeline Warning] Extraction incomplete: {e}")
        
    return {"doc_id": doc_id, "figures": figures}

@app.post("/rate_caption", response_model=RateCaptionResponse)
async def rate_caption(payload: RateCaptionRequest):
    # 1. 优先尝试 AI
    result = analyze_with_llm(payload.caption, payload.context)
    if result:
        return result
    
    # 2. 失败则回退到规则
    print("[Backend] Switching to Heuristic Fallback")
    return heuristic_rate(payload.caption)

@app.post("/generate_captions", response_model=GenerateCaptionResponse)
async def generate_captions(payload: GenerateCaptionRequest):
    result = generate_captions_with_llm(payload.context)
    if result:
        return result
    # Fallback if AI fails
    return GenerateCaptionResponse(
        short_caption=GeneratedItem(text="Generation failed.", rating=0),
        long_caption=GeneratedItem(text="Could not generate captions at this time.", rating=0)
    )

# ---------------- 7. 静态文件服务 ----------------
app.mount("/pdffigures2_output", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
