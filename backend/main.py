from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi import UploadFile, File
import shutil
from uuid import uuid4
from pathlib import Path

from pdf_pipeline.pipeline import extract_from_pdf

app = FastAPI(title="SciCapenter Backend")

# Allow local development from any localhost port
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Health status of the backend")


class RateCaptionRequest(BaseModel):
    caption: str = Field(..., description="Caption text to evaluate")


class CaptionAspectScores(BaseModel):
    helpfulness: int = 0
    faithfulness: int = 0
    completeness: int = 0


class RateCaptionResponse(BaseModel):
    rating: int
    explanation: str
    aspects: CaptionAspectScores | None = None


class ParsePdfRequest(BaseModel):
    pdf_path: str


class ParsePdfResponse(BaseModel):
    results: List[Dict[str, Any]]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload_pdf", response_model=ParsePdfResponse)
async def upload_pdf(file: UploadFile = File(...)) -> ParsePdfResponse:
    """
    真正解析上传的 PDF：
    - 接收 UploadFile
    - 保存到本地 uploaded_pdfs 目录
    - 调用 extract_from_pdf(保存后的路径)
    - 返回 ParsePdfResponse(results=...)
    """
    # 1. 类型检查（只接受 PDF）
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 2. 生成一个临时文件路径，避免文件名冲突
    suffix = Path(file.filename).suffix or ".pdf"
    temp_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    try:
        # 3. 把上传内容写入磁盘
        with temp_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        # 4. 调用现有的 extract_from_pdf 解析“刚保存的这个 PDF”
        results = extract_from_pdf(str(temp_path))

        # 5. 正常返回
        return ParsePdfResponse(results=results)

    except Exception as exc:
        # 解析失败时，抛 500，并附带错误信息
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse uploaded PDF: {exc}",
        ) from exc

    finally:
        # 6. 可选：解析完后删除临时 PDF 文件，避免磁盘堆积
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            # 删除失败就打印一下，不影响主流程
            print(f"[WARN] Failed to delete temp file: {temp_path}")



def _rate_caption(caption: str) -> Tuple[int, str]:
    trimmed = caption.strip()
    words = trimmed.split()
    length = len(words)

    if length == 0:
        return 0, "Caption is empty."
    if length < 5:
        return 1, "The caption is extremely short and likely unhelpful to readers."
    if length < 40:
        return 2, "Caption is too short to be descriptive."
    if length <= 180:
        return 5, "Caption has a balanced length likely sufficient."
    if length <= 320:
        return 4, "Caption is descriptive but may be verbose."
    return 3, "Caption is very long; consider shortening."



@app.post("/rate_caption", response_model=RateCaptionResponse)
async def rate_caption(payload: RateCaptionRequest) -> RateCaptionResponse:
    # 调试日志：打印被调用次数和 caption 的前 80 个字符
    print(
        "[DEBUG] /rate_caption called, "
        f"len={len(payload.caption)}, "
        f"preview={payload.caption[:80]!r}"
    )

    rating, explanation = _rate_caption(payload.caption)
    return RateCaptionResponse(rating=rating, explanation=explanation)

FIGURE_DATA_ROOT = Path(r"D:\pdffigures2_output")
app.mount("/fig_images", StaticFiles(directory=str(FIGURE_DATA_ROOT)), name="fig_images")


@app.post("/parse_pdf", response_model=ParsePdfResponse)
async def parse_pdf_endpoint(request: ParsePdfRequest) -> ParsePdfResponse:
    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")

    try:
        results = extract_from_pdf(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}") from exc

    return ParsePdfResponse(results=results)



if __name__ == "__main__":
    # Run locally with: uvicorn backend.main:app --reload --port 8000
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
