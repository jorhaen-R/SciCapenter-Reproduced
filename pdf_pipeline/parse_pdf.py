"""
PDF parsing helpers for SciCapenter-Reproduced.

功能：
1. 读取 pdffigures2 的 JSON 输出，规范化成统一的 figure 结构；
2. 使用 PyMuPDF 提取 PDF 正文段落（paragraphs）；
3. 从正文中匹配包含 "Figure 1" / "Fig. 1" / "Table 1" 等的段落，
   得到 figure-mentioning paragraphs；
4. 返回一个统一的字典结构，供后续前端 / API 使用。

使用前提：
- 你已经手动在命令行运行过 pdffigures2，例如：
    sbt "runMain org.allenai.pdffigures2.FigureExtractorCli D:\Papers\sci.pdf \
        -d D:\pdffigures2_output\sci\data \
        -g D:\pdffigures2_output\sci\images \
        -m D:\pdffigures2_output\sci\meta.json"

- 本脚本默认假设：
    PDF: D:\Papers\sci.pdf
    pdffigures2 输出根目录: D:\pdffigures2_output\
    具体到此 PDF 的目录:   D:\pdffigures2_output\sci\
    JSON 在:                D:\pdffigures2_output\sci\data\*.json
    图片在:                 D:\pdffigures2_output\sci\images\...

你可以根据需要修改 OUTPUT_ROOT_DEFAULT 等常量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any

# ---------- 依赖库 ----------
# 需要先安装：
#   pip install pymupdf
try:
    import fitz  # PyMuPDF
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "本脚本需要 PyMuPDF，请先执行：pip install pymupdf"
    ) from e


# ============= 配置区域 =============

# pdffigures2 输出根目录（你之前运行 CLI 时用的路径根）
OUTPUT_ROOT_DEFAULT = Path(r"D:\pdffigures2_output")




# ============= 数据结构定义 =============

@dataclass
class FigureInfo:
    figure_id: str        # 例如 "Figure 1", "Table 1"
    fig_type: str         # "Figure" or "Table"
    name: str             # 原始 name 字段，一般是 "1", "2"
    page: int
    caption: str
    image_path: str       # 渲染后的图片路径
    image_text: List[str] # 图内文字 OCR（来自 pdffigures2 的 imageText）


@dataclass
class ParagraphInfo:
    page: int
    text: str
    bbox: Dict[str, float]  # {"x1":..., "y1":..., "x2":..., "y2":...}


@dataclass
class ParsedPdf:
    pdf_path: str
    meta: Dict[str, Any]                  # pdffigures2 meta.json 的内容（如果存在）
    figures: List[FigureInfo]
    paragraphs: List[ParagraphInfo]
    figure_mentions: Dict[str, List[ParagraphInfo]]  # key: figure_id, value: 段落列表


@dataclass
class FigureWithParagraphs(FigureInfo):
    """在 FigureInfo 的基础上，额外挂上它对应的段落列表。"""
    paragraphs: List[ParagraphInfo]


# ============= 核心函数 =============
def extract_figures_with_pdffigures2(
    pdf_path: str | Path,
    output_root: str | Path = OUTPUT_ROOT_DEFAULT,
) -> List[FigureInfo]:
    """
    对外暴露的图/表提取函数（只负责读 pdffigures2 的 JSON，不负责运行 CLI）。

    pipeline.extract_from_pdf 会调用这里。
    """
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    figures_json_path = _discover_pdffigures2_json(pdf_path, output_root)
    figures = _load_pdffigures2_json(figures_json_path)
    return figures


def _discover_pdffigures2_json(pdf_path: Path, output_root: Path) -> Path:
    """
    根据 PDF 文件名和 output_root 推断 pdffigures2 的 JSON 文件路径。

    例如：
        pdf_path = D:\Papers\sci.pdf
        output_root = D:\pdffigures2_output
    则：
        pdf_stem = "sci"
        pdf_output_dir = D:\pdffigures2_output\sci\data\
        在该目录下寻找第一个 *.json 文件。
    """
    pdf_stem = pdf_path.stem
    data_dir = output_root / pdf_stem / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"未找到 pdffigures2 data 目录：{data_dir}")

    json_candidates = sorted(data_dir.glob("*.json"))
    if not json_candidates:
        raise FileNotFoundError(f"{data_dir} 下未找到任何 JSON 文件，请确认 pdffigures2 已经成功运行。")

    # 通常只有一个 json，这里取第一个
    return json_candidates[0]


def _load_pdffigures2_json(json_path: Path) -> List[FigureInfo]:
    """
    读取 pdffigures2 输出 JSON，并转为 List[FigureInfo]。

    JSON 结构类似：
    [
      {
        "caption": "...",
        "figType": "Table",
        "name": "1",
        "page": 5,
        "renderURL": "D:\\pdffigures2_output\\images\\figsci-Table1-1.png",
        "imageText": [...]
      },
      ...
    ]
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    figures: List[FigureInfo] = []
    for item in raw_items:
        fig_type = item.get("figType", "").strip() or "Figure"
        name = str(item.get("name", "")).strip()
        page = int(item.get("page", 0) or 0)
        caption = (item.get("caption") or "").strip()
        image_path = item.get("renderURL", "") or ""
        image_text = item.get("imageText") or []

        # 根据 figType 和 name 构造统一的 figure_id
        if fig_type.lower().startswith("table"):
            figure_id = f"Table {name}".strip()
        else:
            figure_id = f"Figure {name}".strip()

        figures.append(
            FigureInfo(
                figure_id=figure_id,
                fig_type=fig_type,
                name=name,
                page=page,
                caption=caption,
                image_path=image_path,
                image_text=image_text,
            )
        )

    return figures


def _load_pdffigures2_meta(pdf_path: Path, output_root: Path) -> Dict[str, Any]:
    """
    读取 pdffigures2 的 meta.json（如果存在），否则返回空 dict。
    例如：
        D:\pdffigures2_output\sci\meta.json
    """
    pdf_stem = pdf_path.stem
    meta_path = output_root / pdf_stem / "meta.json"
    if meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def extract_paragraphs_with_pymupdf(pdf_path: Path) -> List[ParagraphInfo]:
    """
    使用 PyMuPDF 提取 PDF 中的“段落级”文本。

    实现方式：
    - 使用 page.get_text("blocks")；
    - 每个 block 视为一个段落（也可以后续再细分）。
    """
    doc = fitz.open(pdf_path)
    paragraphs: List[ParagraphInfo] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1  # PDF 页码通常从 1 开始

        # blocks: List[(x0, y0, x1, y1, text, block_no, ...)]
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if not text:
                continue

            # 合并多行，去掉多余空白
            clean_text = " ".join(line.strip() for line in text.splitlines()).strip()
            if not clean_text:
                continue

            paragraphs.append(
                ParagraphInfo(
                    page=page_num,
                    text=clean_text,
                    bbox={"x1": float(x0), "y1": float(y0), "x2": float(x1), "y2": float(y1)},
                )
            )

    doc.close()
    return paragraphs


def _build_aliases_for_figure(fig: FigureInfo) -> List[str]:
    """
    为某个图像构建在正文中可能出现的各种称呼形式。
    例如 Figure 1:
        - "Figure 1"
        - "Fig. 1"
        - "Fig 1"
    Table 1:
        - "Table 1"
        - "Tab. 1"
    """
    aliases: List[str] = []
    try:
        num_str = fig.name.strip()
        # 有些论文会写成 "Fig. 1", "Fig 1"
        if fig.fig_type.lower().startswith("table"):
            aliases.append(f"Table {num_str}")
            aliases.append(f"Tab. {num_str}")
            aliases.append(f"Tab {num_str}")
        else:
            aliases.append(f"Figure {num_str}")
            aliases.append(f"Fig. {num_str}")
            aliases.append(f"Fig {num_str}")
    except Exception:
        # 最差情况就使用 figure_id 作为唯一 alias
        aliases.append(fig.figure_id)
    return aliases


def _build_mentions_map(
    figures: List[FigureInfo],
    paragraphs: List[ParagraphInfo],
) -> Dict[str, List[ParagraphInfo]]:
    """
    内部函数：根据正文段落中是否包含 "Figure 1"/"Fig. 1"/"Table 1" 等字符串，
    建立 figure_id -> paragraphs 的映射。
    """
    mentions: Dict[str, List[ParagraphInfo]] = {fig.figure_id: [] for fig in figures}

    figure_aliases: Dict[str, List[str]] = {
        fig.figure_id: _build_aliases_for_figure(fig) for fig in figures
    }

    for para in paragraphs:
        text = para.text
        for fig in figures:
            aliases = figure_aliases.get(fig.figure_id, [])
            if any(alias in text for alias in aliases):
                mentions[fig.figure_id].append(para)

    return mentions



def link_figures_to_paragraphs(
    figures: List[FigureInfo],
    paragraphs: List[ParagraphInfo],
) -> List[FigureWithParagraphs]:
    """
    对外接口：返回每个 figure 附带 paragraphs 的列表，
    供 pipeline.extract_from_pdf 直接使用。
    """
    mentions = _build_mentions_map(figures, paragraphs)

    result: List[FigureWithParagraphs] = []
    for fig in figures:
        result.append(
            FigureWithParagraphs(
                **asdict(fig),
                paragraphs=mentions.get(fig.figure_id, []),
            )
        )
    return result


def parse_pdf(
    pdf_path: str,
    output_root: str | Path = OUTPUT_ROOT_DEFAULT,
) -> ParsedPdf:
    """
    高层封装：给定 PDF 路径 + pdffigures2 输出根目录，
    返回一个 ParsedPdf 对象，其中包括：
      - meta（pdffigures2 元信息）
      - figures（图 / 表）
      - paragraphs（正文段落）
      - figure_mentions（figure_id -> 段落列表）
    """
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    # 1. 图/表
    figures = extract_figures_with_pdffigures2(pdf_path, output_root)

    # 2. 读取 meta.json（如果有）
    meta = _load_pdffigures2_meta(pdf_path, output_root)

    # 3. 使用 PyMuPDF 抽取正文段落
    paragraphs = extract_paragraphs_with_pymupdf(pdf_path)

    # 4. 建立 figure_id -> paragraphs 的映射
    figure_mentions = _build_mentions_map(figures, paragraphs)

    return ParsedPdf(
        pdf_path=str(pdf_path),
        meta=meta,
        figures=figures,
        paragraphs=paragraphs,
        figure_mentions=figure_mentions,
    )


# ============= 命令行测试入口 =============

if __name__ == "__main__":
    # 你可以在这里修改测试 PDF 路径
    sample_pdf = r"D:\Papers\sci.pdf"

    try:
        parsed = parse_pdf(sample_pdf)

        print(f"[parse_pdf] PDF: {parsed.pdf_path}")
        print(f"[parse_pdf] Figures/Tables: {len(parsed.figures)}")
        print(f"[parse_pdf] Paragraphs: {len(parsed.paragraphs)}")

        # 打印每个 figure 的基本信息和引用情况
        for fig in parsed.figures:
            fig_dict = asdict(fig)
            fig_id = fig_dict["figure_id"]
            caption_preview = fig_dict["caption"][:80].replace("\n", " ")
            print(f"\n=== {fig_id} (page {fig_dict['page']}) ===")
            print(f"Caption: {caption_preview}...")
            print(f"Image:   {fig_dict['image_path']}")
            mentions = parsed.figure_mentions.get(fig_id, [])
            print(f"Mentions: {len(mentions)} 段落")
            for m in mentions[:2]:  # 只展示前两个段落
                print(f"  - [page {m.page}] {m.text[:100]}...")

    except Exception as exc:
        print(f"Failed to parse {sample_pdf}: {exc}")
