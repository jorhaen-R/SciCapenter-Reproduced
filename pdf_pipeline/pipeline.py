"""
High-level PDF pipeline for the SciCapenter reproduction.

This module exposes a single clean function:

    extract_from_pdf(pdf_path: str) -> list[dict]

which will later be called from the backend API.
"""

from __future__ import annotations


from dataclasses import asdict
from typing import List, Dict, Any

from .parse_pdf import parse_pdf, ParsedPdf

from pathlib import Path
from typing import Dict, List

# 从你已经写好的 parse_pdf.py 中导入底层能力
from .parse_pdf import (
    extract_figures_with_pdffigures2,
    extract_paragraphs_with_pymupdf,
    link_figures_to_paragraphs,
)


def extract_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Top-level helper that:
    1. Extracts figures/tables from a PDF via pdffigures2.
    2. Extracts paragraphs from the same PDF via PyMuPDF.
    3. Links figures to all paragraphs that mention them.

    Args:
        pdf_path: Absolute or relative path to the input PDF.

    Returns:
        A list of dictionaries, one per figure/table, with the shape:

        {
            "figure_id": "Figure 1",
            "fig_type": "Figure",
            "page": 3,
            "caption": "...",
            "image_path": "D:\\pdffigures2_output\\sci\\images\\figsci-Figure1-1.png",
            "paragraphs": [
                {
                    "page": 3,
                    "text": "This figure illustrates ...",
                    "bbox": [x1, y1, x2, y2],
                },
                ...
            ],
        }
    """
    pdf_path = str(Path(pdf_path).resolve())

    # 1) 图/表
    figures = extract_figures_with_pdffigures2(pdf_path)

    # 2) 段落
    paragraphs = extract_paragraphs_with_pymupdf(pdf_path)

    # 3) 关联：figure -> paragraphs
    linked = link_figures_to_paragraphs(figures, paragraphs)

    # 4) 把 dataclass / 自定义对象，转换成干净的 dict 列表
    result: List[Dict] = []
    for fig in linked:
        # 这里假设 fig 是一个类似 dataclass 的对象；
        # 如果你现在已经在 parse_pdf.py 里用 dict，就按你现有字段改名就行。
        result.append(
            {
                "figure_id": fig.figure_id,
                "fig_type": fig.fig_type,
                "page": fig.page,
                "caption": fig.caption,
                "image_path": fig.image_path,
                "paragraphs": [
                    {
                        "page": p.page,
                        "text": p.text,
                        "bbox": p.bbox,  # 比如 [x1, y1, x2, y2]
                    }
                    for p in fig.paragraphs
                ],
            }
        )

    return result


def extract_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    High-level API for the rest of the system.

    Args:
        pdf_path: Absolute or relative path to a PDF.

    Returns:
        A list of dicts, each of the form:
        {
            "figure_id": "Figure 1",
            "fig_type": "Figure",
            "page": 3,
            "caption": "...",
            "image_path": "...",
            "image_text": [...],
            "paragraphs": [
                {"page": 3, "text": "...", "bbox": {...}},
                ...
            ],
        }
    """
    parsed: ParsedPdf = parse_pdf(pdf_path)

    results: List[Dict[str, Any]] = []
    for fig in parsed.figures:
        mentions = parsed.figure_mentions.get(fig.figure_id, [])

        results.append(
            {
                "figure_id": fig.figure_id,
                "fig_type": fig.fig_type,
                "page": fig.page,
                "caption": fig.caption,
                "image_path": fig.image_path,
                "image_text": fig.image_text,
                "paragraphs": [
                    {
                        "page": p.page,
                        "text": p.text,
                        "bbox": p.bbox,
                    }
                    for p in mentions
                ],
            }
        )

    return results


# 小型自测，用来代替之前 parse_pdf.py 里那堆 print
if __name__ == "__main__":
    sample_pdf = r"D:\Papers\sci.pdf"
    items = extract_from_pdf(sample_pdf)

    print(f"[pipeline] PDF: {sample_pdf}")
    print(f"[pipeline] Figures/Tables: {len(items)}")
    for item in items:
        print(
            f"- {item['figure_id']} (page {item['page']}), "
            f"{len(item['paragraphs'])} mentioning paragraphs"
        )

