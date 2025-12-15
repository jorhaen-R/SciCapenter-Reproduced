import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class ParagraphInfo:
    page: int
    text: str
    bbox: List[float]

@dataclass
class ParsedFigure:
    figure_id: str
    fig_type: str
    page: int
    caption: str
    image_path: str
    image_text: str = ""

@dataclass
class ParsedPdf:
    pdf_path: str
    figures: List[ParsedFigure] = field(default_factory=list)
    figure_mentions: Dict[str, List[ParagraphInfo]] = field(default_factory=dict)

def load_json_robust(path: Path) -> Any:
    encodings = ["utf-8", "gb18030", "latin1"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return []

def parse_pdf(pdf_path: str, output_root: Path) -> ParsedPdf:
    pdf_path_obj = Path(pdf_path)
    data_dir = output_root / "data"
    
    # 自动搜索数据文件
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"[parse_pdf] Warning: No JSON found in {data_dir}")
        return ParsedPdf(pdf_path=pdf_path)

    json_path = json_files[0]
    data = load_json_robust(json_path)
    parsed = ParsedPdf(pdf_path=pdf_path)

    if not isinstance(data, list):
        data = []

    for entry in data:
        # --- 核心修复: ID 唯一化逻辑 ---
        raw_name = entry.get("name", "")
        fig_type = entry.get("figType", "Figure")  # 通常是 "Figure" 或 "Table"
        
        # 如果 name 只是纯数字 "3"，组合成 "Figure-3"
        # 如果 name 已经是 "Figure 3"，将其规范化为 "Figure-3"
        if not raw_name:
            # 兜底：从 URL 获取
            url = entry.get("renderURL", "Unknown-1")
            raw_name = url.split("-")[-1].replace(".png", "")
        
        # 强制构建唯一 ID: {Type}-{Name}
        # 例如: Figure-3, Table-3
        clean_name = raw_name.replace(" ", "") # 去空格
        if fig_type not in clean_name:
            unique_id = f"{fig_type}-{clean_name}"
        else:
            unique_id = clean_name

        # 图片路径处理
        img_rel_path = entry.get("renderURL", "")
        if img_rel_path.startswith("/") or img_rel_path.startswith("\\"):
             img_rel_path = img_rel_path[1:]
        full_image_path = str(output_root / img_rel_path)

        figure = ParsedFigure(
            figure_id=unique_id, # 使用新的唯一 ID
            fig_type=fig_type,
            page=entry.get("page", 1),
            caption=entry.get("caption", ""),
            image_path=full_image_path,
            image_text=entry.get("regionless-caption", "")
        )
        parsed.figures.append(figure)

    return parsed