import shutil
import subprocess
import re
import os
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent
PDFFIGURES2_JAR = BASE_DIR / "pdffigures2.jar"

# ==============================================================================
# 1. 辅助算法：基于坐标流的文本重构
# ==============================================================================
def get_clean_text_from_page(page) -> str:
    """
    将双栏布局的 PDF 页面重组为单栏线性文本。
    策略：获取所有单词 -> 按 X 轴分左右栏 -> 按 Y 轴排序拼接
    """
    try:
        width = page.width
        height = page.height
        
        # 1. 提取所有单词及其坐标
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False
        )
        
        # 2. 过滤页眉页脚 (Top 8%, Bottom 8%)
        content_words = [
            w for w in words 
            if (height * 0.08) < w['top'] < (height * 0.92)
        ]
        
        # 3. 分栏逻辑
        mid_point = width / 2
        left_col_words = []
        right_col_words = []
        
        for w in content_words:
            word_center = (w['x0'] + w['x1']) / 2
            if word_center < mid_point:
                left_col_words.append(w)
            else:
                right_col_words.append(w)
        
        # 4. 排序 (按 Top 从上到下，如果 Top 相同按 x0 从左到右)
        left_col_words.sort(key=lambda w: (w['top'], w['x0']))
        right_col_words.sort(key=lambda w: (w['top'], w['x0']))
        
        # 5. 拼接
        left_text = " ".join([w['text'] for w in left_col_words])
        right_text = " ".join([w['text'] for w in right_col_words])
        
        return left_text + " " + right_text
        
    except Exception as e:
        print(f"[Pipeline Warning] Page text extraction failed: {e}")
        return ""

# ==============================================================================
# 2. 核心逻辑：提取提及段落 (带噪音过滤)
# ==============================================================================
def extract_mention_paragraphs(pdf_path: str, figures: List[Dict]) -> Dict[str, List[str]]:
    mentions_map = {f['figure_id']: [] for f in figures}
    
    # 1. 准备正则
    fig_patterns = {}
    for fig in figures:
        fid = fig['figure_id']
        if '-' in fid:
            ftype, fnum = fid.split('-', 1)
        else:
            ftype = "Figure"
            fnum = "".join(filter(str.isdigit, fid))
        
        if not fnum: continue

        if "Figure" in ftype:
            pattern = re.compile(rf"(?i)(Figure|Fig\.?)\s*{re.escape(fnum)}\b")
        else:
            pattern = re.compile(rf"(?i)(Table|Tab\.?)\s*{re.escape(fnum)}\b")
        fig_patterns[fid] = pattern

    # --- 🔥 噪音检测函数 ---
    def is_table_noise(text: str, pattern: re.Pattern) -> bool:
        text = text.strip()
        if not text: return True
        
        # 1. 是图注本身吗？(Table 5: Running time...)
        if re.search(rf"^{pattern.pattern}\s*:", text, re.IGNORECASE):
            return True
            
        # 2. 是表格特殊符号吗？(*, KL*, T**)
        if "KL*" in text or "T**" in text or text.startswith("*"):
            return True
            
        # 3. 是表格数据行吗？(数字占比过高)
        # 计算数字字符的比例
        digit_count = sum(c.isdigit() for c in text)
        if len(text) > 0 and (digit_count / len(text) > 0.15): 
            return True
            
        # 4. 太短或者是纯表头
        if len(text) < 40:
            return True
            
        return False

    try:
        # 2. 提取全文
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = get_clean_text_from_page(page)
                full_text += text + " "
        
        full_text = re.sub(r'\s+', ' ', full_text)

        # 3. 语义分句
        sentences = re.split(r'(?<=[.?!])\s+', full_text)
        
        # 4. 匹配与提取
        for i, sent in enumerate(sentences):
            for fid, pattern in fig_patterns.items():
                if pattern.search(sent):
                    # 尝试构建一个段落 (当前句 + 前后句)
                    # 我们先收集候选句子，然后逐个过滤
                    raw_candidates = sentences[max(0, i-1) : min(len(sentences), i+3)]
                    clean_candidates = []
                    
                    for cand in raw_candidates:
                        if not is_table_noise(cand, pattern):
                            clean_candidates.append(cand)
                    
                    if not clean_candidates: continue

                    paragraph = " ".join(clean_candidates)
                    
                    # 最后的整体检查
                    if "[%CODE%]" in paragraph: continue
                    if len(paragraph) < 50: continue

                    if paragraph not in mentions_map[fid]:
                        mentions_map[fid].append(paragraph)

    except Exception as e:
        print(f"[Pipeline Error] Extraction failed: {e}")

    return mentions_map

# ==============================================================================
# 3. pdffigures2 工具调用
# ==============================================================================
def run_pdffigures2(pdf_path: str, output_dir: str) -> bool:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    images_prefix = output_dir / "images" / "figure"
    data_prefix = output_dir / "data" / "data"
    images_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", "-Xmx4g", "-jar", str(PDFFIGURES2_JAR),
        str(pdf_path),
        "-m", str(images_prefix),
        "-d", str(data_prefix)
    ]
    print(f"[pipeline] Running pdffigures2: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[pipeline warning] pdffigures2 stderr:\n{result.stderr}")
    return True

# ==============================================================================
# 4. 核心编排
# ==============================================================================
def extract_from_pdf(pdf_path: str, output_root: Path, static_base: str) -> List[Dict[str, Any]]:
    from .parse_pdf import parse_pdf
    
    parsed = parse_pdf(pdf_path, output_root)
    
    temp_figures = [{"figure_id": fig.figure_id} for fig in parsed.figures]
    mentions_map = extract_mention_paragraphs(pdf_path, temp_figures)

    results = []
    for fig in parsed.figures:
        # Robust Path Construction
        try:
            # Try to get path relative to the UUID folder
            rel_path = Path(fig.image_path).relative_to(output_root)
            web_path = f"{static_base}/{rel_path.as_posix()}"
        except ValueError:
            # Fallback for Linux/Docker relative paths
            # If the path is just "images/figure-1.png", relative_to absolute path fails.
            fname = Path(fig.image_path).name
            if "images" in str(fig.image_path):
                web_path = f"{static_base}/images/{fname}"
            else:
                web_path = f"{static_base}/{fname}"

        paragraphs = mentions_map.get(fig.figure_id, [])
        results.append({
            "figure_id": fig.figure_id,
            "fig_type": fig.fig_type,
            "page": fig.page,
            "caption": fig.caption,
            "image_path": web_path,
            "image_text": fig.image_text,
            "mention_paragraphs": paragraphs
        })
    return results
