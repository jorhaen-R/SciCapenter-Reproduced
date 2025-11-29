BackendFigure (extract_from_pdf 输出单条):

- figure_id: "Figure 1" / "Table 1"（图号）
- fig_type: "Figure" / "Table"
- page: 1,2,3...
- caption: 原始图注（可能被截断）
- image_path: 本地图片路径
- image_text: OCR 文本（对表格尤其重要）
- paragraphs: 数组，每个元素有：
    - page
    - text (figure-mentioning paragraph)
    - bbox (x1,y1,x2,y2)
