# SciCapenter Figure Data Schema

## 1. Core Data Object: `FigureDTO`

后端 API (`/upload_pdf`, `/parse_pdf`) 返回的 `figures` 列表中的每一项都遵循此结构。

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **figure_id** | `string` | 图片的唯一标识符 | `"Figure 1"` |
| **title** | `string` | 用于显示的标题 | `"Figure 1"` |
| **caption** | `string` | 原始论文中的 Caption 文本 | `"Overview of the system pipeline..."` |
| **image_path** | `string` | 图片的相对 Web 路径 (Clean URL) | `"pdffigures2_output/uuid/images/fig1.png"` |
| **mention_paragraphs**| `List[str]`| 提及该图片的段落文本列表 | `["As shown in Figure 1, the system..."]` |
| **page** | `int` | 图片所在的页码 | `3` |

## 2. API Response Format

```json
{
  "doc_id": "uuid-string",
  "figures": [
    {
      "figure_id": "Figure 1",
      "title": "Figure 1",
      "caption": "...",
      "image_path": "pdffigures2_output/...",
      "mention_paragraphs": [...],
      "page": 2
    }
  ]
}