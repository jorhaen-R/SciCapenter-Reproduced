import pandas as pd
import json
import os

# --- 1. 配置代理 (必须匹配你的 Clash 端口) ---
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

def download_parquet_direct():
    print("🚀 正在通过 Parquet 接口 **强制直连** SciCap 真实数据库...")
    print("💡 正在跳过本地缓存，直接从内存处理...")

    # Parquet URL
    url = "https://huggingface.co/api/datasets/CrowdAILab/scicap/parquet/default/train/0.parquet"
    output_file = "backend/real_train.jsonl"

    try:
        # 1. 直接读取远程 Parquet 到内存 (pandas DataFrame)
        # 这步如果不报错，说明网络通了且数据拿到了
        print(f"🔗 正在请求: {url}")
        df = pd.read_parquet(url, engine='pyarrow')
        
        print(f"✅ 成功获取数据表！原始行数: {len(df)}")
        print("🔄 正在清洗并写入 JSONL...")

        # 2. 转换格式
        valid_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for index, row in df.iterrows():
                # 容错获取
                caption = str(row.get('caption', '')).strip()
                paragraph = str(row.get('mention_paragraph', '')).strip()

                # 质量过滤
                if len(caption) < 10 or len(paragraph) < 50:
                    continue

                # 构造标准微调格式
                conversation = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful scientific writing assistant. Generate a concise and accurate caption for the figure based on the provided context."},
                        {"role": "user", "content": f"Context Paragraphs: {paragraph}\n\nTask: Generate a caption."},
                        {"role": "assistant", "content": caption}
                    ]
                }
                f.write(json.dumps(conversation) + "\n")
                valid_count += 1
                
                # 限制 2000 条，防止训练太久
                if valid_count >= 2000:
                    break

        print(f"🎉 完美成功！已生成 {valid_count} 条 **真实的** SciCap 科研数据！")
        print(f"📂 数据集路径: {output_file}")
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        print("如果是 ConnectTimeout，请检查 Clash 是否开启了 TUN 模式或系统代理。")

if __name__ == "__main__":
    download_parquet_direct()