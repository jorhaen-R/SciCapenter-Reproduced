from openai import OpenAI

# 1. 直接填入你的 DeepSeek Key (不要用 os.getenv 了，先排除环境问题)
# 注意：sk-后面那一长串都要复制进去
MY_API_KEY = "sk-5699214e4174482cafe3a1468a5d3428" 

# 2. 初始化客户端 (强制指定 base_url)
client = OpenAI(
    api_key=MY_API_KEY,
    base_url="https://api.deepseek.com"  # <--- 这行代码必须生效，否则就会连去美国
)

try:
    print(f"正在尝试连接: {client.base_url}")
    print("正在呼叫 DeepSeek...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "你好，能收到吗？"},
        ],
        stream=False
    )
    print("✅ 成功回复:", response.choices[0].message.content)
    
except Exception as e:
    print("❌ 依然失败:", e)