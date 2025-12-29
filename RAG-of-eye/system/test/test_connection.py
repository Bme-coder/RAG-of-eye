import os
import sys
from dotenv import load_dotenv
from openai import OpenAI


def _collect_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts).strip()
    return str(content or "").strip()


def test_yunwu_connection():
    # 1. 加载 .env 文件
    print("正在加载环境变量...")
    load_dotenv()

    # 2. 获取配置
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("ANTHROPIC_API_URL")
    model_name = (
        os.getenv("LLM_MODEL_NAME")
        or os.getenv("CLAUDE_MODEL_NAME")
        or "gpt-4o-mini"
    )

    # 3. 打印调试信息 (隐藏部分 Key 防止泄露)
    if not api_key:
        print("❌ 错误: 未找到 OPENAI_API_KEY，请检查 .env 文件")
        return

    masked_key = api_key[:8] + "****" + api_key[-4:] if len(api_key) > 12 else "****"
    print(f"✅检测到 API Key: {masked_key}")
    print(f"✅检测到 Base URL: {base_url}")
    print(f"👉 准备使用的模型 {model_name}")

    # 4. 初始化客户端
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
    except Exception as e:
        print(f"❌客户端初始化失败: {e}")
        return

    # 5. 发送测试请求
    print("\n🚀 正在发起测试请求 (这可能需要几秒钟)...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=128,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a health-check bot that confirms GPT connectivity."},
                {"role": "user", "content": "你好，请回复“连接成功”四个字，并告诉我你当前使用的模型名称。"},
            ],
        )

        # 6. 解析并打印结果
        content = _collect_text(response.choices[0].message.content)
        print("\n" + "=" * 30)
        print("🎉 测试成功！收到了来自 GPT API 的回复：")
        print("=" * 30)
        print(content)
        print("=" * 30)

    except Exception as e:
        print(f"\n❌请求失败: {e}")
        print("提示：")
        print("1. 如果是401 Error: API Key 错误或额度不足。")
        print("2. 如果是404 Error: 模型名称写错了，或者 Base URL 不对。")
        print("3. 如果是Connection Error: 网络问题或 Base URL 拼写错误。")


if __name__ == "__main__":
    test_yunwu_connection()
