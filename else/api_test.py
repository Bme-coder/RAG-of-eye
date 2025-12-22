import os
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

# --- 在这里直接输入您的 API 密钥 ---
# 警告：请勿将包含此密钥的代码分享或上传到 GitHub！
YOUR_API_KEY = "sk-Nl5KbcmhQQZxL7mLzHbuO77aoVAI9T1DqvU8x3l72klHKRei"  # <--- 在这里粘贴您的 OpenAI API 密钥

def test_openai_api_hardcoded():
    """
    一个简单的小程序，用于测试 OpenAI API 是否配置正确。
    (此版本直接在代码中写入 API 密钥)
    """
    print("--- 正在开始测试 OpenAI API (使用硬编码密钥) ---")

    print("✅ 成功: 找到了硬编码的 API 密钥。")

    try:
        # 2. 初始化 OpenAI 客户端
        # (我们现在使用 api_key 参数直接传递密钥)
        client = OpenAI(
            api_key=YOUR_API_KEY
        )
        print("... 正在初始化 OpenAI 客户端...")

        # 3. 发起一个简单的 API 请求
        print("... 正在向 'gpt-4o-mini' 发送测试请求 (请求讲个笑话)...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "帮我讲一个关于程序员的简短笑话。"}
            ],
            temperature=0.7,
            max_tokens=100
        )

        # 4. 解析并打印结果
        answer = response.choices[0].message.content.strip()

        print("\n--- ✅ 测试成功! ---")
        print("🤖 模型的回答:")
        print(answer)

    # 5. 处理常见的 API 错误
    except AuthenticationError:
        print("\n❌ 认证错误 (AuthenticationError):")
        print("您在代码中填写的 API 密钥无效、已过期或已被撤销。")
        print("请登录 OpenAI 平台检查您的 API 密钥设置。")
    
    except RateLimitError:
        print("\n❌ 速率限制错误 (RateLimitError):")
        print("该 API 密钥已超出配额。这可能是因为：")
        print("1. 免费试用额度已用完。")
        print("2. 账户未绑定有效的支付方式。")
        print("请登录 OpenAI 平台检查您的账户状态和用量限制。")
        
    except APIConnectionError:
        print("\n❌ 连接错误 (APIConnectionError):")
        print("无法连接到 OpenAI API。")
        print("请检查您的网络连接、防火墙或代理设置。")

    except Exception as e:
        print(f"\n❌ 发生了一个意外错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")

# 运行主测试函数
if __name__ == "__main__":
    test_openai_api_hardcoded()