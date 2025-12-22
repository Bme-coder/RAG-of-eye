from openai import OpenAI

if __name__ == '__main__':
    client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-Nl5KbcmhQQZxL7mLzHbuO77aoVAI9T1DqvU8x3l72klHKRei',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "what time is it now?",
            }
        ],
        model="gpt-4o-mini", # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动
    )

    print(chat_completion)