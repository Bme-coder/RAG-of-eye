"""
Health-check script that validates the GPT-4o mini completion endpoint
and the OpenAI-compatible embedding endpoint configured in system/.env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_environment() -> None:
    """Load environment variables from system/.env if it exists."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        print(f"✅ 已加载环境变量: {ENV_PATH}")
    else:
        load_dotenv()
        print("⚠️ 未找到 system/.env，回退到默认的环境加载逻辑。")


def _collect_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
        return "".join(texts).strip()
    return str(content or "").strip()


def check_llm_chat() -> Tuple[bool, str]:
    """Send a lightweight prompt to validate the GPT endpoint."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("ANTHROPIC_API_URL")
    model_name = (
        os.getenv("LLM_MODEL_NAME")
        or os.getenv("CLAUDE_MODEL_NAME")
        or "gpt-4o-mini"
    )

    if not api_key:
        return False, "缺少 OPENAI_API_KEY，请检查 .env。"

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")

    client = OpenAI(**client_kwargs)
    try:
        msg: ChatCompletion = client.chat.completions.create(
            model=model_name,
            max_tokens=64,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a connection testing assistant."},
                {"role": "user", "content": "请仅回复：连接成功。"},
            ],
        )
        content = _collect_text(msg.choices[0].message.content)
        if "连接成功" in content:
            return True, f"GPT 模型 {model_name} 正常响应。回复: {content}"
        return True, f"GPT 返回非预期内容: {content}"
    except Exception as exc:  # noqa: BLE001
        return False, f"GPT API 调用失败: {exc}"


def check_embedding_api() -> Tuple[bool, str]:
    """Send a minimal embedding request through the OpenAI-compatible endpoint."""
    provider = (os.getenv("EMBED_PROVIDER") or "local").strip().lower()
    if provider != "api":
        return True, "EMBED_PROVIDER != api，跳过 OpenAI 接口检测。"

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = (os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-large"

    if not api_key:
        return False, "EMBED_PROVIDER=api 但缺少 OPENAI_API_KEY。"

    url = f"{api_base}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": "ping"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        vector_len = len(data["data"][0]["embedding"])
        return True, f"OpenAI embedding 接口可用，向量长度 {vector_len}。"
    except requests.HTTPError as http_err:
        return False, f"OpenAI 兼容接口 HTTP 错误: {http_err}，响应: {http_err.response.text}"
    except Exception as exc:  # noqa: BLE001
        return False, f"OpenAI 兼容接口调用失败: {exc}"


def main() -> None:
    load_environment()
    print("\n=== GPT / OpenAI 连通性检测 ===")
    llm_ok, llm_msg = check_llm_chat()
    status = "✅" if llm_ok else "❌"
    print(f"{status} {llm_msg}")

    print("\n=== OpenAI Embedding 接口检测 ===")
    embed_ok, embed_msg = check_embedding_api()
    status = "✅" if embed_ok else "❌"
    print(f"{status} {embed_msg}")

    print("\n=== 检测总结 ===")
    summary = {
        "llm": {"ok": llm_ok, "message": llm_msg},
        "embedding": {"ok": embed_ok, "message": embed_msg},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
