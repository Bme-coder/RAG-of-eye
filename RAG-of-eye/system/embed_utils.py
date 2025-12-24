from __future__ import annotations

import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

DEFAULT_LOCAL_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-large"


def build_embedding_from_env():
    """
    根据环境变量选择嵌入提供商：
    - EMBED_PROVIDER=local (默认)：加载本地 HuggingFace 模型
    - EMBED_PROVIDER=api：调用 OpenAI Embedding API
    """
    provider = (os.getenv("EMBED_PROVIDER") or "local").strip().lower()
    if provider == "api":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("EMBED_PROVIDER=api 但缺少 OPENAI_API_KEY。")
        api_base = os.getenv("OPENAI_API_BASE")
        model_name = os.getenv("OPENAI_EMBED_MODEL", DEFAULT_OPENAI_EMBED_MODEL)
        return OpenAIEmbedding(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
        )

    model_name = os.getenv("EMBED_MODEL_NAME", DEFAULT_LOCAL_EMBED_MODEL)
    return HuggingFaceEmbedding(model_name=model_name)
