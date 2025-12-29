import os
import sys
import logging
from pathlib import Path
import nest_asyncio

from dotenv import load_dotenv

# --- 核心 LlamaIndex 导入 ---
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser

# --- 集成导入 ---
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.postprocessor import LLMRerank

from embed_utils import build_embedding_from_env

# 应用 nest_asyncio
nest_asyncio.apply()

# --- 配置日志 ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 默认配置参数 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LLM_MODEL = (
    os.getenv("LLM_MODEL_NAME")
    or os.getenv("CLAUDE_MODEL_NAME")
    or "gpt-4o-mini"
)
DEFAULT_PAPERS_DIR = str((DATA_ROOT / "raw" / "medical_papers").resolve())
DEFAULT_STORAGE_DIR = str((DATA_ROOT / "index" / "paper_storage_vlm_structure").resolve())

def check_api_keys():
    """检查必要环境变量"""
    load_dotenv()
    missing = []
    if "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
        missing.append("OPENAI_API_KEY")
    if "LLAMA_CLOUD_API_KEY" not in os.environ:
        missing.append("LLAMA_CLOUD_API_KEY")
    
    if missing:
        print(f"❌ 错误: 缺少环境变量: {missing}")
        return False
    return True

def setup_global_settings():
    """配置全局模型"""
    if Settings.llm is None: 
        print("--- 配置全局模型 (GPT-4o mini) ---")
        
        # 👇 1. 显式获取 .env 里的代理地址
        api_url = os.getenv("OPENAI_API_BASE") or os.getenv("ANTHROPIC_API_URL")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if api_url:
            print(f"🌍 使用代理地址: {api_url}")

        # 👇 2. 传给 Embedding 模型
        Settings.embed_model = build_embedding_from_env()
        
        # 👇 3. 传给 LLM 模型
        Settings.llm = LlamaOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=api_key,
            api_base=api_url,
            max_tokens=1024,
        )

def load_or_create_index(papers_dir, storage_dir):
    """加载或创建索引 (核心 VLM 解析逻辑)"""
    if not os.path.exists(storage_dir):
        print(f"--- 未找到索引，正在解析 '{papers_dir}' ---")
        
        # 1. 配置 VLM Parser (加入指令优化)
        parser = LlamaParse(
            result_type="markdown",
            parsing_instruction="""
            The document is a medical paper. 
            1. Preserve all tables in Markdown format. 
            2. Keep quantitative data (percentages, p-values, sample sizes) accurate.
            3. Ignore headers and footers.
            """,
            verbose=True,
            language="en", 
        )
        
        # 2. 加载数据
        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader(
            input_dir=papers_dir,
            file_extractor=file_extractor
        )
        documents = reader.load_data()

        # 3. 结构化切分 (提取表格)
        print("--- 正在提取文档结构 (表格对象) ---")
        node_parser = MarkdownElementNodeParser(llm=Settings.llm, num_workers=8)
        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        # 4. 建立索引
        index = VectorStoreIndex(nodes=base_nodes + objects)
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"索引已保存至 '{storage_dir}'")
    else:
        print(f"--- 从 '{storage_dir}' 加载现有索引 ---")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        
    return index

def build_query_engine(index, similarity_top_k=10):
    """
    (修改点) 仅构建并返回引擎对象，不运行死循环
    这是给外部接口调用的核心组件
    """
    print("--- 构建查询引擎 (Top-10 + Rerank Top-3) ---")
    
    reranker = LLMRerank(top_n=3)
    
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        node_postprocessors=[reranker],
        response_mode="compact" 
    )
    return query_engine

# ==========================================
# 核心接口：供外部程序 (如 ClinicalAgent) 调用
# ==========================================
def get_rag_engine(
    papers_dir=DEFAULT_PAPERS_DIR,
    storage_dir=DEFAULT_STORAGE_DIR,
    similarity_top_k=10,
):
    """
    对外暴露的唯一接口。
    调用此函数，直接返回一个可以 .query() 的引擎对象。
    """
    if not check_api_keys():
        raise EnvironmentError("Missing API Keys")
    
    # 1. 确保目录存在
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir, exist_ok=True)
        
    # 2. 初始化设置
    setup_global_settings()
    
    # 3. 获取索引
    index = load_or_create_index(papers_dir, storage_dir)
    
    # 4. 返回构建好的引擎
    return build_query_engine(index, similarity_top_k=similarity_top_k)

# ==========================================
# 本地测试入口 (保留原有交互功能)
# ==========================================
if __name__ == "__main__":
    # 只有直接运行此文件时，才会执行下面的交互逻辑
    print("正在初始化 RAG 系统...")
    
    try:
        # 调用自己的接口获取引擎
        engine = get_rag_engine()
        
        print("\n✅ 系统就绪。进入对话模式 (输入 'exit' 退出)。")
        while True:
            query = input("\n🧐 问题: ")
            if query.strip().lower() == 'exit':
                break
            if not query.strip():
                continue
            
            response = engine.query(query)
            print(f"\n💡 答案: {response}")
            
    except Exception as e:
        print(f"启动失败: {e}")
