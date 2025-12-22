import os
import sys
import logging

# --- 核心 LlamaIndex 导入 (不变) ---
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.readers.file import UnstructuredReader # (不变)

# --- (新) 集成导入 ---
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# --- (新) 导入 Re-Ranker ---
from llama_index.postprocessor.cohere_rerank import CohereRerank
# 提醒：您需要一个 Cohere API 密钥并设置 "COHERE_API_KEY" 环境变量

# --- 配置日志 (不变) ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- 定义我们的模型和文件夹 (更新) ---
# 我们现在使用 OpenAI 的模型 API 名称
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini" # 这是一个性价比很高的强大模型

# 文件夹路径 (不变)
PAPERS_DIR = "./medical_papers"
STORAGE_DIR = "./paper_storage_openai" # 建议换个新文件夹，因为 Embedding 模型变了

def check_api_key():
    """检查 OpenAI API 密钥是否已设置"""
    if "OPENAI_API_KEY" not in os.environ:
        print("错误: OPENAI_API_KEY 环境变量未设置。")
        print("请在运行脚本前设置您的 API 密钥：")
        print("  (Mac/Linux) export OPENAI_API_KEY='sk-...'")
        print("  (Windows)   set OPENAI_API_KEY=\"sk-...\"")
        return False
    return True

def setup_global_settings():
    """(已更新) 配置 LlamaIndex 的全局设置 (使用 OpenAI API)"""
    print("--- 1. 配置全局设置 (使用 OpenAI API) ---")
    
    # 设置 Embedding 模型 (使用 OpenAI API)
    Settings.embed_model = OpenAIEmbedding(
        model=EMBED_MODEL
    )
    
    # 设置 LLM (使用 OpenAI API)
    Settings.llm = OpenAI(
        model=LLM_MODEL, 
        temperature=0.1 # 设低一点，让它更准确地基于文献回答
    )

# 
# --- 注意：以下所有函数的代码与“本地版”完全相同！---
# --- LlamaIndex 框架的美妙之处就在于此。---
#

def load_or_create_index(papers_dir, storage_dir):
    """
    (代码不变)
    这是“使用模式”的第一阶段：索引 (Indexing)
    """
    print("--- 2. 加载或创建索引 ---")
    
    # 检查索引是否已在磁盘上
    if not os.path.exists(storage_dir):
        print(f"未找到索引。正在从 '{papers_dir}' 创建新索引...")
        
        # 1. 加载 (Load)
        reader = SimpleDirectoryReader(
            input_dir=papers_dir,
            file_extractor={".pdf": UnstructuredReader()}
        )
        documents = reader.load_data()
        
        # 2. 索引 (Index)
        # LlamaIndex 现在会自动调用 OpenAI API (text-embedding-3-small)
        # 这会产生 API 费用！
        print("正在调用 Embedding API... (这可能会产生费用)")
        index = VectorStoreIndex.from_documents(documents)
        
        # 3. 持久化 (Persist)
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"索引已创建并保存到 '{storage_dir}'。")
        
    else:
        print(f"正在从 '{storage_dir}' 加载现有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print("索引加载成功。")
        
    return index

# 升级代码只需要修改这里的内容
def run_query_engine(index):
    """
    (已更新)
    这是“使用模式”的第二阶段：查询 (Querying)
    我们现在添加了 Re-Ranking 步骤 (对应 Notebook Level 3)
    """
    print("--- 3. 启动查询引擎 (带 Re-Ranking 功能) ---")
    
    # 1. (新) 配置 Re-Ranker
    # 您需要一个 Cohere API 密钥。
    # top_n=3 意味着它会从上一步返回的文档中筛选出最好的 3 个。
    reranker = CohereRerank(top_n=3)
    
    # 2. (更新) 配置查询引擎
    query_engine = index.as_query_engine(
        # (更新) "广撒网": 
        # 先从向量库检索 10 个候选文档
        similarity_top_k=10, 
        
        # (新) "智能过滤": 
        # 将 Reranker 作为后处理器添加
        node_postprocessors=[reranker] 
    )
    
    print("查询引擎已就绪。输入您的问题，或输入 'exit' 退出。")
    
    # ... (while True 循环部分的代码完全不变) ...
    while True:
        try:
            query = input("\n🧐 您的问题: ")
            if query.strip().lower() == 'exit':
                break
            
            # LlamaIndex 现在会自动执行 "Retrieve (k=10)" -> "Re-rank (n=3)" -> "Generate"
            print("正在调用 API (检索, 重排, 生成)... (这可能会产生费用)")
            response = query_engine.query(query)
            
            print("\n💡 答案:")
            print(str(response))
                
        except KeyboardInterrupt:
            break

def main():
    if not check_api_key(): # 检查密钥
        return

    if not os.path.exists(PAPERS_DIR) or not os.listdir(PAPERS_DIR): # (不变)
        print(f"错误: '{PAPERS_DIR}' 文件夹不存在或为空。")
        print("请创建该文件夹，并放入您的医学文献 PDF。")
        return

    setup_global_settings()
    index = load_or_create_index(papers_dir=PAPERS_DIR, storage_dir=STORAGE_DIR)
    run_query_engine(index)
    print("\n程序已退出。")

if __name__ == "__main__":
    main()