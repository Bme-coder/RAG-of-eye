import os
import sys
import logging
import nest_asyncio

# --- 核心 LlamaIndex 导入 ---
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
# (新) 引入 VLM 解析器
from llama_parse import LlamaParse
# (新) 引入 Markdown 结构化节点解析器 (处理表格的神器)
from llama_index.core.node_parser import MarkdownElementNodeParser

# --- 集成导入 ---
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import LLMRerank

# 应用 nest_asyncio，防止在某些环境(如Notebook/复杂脚本)下 LlamaParse 异步报错
nest_asyncio.apply()

# --- 配置日志 ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# 过滤掉部分 HTTP 请求日志，保持清爽
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 配置参数 ---
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini" 

# 文件夹路径
PAPERS_DIR = "./medical_papers"
# 建议修改存储路径名，以区分旧版索引
STORAGE_DIR = "./paper_storage_vlm_structure" 

def check_api_keys():
    """(更新) 检查 OpenAI 和 LlamaCloud API 密钥"""
    missing = []
    if "OPENAI_API_KEY" not in os.environ:
        missing.append("OPENAI_API_KEY")
    if "LLAMA_CLOUD_API_KEY" not in os.environ:
        missing.append("LLAMA_CLOUD_API_KEY")
    
    if missing:
        print("❌ 错误: 缺少以下环境变量:")
        for k in missing:
            print(f"   - {k}")
        print("\n请设置环境变量 (LlamaKey 去 https://cloud.llamaindex.ai 免费申请):")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export LLAMA_CLOUD_API_KEY='llx-...'")
        return False
    return True

def setup_global_settings():
    """配置 LlamaIndex 全局设置"""
    print("--- 1. 配置全局模型 (OpenAI) ---")
    
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    # 温度设为 0，学术问答需要严谨
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0) 

def load_or_create_index(papers_dir, storage_dir):
    """
    (核心修改) 加载或创建索引
    这里实现了 VLM 解析 -> 结构化切分 -> 递归索引 的全流程
    """
    print("--- 2. 加载或创建索引 ---")
    
    if not os.path.exists(storage_dir):
        print(f"未找到索引。正在使用 VLM 解析 '{papers_dir}' 中的文档...")
        print("这可能需要几分钟，因为要将 PDF 上传到云端进行视觉分析...")
        
        # --- 步骤 A: 配置 VLM 解析器 (LlamaParse) ---
        parser = LlamaParse(
            result_type="markdown", # 输出 Markdown
            verbose=True,
            language="en",          # 如果是中文论文改为 "ch"
            num_workers=4           # 并行解析
        )
        
        # --- 步骤 B: 加载数据 (使用 LlamaParse 替换 Unstructured) ---
        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader(
            input_dir=papers_dir,
            file_extractor=file_extractor # 关键：指定 PDF 用 LlamaParse 处理
        )
        documents = reader.load_data()
        print(f"文档解析完成，共加载 {len(documents)} 页/部分。")

        # --- 步骤 C: 结构化切分 (提取表格和层级) ---
        print("正在分析文档结构 (提取表格对象)...")
        node_parser = MarkdownElementNodeParser(
            llm=Settings.llm, 
            num_workers=8
        )
        
        # 获取原始节点
        nodes = node_parser.get_nodes_from_documents(documents)
        # 分离出“基础文本节点”和“表格对象节点”
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        print(f"解析出 {len(base_nodes)} 个文本块和 {len(objects)} 个结构化对象(如表格)。")

        # --- 步骤 D: 建立递归索引 ---
        # 将所有节点（包括表格对象）一起建立索引
        # LlamaIndex 会自动处理摘要->原始表格的映射
        index = VectorStoreIndex(nodes=base_nodes + objects)
        
        # --- 步骤 E: 持久化 ---
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"索引已创建并保存到 '{storage_dir}'。")
        
    else:
        print(f"正在从 '{storage_dir}' 加载现有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print("索引加载成功。")
        
    return index

def run_query_engine(index):
    """
    (保持逻辑) 启动查询引擎
    """
    print("--- 3. 启动查询引擎 (OpenAI Rerank + 结构化检索) ---")
    
    # 配置 Reranker
    reranker = LLMRerank(top_n=3)
    
    # 配置查询引擎
    # 这里的 recursive=True 是默认开启的，保证能从表格摘要检索回原表格
    query_engine = index.as_query_engine(
        similarity_top_k=10, 
        node_postprocessors=[reranker],
        response_mode="compact" 
    )
    
    print("✅ 系统就绪。支持查询论文细节、公式参数、表格数据。")
    
    while True:
        try:
            query = input("\n🧐 问题 (输入 'exit' 退出): ")
            if query.strip().lower() == 'exit':
                break
            if not query.strip():
                continue
            
            print("正在检索并生成答案...")
            response = query_engine.query(query)
            
            print("\n💡 答案:")
            print(str(response))
            
            # (可选) 打印出引用的来源，看看它是从表格里拿的还是文本里拿的
            # print("\n--- 参考来源 ---")
            # for source in response.source_nodes:
            #     print(f"[Score: {source.score:.2f}] {source.node.get_content()[:50]}...")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    if not check_api_keys():
        return

    if not os.path.exists(PAPERS_DIR):
        os.makedirs(PAPERS_DIR)
        print(f"已创建目录 '{PAPERS_DIR}'。请将 PDF 放入该目录后重试。")
        return
        
    if not os.listdir(PAPERS_DIR):
        print(f"错误: '{PAPERS_DIR}' 是空的。请放入 PDF。")
        return

    setup_global_settings()
    index = load_or_create_index(papers_dir=PAPERS_DIR, storage_dir=STORAGE_DIR)
    run_query_engine(index)
    print("\n程序已退出。")

if __name__ == "__main__":
    main()