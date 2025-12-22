# 🚀 VLM-Enhanced Academic RAG (基于视觉理解的学术文献问答系统)

这是一个基于 **LlamaIndex 2.0** 架构构建的高级 RAG 系统，专为处理**学术论文、复杂研报和技术手册**而设计。

与传统的基于 OCR/文本提取的 RAG 系统不同，本项目引入了 **VLM (Vision-Language Model)** 技术，能够像人眼一样“看懂”文档中的复杂排版、双栏结构、数学公式和统计表格，并将其转化为结构化的知识库。

## ✨ 核心特性

  * **👁️ VLM 视觉解析 (LlamaParse):** 使用视觉大模型替代传统 OCR，完美解决 PDF **双栏错乱、公式乱码、页眉页脚干扰**等顽疾。
  * **📊 结构化表格索引 (MarkdownElementNodeParser):** 自动识别文档中的表格，将其提取为独立对象。系统能理解表格的行列关系，支持对具体实验数据的精准检索。
  * **🧠 递归检索 (Recursive Retrieval):** 采用“摘要检索 -\> 原文召回”策略。检索时匹配表格摘要，生成时注入完整 Markdown 表格，大幅减少幻觉。
  * **🎯 二阶段重排序 (LLM Reranking):** 集成 OpenAI 的重排序能力，从向量初筛结果中精选 Top-3，确保回答的相关性。

## 🆚 架构对比：为什么升级？

| 特性 | 🟢 本项目 (RAG 2.0) | 🔴 传统方案 (RAG 1.0 / PyPDF2) |
| :--- | :--- | :--- |
| **解析方式** | **视觉理解 (Vision)** | 文本流提取 (Text Stream) |
| **表格处理** | **转为标准 Markdown 表格** | 经常变成一堆乱码字符串 |
| **多栏排版** | **自动识别分栏，阅读顺序正确** | 容易发生跨栏文字拼接错误 |
| **检索粒度** | **结构化节点 (表格/文本分离)** | 纯文本切片 (容易断章取义) |

## 🛠️ 环境准备

### 1\. 申请 API 密钥

你需要两个服务的 API Key 才能运行此项目：

1.  **OpenAI API Key:** 用于 LLM 推理和 Embedding。
      * [获取 OpenAI Key](https://platform.openai.com/)
2.  **LlamaCloud API Key:** 用于 VLM 文档解析 (每天免费 1000 页)。
      * [获取 LlamaCloud Key](https://www.google.com/search?q=https://cloud.llamaindex.ai/)

### 2\. 安装依赖

推荐使用 Conda 创建虚拟环境：

```bash
# 创建并激活环境
conda create -n rag_vlm python=3.10
conda activate rag_vlm

# 安装核心库
pip install llama-index llama-parse llama-index-embeddings-openai llama-index-llms-openai llama-index-postprocessor-flag-embedding nest_asyncio
```

## 🏃‍♂️ 快速开始

### 第一步：配置环境变量

在终端中设置你的 API Key (或者将其写入系统的环境变量配置中)：

**Mac/Linux:**

```bash
export OPENAI_API_KEY='sk-proj-...'
export LLAMA_CLOUD_API_KEY='llx-...'
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="sk-proj-..."
$env:LLAMA_CLOUD_API_KEY="llx-..."
```

### 第二步：准备数据

在项目根目录下创建一个名为 `medical_papers` 的文件夹，并将你的 PDF 论文放入其中。

```bash
mkdir medical_papers
# 将你的 paper.pdf 复制进去
```

### 第三步：运行系统

运行主程序：

```bash
python main.py
```

系统将自动执行以下流程：

1.  上传 PDF 到 LlamaCloud 进行视觉解析。
2.  下载解析后的 Markdown 数据。
3.  提取表格和文本结构。
4.  生成本地向量索引 (保存在 `paper_storage_vlm_structure` 目录)。
5.  启动问答对话框。

用户问题 $\xrightarrow{\text{embedding}}$ 向量库 $\xrightarrow{\text{top\_k=10}}$ 10 个粗选片段 $\xrightarrow{\text{LLMRerank}}$ 3 个精选片段 (top_n=3) $\xrightarrow{\text{Context}}$ GPT-4o-mini $\rightarrow$ 最终答案

## 📂 项目结构

```text
.
├── main.py                      # 主程序代码
├── medical_papers/              # [输入] 存放 PDF 文件
│   └── attention_is_all_you_need.pdf
├── paper_storage_vlm_structure/ # [输出] 自动生成的向量索引数据 (持久化存储)
└── README.md                    # 说明文档
```

## ⚠️ 成本与隐私提示

  * **API 成本:** 解析过程会调用 OpenAI Embedding 接口，问答过程会调用 GPT-4o-mini。请关注你的 OpenAI 余额。
  * **LlamaParse 配额:** 免费版账号每天限制解析 1000 页 PDF，超过需要付费。
  * **数据隐私:** 使用本方案时，PDF 文件会被上传至 LlamaCloud 进行解析，文本会发送至 OpenAI。**请勿用于处理绝密/涉密科研数据**。

-----
# 后续提升点
这是一份专门为你整理的 Markdown 文档。

我特意采用了\*\*“设计意图 (Design Intent) + 技术实现 (Implementation)”**的结构。这种写法不仅适合人类阅读，更重要的是，当未来有 AI（包括我或其他编程 Agent）读取你的 README 时，能通过这段文字彻底理解你**“为什么要这样配置”**以及**“期望达到什么效果”\*\*，从而生成更精准的代码。

你可以将以下内容直接添加到你 README 的 **Advanced Configuration** 或 **Optimization Strategy** 章节中。

-----

## 🧠 Advanced Optimization Strategies (LlamaParse 进阶配置)

> **Note for AI Collaborators:** This section outlines the advanced parsing logic derived from `demo_advanced.ipynb`. The goal is to move from generic PDF parsing to **domain-specific (Academic) parsing** by leveraging LlamaParse's promptability and multimodal capabilities.

我们通过分析 LlamaParse 的高级用法，对本系统的解析模块进行了以下三点核心优化，以适应学术文献的特殊性：

### 1\. Promptable Parsing (可提示的解析策略)

**设计意图 (Design Intent):**
LlamaParse 不仅仅是一个转换工具，它是一个接受自然语言指令的 Agent。对于学术论文，默认解析往往会保留页眉、页脚（如 "Page 1 of 10", "Confidential"）以及参考文献列表，这些对于 RAG 检索来说是高噪音数据。同时，我们需要强制 LaTeX 格式以确保数学公式的语义完整性。

**技术实现 (Implementation):**
利用 `parsing_instruction` 参数注入领域知识：

```python
parsing_instruction = """
You are parsing an academic paper for a RAG system.
Requirements:
1.  **Noise Removal:** Strictly ignore and do not output page headers, footers, and page numbers.
2.  **Math Formatting:** Convert all mathematical formulas and symbols into LaTeX format (e.g., $x^2$).
3.  **Structure:** Preserve the hierarchical structure of sections (Abstract, Introduction, Methodology).
4.  **Tables:** Extract tables into standard Markdown format, ensuring row/column alignment is preserved.
"""
```

### 2\. Deep Multimodal Understanding (深度多模态理解)

**设计意图 (Design Intent):**
学术论文中的信息密度往往集中在图表（Charts）和架构图（Diagrams）中。默认的快速解析模型可能只提取图片中的 OCR 文字，而忽略了图片传达的“趋势”或“逻辑”。我们需要启用更强的视觉模型来生成详细的 Image Caption。

**技术实现 (Implementation):**
启用 `gpt-4o` 模式（或类似的高级多模态模型）处理复杂文档：

```python
# Configuration to enable deep visual understanding
use_vendor_multimodal_model=True
vendor_multimodal_model_name="openai-gpt-4o"
```

### 3\. Structured Metadata Extraction (结构化元数据提取)

**设计意图 (Design Intent):**
虽然 RAG 主要依赖 Markdown 进行语义检索，但在某些场景下（如筛选特定年份、特定作者的论文），我们需要从非结构化 PDF 中精确提取 JSON 格式的元数据。

**未来扩展 (Future Scope):**
参考 `result_type="json"` 模式，未来可增加一个预处理步骤，专门提取 Metadata 存入 SQL/NoSQL 数据库，实现 "Hybrid Search (Vector + Metadata Filter)"。

-----

### 🚀 Recommended Configuration Code (最终推荐配置)

基于以上启发，本系统核心解析器的最佳配置如下：

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    # Output Format
    result_type="markdown",
    
    # Strategy 1: Domain-Specific Instructions
    parsing_instruction="""
    The document is an academic paper.
    1. Strictly preserve mathematical formulas in LaTeX format ($...$).
    2. Do not output page headers, footers, or page numbers.
    3. Provide descriptive summaries for images and diagrams.
    4. Ensure tables are formatted as Markdown tables.
    """,
    
    # Strategy 2: Enhanced Vision Capability (Optional, costs more but better for charts)
    # use_vendor_multimodal_model=True,
    # vendor_multimodal_model_name="openai-gpt-4o",
    
    verbose=True,
    language="en",
    num_workers=4
)
```