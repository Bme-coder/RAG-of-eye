# 🏅 Intelligent Medical CDSS & Myopia Prediction Engine

> 基于 RAG 2.0 的临床决策支持与近视预测系统  
> 既能实时辅助医生，也能离线批量产出覆盖所有年龄/民族/度数组合的预测数据库。

---

## 🧭 系统概览

```mermaid
graph TD
    PDF[Medical Papers PDF] -->|1. 解析/切块/入库| Ingest[ingest_pipeline.py]
    Ingest -->|2. 提供向量库 (Chroma)| Core[rag_core.py]
    Core -->|3. 检索接口| Agent[clinical_system.py]
    Agent -->|4. 实时推理| Doctor[病例评估]
    Agent -->|5. 批量抽取| Miner[mining_task.py]
    Miner -->|6. 生成| Config[medical_config.json]
    Config -->|7. 全量遍历| Builder[myopia_builder.py]
    Builder -->|8. 输出| DB[myopia_db.json]
    DB -->|9. 零延迟调用| Frontend[Web / Desktop]
```

- **GPT-4o mini**：所有 LLM 推理统一走 OpenAI 兼容 API（`.env` 中配置 `OPENAI_API_KEY`/`OPENAI_API_BASE`）。
- **Embedding 双模切换**：`embed_utils.py` 封装了 `EMBED_PROVIDER`（`local`=HuggingFace、`api`=OpenAI Embedding）。只改 `.env` 即可切换。
- **ingest_pipeline**：PDF→Claude 元数据→SentenceSplitter→Chroma，保证后续检索能按民族/年龄/治疗过滤。

---

## 📌 解决的核心需求

| 痛点 | 解决方案 | 模块 |
| --- | --- | --- |
| 读不懂 PDF 表格 | LlamaParse + Markdown 表格还原 | `rag_core.py` |
| 海量病情组合查询 | 动态 Prompt + Chroma 检索 | `clinical_system.py` |
| 结构化抽取 | Claude JSON Mode + Pydantic 校验 | `clinical_system.py` |
| 方案评分与可视化 | Python 加权评分 + Claude 医学点评 + Scipy 曲线拟合 | `clinical_system.py` |
| 治疗曲线构建 | LLM 抽点 + 数学模型 | `clinical_system.py` |
| 全量预测数据库 | RAG 抽 KPI + Python 四层遍历 | `mining_task.py` + `myopia_builder.py` |

---

## 📂 文件详解

### 1. `ingest_pipeline.py`（文献入库）
- **职责**：将 `data/raw/medical_papers` 内的 PDF 批量解析，Claude 自动生成民族/性别/年龄段/治疗标签，SentenceSplitter 切块后写入 Chroma。
- **关键函数**
  - `_ensure_env_and_settings()`：加载 `.env` 的 Claude/Embedding 配置，保持与主系统一致。
  - `_claude_text()`：统一处理 `messages.create` 的输出，方便 JSON/纯文本切换。
  - `chunk_and_tag_documents()`：Chunk → metadata merge → `ChromaVectorStore`。
- **下游依赖**：`rag_core.py` 和 `clinical_system.py` 的向量检索全部基于该流程产出的集合，metadata filter 也在此生成。

### 2. `rag_core.py`（能力内核）
- 调 `ingest_pipeline` 产出的 ChromaDB，加载/持久化索引，提供 `get_rag_engine()` 供上层使用。
- 内置 `LLMRerank(top_n=3)`，保证 Claude 能看到最相关段落。

### 3. `clinical_system.py`（业务大脑）
- `ClinicalAgent` 完整封装：检索 → Claude JSON 抽取 → 结构化评分 → Scipy 曲线拟合。
- `invoke_llm()` 供其它脚本（如 `mining_task.py`）直接调用 Claude。

### 4. `mining_task.py`（知识矿工）
- 将民族/性别/年龄/治疗组合写成 Prompt，循环调用 `ClinicalAgent.research_general_knowledge()` 生成 `medical_config.json`。

### 5. `myopia_builder.py`（数据工厂）
- 读取 `medical_config.json`，遍历民族 × 性别 × 年龄 × 度数，输出 `myopia_db.json`，供前端零延迟查询。

### 6. `index.html` / `new_index.py`
- Web / Desktop 端示例，直接读取 `myopia_db.json` 并渲染红绿对比曲线。

---

## ⚙️ 使用指南

### 第 1 步：环境配置
1. 创建 `.env`：
   ```ini
   ANTHROPIC_API_KEY=sk-...
   ANTHROPIC_API_URL=https://api.anthropic.com/v1
   LLAMA_CLOUD_API_KEY=llx-...

   # Embedding：local（默认，本地 HuggingFace）| api（OpenAI Embedding）
   EMBED_PROVIDER=local
   EMBED_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
   OPENAI_API_KEY=sk-...              # 仅在 EMBED_PROVIDER=api 时需要
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_EMBED_MODEL=text-embedding-3-large
   ```
2. 运行 `python system/test/test_connection.py` 快速验证 Claude/API/代理配置。
3. 将 PDF 放入 `data/raw/medical_papers/`。

### 第 2 步：执行文献入库（Ingest Phase）
```bash
python system/ingest_pipeline.py
```
- **结果**：`data/chroma_db/` 下生成/更新 `myopia_medical_papers` 集合。
- **提示**：首次运行会下载 Claude/HuggingFace 模型并构建向量库，可能需要几分钟；之后增量写入很快。

### 第 3 步：运行知识挖掘（Mining Phase）
```bash
python system/mining_task.py
```
- **输出**：`artifacts/medical_config.json`。可自行审校 Claude 抽取的 base rate/efficacy 是否合理。

### 第 4 步：运行全量构建（Building Phase）
```bash
python system/myopia_builder.py
```
- **输出**：`artifacts/myopia_db.json`，包含近 2 万组合的曲线参数。

### 一键运行（Pipeline Codex）
```bash
python system/pipeline_codex.py
```
- 默认顺序等价于 `ingest → mining → builder`，可通过 `--steps mining builder` 之类的参数执行子集，`--ignore-errors` 可在某步失败时继续。

### 第 5 步（可选）：前端集成
- Web：`python -m http.server` 后访问 `system/index.html`。
- Desktop：`python system/new_index.py`（PyQt6 GUI）。

### 第 6 步：下次怎么用？（最佳实践）
> 终端默认保持“干净”状态，不主动走代理。按照场景输入暗号即可切换到对应模式。

| 场景 | 指令 | 说明 |
| --- | --- | --- |
| 跑 RAG 代码，需要连 OpenAI | `proxy_on` | 一键开启工作模式，然后直接运行 Python。若提示连接被拒绝，回到 Windows 端 `ipconfig` 查看最新局域网 IP，并同步更新 `.bashrc` 中的代理地址。 |
| 下载/同步大模型，国内镜像更快 | `use_mirror` | 切换到镜像模式后再执行下载脚本。下载完成若要继续跑代码，再次输入 `proxy_on` 切回代理。 |
| 本地改代码，不需联网 | 无操作 | 默认即为最干净的环境，保持原样即可。 |

- `proxy_on`：干活模式，联通 OpenAI/Claude 所需的所有代理环境变量。
- `use_mirror`：下载模式，路由到国内镜像源以加速模型或依赖获取。
- `proxy_off`：回到原点，清理代理相关变量，确保终端不乱连。

---

## ⚠️ 注意事项
1. 首次运行 `ingest_pipeline` / `mining_task` / `clinical_system` 时会触发 Chroma 构建与 Claude 请求，耗时较长但只需一次。
2. 若发现 `myopia_db.json` 中某些组合不合理，可直接编辑 `medical_config.json`，再重新执行 builder 即可。
3. 切换 Embedding 供应商时，只需修改 `.env` 并重新运行 ingest，以确保向量库与实际模型一致。
