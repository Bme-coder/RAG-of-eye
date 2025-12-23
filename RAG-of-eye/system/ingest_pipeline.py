"""
ChromaDB ingestion pipeline with auto-tagging metadata.

Usage:
    python ingest_pipeline.py

The script will read PDFs under data/raw/medical_papers, chunk them with
LlamaIndex, enrich each chunk with LLM-generated metadata (ethnicity, age
range, etc.), and persist them into a Chroma vector store for hybrid search.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline, IngestionComponent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_PAPERS_DIR = DATA_ROOT / "raw" / "medical_papers"
CHROMA_DIR = DATA_ROOT / "chroma_db"

AUTO_TAG_PROMPT = """
Analyze the following medical excerpt and respond with JSON:
{
  "ethnicity": "Asian" | "Caucasian" | "Other",
  "age_min": int,
  "age_max": int,
  "treatment_tags": [str],
  "study_type": "Cohort" | "Review" | "Trial"
}
Rules:
- Map Chinese, Taiwanese, Hong Kong, Singaporean, Japanese, Korean to "Asian".
- Map White, European, Western to "Caucasian".
- Map Hispanic, Black, African-American to "Other".
If data is missing, make a conservative estimate but never leave fields null.
Context:
{context}
"""


def _ensure_env_and_settings() -> OpenAI:
    load_dotenv()
    api_base = os.getenv("OPENAI_API_BASE")
    llm_client = OpenAI(base_url=api_base)

    Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0, api_base=api_base)
    Settings.embed_model = None
    return llm_client


def auto_tag_text(llm_client: OpenAI, text: str) -> Dict[str, Any]:
    prompt = AUTO_TAG_PROMPT.format(context=text[:3000])
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical metadata extractor. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        payload = json.loads(response.choices[0].message.content)
        payload.setdefault("treatment_tags", [])
        return payload
    except Exception:
        return {"ethnicity": "Other", "age_min": 0, "age_max": 0, "treatment_tags": [], "study_type": "Review"}


class AutoTagComponent(IngestionComponent):
    def __init__(self, llm_client: OpenAI):
        super().__init__()
        self.llm_client = llm_client

    def process(self, documents: list[Document]) -> list[Document]:
        for doc in documents:
            metadata = auto_tag_text(self.llm_client, doc.text)
            doc.metadata.update(metadata)
        return documents


def build_ingestion_pipeline(llm_client: OpenAI) -> IngestionPipeline:
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    components = [
        splitter,
        AutoTagComponent(llm_client),
    ]
    return IngestionPipeline(components=components)


def persist_documents(documents: list[Document]) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    vector_store = ChromaVectorStore(
        collection_name="myopia_medical_papers",
        chroma_client=client,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )


def run_ingest(papers_dir: Path = RAW_PAPERS_DIR) -> None:
    if not papers_dir.exists():
        raise FileNotFoundError(f"Medical papers directory not found: {papers_dir}")

    llm_client = _ensure_env_and_settings()
    reader = SimpleDirectoryReader(input_dir=str(papers_dir))
    raw_documents = reader.load_data()

    pipeline = build_ingestion_pipeline(llm_client)
    processed_docs = pipeline.run(raw_documents)
    persist_documents(processed_docs)
    print(f"Ingested {len(processed_docs)} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    run_ingest()
