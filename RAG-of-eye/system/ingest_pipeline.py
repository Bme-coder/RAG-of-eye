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
import re
from typing import Any, Dict

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from anthropic import Anthropic
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.vector_stores.chroma import ChromaVectorStore

from embed_utils import build_embedding_from_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_PAPERS_DIR = DATA_ROOT / "raw" / "medical_papers"
CHROMA_DIR = DATA_ROOT / "chroma_db"
LLM_AVAILABLE = True
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-sonnet-20241022")

AUTO_TAG_PROMPT = """
Analyze the following medical excerpt and respond with JSON:
{{
  "ethnicity": "Asian" | "Caucasian" | "Other",
  "age_min": int,
  "age_max": int,
  "treatment_tags": [str],
  "study_type": "Cohort" | "Review" | "Trial"
}}
Rules:
- Map Chinese, Taiwanese, Hong Kong, Singaporean, Japanese, Korean to "Asian".
- Map White, European, Western to "Caucasian".
- Map Hispanic, Black, African-American to "Other".
If data is missing, make a conservative estimate but never leave fields null.
Context:
{context}
"""


def _ensure_env_and_settings() -> Anthropic:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing ANTHROPIC_API_KEY for Claude access.")
    api_url = os.getenv("ANTHROPIC_API_URL")

    client_kwargs = {"api_key": api_key}
    if api_url:
        client_kwargs["base_url"] = api_url
    llm_client = Anthropic(**client_kwargs)

    Settings.llm = LlamaAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,
        api_key=api_key,
        base_url=api_url,
        max_tokens=1024,
    )
    Settings.embed_model = build_embedding_from_env()
    return llm_client


def _claude_text(
    llm_client: Anthropic,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 800,
) -> str:
    response = llm_client.messages.create(
        model=CLAUDE_MODEL,
        max_output_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
        elif isinstance(block, dict) and block.get("text"):
            chunks.append(str(block["text"]))
    return "".join(chunks).strip()


def auto_tag_text(llm_client: Anthropic, text: str) -> Dict[str, Any]:
    global LLM_AVAILABLE
    prompt = AUTO_TAG_PROMPT.format(context=text[:3000])
    if LLM_AVAILABLE:
        try:
            payload_text = _claude_text(
                llm_client,
                system_prompt="You are a medical metadata extractor. Output JSON only.",
                user_prompt=prompt,
                max_tokens=600,
            )
            payload = json.loads(payload_text)
            payload.setdefault("treatment_tags", [])
            return payload
        except Exception:
            LLM_AVAILABLE = False
    return _heuristic_tags(text)


def _heuristic_tags(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["chinese", "taiwan", "hong kong", "singapore", "japan", "korea"]):
        ethnicity = "Asian"
    elif any(keyword in lowered for keyword in ["caucasian", "white", "europe", "western", "usa", "american"]):
        ethnicity = "Caucasian"
    else:
        ethnicity = "Other"

    age_min, age_max = 6, 16
    range_match = re.search(r"ages?\s*(\d{1,2})\s*(?:-|to|–|—)\s*(\d{1,2})", lowered)
    if range_match:
        age_min = int(range_match.group(1))
        age_max = int(range_match.group(2))
    else:
        single_match = re.findall(r"(\d{1,2})\s*(?:years|yrs)", lowered)
        if len(single_match) >= 2:
            age_min = int(min(single_match[:2]))
            age_max = int(max(single_match[:2]))

    treatment_keywords = {
        "atropine": "Low-dose Atropine",
        "orthokeratology": "Ortho-K",
        "red light": "Red Light Therapy",
        "defocus": "Defocus Glasses",
    }
    treatment_tags = [label for keyword, label in treatment_keywords.items() if keyword in lowered]

    if "trial" in lowered or "randomized" in lowered:
        study_type = "Trial"
    elif "meta-analysis" in lowered or "systematic review" in lowered:
        study_type = "Review"
    else:
        study_type = "Cohort"

    return {
        "ethnicity": ethnicity,
        "age_min": age_min,
        "age_max": age_max,
        "treatment_tags": treatment_tags,
        "study_type": study_type,
    }


def chunk_and_tag_documents(llm_client: Anthropic, documents: list[Document]) -> list[Document]:
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    processed_docs: list[Document] = []
    for doc in documents:
        base_metadata = dict(doc.metadata)
        base_metadata.setdefault("source_id", base_metadata.get("doc_id") or base_metadata.get("file_name") or doc.doc_id)
        auto_tags = auto_tag_text(llm_client, doc.text[:5000])
        base_metadata.update(auto_tags)
        if isinstance(base_metadata.get("treatment_tags"), list):
            base_metadata["treatment_tags"] = ",".join(base_metadata["treatment_tags"])
        chunks = splitter.split_text(doc.text)
        for idx, chunk in enumerate(chunks):
            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["source_id"] = f"{base_metadata['source_id']}#chunk_{idx}"
            processed_docs.append(Document(text=chunk, metadata=chunk_metadata))
    return processed_docs


def persist_documents(documents: list[Document]) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("myopia_medical_papers")
    vector_store = ChromaVectorStore(chroma_collection=collection)
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

    processed_docs = chunk_and_tag_documents(llm_client, raw_documents)
    persist_documents(processed_docs)
    print(f"Ingested {len(processed_docs)} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    run_ingest()
