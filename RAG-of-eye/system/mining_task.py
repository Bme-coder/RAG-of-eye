"""
Saturated mining pipeline that loops through ethnicity/age/treatment matrices,
extracts records via the ClinicalAgent + LLM, and aggregates them using the
analytics helpers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from analytics import calculate_weighted_statistics
from clinical_system import ClinicalAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "medical_config.json"

ETHNICITIES = ["Asian", "Caucasian"]
AGES = list(range(6, 17))

TREATMENTS = [
    {"name": "Control", "key": "Natural", "query": "(natural progression OR 控制组 OR 无干预 OR 对照组)", "is_treatment": False},
    {"name": "Low-dose Atropine", "key": "Atropine_Low", "query": "(Low-dose Atropine OR 低浓度阿托品)", "is_treatment": True},
    {"name": "Ortho-K", "key": "Ortho_K", "query": "(Orthokeratology OR 角膜塑形镜 OR OK镜)", "is_treatment": True},
    {"name": "Defocus Glasses", "key": "Defocus_Glasses", "query": "(Defocus spectacle OR 离焦眼镜)", "is_treatment": True},
    {"name": "Red Light Therapy", "key": "Red_Light_Therapy", "query": "(red light therapy OR 低能量红光)", "is_treatment": True},
]

EXTRACTION_PROMPT = """
You are a clinical research meta-analyst. Extract all available numerical datapoints
for the cohort described below. Respond with JSON matching:
{
  "records": [
    {
      "source_id": "Paper_ID_or_DOI",
      "mean": -0.85,
      "sd": 0.22,
      "n": 120,
      "treatment_detail": "0.01% atropine nightly",
      "notes": "Any caveats or CI conversion logic"
    }
  ]
}
Rules:
- Values represent diopters/year (negative for progression).
- Include SD whenever possible. If only CI is provided, convert to SD (SD ≈ (upper-lower)/3.92).
- Never omit `n`, and set conservative estimates if exact sample size is missing.
- Focus exclusively on {ethnicity} patients around age {age} undergoing "{treatment}".
- Include multiple records per study if different cohorts exist.

Context:
{context}
"""


def build_query_text(ethnicity: str, age: int, treatment_query: str) -> str:
    return f"{ethnicity} children age {age} myopia progression rate {treatment_query}"


def extract_records(agent: ClinicalAgent, ethnicity: str, age: int, treatment_cfg: Dict[str, str]) -> List[Dict]:
    engine = agent.build_query_engine(similarity_top_k=50, ethnicity=ethnicity, age=age)
    query_text = build_query_text(ethnicity, age, treatment_cfg["query"])
    try:
        rag_response = engine.query(query_text)
    except Exception as exc:
        print(f"[WARN] 检索失败: {ethnicity} age {age} {treatment_cfg['name']} -> {exc}")
        return []

    prompt = EXTRACTION_PROMPT.format(
        ethnicity=ethnicity,
        age=age,
        treatment=treatment_cfg["name"],
        context=str(rag_response)[:5000],
    )
    try:
        response_text = agent.invoke_llm(
            system_prompt="You are a structured data extractor. Output JSON only.",
            user_prompt=prompt,
            max_tokens=1200,
        )
        payload = json.loads(response_text)
        return payload.get("records", [])
    except Exception as exc:
        print(f"[WARN] LLM 抽取失败: {ethnicity} age {age} {treatment_cfg['name']} -> {exc}")
        return []


def run_mining_job():
    print("[INFO] 正在启动饱和式挖掘 (Myopia Task)...")
    agent = ClinicalAgent()
    aggregated_rows: List[Dict] = []

    for ethnicity in ETHNICITIES:
        for age in AGES:
            for treatment in TREATMENTS:
                print(f"→ {ethnicity} | Age {age} | {treatment['name']}")
                raw_records = extract_records(agent, ethnicity, age, treatment)
                stats = calculate_weighted_statistics(raw_records, is_treatment=treatment["is_treatment"])
                if not stats:
                    continue
                aggregated_rows.append(
                    {
                        "ethnicity": ethnicity,
                        "age": age,
                        "treatment": treatment["name"],
                        "treatment_key": treatment["key"],
                        **stats,
                    }
                )

    if not aggregated_rows:
        print("[ERROR] 未生成任何有效数据，请检查文献或提示词。")
        return

    meta_schema = {
        "dimensions": ["Ethnicity", "Age", "Treatment"],
        "ethnicities": ETHNICITIES,
        "ages": AGES,
        "treatments": [{"name": t["name"], "key": t["key"]} for t in TREATMENTS],
    }

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "meta_schema": meta_schema,
        "base_rates": aggregated_rows,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 饱和式挖掘完成，共生成 {len(aggregated_rows)} 条聚合记录 -> {CONFIG_PATH}")


if __name__ == "__main__":
    run_mining_job()
