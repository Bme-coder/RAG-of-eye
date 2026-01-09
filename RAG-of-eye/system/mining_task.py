"""
Saturated mining pipeline that loops through ethnicity/age/treatment matrices,
extracts records via the ClinicalAgent + LLM, and aggregates them using the
analytics helpers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from analytics import calculate_weighted_statistics
from clinical_system import ClinicalAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "medical_config.json"

ETHNICITIES = ["Asian", "Caucasian"]
GENDERS = ["Female", "Male"]
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
{{
  "records": [
    {{
      "mean": -0.50,
      "n": 120,
      "source_id": "Paper_ID_or_DOI",
      "treatment_detail": "0.01% atropine nightly",
      "confidence": "High",
      "full_passage": "The LAMP study enrolled 438 children... [COPY THE FULL PARAGRAPH HERE] ...mean progression was -0.50D.", 
      "notes": "Any caveats"
    }}
  ]
}}
Rules:
- Values represent diopters/year (negative for progression).
- CRITICAL: For `full_passage`, you MUST copy the **ENTIRE PARAGRAPH** or logical text block that contains the data point. Do not truncate it. I need the full context.
- Never omit `n`.
- Focus exclusively on {ethnicity} patients {gender_focus} around age {age} undergoing "{treatment}".
Context:
{context}
"""


def build_query_text(ethnicity: str, gender: Optional[str], age: int, treatment_query: str) -> str:
    subject_parts = [ethnicity]
    if gender:
        subject_parts.append(f"{gender.lower()} children")
    else:
        subject_parts.append("children")
    subject = " ".join(subject_parts)
    return f"{subject} age {age} myopia progression rate {treatment_query}"


def extract_records(
    agent: ClinicalAgent,
    ethnicity: str,
    gender: Optional[str],
    age: int,
    treatment_cfg: Dict[str, str],
) -> List[Dict]:
    engine = agent.build_query_engine(similarity_top_k=50, ethnicity=ethnicity, age=age, gender=gender)
    query_text = build_query_text(ethnicity, gender, age, treatment_cfg["query"])
    try:
        rag_response = engine.query(query_text)
    except Exception as exc:
        print(f"[WARN] 检索失败: {ethnicity} {gender or 'All'} age {age} {treatment_cfg['name']} -> {exc}")
        return []

    gender_focus = {
        "Female": "who are female",
        "Male": "who are male",
    }.get(gender, "of any gender")

    prompt = EXTRACTION_PROMPT.format(
        ethnicity=ethnicity,
        age=age,
        gender_focus=gender_focus,
        treatment=treatment_cfg["name"],
        context=str(rag_response)[:5000],
    )
    try:
        response_text = agent.invoke_llm(
            system_prompt="You are a structured data extractor. Output JSON only.",
            user_prompt=prompt,
            max_tokens=1200,
        )
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            # pattern: ```json\n{...}\n```
            if len(parts) >= 3:
                cleaned = parts[1]
            else:
                cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        if not cleaned:
            raise ValueError("LLM 返回为空，无法解析 JSON。")
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[DEBUG] 原始 LLM 输出 ({ethnicity}, age {age}, {treatment_cfg['name']}): {response_text!r}")
            raise
        return payload.get("records", [])
    except Exception as exc:
        print(f"[WARN] LLM 抽取失败: {ethnicity} {gender or 'All'} age {age} {treatment_cfg['name']} -> {exc}")
        return []


def run_mining_job():
    print("[INFO] 正在启动饱和式挖掘 (Myopia Task)...")
    agent = ClinicalAgent()
    aggregated_rows: List[Dict] = []

    for ethnicity in ETHNICITIES:
        for gender in GENDERS:
            for age in AGES:
                for treatment in TREATMENTS:
                    print(f"→ {ethnicity} | {gender} | Age {age} | {treatment['name']}")
                    raw_records = extract_records(agent, ethnicity, gender, age, treatment)
                    if not raw_records:
                        fallback_records = extract_records(agent, ethnicity, None, age, treatment)
                        if fallback_records:
                            print(f"[INFO] 性别 {gender} 缺少样本，使用性别无关数据兜底。")
                        raw_records = fallback_records
                    stats = calculate_weighted_statistics(raw_records, is_treatment=treatment["is_treatment"])
                    if not stats:
                        continue
                    aggregated_rows.append(
                        {
                            "ethnicity": ethnicity,
                            "gender": gender,
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
        "dimensions": ["Ethnicity", "Gender", "Age", "Treatment"],
        "ethnicities": ETHNICITIES,
        "genders": GENDERS,
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
