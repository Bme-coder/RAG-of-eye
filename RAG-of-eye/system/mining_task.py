"""
Saturated mining pipeline that loops through ethnicity/age/treatment matrices,
extracts records via the ClinicalAgent + LLM, and aggregates them using the
analytics helpers.
"""

from __future__ import annotations

import json
from collections import Counter
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
    {
        "name": "Control",
        "key": "Natural",
        "query": "(natural progression OR 控制组 OR 无干预 OR 对照组)",
        "is_treatment": False,
        "required_keywords": ["control", "placebo", "natural", "对照", "untreated"],
        "forbidden_keywords": [],
    },
    {
        "name": "Low-dose Atropine",
        "key": "Atropine_Low",
        "query": "(Low-dose Atropine OR 低浓度阿托品)",
        "is_treatment": True,
        "required_keywords": ["atropine", "阿托品"],
        "forbidden_keywords": ["ortho", "orthoker", "ok镜", "defocus", "red light", "红光"],
    },
    {
        "name": "Ortho-K",
        "key": "Ortho_K",
        "query": "(Orthokeratology OR 角膜塑形镜 OR OK镜)",
        "is_treatment": True,
        "required_keywords": ["ortho", "orthoker", "ok lens", "ok镜", "角膜塑形"],
        "forbidden_keywords": ["atropine", "阿托品"],
    },
    {
        "name": "Defocus Glasses",
        "key": "Defocus_Glasses",
        "query": "(Defocus spectacle OR 离焦眼镜)",
        "is_treatment": True,
        "required_keywords": ["defocus", "dims", "hal", "离焦"],
        "forbidden_keywords": ["atropine", "阿托品"],
    },
    {
        "name": "Red Light Therapy",
        "key": "Red_Light_Therapy",
        "query": "(red light therapy OR 低能量红光)",
        "is_treatment": True,
        "required_keywords": ["red light", "红光"],
        "forbidden_keywords": ["atropine", "阿托品"],
    },
]

EXTRACTION_PROMPT = """
You are a strict medical data auditor. Extract structured measurements ONLY if the
text explicitly discusses the requested treatment and cohort.

TARGET CONTEXT:
- Ethnicity: {ethnicity}
- Gender focus: {gender_focus}
- Age: {age}
- Treatment: {treatment}
- Treatment keywords: {keyword_hint}

SOURCE CONTEXT (verbatim snippet):
{context}

RULES:
1. If the passage does not clearly mention the target treatment (or synonyms),
   return an empty array [].
2. Never hallucinate numbers. When mean/n/sd is not explicitly stated, output
   null for that field.
3. Copy the entire paragraph containing the numbers into `full_passage` and do
   not paraphrase.
4. Provide a concrete `source_id` (DOI/PMID/title). Placeholders such as
   "Paper_ID_or_DOI" are forbidden.
5. Output pure JSON: a list of objects with keys
   [mean, n, sd, source_id, treatment_detail, confidence, full_passage, notes].
6. If no compliant evidence exists, respond with [] exactly.
"""

PLACEHOLDER_SOURCE_IDS = {"", "N/A", "n/a", "Paper_ID_or_DOI", "UNKNOWN", "Unknown"}
SUSPICIOUS_MEAN_VALUES = {-0.5, -0.50, -0.500}
SUSPICIOUS_SAMPLE_SIZES = {120}


def build_query_text(ethnicity: str, gender: Optional[str], age: int, treatment_query: str) -> str:
    subject_parts = [ethnicity]
    if gender:
        subject_parts.append(f"{gender.lower()} children")
    else:
        subject_parts.append("children")
    subject = " ".join(subject_parts)
    return f"{subject} age {age} myopia progression rate {treatment_query}"


def _context_has_keywords(context: str, *, required_keywords: List[str]) -> bool:
    if not required_keywords:
        return True
    lowered = context.lower()
    return any(keyword.lower() in lowered for keyword in required_keywords)


def _context_has_forbidden(context: str, *, forbidden_keywords: List[str]) -> bool:
    if not forbidden_keywords:
        return False
    lowered = context.lower()
    return any(keyword.lower() in lowered for keyword in forbidden_keywords)


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

    rag_text = str(rag_response)
    if not _context_has_keywords(rag_text, required_keywords=treatment_cfg.get("required_keywords", [])):
        print(f"[INFO] 检索上下文未包含目标治疗关键词，跳过: {ethnicity} age {age} {treatment_cfg['name']}")
        return []
    if _context_has_forbidden(rag_text, forbidden_keywords=treatment_cfg.get("forbidden_keywords", [])):
        print(f"[INFO] 检索上下文命中禁止关键词，跳过: {ethnicity} age {age} {treatment_cfg['name']}")
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
        context=rag_text[:5000],
        keyword_hint=", ".join(treatment_cfg.get("required_keywords", [])) or "N/A",
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
        data = payload if isinstance(payload, list) else payload.get("records", [])
        if not isinstance(data, list):
            return []
        return data
    except Exception as exc:
        print(f"[WARN] LLM 抽取失败: {ethnicity} {gender or 'All'} age {age} {treatment_cfg['name']} -> {exc}")
        return []



def validate_extracted_record(record: Dict, treatment_cfg: Dict[str, str]) -> tuple[bool, str]:
    source_id = (record.get("source_id") or "").strip()
    if not source_id or source_id in PLACEHOLDER_SOURCE_IDS:
        return False, "缺少有效来源 ID"

    mean = record.get("mean")
    n_value = record.get("n") or record.get("sample_size")
    try:
        mean_val = float(mean) if mean is not None else None
    except (TypeError, ValueError):
        mean_val = None
    try:
        n_val = int(float(n_value)) if n_value is not None else None
    except (TypeError, ValueError):
        n_val = None

    if mean_val is not None and n_val is not None:
        if mean_val in SUSPICIOUS_MEAN_VALUES and n_val in SUSPICIOUS_SAMPLE_SIZES:
            return False, "疑似默认均值/样本"

    evidence_blob = " ".join(
        [
            str(record.get("treatment_detail", "")),
            str(record.get("full_passage", "")),
            str(record.get("notes", "")),
        ]
    ).lower()

    required_keywords = [kw.lower() for kw in treatment_cfg.get("required_keywords", [])]
    if required_keywords and not any(keyword in evidence_blob for keyword in required_keywords):
        return False, "证据未提及目标治疗"

    forbidden_keywords = [kw.lower() for kw in treatment_cfg.get("forbidden_keywords", [])]
    if forbidden_keywords and any(keyword in evidence_blob for keyword in forbidden_keywords):
        return False, "证据包含冲突治疗"

    return True, "OK"


def audit_base_rates(base_rates: List[Dict]) -> None:
    if not base_rates:
        return

    numeric_means = [round(float(item["mean"]), 3) for item in base_rates if isinstance(item.get("mean"), (int, float))]
    if numeric_means:
        counter = Counter(numeric_means)
        value, count = counter.most_common(1)[0]
        if len(numeric_means) >= 10 and count / len(numeric_means) > 0.5:
            raise RuntimeError(f"base_rates 中 {value} 重复比例 {count}/{len(numeric_means)} 过高，疑似幻觉")

    sample_sizes = [int(item["total_n"]) for item in base_rates if isinstance(item.get("total_n"), (int, float))]
    if sample_sizes:
        counter = Counter(sample_sizes)
        value, count = counter.most_common(1)[0]
        if len(sample_sizes) >= 10 and count / len(sample_sizes) > 0.5:
            raise RuntimeError(f"样本量 {value} 重复比例 {count}/{len(sample_sizes)} 过高，疑似默认值")


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
                    if not raw_records:
                        continue

                    valid_records = []
                    for record in raw_records:
                        ok, reason = validate_extracted_record(record, treatment)
                        if ok:
                            valid_records.append(record)
                        else:
                            print(
                                f"[INFO] 丢弃记录: {ethnicity} age {age} {treatment['name']} -> {reason}"
                            )

                    if not valid_records:
                        continue

                    stats = calculate_weighted_statistics(
                        valid_records, is_treatment=treatment["is_treatment"]
                    )
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

    audit_base_rates(aggregated_rows)

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
