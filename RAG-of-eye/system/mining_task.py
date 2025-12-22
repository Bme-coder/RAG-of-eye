# mining_task.py
import json
from pathlib import Path
from typing import Dict, List
from clinical_system import ClinicalAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "medical_config.json"


def build_base_rates_prompt(selected_variables: List[str]) -> str:
    axes_text = ", ".join(selected_variables) if selected_variables else "Age, Ethnicity"
    return f"""
    **Role:** You are a medical data analyst. Your task is to extract "Annual Myopia Progression Rates" (Spherical Equivalent Refraction, SER, in Diopters/year) from the provided medical literature context.

    **Focus Variables (chosen by clinician):** {axes_text}.
    Explicitly reference these variables when reasoning about which datapoints belong to each group.

    **Normalization Rules:**
    - Map Chinese / East Asian / Singaporean / Taiwanese / Hong Kong -> **Asian**.
    - Map White / European / UK / USA (White) / Western -> **Caucasian**.

    **Extraction Requirements:**
    - Cover Ages **6 to 16** for **Male/Female** within Asian and Caucasian cohorts.
    - You MUST extract `sample_size` (n) for every datapoint. If the text omits it, infer a conservative estimate based on the study description but never leave it empty.
    - Values must be negative floats (diopters/year). Use interpolation only when necessary and describe the rule in `notes`.

    **Strict JSON Output:**
    {{
        "data_points": [
            {{
                "label": "Asian|Female|6",
                "ethnicity": "Asian",
                "sex": "Female",
                "age": 6,
                "rate": -0.80,
                "sample_size": 120,
                "variables": {{"Age": 6, "Ethnicity": "Asian", "Parental Myopia": "yes"}},
                "notes": "source sentence or interpolation rule"
            }}
        ]
    }}
    """


def run_mining_job(interactive_mode: bool = True):
    print("⛏️  正在启动知识挖掘任务 (Myopia Task)...")
    
    agent = ClinicalAgent()

    # ===== Phase 1: Survey =====
    selected_variables: List[str] = []
    if interactive_mode:
        survey_results = agent.survey_covariates()
        if survey_results:
            print("\n📊 变量普查结果:")
            for name, freq in survey_results.items():
                print(f"   • {name}: {freq}")
        else:
            print("\n⚠️ 未能获得变量普查结果，使用默认轴 (Age, Ethnicity)")

        user_choice = input("\nWhich variables do you want to use as standard axes? (comma separated)\n> ").strip()
        if user_choice:
            selected_variables = [item.strip() for item in user_choice.split(",") if item.strip()]

    if not selected_variables:
        selected_variables = ["Age", "Ethnicity"]

    # ===== Phase 2: Dynamic BASE_RATES prompt =====
    base_prompt = build_base_rates_prompt(selected_variables)
    print("\n--- 正在挖掘: BASE_RATES ---")
    base_result = agent.research_general_knowledge(
        base_prompt,
        enforce_validation=True,
        value_field="rate",
        hard_floor=-2.0,
    )
    base_records = base_result.get("data_points", []) if base_result else []
    if not base_records:
        print("⚠️ BASE_RATES 未获得有效数据")

    # ===== Phase 3: Treatment efficacy remains scripted =====
    treatments_prompt = """
    Extract the **efficacy rates** (percentage reduction in myopia progression) for the following treatments based on the provided literature.
    
    If the provided text does not contain specific efficacy numbers for a treatment, use general medical consensus knowledge to fill it (but prioritize the text).

    Required JSON Format:
    {
        "Atropine_Low": {"name": "Low-dose Atropine", "efficacy": 0.XX},
        "Ortho_K": {"name": "Ortho-K", "efficacy": 0.XX},
        "Defocus_Glasses": {"name": "Defocus Glasses", "efficacy": 0.XX}
    }
    (Efficacy should be a float between 0 and 1, e.g., 0.50 for 50%)
    """
    print("\n--- 正在挖掘: TREATMENTS ---")
    treatments_result = agent.research_general_knowledge(treatments_prompt)

    mined_data = {}
    if base_records:
        mined_data["base_rates"] = base_records
        print("✅ BASE_RATES 获取成功!")
    if treatments_result:
        mined_data["treatments"] = treatments_result
        print("✅ TREATMENTS 获取成功!")

    if not mined_data:
        print("❌ 未生成有效配置")
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(mined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 挖掘完成！已生成配置文件: {CONFIG_PATH}")


if __name__ == "__main__":
    run_mining_job()
