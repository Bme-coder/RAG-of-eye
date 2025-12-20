# mining_task.py
import json
import os
from clinical_system import ClinicalAgent

def run_mining_job():
    print("⛏️  正在启动知识挖掘任务 (Myopia Task)...")
    
    # 1. 初始化 Agent
    agent = ClinicalAgent()
    
    tasks = {
        # --- 通用化挖掘任务 (BASE_RATES) ---
        "BASE_RATES": """
        **Role:** You are a medical data analyst. Your task is to extract "Annual Myopia Progression Rates" (Spherical Equivalent Refraction, SER, in Diopters/year) from the provided medical literature context.

        **Target Schema:**
        We need a complete dataset for **Ages 6 to 16**, for **Male/Female**, across two specific ethnicity groups: **Asian** and **Caucasian**.

        **1. Ethnicity Mapping Rules (Normalization):**
        - If the text mentions "Chinese", "East Asian", "Singaporean", "Taiwanese", or "Hong Kong", map data to -> **"Asian"**.
        - If the text mentions "White", "European", "UK", "USA (White)", or "Western", map data to -> **"Caucasian"**.

        **2. Data Extraction Logic (Priority Order):**
        - **Priority A (Specifics):** If the text provides a table/chart with age-specific rates, use those exact numbers.
        - **Priority B (Formulas):** If the text provides a regression model (e.g., "progression slows by 0.05D per year of age"), use it to calculate rates for ages 6-16.
        - **Priority C (Averages):** If the text only gives a *mean annual progression* (e.g., "-0.80 D/year for Asians"), use this mean value for ALL ages, or apply a logical decay trend if the text suggests one (younger children progress faster).

        **3. Required Output Format (Strict JSON):**
        - Values must be **negative floats** (e.g., -0.65).
        - Use logical interpolation if specific years are missing. Do not leave values as null.

        {
            "Asian": {
                "Female": {"6": -X.X, "7": -X.X, ..., "16": -X.X},
                "Male":   {"6": -X.X, ..., "16": -X.X}
            },
            "Caucasian": {
                "Female": {"6": -X.X, ..., "16": -X.X},
                "Male":   {"6": -X.X, ..., "16": -X.X}
            }
        }
        """,
        
        # --- 通用化挖掘任务 (TREATMENTS) ---
        "TREATMENTS": """
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
    }
    
    mined_data = {}
    
    for key, prompt in tasks.items():
        print(f"\n--- 正在挖掘: {key} ---")
        result = agent.research_general_knowledge(prompt)
        if result:
            print(f"✅ {key} 获取成功!")
            mined_data[key] = result
        else:
            print(f"⚠️ {key} 获取失败")

    with open("medical_config.json", "w", encoding="utf-8") as f:
        json.dump(mined_data, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 挖掘完成！已生成配置文件: medical_config.json")

if __name__ == "__main__":
    run_mining_job()