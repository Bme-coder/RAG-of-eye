import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from algo_core import fit_gpr_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "medical_config.json"
DB_PATH = ARTIFACTS_DIR / "myopia_db.json"


def load_config(path: Path = CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        print(f"⚠️ 错误: 找不到{path}")
        return pd.DataFrame(), {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        base_records = data
        treatments = {}
    else:
        base_records = data.get("base_rates", [])
        # 优先读取新的小写键，但兼容旧版本的大写
        treatments = data.get("treatments") or data.get("TREATMENTS", {})

    df = pd.DataFrame(base_records)
    return df, treatments


def calculate_progression(rate_lookup: Dict[int, float], start_age: int, start_diopter: float, efficacy: float):
    target_age = 18
    timeline = list(range(start_age, target_age + 1))
    natural = [round(start_diopter, 2)]
    managed = [round(start_diopter, 2)]
    curr_nat = start_diopter
    curr_man = start_diopter
    fallback_rate = rate_lookup[max(rate_lookup.keys())] if rate_lookup else -0.5

    for age in timeline[:-1]:
        rate = rate_lookup.get(age, fallback_rate)
        curr_nat += rate
        curr_man += rate * (1 - efficacy)
        natural.append(round(curr_nat, 2))
        managed.append(round(curr_man, 2))

    return {
        "timeline": timeline,
        "natural": natural,
        "managed": managed,
    }


def sanitize_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return str(value).replace(" ", "")


def build_dictionary():
    print("🏭 正在启动全量遍历构建...")

    df, treatments = load_config()
    if df.empty:
        print("⚠️ base_rates 数据为空")
        return

    group_cols = [col for col in df.columns if col not in ["rate", "n", "age", "weight_alpha"]]
    if not group_cols:
        df["_global"] = "ALL"
        group_cols = ["_global"]

    start_ages = range(6, 17)
    start_diopters = np.arange(-0.50, -6.25, -0.50)
    db: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    count = 0

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        fit_result = fit_gpr_curve(group_df)
        if not fit_result["timeline"]:
            continue
        rate_lookup = {
            int(age): float(rate) for age, rate in zip(fit_result["timeline"], fit_result["mean_curve"])
        }
        label_parts = [sanitize_value(val) for val in group_key]
        base_label = "_".join(part for part in label_parts if part and part != "NA") or "GLOBAL"

        for age in start_ages:
            for dio in start_diopters:
                dio = round(float(dio), 2)
                user_key = f"{base_label}_{age}_{dio}"
                entry = db.setdefault(user_key, {})

                if "Natural" not in entry:
                    entry["Natural"] = calculate_progression(rate_lookup, age, dio, efficacy=0.0)

                for t_key, t_val in treatments.items():
                    efficacy = t_val.get("efficacy", 0.0)
                    entry[t_key] = calculate_progression(rate_lookup, age, dio, efficacy)

                count += 1
                if count % 1000 == 0:
                    print(f"   已生成{count} 条...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(db, f)

    print(f"✅ 构建完成！最终生成{count} 条数据 -> {DB_PATH}")


if __name__ == "__main__":
    build_dictionary()
