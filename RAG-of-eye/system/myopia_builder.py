import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from algo_core import fit_gpr_curve

# ================= 配置与常量 =================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "medical_config.json"
DB_PATH = ARTIFACTS_DIR / "myopia_db.json"

# 治疗方案归一化映射
TREATMENT_KEY_ALIASES = {
    "Control": "Natural",
    "Natural": "Natural",
    "Low-dose Atropine": "Atropine_Low",
    "Atropine": "Atropine_Low",
    "Ortho-K": "Ortho_K",
    "OK Lens": "Ortho_K",
    "Defocus Glasses": "Defocus_Glasses",
    "Red Light Therapy": "Red_Light_Therapy",
}

# 预定义的治疗有效率 (Efficacy)
# 如果 medical_config.json 里没有具体的 efficacy 数据，则使用这些默认值进行推演
TREATMENT_EFFICACIES = {
    "Natural": 0.0,
    "Atropine_Low": 0.37,      # ~37% 延缓效果
    "Ortho_K": 0.50,           # ~50%
    "Defocus_Glasses": 0.25,   # ~25%
    "Red_Light_Therapy": 0.60, # ~60%
}

# ================= 工具函数 =================

def load_config(path: Path = CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        print(f"⚠️ 错误: 找不到 {path}")
        return pd.DataFrame(), {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容两种格式：直接是 list 或是 {"base_rates": [...]}
    if isinstance(data, list):
        base_records = data
        meta = {}
    else:
        base_records = data.get("base_rates", [])
        meta = data.get("meta_schema") or {}

    df = pd.DataFrame(base_records)
    
    if df.empty:
        return df, meta

    # 字段标准化
    if "mean" in df.columns and "rate" not in df.columns:
        df["rate"] = df["mean"]
    if "total_n" in df.columns and "n" not in df.columns:
        df["n"] = df["total_n"]
        
    # 计算权重 alpha (样本量越大，alpha 越小，权重越高)
    if "weight_alpha" not in df.columns and "n" in df.columns:
        df["weight_alpha"] = df["n"].apply(
            lambda val: float(1.0 / max(1.0, np.sqrt(val))) if pd.notna(val) and val else 0.5
        )
        
    # 处理 Treatment Key
    if "treatment" not in df.columns:
        df["treatment"] = "Control"
    if "treatment_key" not in df.columns:
        df["treatment_key"] = df["treatment"].apply(
            lambda val: TREATMENT_KEY_ALIASES.get(val, sanitize_value(val))
        )
    else:
        # 确保 treatment_key 也是标准化的
        df["treatment_key"] = df["treatment_key"].apply(
            lambda val: TREATMENT_KEY_ALIASES.get(val, val)
        )

    # 清理非标量列 (防止 groupby 报错)
    drop_cols = []
    for col in df.columns:
        if col in {"rate", "n", "age", "weight_alpha", "mean", "sd", "total_n"}:
            continue
        # 检查是否包含列表或字典
        if df[col].apply(lambda val: isinstance(val, (dict, list, set))).any():
            drop_cols.append(col)
            
    if drop_cols:
        df = df.drop(columns=drop_cols)
        
    # 清理纯文本噪点列
    text_noise_cols = [col for col in ("label", "notes", "source_id", "source_ids") if col in df.columns]
    if text_noise_cols:
        df = df.drop(columns=text_noise_cols)
        
    return df, meta

def sanitize_value(value) -> str:
    """将数值或字符串转换为 URL 安全的字符串"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return (
        str(value)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
        .replace("/", "_")
    )

def calculate_progression(
    rate_lookup: Dict[int, float],
    upper_lookup: Dict[int, float],
    lower_lookup: Dict[int, float],
    start_age: int,
    start_diopter: float,
    efficacy: float,
):
    """
    核心积分引擎：根据基准增长率和疗效，计算未来的屈光度曲线。
    """
    target_age = 18 # 预测到 18 岁
    timeline = list(range(start_age, target_age + 1))
    scale = 1.0 - efficacy # 延缓系数 (e.g. 效力0.3 -> scale 0.7)
    
    # 兜底逻辑：如果没有数据，假设每年 -0.5D
    if not rate_lookup:
        rate_lookup = {start_age: -0.5}
    if not upper_lookup:
        upper_lookup = rate_lookup
    if not lower_lookup:
        lower_lookup = rate_lookup

    def _fallback(lookup: Dict[int, float], default: float) -> float:
        return lookup.get(max(lookup.keys())) if lookup else default

    fallback_rate = _fallback(rate_lookup, -0.5)
    fallback_upper = _fallback(upper_lookup, fallback_rate)
    fallback_lower = _fallback(lower_lookup, fallback_rate)

    def build_series(lookup: Dict[int, float], fallback_value: float) -> List[float]:
        values = [round(start_diopter, 2)]
        curr = start_diopter
        # 逐年积分
        for age in timeline[:-1]:
            # 获取当年的自然增长率 (mean/upper/lower)
            base_rate = lookup.get(age, fallback_value)
            # 应用治疗效果 (Natural 则是 * 1.0)
            real_rate = base_rate * scale
            curr += real_rate
            values.append(round(curr, 2))
        return values

    return {
        "timeline": timeline,
        "mean": build_series(rate_lookup, fallback_rate),
        "upper": build_series(upper_lookup, fallback_upper),
        "lower": build_series(lower_lookup, fallback_lower),
    }

# ================= 主构建逻辑 =================

def build_dictionary():
    print("🏭 正在启动全量构建 (Pivot Mode)...")

    df, meta_schema = load_config()
    if df.empty:
        print("⚠️ base_rates 数据为空，跳过构建。")
        return

    # 1. 确定分组列 (Cohort Dimensions)
    # 我们要排除掉 rate, treatment 等变化量，只保留人口学特征 (Ethnicity, Gender 等)
    exclude_cols = {
        "rate", "n", "age", "weight_alpha", "mean", "sd", "total_n",
        "treatment", "treatment_key", "records_used"
    }
    group_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 如果没有维度列 (比如只有全局数据)，加一个 dummy 列
    if not group_cols:
        df["_global"] = "ALL"
        group_cols = ["_global"]

    print(f"📊 分组维度: {group_cols}")

    # 2. 准备遍历空间
    start_ages = range(6, 17) # 6岁到16岁
    start_diopters = np.arange(-0.50, -6.25, -0.50) # -0.5D 到 -6.0D
    
    db: Dict[str, Any] = {}
    count = 0

    # 3. 按人群 (Cohort) 循环
    # 例如：Group = ("Asian", "Male")
    for group_values, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        # --- A. 提取基准数据 (Natural/Control) ---
        # 我们只用 Control 组的数据来拟合基准曲线
        natural_df = group_df[
            group_df["treatment_key"].isin(["Natural", "Control", "NA"])
        ]
        
        # 如果该组完全没有 Control 数据，是否要用所有数据混杂拟合？
        # 严格模式下：没有 Control 就不拟合。
        # 宽松模式下：如果为空，尝试用 data 兜底 (这里暂用严格模式+fallback)
        if natural_df.empty:
            # print(f"⚠️ 跳过组 {group_values}: 缺少 Natural 数据")
            # 可以在这里做一个 Global Fallback，或者暂时跳过
            # 为了演示健壮性，我们尝试用 group_df (假设里面隐含了基准)
            fit_df = group_df
        else:
            fit_df = natural_df

        # --- B. 拟合 GPR 模型 ---
        fit_result = fit_gpr_curve(fit_df)
        if not fit_result["timeline"]:
            continue

        # 建立快速查找表 (Lookup Tables)
        rate_lookup = {int(a): float(r) for a, r in zip(fit_result["timeline"], fit_result["mean_curve"])}
        upper_lookup = {int(a): float(v) for a, v in zip(fit_result["timeline"], fit_result.get("upper", []))}
        lower_lookup = {int(a): float(v) for a, v in zip(fit_result["timeline"], fit_result.get("lower", []))}

        # --- C. 生成 Key 前缀 ---
        # 格式: Asian_Male (不含 treatment)
        label_parts = [sanitize_value(val) for val in group_values]
        base_key = "_".join(part for part in label_parts if part and part != "NA") or "Global"

        # --- D. 遍历所有起始状态 (Age x Diopter) ---
        for age in start_ages:
            for dio in start_diopters:
                dio = round(float(dio), 2)
                
                # 最终 Key: Asian_Male_8_-2.0
                user_key = f"{base_key}_{age}_{dio}"
                
                # 容器: 存放该条件下所有治疗方案的曲线
                entry = {} 

                # --- E. 遍历所有已知治疗方案并生成曲线 ---
                # 1. Natural (Efficacy = 0)
                entry["Natural"] = calculate_progression(
                    rate_lookup, upper_lookup, lower_lookup,
                    start_age=age, start_diopter=dio, efficacy=0.0
                )

                # 2. Treatments (应用 Efficacy)
                for t_key, efficacy in TREATMENT_EFFICACIES.items():
                    if t_key == "Natural": continue
                    
                    entry[t_key] = calculate_progression(
                        rate_lookup, upper_lookup, lower_lookup,
                        start_age=age, start_diopter=dio, efficacy=efficacy
                    )
                
                # 写入大表
                db[user_key] = entry
                count += 1
        
        if count % 5000 == 0:
            print(f"   已生成 {count} 个预测组合...")

    # 4. 保存
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 构造最终 JSON 结构
    final_payload = {
        "meta_schema": meta_schema,
        "data_lookup": db # 将所有数据放入 data_lookup 键下，符合前端预期
    }
    
    # 如果 meta_schema 里没有 dimensions，补一下，方便前端自动生成下拉框
    if "dimensions" not in final_payload["meta_schema"]:
        final_payload["meta_schema"]["dimensions"] = group_cols
        # 简单的 options 推断
        options = {}
        for col in group_cols:
            options[col] = sorted(df[col].dropna().unique().tolist())
        final_payload["meta_schema"]["options"] = options

    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f)

    print(f"✅ 构建完成！")
    print(f"   - 输出文件: {DB_PATH}")
    print(f"   - 总 Key 数: {len(db)}")
    print(f"   - 示例 Key: {list(db.keys())[0] if db else 'None'}")


if __name__ == "__main__":
    build_dictionary()