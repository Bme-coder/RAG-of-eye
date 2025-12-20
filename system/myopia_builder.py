# myopia_bulider.py (建议改名为 myopia_builder.py)
import json
import numpy as np
import os

# ================= 配置加载 =================
def load_config():
    if not os.path.exists("medical_config.json"):
        print("❌ 错误: 找不到 medical_config.json")
        return {}, {}
    
    with open("medical_config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rates = data.get("BASE_RATES", {})
    treatments = data.get("TREATMENTS", {})
    
    # 清洗 Key (str -> int)
    try:
        for eth in rates:
            for sex in rates[eth]:
                rates[eth][sex] = {int(k): float(v) for k, v in rates[eth][sex].items()}
    except:
        print("⚠️ 配置格式可能有误")
        
    return rates, treatments

# ================= 核心计算逻辑 =================
def calculate_progression(ethnicity, gender, start_age, start_diopter, efficacy, base_rates):
    target_age = 17
    years = list(range(start_age, target_age + 1))
    
    red_curve = [start_diopter]
    green_curve = [start_diopter]
    
    curr_red = start_diopter
    curr_green = start_diopter
    
    for age in years[:-1]:
        # 查表：如果字典里没有该种族数据，默认给 -0.5
        # 注意：这里 ethnicity 变量会传入 "Asian" 或 "Caucasian"
        rate = base_rates.get(ethnicity, {}).get(gender, {}).get(age, -0.5)
        
        curr_red += rate
        red_curve.append(round(curr_red, 2))
        
        curr_green += rate * (1 - efficacy)
        green_curve.append(round(curr_green, 2))
        
    return {
        "timeline": years,
        "natural": red_curve,
        "managed": green_curve
    }

# ================= 构建主程序 =================
def build_dictionary():
    print("🏭 正在启动全量遍历构建...")
    
    base_rates, treatments = load_config()
    if not base_rates: return
    
    db = {}
    count = 0
    
    # --- 修改点：完全对齐你的需求范围 ---
    
    # 1. 种族：Asian 和 Caucasian
    # (注意：前提是 mining_task 挖到了 Caucasian 的数据，否则会用默认值 -0.5)
    ethnicities = ["Asian", "Caucasian"] 
    
    # 2. 性别
    genders = ["Female", "Male"]
    
    # 3. 年龄：6岁 到 16岁 (range不含结尾，所以写17)
    start_ages = range(6, 17) 
    
    # 4. 度数：-0.5 到 -6.0 (步长0.5)
    # np.arange(-0.5, -6.1, -0.5) 才能包含 -6.0
    start_diopters = np.arange(-0.50, -6.25, -0.50) 
    
    # 计算总组合数
    total = len(ethnicities) * len(genders) * len(start_ages) * len(start_diopters)
    print(f"   预计生成组合数: {total} 种情况")
    
    for eth in ethnicities:
        for sex in genders:
            for age in start_ages:
                for dio in start_diopters:
                    dio = round(float(dio), 2)
                    
                    # 生成 Key: "Asian_Male_16_-5.5"
                    user_key = f"{eth}_{sex}_{age}_{dio}"
                    db[user_key] = {}
                    
                    for t_key, t_val in treatments.items():
                        res = calculate_progression(
                            eth, sex, age, dio, 
                            t_val["efficacy"], 
                            base_rates
                        )
                        db[user_key][t_key] = res
                    
                    count += 1
                    if count % 1000 == 0: print(f"   已生成 {count} 条...")
    
    with open("myopia_db.json", "w", encoding="utf-8") as f:
        json.dump(db, f)
        
    print(f"✅ 构建完成！最终生成 {count} 条数据 -> myopia_db.json")

if __name__ == "__main__":
    build_dictionary()