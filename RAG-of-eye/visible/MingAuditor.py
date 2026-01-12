import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

class AdvancedDataAuditor:
    def __init__(self, config_path="/home/disks/sdg/zcw/projects/RAG-of-eye/RAG-of-eye/artifacts/medical_config.json"):
        self.config_path = config_path
        self.output_path = os.path.join(os.path.dirname(config_path), "data_health_report.md")
        self.raw_data = self._load_json()
        self.df = self._flatten_data()
        
        # 存储分析结果容器
        self.coverage_matrix = {}
        self.critical_errors = []
        self.warnings = []
        self.action_plan = []

    def _load_json(self):
        if not os.path.exists(self.config_path):
            print(f"❌ 错误: 找不到文件 {self.config_path}")
            return {}
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _flatten_data(self):
        """将 JSON 展平为 DataFrame，方便分析"""
        rows = []
        if "base_rates" not in self.raw_data:
            return pd.DataFrame()

        for entry in self.raw_data["base_rates"]:
            # 组合唯一分组键
            group_key = f"{entry.get('ethnicity', 'Unk')}_{entry.get('gender', 'Unk')}_{entry.get('treatment_key', 'Unk')}"
            
            base_info = {
                "group_id": group_key,
                "ethnicity": entry.get('ethnicity'),
                "gender": entry.get('gender'),
                "treatment": entry.get('treatment_key'),
                "age": entry.get('age'),
                "mean": entry.get('mean'),
                "total_n": entry.get("total_n", 0),
            }
            
            # 如果没有 raw_evidence，添加一条只有 base_info 的记录
            evidence_list = entry.get("raw_evidence", [])
            if not evidence_list:
                rows.append({**base_info, "has_evidence": False, "passage": ""})
            else:
                for ev in evidence_list:
                    rows.append({
                        **base_info,
                        "has_evidence": True,
                        "evidence_val": ev.get('mean'),
                        "evidence_n": ev.get('n', 0),
                        "passage": ev.get('full_passage', ""),
                        "source": ev.get('source_id', "Unknown")
                    })
        return pd.DataFrame(rows)

    def run_audit(self):
        """执行全流程审计"""
        print(f"🕵️‍♂️ 正在审计数据: {self.config_path} ...")
        
        if self.df.empty:
            print("❌ 数据为空，无法分析。")
            return

        # 1. 分析覆盖率 (生成矩阵)
        self._analyze_coverage()
        
        # 2. 分析数据质量 (幻觉与生理异常)
        self._analyze_quality()
        
        # 3. 生成行动建议 (只生成指令文本)
        self._generate_next_steps()
        
        # 4. 写入 Markdown 报告
        self._write_report()
        
        print(f"✅ 审计完成！报告已生成: {self.output_path}")

    # =========================================================
    # 模块 1: 覆盖率矩阵 (Data Coverage Matrix)
    # =========================================================
    def _analyze_coverage(self):
        # 获取所有唯一的分组
        groups = self.df['group_id'].unique()
        ages = sorted(self.df['age'].unique())
        
        # 构建矩阵数据
        matrix = {g: {a: "❌" for a in range(6, 17)} for g in groups} # 默认 6-16 岁全空
        
        for _, row in self.df.drop_duplicates(subset=['group_id', 'age']).iterrows():
            gid = row['group_id']
            age = row['age']
            n = row['total_n']
            
            if age < 6 or age > 16: continue # 忽略范围外年龄
            
            if n == 0:
                matrix[gid][age] = "❌" # 有记录但 N=0
            elif n < 50:
                matrix[gid][age] = f"⚠️ ({int(n)})" # 样本不足
            else:
                matrix[gid][age] = f"✅ ({int(n)})" # 样本充足
        
        self.coverage_matrix = matrix

    # =========================================================
    # 模块 2: 质量深度检查 (Deep Quality Check)
    # =========================================================
    def _analyze_quality(self):
        # A. 幻觉检测 (Hallucination Check)
        evidence_rows = self.df[self.df['has_evidence'] == True]
        for _, row in evidence_rows.iterrows():
            val = str(abs(row.get('evidence_val', 0))) # 取绝对值比对
            passage = row.get('passage', "")
            
            if len(passage) > 20: # 只有当有长文本时才检查
                # 简单模糊匹配：检查数值字符串是否在文本中
                # 生产环境建议用 Regex: re.search(rf"{val}", passage)
                if val not in passage and f"{row.get('evidence_val', 0)}" not in passage:
                     self.critical_errors.append({
                        "type": "幻觉警报 (Hallucination)",
                        "group": row['group_id'],
                        "age": row['age'],
                        "details": f"提取值 **{row.get('evidence_val')}** 未在原文中找到。",
                        "context": passage[:100] + "..." # 截取前100字
                    })

        # B. 生理异常检测 (Physiological Check)
        for _, row in self.df.drop_duplicates(subset=['group_id', 'age']).iterrows():
            mean_val = row['mean']
            # 异常 1: 远视漂移 (近视治疗很少会让度数变正)
            if mean_val > 0.2: 
                self.critical_errors.append({
                    "type": "生理悖论 (Positive Progression)",
                    "group": row['group_id'],
                    "age": row['age'],
                    "details": f"数值 **{mean_val}** 为正数 (通常应为负)。请检查是否为轴长(mm)而非屈光度(D)。",
                    "context": "N/A"
                })
            # 异常 2: 剧烈进展 (一年加深超过 200度/-2.0D 极罕见)
            if mean_val < -2.0:
                 self.warnings.append(f"⚠️ **{row['group_id']} @ Age {row['age']}**: 数值 {mean_val} 过低，疑似病理性近视或数据错误。")

    # =========================================================
    # 模块 3: 生成行动建议 (Action Plan Generator)
    # =========================================================
    def _generate_next_steps(self):
        # 策略：找到覆盖率低的分组
        for gid, age_map in self.coverage_matrix.items():
            missing_ages = [a for a, status in age_map.items() if "❌" in status]
            low_sample_ages = [a for a, status in age_map.items() if "⚠️" in status]
            
            parts = gid.split('_')
            if len(parts) >= 3:
                eth, gen, treat = parts[0], parts[1], "_".join(parts[2:])
                
                if len(missing_ages) > 5: # 缺得太多
                    cmd = f"python system/mining_task.py --ethnicity {eth} --gender {gen} --treatment {treat} --target_ages {min(missing_ages)}-{max(missing_ages)}"
                    self.action_plan.append(f"🔴 **严重缺失**: {gid} 缺失大量数据。\n   > 建议指令: `{cmd}`")
                elif missing_ages:
                    ages_str = ",".join(map(str, missing_ages))
                    cmd = f"python system/mining_task.py --ethnicity {eth} --gender {gen} --treatment {treat} --target_ages {ages_str}"
                    self.action_plan.append(f"🟠 **填补空白**: {gid} 缺失年龄 {ages_str}。\n   > 建议指令: `{cmd}`")

    # =========================================================
    # 模块 4: 写报告 (Report Writer)
    # =========================================================
    def _write_report(self):
        with open(self.output_path, 'w') as f:
            # Title
            f.write(f"# 🏥 RAG-of-Eye 数据健康体检报告\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Section 1: Coverage Matrix
            f.write("## 1. 📊 数据覆盖率全景图 (Data Coverage Matrix)\n")
            f.write("> 图例: ❌ = 缺失 | ⚠️ = 样本不足(<50) | ✅ = 充足\n\n")
            
            # Header
            header = "| Group | " + " | ".join([f"Age {a}" for a in range(6, 17)]) + " |"
            separator = "| :--- | " + " | ".join([":---:" for _ in range(6, 17)]) + " |"
            f.write(header + "\n" + separator + "\n")
            
            # Rows
            for gid, age_map in self.coverage_matrix.items():
                row_str = f"| **{gid}** | " + " | ".join([age_map[a] for a in range(6, 17)]) + " |"
                f.write(row_str + "\n")
            f.write("\n")
            
            # Section 2: Critical Issues
            f.write("## 2. 🚨 严重阻断性问题 (Critical Issues)\n")
            if not self.critical_errors:
                f.write("✅ **太棒了！未发现严重的阻断性数据问题。**\n\n")
            else:
                f.write("**以下问题可能会导致建模失败或误导医生，请务必处理：**\n\n")
                for err in self.critical_errors:
                    f.write(f"### 🛑 {err['type']}\n")
                    f.write(f"- **坐标**: `{err['group']}` @ Age {err['age']}\n")
                    f.write(f"- **详情**: {err['details']}\n")
                    f.write(f"- **原文片段**: _{err['context']}_\n\n")
            
            # Section 3: Warnings
            if self.warnings:
                f.write("## 3. ⚠️ 潜在风险警告 (Warnings)\n")
                for w in self.warnings:
                    f.write(f"- {w}\n")
                f.write("\n")

            # Section 4: Next Steps
            f.write("## 4. 📝 自动生成的下一步指令 (Action Plan)\n")
            f.write("**请复制以下指令到终端运行，以填补数据空缺：**\n\n")
            if not self.action_plan:
                f.write("🎉 无需额外操作，数据覆盖率已达标！\n")
            else:
                for plan in self.action_plan:
                    f.write(f"{plan}\n")

if __name__ == "__main__":
    auditor = AdvancedDataAuditor()
    auditor.run_audit()