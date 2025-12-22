import json
import os
import numpy as np
from scipy.optimize import curve_fit
from pydantic import BaseModel, Field, ValidationError, root_validator
from typing import List, Dict, Optional, Any
from openai import OpenAI
from rag_core import get_rag_engine

# 假设这是你以前写好的 RAG 内核
# from rag_core import get_engine 
# engine = get_engine()

# 为了演示，我们模拟一个 engine 对象 (实际使用时替换为真实的)
# class MockEngine:
#     def query(self, q):
#         return "模拟 RAG 返回的关于治疗方案和疗效数据的文本..."
# engine = MockEngine() 

# ================= 1. 数据结构定义 (Structured Output) =================

class TreatmentPlan(BaseModel):
    name: str = Field(..., description="治疗方案名称")
    conditions: str = Field(..., description="实施条件")
    cure_rate: float = Field(..., description="治愈率 (0-1.0)")
    side_effects: float = Field(..., description="副作用/并发症率 (0-1.0)")
    time_points: List[float] = Field(..., description="随访时间点(月)")
    recovery_values: List[float] = Field(..., description="对应时间点的视力恢复值")

class EvaluationResult(BaseModel):
    plan_name: str
    score_a: float
    score_b: float
    reason: str
    curve_params_with: Dict[str, float] # 拟合参数
    curve_params_without: Dict[str, float]

class FuzzyDataPoint(BaseModel):
    rate: float
    age: float
    sample_size: Optional[int] = Field(default=None, alias="n")
    ethnicity: Optional[str] = None
    gender: Optional[str] = None
    parental_myopia: Optional[str] = None
    treatment: Optional[str] = None
    notes: Optional[str] = None
    weight_alpha: float = Field(default=0.5)

    @root_validator(pre=False, skip_on_failure=True)
    def compute_uncertainty(cls, values):
        sample_size = values.get("sample_size")
        if sample_size is None or sample_size < 30:
            values["weight_alpha"] = 0.5
        else:
            values["weight_alpha"] = float(1.0 / max(1.0, np.sqrt(sample_size)))
        return values

    def to_record(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = {
            "rate": self.rate,
            "age": self.age,
            "n": self.sample_size,
            "weight_alpha": self.weight_alpha,
        }
        optional_fields = {
            "ethnicity": self.ethnicity,
            "gender": self.gender,
            "parental_myopia": self.parental_myopia,
            "treatment": self.treatment,
            "notes": self.notes,
        }
        for key, value in optional_fields.items():
            if value is not None:
                record[key] = value

        if extra:
            for key, value in extra.items():
                if key not in record and value is not None:
                    record[key] = value
        return record

class CovariateStat(BaseModel):
    name: str
    frequency: str

class CovariateSurvey(BaseModel):
    variables: List[CovariateStat]

class ValidatedDataPoint(BaseModel):
    label: str
    rate: float
    sample_size: int = Field(..., ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ================= 2. 临床 Agent 类 (核心逻辑) =================

class ClinicalAgent:
    def __init__(self, ):
        from dotenv import load_dotenv
        load_dotenv()
        print("正在连接 RAG 内核...")
        self.engine = get_rag_engine()
        self.survey_engine = None
        self.survey_top_k = 40

        # 👇 1. 获取代理地址
        api_base = os.getenv("OPENAI_API_BASE")
        
        # 👇 2. 初始化 OpenAI 客户端时传入 base_url
        self.llm_client = OpenAI(
            base_url=api_base # <--- 强制指定
        )

    def survey_covariates(self, max_nodes: int = 40) -> Dict[str, str]:
        """
        扫描文献，统计回归分析中常见的自变量及其出现频率。
        """
        print(f"🧭 正在执行变量普查，Top-{max_nodes} 节点 ...")
        if self.survey_engine is None or max_nodes != self.survey_top_k:
            self.survey_engine = get_rag_engine(similarity_top_k=max_nodes)
            self.survey_top_k = max_nodes

        survey_query = (
            "Identify independent variables or covariates that appear in "
            "regression analyses of myopia development, treatment efficacy, "
            "or axial elongation."
        )
        rag_response = self.survey_engine.query(survey_query)

        survey_prompt = f"""
        Review the provided medical papers regarding Myopia. List all independent variables used in regression analysis
        (e.g., Age, AL, Parental Myopia, Outdoor Time). For each variable, estimate the percentage of papers that include it
        based on the context. Return a JSON object with `variables` as a list of objects containing `name` and `frequency`
        (the latter should be a percentage string such as "85%").

        Context:
        {str(rag_response)[:4000]}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical literature surveyor. Respond with JSON only."},
                    {"role": "user", "content": survey_prompt}
                ],
                response_format={"type": "json_object"},
            )
            payload = response.choices[0].message.content
            survey = CovariateSurvey.model_validate_json(payload)
            return {item.name: item.frequency for item in survey.variables}
        except Exception as exc:
            print(f"⚠️ 变量普查失败: {exc}")
            return {}

    def _generate_search_query(self, patient_features: Dict) -> str:
        """
        [条件映射] 将患者特征映射为 RAG 搜索词
        不需要修改内核，而是通过 Prompt Engineering 优化输入
        """
        # 这里可以用简单的规则，也可以用 LLM 扩展
        query = f"针对 {patient_features['disease']} (阶段: {patient_features['stage']}) 的最新临床治疗方案、疗效数据及对比研究。"
        return query

    def _extract_structured_plans(self, rag_text: str) -> List[TreatmentPlan]:
        """
        [结构化提取] 利用 LLM 将 RAG 返回的纯文本转化为 JSON 对象
        """
        prompt = f"""
        从以下医学文献片段中，提取具体的治疗方案、疗效指标和时间曲线数据。
        文献内容: {rag_text[:3000]} 
        
        请严格返回 JSON 格式，包含 name, conditions, cure_rate, side_effects, time_points, recovery_values。
        """
        
        # 使用 OpenAI 的 JSON Mode (保证格式稳定)
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a medical data extractor. Output JSON."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # 这里省略了复杂的 JSON 解析和错误处理，直接假设提取成功
        # 实际开发中建议使用 LangChain 的 OutputParser 或 Pydantic Parser
        try:
            data = json.loads(response.choices[0].message.content)
            plans = [TreatmentPlan(**p) for p in data.get('plans', [])]
            return plans
        except:
            return []

    def _score_algorithm_a(self, plan: TreatmentPlan) -> float:
        """
        [算法 A] 指标驱动的硬计算
        """
        # 简单的加权公式：治愈率 * 70 - 副作用 * 30
        score = (plan.cure_rate * 100 * 0.7) - (plan.side_effects * 100 * 0.3)
        return round(max(0, score), 2)

    def _score_algorithm_b(self, plan: TreatmentPlan) -> tuple[float, str]:
        """
        [算法 B] 模型评估 (主观打分)
        """
        prompt = f"作为主任医师，请对方案 '{plan.name}' 进行综合打分(0-100)并给出简短理由。治愈率:{plan.cure_rate}, 副作用:{plan.side_effects}。"
        res = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = res.choices[0].message.content
        # 这里需要再用逻辑提取分数，简化起见直接模拟
        return 85.0, content

    def _fit_curve(self, x_data, y_data):
        """
        [疗效曲线建模] 使用 Scipy 进行数学拟合
        """
        if len(x_data) < 2: return {"a": 0, "b": 0} # 数据点太少无法拟合
        
        # 定义对数模型: y = a * ln(x) + b
        def log_func(t, a, b):
            return a * np.log(t) + b
            
        try:
            popt, _ = curve_fit(log_func, x_data, y_data, maxfev=5000)
            return {"func": "log", "a": popt[0], "b": popt[1]}
        except:
            return {"error": "fit_failed"}
        
    def validate_and_filter(
        self,
        data_points: List[Dict[str, Any]],
        value_field: str = "rate",
        hard_floor: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        数据清洗：过滤样本量不足和异常值的点位。
        """
        if not data_points:
            return []

        parsed: List[tuple[Dict[str, Any], ValidatedDataPoint]] = []
        for idx, entry in enumerate(data_points):
            try:
                payload = ValidatedDataPoint(
                    label=str(entry.get("label") or entry.get("id") or entry.get("ethnicity") or f"point_{idx}"),
                    rate=float(entry[value_field]),
                    sample_size=int(entry["sample_size"]),
                    metadata=entry.get("metadata") or {},
                )
                parsed.append((entry, payload))
            except (KeyError, TypeError, ValueError, ValidationError):
                continue

        # 样本量过滤
        parsed = [(raw, clean) for raw, clean in parsed if clean.sample_size >= 30]
        if not parsed:
            return []

        values = np.array([clean.rate for _, clean in parsed], dtype=float)
        mean = float(values.mean())
        std = float(values.std())

        cleaned: List[Dict[str, Any]] = []
        for raw, clean in parsed:
            value = clean.rate
            if std > 0 and abs(value - mean) > 3 * std:
                continue
            if hard_floor is not None and value < hard_floor:
                continue

            age_value = raw.get("age") or raw.get("Age")
            if age_value is None:
                continue

            try:
                fuzzy = FuzzyDataPoint(
                    rate=float(value),
                    age=float(age_value),
                    sample_size=clean.sample_size,
                    ethnicity=raw.get("ethnicity") or raw.get("Ethnicity"),
                    gender=raw.get("gender") or raw.get("sex"),
                    parental_myopia=raw.get("parental_myopia"),
                    treatment=raw.get("treatment"),
                    notes=raw.get("notes") or raw.get("commentary"),
                )

                extra_fields = {
                    k: v for k, v in raw.items()
                    if k not in {
                        "rate", "Rate", value_field, "sample_size", "n",
                        "ethnicity", "Ethnicity", "gender", "sex",
                        "age", "Age", "parental_myopia", "treatment", "notes", "commentary"
                    }
                }
                cleaned.append(fuzzy.to_record(extra=extra_fields))
            except (ValidationError, TypeError, ValueError):
                kept = dict(raw)
                kept[value_field] = float(value)
                kept["sample_size"] = int(clean.sample_size)
                cleaned.append(kept)

        return cleaned

    def research_general_knowledge(
        self,
        topic_prompt: str,
        enforce_validation: bool = False,
        value_field: str = "rate",
        hard_floor: Optional[float] = None,
    ) -> dict:
        """
        通用知识调研能力。
        输入：一段自然语言指令（比如：提取亚洲近视增长率...）
        输出：清洗好的 JSON 字典
        """
        print(f"🔎 Agent 正在调研: {topic_prompt[:20]}...")
        
        # 1. 复用你的 RAG 内核进行检索
        # 这里直接把调研指令作为 Query 发给 RAG
        rag_response = self.engine.query(topic_prompt)
        
        # 2. 复用 LLM 能力进行 JSON 提取
        # 我们构建一个专门的 Prompt 强制输出 JSON
        extraction_prompt = f"""
        Based on the following medical context, extract the information requested.
        
        Context:
        {str(rag_response)[:4000]}
        
        Request:
        {topic_prompt}
        
        IMPORTANT: You must output a VALID JSON object only. No markdown, no explanation.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise data extractor. Output JSON only."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format={"type": "json_object"} # 强制 JSON 模式
            )
            
            json_str = response.choices[0].message.content
            result = json.loads(json_str)

            if enforce_validation and isinstance(result.get("data_points"), list):
                result["data_points"] = self.validate_and_filter(
                    result["data_points"],
                    value_field=value_field,
                    hard_floor=hard_floor,
                )
            return result
            
        except Exception as e:
            print(f"❌ 调研失败: {e}")
            return {}

    # ================= 主流程 =================
    def process_patient(self, patient_features: Dict):
        print(f"正在分析患者: {patient_features} ...")
        
        # 1. RAG 检索 (调用你的内核)
        query = self._generate_search_query(patient_features)
        print(f"检索 Query: {query}")
        rag_response = self.engine.query(query) # <--- 关键：这里调用了旧代码
        
        # 2. 结构化提取
        plans = self._extract_structured_plans(str(rag_response))
        print(f"提取到 {len(plans)} 个候选方案")
        
        results = []
        for plan in plans:
            # 3. 打分
            score_a = self._score_algorithm_a(plan)
            score_b, reason = self._score_algorithm_b(plan)
            
            # 4. 曲线拟合 (With Management)
            curve_params = self._fit_curve(plan.time_points, plan.recovery_values)
            
            # (模拟) Without Management 曲线通常基于文献中的对照组数据，逻辑同上
            curve_params_ctrl = {"func": "log", "a": 0.1, "b": 0.1} 
            
            result = EvaluationResult(
                plan_name=plan.name,
                score_a=score_a,
                score_b=score_b,
                reason=reason,
                curve_params_with=curve_params,
                curve_params_without=curve_params_ctrl
            )
            results.append(result)
            
        return results

# ================= 3. 运行示例 =================

if __name__ == "__main__":
    # 初始化你的 RAG 内核 (假设你有 get_engine 函数)
    # from main import load_or_create_index, setup_global_settings
    # setup_global_settings()
    # real_engine = load_or_create_index(...).as_query_engine(...)
    
    # 这里用 Mock 演示
    agent = ClinicalAgent()
    
    patient_info = {
        "disease": "原发性开角型青光眼",
        "stage": "中期",
        "age": 45
    }
    
    final_report = agent.process_patient(patient_info)
    
    # 模拟存入数据库
    print("\n=== 最终分析报告 (即将写入 DB) ===")
    for item in final_report:
        print(item.json(indent=2, ensure_ascii=False))
