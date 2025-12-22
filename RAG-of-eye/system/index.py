import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QFrame, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

# 引入 Matplotlib 用于画图
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置中文字体 (防止 Matplotlib 中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = ARTIFACTS_DIR / "myopia_db.json"

class MyopiaCalculator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("近视进展预测计算器 (Myopia Progression CDSS)")
        self.resize(1200, 800)
        
        # 1. 加载数据
        self.db = self.load_database()
        
        # 2. 初始化 UI
        self.init_ui()
        
        # 3. 首次渲染
        self.update_display()

    def load_database(self):
        """加载 JSON 字典"""
        db_path = DB_PATH
        if not db_path.exists():
            QMessageBox.warning(self, "数据缺失", f"找不到 {db_path}，将使用模拟数据演示。")
            return self.get_mock_data()
        
        try:
            with db_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")
            return {}

    def get_mock_data(self):
        """兜底的模拟数据"""
        return {
            "Asian_Female_8_-2.5": {
                "Atropine_Low": {
                    "timeline": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                    "natural": [-2.5, -3.4, -4.2, -4.9, -5.5, -6.0, -6.4, -6.7, -6.9, -7.0],
                    "managed": [-2.5, -3.0, -3.5, -3.9, -4.2, -4.5, -4.7, -4.8, -4.9, -5.0]
                }
            }
        }

    def init_ui(self):
        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- 顶部：标题 ---
        title = QLabel("近视进展预测模型")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50;")
        main_layout.addWidget(title)

        # --- 中部：选择过滤器 (Filters) ---
        filter_frame = QFrame()
        filter_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 10px; padding: 10px;")
        filter_layout = QHBoxLayout(filter_frame)
        
        # 创建下拉框
        self.combo_ethnicity = self.create_combo("种族", ["Asian", "Caucasian"], filter_layout)
        self.combo_age = self.create_combo("年龄", [str(i) for i in range(6, 17)], filter_layout, default="8")
        self.combo_gender = self.create_combo("性别", ["Female", "Male"], filter_layout)
        
        # 度数下拉框 (-0.5 到 -6.0)
        diopters = [f"{i:.2f}" for i in  [x * -0.5 for x in range(1, 13)]] # -0.5, -1.0 ...
        self.combo_diopter = self.create_combo("初始度数 (D)", diopters, filter_layout, default="-2.50")
        
        # 治疗方案
        self.combo_treatment = self.create_combo(
            "防控方案", 
            ["低浓度阿托品 (Atropine_Low)", "角膜塑形镜 (Ortho_K)", "离焦框架镜 (Defocus_Glasses)"], 
            filter_layout
        )
        # 映射显示文本到 Key
        self.treat_map = {
            "低浓度阿托品 (Atropine_Low)": "Atropine_Low",
            "角膜塑形镜 (Ortho_K)": "Ortho_K",
            "离焦框架镜 (Defocus_Glasses)": "Defocus_Glasses"
        }

        main_layout.addWidget(filter_frame)

        # --- 底部：图表 + 数据面板 ---
        content_layout = QHBoxLayout()
        
        # 1. 左侧图表 (Matplotlib)
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor='#ffffff')
        self.canvas = FigureCanvas(self.figure)
        content_layout.addWidget(self.canvas, stretch=2)

        # 2. 右侧数据面板
        stats_group = QGroupBox("预测详情 (至17岁)")
        stats_group.setStyleSheet("""
            QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid #ddd; border-radius: 8px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(20)

        # 统计项
        self.lbl_natural = self.create_stat_item("自然进展 (无干预)", "#e74c3c", stats_layout)
        self.lbl_managed = self.create_stat_item("干预后预计", "#27ae60", stats_layout)
        self.lbl_saved = self.create_stat_item("预计保留度数", "#2980b9", stats_layout)
        self.lbl_efficacy = self.create_stat_item("方案有效率", "#34495e", stats_layout)
        
        stats_layout.addStretch() # 顶上去
        content_layout.addWidget(stats_group, stretch=1)

        main_layout.addLayout(content_layout)

    def create_combo(self, label_text, items, parent_layout, default=None):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)
        
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #7f8c8d; font-weight: bold; font-size: 12px;")
        layout.addWidget(lbl)
        
        combo = QComboBox()
        combo.addItems(items)
        combo.setStyleSheet("""
            QComboBox { padding: 5px; border: 1px solid #dce0e6; border-radius: 4px; min-width: 120px; }
            QComboBox:focus { border: 1px solid #3498db; }
        """)
        if default and default in items:
            combo.setCurrentText(default)
            
        # 绑定事件
        combo.currentIndexChanged.connect(self.update_display)
        
        layout.addWidget(combo)
        parent_layout.addWidget(container)
        return combo

    def create_stat_item(self, title, color, parent_layout):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #555; font-size: 14px;")
        
        lbl_val = QLabel("--")
        lbl_val.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        lbl_val.setStyleSheet(f"color: {color};")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        parent_layout.addWidget(container)
        return lbl_val

    def update_display(self):
        # 1. 获取当前选择
        eth = self.combo_ethnicity.currentText()
        age = self.combo_age.currentText()
        gen = self.combo_gender.currentText()
        
        # 处理度数：界面显示 "-2.50"，Key可能是 "-2.5"
        raw_dio = self.combo_diopter.currentText()
        # 转一下 float 再转 string，去除多余的 0，匹配 Python 生成的 key 风格
        dio = str(float(raw_dio)) 
        
        treat_text = self.combo_treatment.currentText()
        treat_key = self.treat_map[treat_text]

        # 2. 拼装 Key
        key = f"{eth}_{gen}_{age}_{dio}"
        # print(f"查询 Key: {key}") # 调试用

        # 3. 查库
        if key in self.db and treat_key in self.db[key]:
            data = self.db[key][treat_key]
            self.plot_chart(data)
            self.render_stats(data, treat_key)
        else:
            # 数据没找到 (可能是 Mock 数据的 Key 不对，或者范围溢出)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "暂无该组合数据", ha='center', va='center', fontsize=14)
            self.canvas.draw()
            self.lbl_natural.setText("--")

    def plot_chart(self, data):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 绘制红线 (自然)
        ax.plot(data['timeline'], data['natural'], color='#e74c3c', linewidth=3, label='自然进展 (无干预)')
        ax.fill_between(data['timeline'], data['natural'],  max(data['natural'] + data['managed']), color='#e74c3c', alpha=0.1)

        # 绘制绿线 (干预)
        ax.plot(data['timeline'], data['managed'], color='#27ae60', linewidth=3, label='干预后 (With Management)')
        ax.fill_between(data['timeline'], data['managed'], min(data['natural'] + data['managed']), color='#27ae60', alpha=0.1)

        # 设置图表样式
        ax.set_title("屈光度预测曲线 (Refractive Error)", fontsize=12, pad=15)
        ax.set_xlabel("年龄 (Age)")
        ax.set_ylabel("度数 (Diopters)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
        # 设置 Y 轴范围 (让 -8 在下面，0 在上面)
        ax.set_ylim(-9.0, 0.5)
        
        self.canvas.draw()

    def render_stats(self, data, treat_key):
        final_natural = data['natural'][-1]
        final_managed = data['managed'][-1]
        saved = final_managed - final_natural # 比如 -4 - (-6) = +2
        
        self.lbl_natural.setText(f"{final_natural:.2f} D")
        self.lbl_managed.setText(f"{final_managed:.2f} D")
        self.lbl_saved.setText(f"+{saved:.2f} D")
        
        # 有效率回显
        efficacy_map = {"Atropine_Low": "37%", "Ortho_K": "50%", "Defocus_Glasses": "25%"}
        self.lbl_efficacy.setText(efficacy_map.get(treat_key, "--"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局字体
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MyopiaCalculator()
    window.show()
    sys.exit(app.exec())
