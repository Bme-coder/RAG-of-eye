import sys
import matplotlib

# 设置 matplotlib 后端
matplotlib.use('QtAgg')

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QComboBox, QFrame, QSizePolicy, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. 扩充后的 Mock 数据 (延伸至 17 岁)
# ==========================================
MOCK_JSON_DB = {
  "meta_schema": {
    "dimensions": ["Ethnicity", "Gender"],
    "options": {
      "Ethnicity": ["Asian", "Caucasian"],
      "Gender": ["Male", "Female"]
    }
  },
  "data_lookup": {
    "Asian_Male": {
      "Natural": {
        "timeline": [6, 8, 10, 12, 14, 16, 17],
        "mean":     [-1.0, -3.0, -4.5, -5.3, -6.0, -6.5, -6.66],
        "upper":    [-0.8, -2.2, -3.5, -4.5, -5.0, -5.5, -5.80],
        "lower":    [-1.2, -3.8, -5.5, -6.5, -7.5, -8.0, -8.20]
      },
      "Atropine_Low": {
        "timeline": [6, 8, 10, 12, 14, 16, 17],
        "mean":     [-1.0, -2.0, -2.8, -3.2, -3.8, -4.2, -4.37],
        "upper":    [-0.9, -1.6, -2.3, -2.7, -3.2, -3.6, -3.80],
        "lower":    [-1.1, -2.4, -3.3, -3.8, -4.4, -4.8, -5.00]
      }
    },
    "Asian_Female": {
      "Natural": {
        "timeline": [6, 8, 10, 12, 14, 16, 17],
        "mean":     [-0.5, -2.5, -3.5, -4.5, -5.2, -5.8, -6.10],
        "upper":    [-0.3, -1.8, -2.8, -3.8, -4.5, -5.0, -5.30],
        "lower":    [-0.7, -3.2, -4.5, -5.5, -6.2, -7.0, -7.50]
      }
      # 模拟无治疗数据的情况
    }
  }
}

# ==========================================
# 2. 右侧数据面板 (Info Panel)
# ==========================================
class InfoCard(QFrame):
    """自定义卡片组件，用于显示单个指标"""
    def __init__(self, title, color_code):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2D2D2D;
                border-radius: 8px;
                border: 1px solid #3E3E3E;
                padding: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # 标题
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #AAAAAA; font-size: 12px; font-weight: normal; border: none;")
        layout.addWidget(lbl_title)
        
        # 数值
        self.lbl_value = QLabel("--")
        self.lbl_value.setStyleSheet(f"color: {color_code}; font-size: 24px; font-weight: bold; border: none;")
        layout.addWidget(self.lbl_value)

    def set_value(self, text):
        self.lbl_value.setText(text)

class DataPanel(QWidget):
    """右侧总控面板"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 10)
        self.layout.setSpacing(15)

        # 标题
        title = QLabel("Expected Progression")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.layout.addWidget(title)
        
        # 说明文字
        desc = QLabel("Projected refractive error at age 17:")
        desc.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 5px;")
        desc.setWordWrap(True)
        self.layout.addWidget(desc)

        # 卡片 1: 自然病程 (红)
        self.card_natural = InfoCard("Without Management", "#FF5252")
        self.layout.addWidget(self.card_natural)

        # 卡片 2: 干预后 (绿)
        self.card_treated = InfoCard("With Management", "#69F0AE")
        self.layout.addWidget(self.card_treated)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444;")
        self.layout.addWidget(line)

        # 卡片 3: 挽回度数 (蓝/白)
        self.card_saved = InfoCard("Saved Diopters", "#FFFFFF")
        self.layout.addWidget(self.card_saved)

        # 减缓百分比徽章
        self.lbl_percent = QLabel("-0%")
        self.lbl_percent.setAlignment(Qt.AlignCenter)
        self.lbl_percent.setStyleSheet("""
            background-color: #1976D2;
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 15px;
            padding: 10px;
        """)
        self.layout.addWidget(self.lbl_percent)
        
        self.layout.addStretch() # 底部顶起

    def update_metrics(self, natural_data, treatment_data):
        """核心算法：计算并更新界面"""
        if not natural_data:
            self.reset_metrics()
            return

        # 获取终点数据 (假设数组最后一位是最终预测值)
        nat_end = natural_data['mean'][-1]
        
        # 更新自然病程
        self.card_natural.set_value(f"{nat_end:.2f} D")

        if treatment_data:
            treat_end = treatment_data['mean'][-1]
            
            # 计算差异 (注意负数逻辑: -4.37 - (-6.66) = +2.29)
            diff = treat_end - nat_end
            
            # 计算百分比: 挽回量 / |自然病程总量|
            if nat_end != 0:
                percent = (diff / abs(nat_end)) * 100
            else:
                percent = 0

            # 更新 UI
            self.card_treated.set_value(f"{treat_end:.2f} D")
            self.card_saved.set_value(f"+{diff:.2f} D")
            self.lbl_percent.setText(f"-{int(percent)}% Progression")
            self.lbl_percent.setStyleSheet("background-color: #1565C0; color: white; border-radius: 10px; padding: 10px; font-weight:bold; font-size: 18px;")
            self.lbl_percent.setVisible(True)
        else:
            self.card_treated.set_value("--")
            self.card_saved.set_value("--")
            self.lbl_percent.setVisible(False)

    def reset_metrics(self):
        self.card_natural.set_value("--")
        self.card_treated.set_value("--")
        self.card_saved.set_value("--")
        self.lbl_percent.setVisible(False)


# ==========================================
# 3. 绘图组件 (Matplotlib)
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # 深度暗黑风格配置
        self.bg_color = "#1E1E1E"    # 与窗口背景一致
        self.plot_bg = "#1E1E1E"     # 绘图区背景
        self.text_color = "#E0E0E0"  # 文字颜色
        self.grid_color = "#444444"  # 网格颜色
        
        self.fig.patch.set_facecolor(self.bg_color)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(self.plot_bg)
        
        self._setup_style()
        super(MplCanvas, self).__init__(self.fig)

    def _setup_style(self):
        self.axes.tick_params(colors=self.text_color, which='both')
        for spine in self.axes.spines.values():
            spine.set_edgecolor(self.grid_color)
        
        self.axes.xaxis.label.set_color(self.text_color)
        self.axes.yaxis.label.set_color(self.text_color)
        self.axes.title.set_color(self.text_color)
        
        # 开启虚线网格
        self.axes.grid(True, linestyle='--', alpha=0.4, color=self.grid_color)

    def plot_data(self, natural_data, treatment_data=None, treatment_name="Treatment"):
        self.axes.clear()
        self._setup_style()
        
        # 1. Natural (Red)
        if natural_data:
            x = natural_data['timeline']
            self.axes.plot(x, natural_data['mean'], 
                          color='#FF5252', linewidth=3, label='Natural Path')
            self.axes.fill_between(x, natural_data['lower'], natural_data['upper'], 
                                  color='#FF5252', alpha=0.15, edgecolor=None)

        # 2. Treatment (Green)
        if treatment_data:
            x_t = treatment_data['timeline']
            self.axes.plot(x_t, treatment_data['mean'], 
                          color='#69F0AE', linewidth=3, linestyle='--', label=treatment_name)
            self.axes.fill_between(x_t, treatment_data['lower'], treatment_data['upper'], 
                                  color='#69F0AE', alpha=0.15, edgecolor=None)

        # 3. 布局美化
        self.axes.set_xlabel("Age (Years)")
        self.axes.set_ylabel("Refractive Error (Diopters)")
        
        # 图例放在上方外部
        self.axes.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), 
                         mode="expand", borderaxespad=0, ncol=2, frameon=False,
                         labelcolor=self.text_color)
        
        # 动态撑满但留白
        self.axes.margins(x=0.05, y=0.15)
        self.draw()

# ==========================================
# 4. 主窗口逻辑
# ==========================================
class MyopiaVisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = MOCK_JSON_DB
        self.current_selectors = {}
        self.dimension_order = self.db["meta_schema"]["dimensions"]
        
        self.setWindowTitle("Myopia Prediction - Professional Edition")
        self.resize(1100, 750)
        self.setup_theme()
        self.init_ui()
        
        # 触发初始渲染
        self.on_dimension_changed()

    def setup_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; }
            QComboBox { 
                background-color: #333; color: white; border: 1px solid #555; 
                padding: 6px; border-radius: 4px; min-width: 120px;
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #333; color: white; selection-background-color: #555; }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Top: Filter Bar ---
        filter_frame = QFrame()
        filter_frame.setStyleSheet("background-color: #252525; border-radius: 8px;")
        filter_layout = QHBoxLayout(filter_frame)
        
        # Dimensions
        for dim in self.dimension_order:
            lbl = QLabel(f"{dim}:")
            lbl.setStyleSheet("color: #AAA; font-weight: bold;")
            combo = QComboBox()
            combo.addItems(self.db["meta_schema"]["options"].get(dim, []))
            combo.currentTextChanged.connect(self.on_dimension_changed)
            filter_layout.addWidget(lbl)
            filter_layout.addWidget(combo)
            filter_layout.addSpacing(15)
            self.current_selectors[dim] = combo

        # Treatment Selector
        t_lbl = QLabel("|  Treatment:")
        t_lbl.setStyleSheet("color: #69F0AE; font-weight: bold;")
        self.treatment_combo = QComboBox()
        self.treatment_combo.currentTextChanged.connect(self.on_treatment_changed)
        filter_layout.addWidget(t_lbl)
        filter_layout.addWidget(self.treatment_combo)
        
        filter_layout.addStretch()
        main_layout.addWidget(filter_frame)

        # --- Content: Split View (Left Chart, Right Data) ---
        content_layout = QHBoxLayout()
        
        # Left: Chart (70%)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(self.canvas, stretch=7)
        
        # Right: Data Panel (30%)
        self.data_panel = DataPanel()
        content_layout.addWidget(self.data_panel, stretch=3)
        
        main_layout.addLayout(content_layout)

    def get_current_key(self):
        return "_".join([self.current_selectors[d].currentText() for d in self.dimension_order])

    def on_dimension_changed(self):
        key = self.get_current_key()
        lookup_data = self.db["data_lookup"].get(key)
        
        self.treatment_combo.blockSignals(True)
        self.treatment_combo.clear()
        
        if lookup_data:
            treatments = [k for k in lookup_data.keys() if k != "Natural"]
            if treatments:
                self.treatment_combo.addItems(treatments)
                self.treatment_combo.setEnabled(True)
            else:
                self.treatment_combo.addItem("None Available")
                self.treatment_combo.setEnabled(False)
        else:
            self.treatment_combo.addItem("No Data")
            self.treatment_combo.setEnabled(False)
            
        self.treatment_combo.blockSignals(False)
        self.on_treatment_changed()

    def on_treatment_changed(self):
        key = self.get_current_key()
        t_name = self.treatment_combo.currentText()
        
        full_record = self.db["data_lookup"].get(key)
        if not full_record:
            self.canvas.axes.clear()
            self.canvas.draw()
            self.data_panel.reset_metrics()
            return

        nat_data = full_record.get("Natural")
        treat_data = full_record.get(t_name)
        
        # Update Chart
        self.canvas.plot_data(nat_data, treat_data, t_name)
        
        # Update Right Panel Metrics
        self.data_panel.update_metrics(nat_data, treat_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyopiaVisApp()
    window.show()
    sys.exit(app.exec())