import sys
import os
import json
import matplotlib

# 设置 matplotlib 后端为 QtAgg
matplotlib.use('QtAgg')

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QComboBox, QFrame, QSizePolicy, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. 配置区域
# ==========================================
# 您的真实数据路径
DATA_FILE_PATH = "/home/disks/sdg/zcw/projects/RAG-of-eye/RAG-of-eye/artifacts/myopia_db.json"

# ==========================================
# 2. 右侧数据面板 (Info Panel) - 保持不变
# ==========================================
class InfoCard(QFrame):
    """显示单个核心指标的卡片"""
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
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #AAAAAA; font-size: 12px; font-weight: normal; border: none;")
        layout.addWidget(lbl_title)
        
        self.lbl_value = QLabel("--")
        self.lbl_value.setStyleSheet(f"color: {color_code}; font-size: 24px; font-weight: bold; border: none;")
        layout.addWidget(self.lbl_value)

    def set_value(self, text):
        self.lbl_value.setText(text)

class DataPanel(QWidget):
    """右侧统计面板"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 10)
        self.layout.setSpacing(15)

        # 标题区
        title = QLabel("Expected Progression")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        self.layout.addWidget(title)
        
        desc = QLabel("Projected refractive error (SER) at end of timeline:")
        desc.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 5px;")
        desc.setWordWrap(True)
        self.layout.addWidget(desc)

        # 卡片区
        self.card_natural = InfoCard("Without Management", "#FF5252") # 红
        self.layout.addWidget(self.card_natural)

        self.card_treated = InfoCard("With Management", "#69F0AE") # 绿
        self.layout.addWidget(self.card_treated)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444;")
        self.layout.addWidget(line)

        # 结论区
        self.card_saved = InfoCard("Saved Diopters", "#FFFFFF")
        self.layout.addWidget(self.card_saved)

        self.lbl_percent = QLabel("-0%")
        self.lbl_percent.setAlignment(Qt.AlignCenter)
        self.lbl_percent.setStyleSheet("""
            background-color: #1565C0; color: white; border-radius: 10px; 
            padding: 10px; font-weight:bold; font-size: 18px;
        """)
        self.lbl_percent.setVisible(False)
        self.layout.addWidget(self.lbl_percent)
        
        self.layout.addStretch()

    def update_metrics(self, natural_data, treatment_data):
        if not natural_data:
            self.reset_metrics()
            return

        # 自动获取数组最后一个点作为终点数据
        nat_end = natural_data['mean'][-1]
        self.card_natural.set_value(f"{nat_end:.2f} D")

        if treatment_data:
            treat_end = treatment_data['mean'][-1]
            diff = treat_end - nat_end # 比如 -4 - (-6) = +2
            
            if nat_end != 0:
                percent = (diff / abs(nat_end)) * 100
            else:
                percent = 0

            self.card_treated.set_value(f"{treat_end:.2f} D")
            self.card_saved.set_value(f"+{diff:.2f} D")
            self.lbl_percent.setText(f"-{int(percent)}% Progression")
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
        
        # 这里的颜色需要和 Qt 窗口背景融合
        self.bg_color = "#1E1E1E"
        self.text_color = "#E0E0E0"
        self.grid_color = "#444444"
        
        self.fig.patch.set_facecolor(self.bg_color)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(self.bg_color)
        
        self._setup_style()
        super(MplCanvas, self).__init__(self.fig)

    def _setup_style(self):
        self.axes.tick_params(colors=self.text_color, which='both')
        for spine in self.axes.spines.values():
            spine.set_edgecolor(self.grid_color)
        
        self.axes.xaxis.label.set_color(self.text_color)
        self.axes.yaxis.label.set_color(self.text_color)
        self.axes.title.set_color(self.text_color)
        self.axes.grid(True, linestyle='--', alpha=0.4, color=self.grid_color)

    def plot_data(self, natural_data, treatment_data=None, treatment_name="Treatment"):
        self.axes.clear()
        self._setup_style()
        
        # 1. 绘制自然病程 (红色)
        if natural_data:
            x = natural_data['timeline']
            self.axes.plot(x, natural_data['mean'], 
                          color='#FF5252', linewidth=3, label='Natural Path')
            self.axes.fill_between(x, natural_data['lower'], natural_data['upper'], 
                                  color='#FF5252', alpha=0.15, edgecolor=None)

        # 2. 绘制治疗方案 (绿色)
        if treatment_data:
            x_t = treatment_data['timeline']
            self.axes.plot(x_t, treatment_data['mean'], 
                          color='#69F0AE', linewidth=3, linestyle='--', label=treatment_name)
            self.axes.fill_between(x_t, treatment_data['lower'], treatment_data['upper'], 
                                  color='#69F0AE', alpha=0.15, edgecolor=None)

        # 3. 布局调整
        self.axes.set_xlabel("Age (Years)")
        self.axes.set_ylabel("Refractive Error (D)")
        
        # 图例置于上方
        self.axes.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), 
                         mode="expand", borderaxespad=0, ncol=2, frameon=False,
                         labelcolor=self.text_color)
        
        # 自动留白
        self.axes.margins(x=0.05, y=0.15)
        self.draw()

# ==========================================
# 4. 主程序窗口
# ==========================================
class MyopiaVisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = {}
        self.current_selectors = {}
        self.dimension_order = []
        
        self.setWindowTitle("Myopia Prediction System")
        self.resize(1200, 800)
        self.setup_theme()
        
        # 1. 先尝试加载数据，如果失败则不初始化 UI
        if self.load_data_from_disk():
            self.init_ui()
            # 触发第一次渲染
            self.on_dimension_changed()
        else:
            # 数据加载失败，已弹窗提示，这里可以做额外处理
            pass

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
            QMessageBox { background-color: #333; color: white; }
        """)

    def load_data_from_disk(self):
        """核心：从指定路径加载 JSON"""
        target_path = DATA_FILE_PATH
        
        # 容错：如果绝对路径找不到，试试当前目录（方便调试）
        if not os.path.exists(target_path):
            local_path = "myopia_db.json"
            if os.path.exists(local_path):
                target_path = local_path
            else:
                QMessageBox.critical(self, "Error", f"Data file not found at:\n{target_path}")
                return False

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                self.db = json.load(f)
                
            # 简单校验数据完整性
            if "meta_schema" not in self.db or "data_lookup" not in self.db:
                raise ValueError("Invalid JSON structure: missing meta_schema or data_lookup")
                
            self.dimension_order = self.db["meta_schema"]["dimensions"]
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Data Error", f"Failed to load JSON:\n{str(e)}")
            return False

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Top Filter Bar ---
        filter_frame = QFrame()
        filter_frame.setStyleSheet("background-color: #252525; border-radius: 8px;")
        filter_layout = QHBoxLayout(filter_frame)
        
        # 根据 meta_schema 动态生成维度下拉框
        options_map = self.db["meta_schema"]["options"]
        for dim in self.dimension_order:
            lbl = QLabel(f"{dim}:")
            lbl.setStyleSheet("color: #AAA; font-weight: bold;")
            combo = QComboBox()
            # 安全获取选项，防止 Key 不存在
            items = options_map.get(dim, [])
            combo.addItems(items)
            combo.currentTextChanged.connect(self.on_dimension_changed)
            
            filter_layout.addWidget(lbl)
            filter_layout.addWidget(combo)
            filter_layout.addSpacing(15)
            self.current_selectors[dim] = combo

        # 治疗方案选择器
        t_lbl = QLabel("|  Treatment:")
        t_lbl.setStyleSheet("color: #69F0AE; font-weight: bold;")
        self.treatment_combo = QComboBox()
        self.treatment_combo.currentTextChanged.connect(self.on_treatment_changed)
        filter_layout.addWidget(t_lbl)
        filter_layout.addWidget(self.treatment_combo)
        
        filter_layout.addStretch()
        main_layout.addWidget(filter_frame)

        # --- Main Content (Split View) ---
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
        """按顺序拼接 Key, 例如: 'Asian_Male'"""
        parts = []
        for dim in self.dimension_order:
            val = self.current_selectors[dim].currentText()
            parts.append(val)
        return "_".join(parts)

    def on_dimension_changed(self):
        key = self.get_current_key()
        lookup_data = self.db["data_lookup"].get(key)
        
        self.treatment_combo.blockSignals(True)
        self.treatment_combo.clear()
        
        if lookup_data:
            # 过滤出除 Natural 外的所有 Key 作为治疗方案
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
        # 维度改变时，手动触发一次绘图更新
        self.on_treatment_changed()

    def on_treatment_changed(self):
        key = self.get_current_key()
        t_name = self.treatment_combo.currentText()
        
        full_record = self.db["data_lookup"].get(key)
        
        # 空值保护
        if not full_record:
            self.canvas.axes.clear()
            self.canvas.draw()
            self.data_panel.reset_metrics()
            return

        nat_data = full_record.get("Natural")
        treat_data = full_record.get(t_name)
        
        # 1. 更新图表
        self.canvas.plot_data(nat_data, treat_data, t_name)
        
        # 2. 更新右侧数字
        self.data_panel.update_metrics(nat_data, treat_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyopiaVisApp()
    window.show()
    sys.exit(app.exec())