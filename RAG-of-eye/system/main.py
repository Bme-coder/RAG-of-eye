import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib

# Ensure we render inside Qt
matplotlib.use("QtAgg")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def locate_data_file(filename: str = "myopia_db.json") -> Optional[Path]:
    """Search current, parent, and nested directories for the given filename."""
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    visited: Set[Path] = set()

    for root in search_roots:
        current = root
        while current not in visited:
            visited.add(current)
            candidate = current / filename
            if candidate.exists():
                return candidate.resolve()
            if current.parent == current:
                break
            current = current.parent

    for root in search_roots:
        try:
            match = next(root.rglob(filename))
            return match.resolve()
        except StopIteration:
            continue
        except PermissionError:
            continue
    return None


def load_database(explicit_path: Optional[Path] = None) -> Dict[str, Any]:
    search_order: List[Path] = []

    if explicit_path:
        search_order.append(Path(explicit_path).expanduser())

    env_path = os.getenv("MYOPIA_DB_PATH")
    if env_path:
        search_order.append(Path(env_path).expanduser())

    for candidate in search_order:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    data_path = locate_data_file()
    if data_path:
        with data_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    locations = [str(path) for path in search_order if path]
    raise FileNotFoundError(
        "Unable to locate myopia_db.json. "
        "Tried CLI/env paths and recursive search under current/project directories. "
        f"Checked: {locations or 'None provided'}"
    )


def _sort_numeric_labels(values: List[str]) -> List[str]:
    def sort_key(val: str) -> Tuple[int, Union[float, str]]:
        try:
            return (0, float(val))
        except ValueError:
            return (1, val)

    return [value for value in sorted(values, key=sort_key)]


def _parse_series_entry_key(raw_key: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (ethnicity, profile, age, baseline) parsed from DB key."""
    if not raw_key:
        return None
    text = raw_key.strip()
    ethnicity = text
    remainder = ""
    if "__" in text:
        ethnicity, remainder = text.split("__", 1)
    elif "_" in text:
        ethnicity, remainder = text.split("_", 1)
    else:
        return None

    parts = [segment for segment in remainder.split("_") if segment]
    if len(parts) < 2:
        return None

    age = parts[-2]
    baseline = parts[-1]
    profile = "_".join(parts[:-2]) if len(parts) > 2 else "General"
    ethnicity = ethnicity or "Unknown"
    profile = profile or "General"
    return ethnicity, profile, age, baseline


class MyopiaDataIndex:
    def __init__(self, raw_db: Dict[str, Any]):
        self.meta = raw_db.get("__meta__", {})
        self._tree: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = {}
        self._build_index(raw_db)

    def _build_index(self, raw_db: Dict[str, Any]) -> None:
        for key, payload in raw_db.items():
            if key.startswith("__"):
                continue
            if not isinstance(payload, dict):
                continue

            parsed = _parse_series_entry_key(key)
            if not parsed:
                continue

            ethnicity, profile, age, baseline = parsed
            profile_map = self._tree.setdefault(ethnicity, {})
            age_map = profile_map.setdefault(profile, {})
            baseline_map = age_map.setdefault(age, {})
            aggregated_entry = baseline_map.setdefault(baseline, {})

            for treatment_name, curve in payload.items():
                if not isinstance(curve, dict):
                    continue
                aggregated_entry[treatment_name] = curve

    def is_empty(self) -> bool:
        return not self._tree

    def ethnicities(self) -> List[str]:
        return sorted(self._tree.keys())

    def profiles(self, ethnicity: str) -> List[str]:
        return sorted(self._tree.get(ethnicity, {}))

    def ages(self, ethnicity: str, profile: str) -> List[str]:
        ages = list(self._tree.get(ethnicity, {}).get(profile, {}).keys())
        return _sort_numeric_labels(ages)

    def baselines(self, ethnicity: str, profile: str, age: str) -> List[str]:
        baselines = list(self._tree.get(ethnicity, {}).get(profile, {}).get(age, {}).keys())
        return _sort_numeric_labels(baselines)

    def get_record(self, ethnicity: str, profile: str, age: str, baseline: str) -> Optional[Dict[str, Any]]:
        return (
            self._tree.get(ethnicity, {})
            .get(profile, {})
            .get(age, {})
            .get(baseline)
        )


class InfoCard(QFrame):
    def __init__(self, title: str, color: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #2D2D2D;
                border-radius: 8px;
                border: 1px solid #3E3E3E;
                padding: 12px;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #B0B0B0; font-size: 12px;")
        layout.addWidget(lbl_title)

        self.lbl_value = QLabel("--")
        self.lbl_value.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.lbl_value)

    def set_value(self, text: str) -> None:
        self.lbl_value.setText(text)


class DataPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 10)
        layout.setSpacing(15)

        title = QLabel("Expected Progression")
        title.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel("Projected spherical equivalent refraction at end of timeline:")
        desc.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.card_natural = InfoCard("Without Management", "#FF5252")
        self.card_treated = InfoCard("With Management", "#69F0AE")
        self.card_saved = InfoCard("Saved Diopters", "#FFFFFF")

        layout.addWidget(self.card_natural)
        layout.addWidget(self.card_treated)
        layout.addWidget(self.card_saved)

        self.lbl_percent = QLabel("--")
        self.lbl_percent.setAlignment(Qt.AlignCenter)
        self.lbl_percent.setStyleSheet(
            """
            QLabel {
                background-color: #1565C0;
                color: white;
                border-radius: 12px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            """
        )
        layout.addWidget(self.lbl_percent)
        layout.addStretch()

    def reset(self) -> None:
        self.card_natural.set_value("--")
        self.card_treated.set_value("--")
        self.card_saved.set_value("--")
        self.lbl_percent.setText("--")

    def update_metrics(self, natural: Dict[str, List[float]], treatment: Optional[Dict[str, List[float]]]) -> None:
        if not natural:
            self.reset()
            return

        nat_end = natural["mean"][-1]
        self.card_natural.set_value(f"{nat_end:.2f} D")

        if treatment:
            treat_end = treatment["mean"][-1]
            diff = treat_end - nat_end
            percent = (diff / abs(nat_end)) * 100 if nat_end != 0 else 0.0

            self.card_treated.set_value(f"{treat_end:.2f} D")
            self.card_saved.set_value(f"{diff:+.2f} D")

            label = f"{percent:+.0f}% change vs natural"
            bg_color = "#1565C0" if percent >= 0 else "#C62828"
            self.lbl_percent.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {bg_color};
                    color: white;
                    border-radius: 12px;
                    padding: 10px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                """
            )
            self.lbl_percent.setText(label)
        else:
            self.card_treated.set_value("--")
            self.card_saved.set_value("--")
            self.lbl_percent.setText("No intervention data")


class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None, width: float = 5, height: float = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.bg_color = "#1E1E1E"
        self.text_color = "#E0E0E0"
        self.grid_color = "#444444"
        self.fig.patch.set_facecolor(self.bg_color)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._configure_axes()

    def _configure_axes(self) -> None:
        self.axes.set_facecolor(self.bg_color)
        self.axes.tick_params(colors=self.text_color, which="both")
        for spine in self.axes.spines.values():
            spine.set_edgecolor(self.grid_color)
        self.axes.grid(True, linestyle="--", alpha=0.4, color=self.grid_color)
        self.axes.xaxis.label.set_color(self.text_color)
        self.axes.yaxis.label.set_color(self.text_color)
        self.axes.title.set_color(self.text_color)

    def show_empty(self, message: str) -> None:
        self.axes.clear()
        self._configure_axes()
        self.axes.text(0.5, 0.5, message, ha="center", va="center", color=self.text_color, fontsize=14)
        self.draw()

    def plot_curves(
        self,
        natural: Optional[Dict[str, List[float]]],
        treatment: Optional[Dict[str, List[float]]],
        treatment_name: Optional[str],
    ) -> None:
        if not natural:
            self.show_empty("Natural progression data missing.")
            return

        self.axes.clear()
        self._configure_axes()

        x_nat = natural["timeline"]
        self.axes.plot(x_nat, natural["mean"], color="#FF5252", linewidth=3, label="Natural Path")
        self.axes.fill_between(x_nat, natural["lower"], natural["upper"], color="#FF5252", alpha=0.15)

        if treatment:
            x_trt = treatment["timeline"]
            label = treatment_name or "Treatment"
            self.axes.plot(
                x_trt,
                treatment["mean"],
                color="#69F0AE",
                linewidth=3,
                linestyle="--",
                label=label,
            )
            self.axes.fill_between(x_trt, treatment["lower"], treatment["upper"], color="#69F0AE", alpha=0.15)

        self.axes.set_xlabel("Age (years)")
        self.axes.set_ylabel("Refractive Error (D)")
        self.axes.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            mode="expand",
            borderaxespad=0,
            ncol=2,
            frameon=False,
            labelcolor=self.text_color,
        )
        self.axes.margins(x=0.05, y=0.15)
        self.draw()


def _extract_db_override(argv: List[str]) -> Optional[Path]:
    for idx, arg in enumerate(argv[1:], start=1):
        raw_value: Optional[str] = None
        if arg.startswith("--db="):
            raw_value = arg.split("=", 1)[1]
        elif arg == "--db" and idx + 1 < len(argv):
            raw_value = argv[idx + 1]

        if raw_value:
            return Path(raw_value).expanduser()
    return None


class MyopiaPredictionApp(QMainWindow):
    def __init__(self, db_override: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Myopia Prediction System")
        self.resize(1280, 840)

        try:
            raw_db = load_database(db_override)
        except Exception as exc:
            QMessageBox.critical(self, "Data Error", str(exc))
            raw_db = {}

        self.data_index = MyopiaDataIndex(raw_db)
        self._setup_theme()
        self._init_ui()
        self._populate_initial_filters()

    def _setup_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1E1E1E; }
            QLabel { color: #E0E0E0; }
            QComboBox {
                background-color: #333333;
                border: 1px solid #555555;
                padding: 6px;
                border-radius: 4px;
                color: #FFFFFF;
                min-width: 140px;
            }
            QComboBox QAbstractItemView {
                background-color: #2C2C2C;
                color: white;
                selection-background-color: #444444;
            }
            """
        )

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        # Filter bar
        filter_frame = QFrame()
        filter_frame.setStyleSheet("QFrame { background-color: #252525; border-radius: 8px; }")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(20, 16, 20, 16)
        filter_layout.setSpacing(20)

        self.combo_ethnicity = self._create_filter_combo(filter_layout, "Ethnicity", self.on_ethnicity_changed)
        self.combo_profile = self._create_filter_combo(filter_layout, "Profile", self.on_profile_changed)
        self.combo_age = self._create_filter_combo(filter_layout, "Age", self.on_age_changed)
        self.combo_baseline = self._create_filter_combo(filter_layout, "Baseline (D)", self.on_baseline_changed)
        self.combo_treatment = self._create_filter_combo(filter_layout, "Treatment", self.on_treatment_changed)

        filter_layout.addStretch()
        main_layout.addWidget(filter_frame)

        # Split content
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(self.canvas, stretch=7)

        self.data_panel = DataPanel()
        content_layout.addWidget(self.data_panel, stretch=3)

    def _create_filter_combo(self, parent_layout: QHBoxLayout, label: str, slot) -> QComboBox:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #B0B0B0; font-size: 12px;")
        layout.addWidget(lbl)

        combo = QComboBox()
        combo.currentIndexChanged.connect(slot)
        layout.addWidget(combo)
        parent_layout.addWidget(container)
        return combo

    def _set_combo_items(self, combo: QComboBox, items: List[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if current in items:
            combo.setCurrentText(current)
        elif items:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _populate_initial_filters(self) -> None:
        if self.data_index.is_empty():
            self.canvas.show_empty("No data available.")
            self.data_panel.reset()
            return
        self._set_combo_items(self.combo_ethnicity, self.data_index.ethnicities())
        self.on_ethnicity_changed()

    def on_ethnicity_changed(self, *args) -> None:
        ethnicity = self.combo_ethnicity.currentText()
        profiles = self.data_index.profiles(ethnicity)
        self._set_combo_items(self.combo_profile, profiles)
        self.on_profile_changed()

    def on_profile_changed(self, *args) -> None:
        ethnicity = self.combo_ethnicity.currentText()
        profile = self.combo_profile.currentText()
        ages = self.data_index.ages(ethnicity, profile)
        self._set_combo_items(self.combo_age, ages)
        self.on_age_changed()

    def on_age_changed(self, *args) -> None:
        ethnicity = self.combo_ethnicity.currentText()
        profile = self.combo_profile.currentText()
        age = self.combo_age.currentText()
        baselines = self.data_index.baselines(ethnicity, profile, age)
        self._set_combo_items(self.combo_baseline, baselines)
        self.on_baseline_changed()

    def on_baseline_changed(self, *args) -> None:
        self._update_treatment_options()
        self.update_display()

    def on_treatment_changed(self, *args) -> None:
        self.update_display()

    def _update_treatment_options(self) -> None:
        record = self._current_record()
        treatments: List[str] = []
        if record:
            for key in sorted(record.keys()):
                if key.lower() == "natural":
                    continue
                treatments.append(key)

        if treatments:
            self.combo_treatment.setEnabled(True)
            self._set_combo_items(self.combo_treatment, treatments)
        else:
            self.combo_treatment.setEnabled(False)
            self._set_combo_items(self.combo_treatment, ["No Interventions"])

    def _current_record(self) -> Optional[Dict[str, Any]]:
        ethnicity = self.combo_ethnicity.currentText()
        profile = self.combo_profile.currentText()
        age = self.combo_age.currentText()
        baseline = self.combo_baseline.currentText()
        if not all([ethnicity, profile, age, baseline]):
            return None
        return self.data_index.get_record(ethnicity, profile, age, baseline)

    @staticmethod
    def _find_curve(record: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        if label in record:
            return record[label]
        for key, value in record.items():
            if key.lower() == label.lower():
                return value
        return None

    def update_display(self) -> None:
        record = self._current_record()
        if not record:
            self.canvas.show_empty("No matching record.")
            self.data_panel.reset()
            return

        natural_curve = self._normalize_curve(self._find_curve(record, "Natural"))

        treatment_curve = None
        treatment_name = None
        if self.combo_treatment.isEnabled():
            treatment_name = self.combo_treatment.currentText()
            treatment_curve = self._normalize_curve(self._find_curve(record, treatment_name))

        self.canvas.plot_curves(natural_curve, treatment_curve, treatment_name)
        self.data_panel.update_metrics(natural_curve or {}, treatment_curve)

    @staticmethod
    def _normalize_curve(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[float]]]:
        if not entry:
            return None

        timeline = entry.get("timeline") or []
        if not timeline:
            return None

        mean = entry.get("mean") or entry.get("managed") or entry.get("natural")
        if mean is None:
            return None

        upper = entry.get("upper") or entry.get("upper_bound") or entry.get("lower") or mean
        lower = entry.get("lower") or entry.get("lower_bound") or entry.get("upper") or mean

        return {
            "timeline": list(timeline),
            "mean": list(mean),
            "upper": list(upper),
            "lower": list(lower),
        }


if __name__ == "__main__":
    db_override = _extract_db_override(sys.argv)
    app = QApplication(sys.argv)
    window = MyopiaPredictionApp(db_override=db_override)
    window.show()
    sys.exit(app.exec())
