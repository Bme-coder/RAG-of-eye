import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons
import pandas as pd
import seaborn as sns
import numpy as np

# --- 0. Config & Style ---
# Removed Chinese font settings to rely on standard English fonts (Arial/DejaVu Sans)
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'sans-serif' 

# --- 1. Data Loading (With English Mock Data) ---
def load_data():
    """Load config. Use English Mock data if file is missing."""
    config_path = "medical_config.json"
    
    if os.path.exists(config_path):
        print(f"✅ Loading local file: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("⚠️ Config not found. Using Mock Data for demo...")
        return {
            "BASE_RATES": {
                "Asian": {
                    "Female": {str(i): round(-0.9 + (i-6)*0.06, 2) for i in range(6,17)},
                    "Male":   {str(i): round(-0.8 + (i-6)*0.06, 2) for i in range(6,17)}
                },
                "Caucasian": {
                    "Female": {str(i): round(-0.6 + (i-6)*0.04, 2) for i in range(6,17)},
                    "Male":   {str(i): round(-0.5 + (i-6)*0.04, 2) for i in range(6,17)}
                }
            },
            "TREATMENTS": {
                "Atropine_Low": {"name": "Low-dose Atropine", "efficacy": 0.37},
                "Ortho_K": {"name": "Ortho-K (OK Lens)", "efficacy": 0.50},
                "Defocus_Glasses": {"name": "Defocus Glasses", "efficacy": 0.25}
            }
        }

CONFIG = load_data()

# --- 2. Main Dashboard Function ---
def run_dashboard():
    # Setup Figure
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("🏥 AI Knowledge Mining & Inference Pipeline Inspector", fontsize=18, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.15)

    # ==========================
    # Part A: Knowledge Matrix (Heatmap)
    # ==========================
    ax_heat = fig.add_subplot(gs[0, 0])
    
    base_rates = CONFIG.get("BASE_RATES", {})
    rows = []
    for eth in base_rates:
        for sex in base_rates[eth]:
            row = {"Group": f"{eth} {sex}"}
            for age in range(6, 17):
                val = base_rates[eth][sex].get(str(age), -0.5)
                row[str(age)] = float(val)
            rows.append(row)
    df_heat = pd.DataFrame(rows).set_index("Group")
    
    sns.heatmap(df_heat, annot=True, cmap="Reds_r", fmt=".2f", ax=ax_heat, cbar=False)
    ax_heat.set_title("1. Extracted Base Rates Matrix (D/Year)", fontsize=12, fontweight='bold')
    ax_heat.set_xlabel("Age")
    ax_heat.set_ylabel("")

    # ==========================
    # Part B: Treatment Efficacy (Bar Chart)
    # ==========================
    ax_bar = fig.add_subplot(gs[0, 1])
    
    treats = CONFIG.get("TREATMENTS", {})
    # Extract names (ensure English if using Mock, or whatever is in JSON)
    names = [t.get("name", k) for k, t in treats.items()]
    effs = [t.get("efficacy", 0) * 100 for t in treats.values()]
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    
    bars = ax_bar.bar(names, effs, color=colors[:len(names)], alpha=0.8)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_title("2. Extracted Treatment Efficacy (%)", fontsize=12, fontweight='bold')
    ax_bar.set_ylabel("Efficacy (%)")
    
    for bar in bars:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f"{bar.get_height():.0f}%", ha='center', fontsize=11, fontweight='bold')

    # ==========================
    # Part C: Interactive Inference Engine (Trace Table)
    # ==========================
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], width_ratios=[0.5, 0.5, 0.8, 4])
    
    # --- Control Panels ---
    ax_ctrl_eth = fig.add_subplot(gs_bottom[0])
    ax_ctrl_sex = fig.add_subplot(gs_bottom[1])
    ax_ctrl_trt = fig.add_subplot(gs_bottom[2])
    ax_table = fig.add_subplot(gs_bottom[3])
    ax_table.axis('off')

    # 1. Ethnicity Selector
    ax_ctrl_eth.set_facecolor('#f0f0f0')
    radio_eth = RadioButtons(ax_ctrl_eth, list(base_rates.keys()))
    ax_ctrl_eth.set_title("Ethnicity", fontsize=10, fontweight='bold')

    # 2. Gender Selector
    ax_ctrl_sex.set_facecolor('#f0f0f0')
    radio_sex = RadioButtons(ax_ctrl_sex, ["Female", "Male"])
    ax_ctrl_sex.set_title("Gender", fontsize=10, fontweight='bold')

    # 3. Treatment Selector
    ax_ctrl_trt.set_facecolor('#f0f0f0')
    treat_map = {v['name']: k for k, v in treats.items()}
    radio_trt = RadioButtons(ax_ctrl_trt, list(treat_map.keys()))
    ax_ctrl_trt.set_title("Treatment Plan", fontsize=10, fontweight='bold')

    # --- Logic: Update Table ---
    def update_trace(val=None):
        ax_table.clear()
        ax_table.axis('off')
        
        # Get selections
        sel_eth = radio_eth.value_selected
        sel_sex = radio_sex.value_selected
        sel_trt_name = radio_trt.value_selected
        sel_trt_key = treat_map[sel_trt_name]
        
        # Get params
        rates_dict = base_rates.get(sel_eth, {}).get(sel_sex, {})
        efficacy = treats[sel_trt_key]['efficacy']
        
        # Simulate Logic
        start_age = 8
        curr_diopter = -2.00
        
        table_data = []
        col_labels = ["Step", "Age", "Start D", "Extracted Rate", "Logic Factor", "Net Change", "End Prediction"]
        
        for i in range(5): # Simulate 5 years
            age = start_age + i
            rate = float(rates_dict.get(str(age), -0.5))
            
            # THE CORE FORMULA
            real_change = rate * (1 - efficacy)
            next_diopter = curr_diopter + real_change
            
            row = [
                f"T + {i}",
                f"{age} yo",
                f"{curr_diopter:.2f} D",
                f"⬇ {rate:.2f} D",
                f"× (1 - {efficacy:.2f})",
                f"{real_change:.2f} D",
                f"➜ {next_diopter:.2f} D"
            ]
            table_data.append(row)
            curr_diopter = next_diopter

        # Draw Table
        t = ax_table.table(cellText=table_data, colLabels=col_labels, 
                           loc='center', cellLoc='center', 
                           bbox=[0, 0.1, 1, 0.8])
        
        t.auto_set_font_size(False)
        t.set_fontsize(11)
        
        # Styling
        for (row, col), cell in t.get_celld().items():
            cell.set_edgecolor('#dcdcdc')
            if row == 0: # Header
                cell.set_facecolor('#404040')
                cell.set_text_props(color='white', weight='bold')
                cell.set_height(0.12)
            else:
                cell.set_height(0.1)
                if row % 2 == 0:
                    cell.set_facecolor('#f8f9fa')

        ax_table.set_title(f"3. Real-time Inference Engine Trace\nLogic: {sel_eth} {sel_sex} with {sel_trt_name}", fontsize=12, fontweight='bold')
        fig.canvas.draw_idle()

    # Bind Events
    radio_eth.on_clicked(update_trace)
    radio_sex.on_clicked(update_trace)
    radio_trt.on_clicked(update_trace)

    # Initial Draw
    update_trace()

    plt.show()

if __name__ == "__main__":
    run_dashboard()