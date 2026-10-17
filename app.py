import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(
    page_title="LPG Recovery Process Optimization Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 600;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    </style>
""", unsafe_allow_html=True)

class FullPlantSurrogateNet(nn.Module):
    def __init__(self, in_features=2, out_features=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)

BASE_DIR = r"C:\Users\HomePC\LPG_Optimization_Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_system_artifacts():
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))

    model = FullPlantSurrogateNet()
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "full_plant_surrogate_best.pth"), map_location="cpu")
    )
    model.eval()

    df_all = pd.read_csv(os.path.join(DATA_DIR, "LPG_Energy_Data.csv"))
    df_clean = pd.read_csv(os.path.join(DATA_DIR, "LPG_Energy_Data_Clean.csv"))

    clf_path = os.path.join(MODEL_DIR, "feasibility_classifier.pkl")
    if os.path.exists(clf_path):
        clf_feas = joblib.load(clf_path)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf_feas = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_feas.fit(
            df_all[["Cooler_Outlet_Temperature_C", "Compressor_Outlet_Pressure_bar"]].values,
            (df_all["Solve_Status"] == "Success").astype(int).values
        )

    return model, scaler_X, scaler_y, clf_feas, df_all, df_clean

model, scaler_X, scaler_y, clf_feas, df_all, df_clean = load_system_artifacts()

@st.cache_data
def evaluate_dense_grid(w_comp, w_deeth_c, w_debut_c, w_reb):
    t_vals = np.linspace(15.0, 45.0, 200)
    p_vals = np.linspace(15.0, 35.0, 200)
    T_mesh, P_mesh = np.meshgrid(t_vals, p_vals)
    X_grid = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])

    feas_mask = clf_feas.predict(X_grid) == 1
    X_feas = X_grid[feas_mask]

    t0 = time.perf_counter()
    with torch.no_grad():
        X_s = scaler_X.transform(X_feas)
        preds_s = model(torch.tensor(X_s, dtype=torch.float32)).numpy()
        preds = scaler_y.inverse_transform(preds_s)
    eval_ms = (time.perf_counter() - t0) * 1000.0

    T_c = X_feas[:, 0]
    P_b = X_feas[:, 1]

    comp_kw    = preds[:, 0]
    cool_kw    = preds[:, 1]
    deeth_c_kw = preds[:, 2]
    deeth_r_kw = preds[:, 3]
    debut_c_kw = preds[:, 4]
    debut_r_kw = preds[:, 5]
    lpg_kgh    = preds[:, 6]

    tot_thermal_kw = comp_kw + cool_kw + deeth_c_kw + deeth_r_kw + debut_c_kw + debut_r_kw
    sec_thermal = tot_thermal_kw / np.maximum(lpg_kgh, 1e-3)

    cop_chiller = np.clip(273.15 + T_c, 280, 320) / (45.0 - T_c + 15.0) * 0.45
    equiv_elec_kw = (
        w_comp * comp_kw +
        (1.0 / cop_chiller) * cool_kw +
        w_deeth_c * deeth_c_kw +
        w_debut_c * debut_c_kw +
        w_reb * (deeth_r_kw + debut_r_kw)
    )
    sec_elec = equiv_elec_kw / np.maximum(lpg_kgh, 1e-3)

    grid_df = pd.DataFrame({
        "Temp_C": T_c,
        "Press_bar": P_b,
        "Compressor_kW": comp_kw,
        "Cooler_kW": cool_kw,
        "Deeth_Cond_kW": deeth_c_kw,
        "Deeth_Reb_kW": deeth_r_kw,
        "Debut_Cond_kW": debut_c_kw,
        "Debut_Reb_kW": debut_r_kw,
        "Total_Thermal_kW": tot_thermal_kw,
        "Equiv_Elec_kWe": equiv_elec_kw,
        "LPG_Recovery_kgh": lpg_kgh,
        "SEC_Thermal_kWh_kg": sec_thermal,
        "SEC_Elec_kWhe_kg": sec_elec
    })
    return grid_df, eval_ms

st.sidebar.markdown("### Process Operating Controls")
st.sidebar.caption("Manual Setpoint Evaluation (Point Simulation)")

input_temp = st.sidebar.slider(
    "Feed Cooler Outlet Temperature (°C)",
    min_value=15.0, max_value=45.0, value=16.22, step=0.10
)
input_press = st.sidebar.slider(
    "Compressor Outlet Pressure (bar)",
    min_value=15.0, max_value=35.0, value=30.88, step=0.10
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Optimization Constraints")
target_min_lpg = st.sidebar.slider(
    "Minimum LPG Production Target (kg/h)",
    min_value=45.0, max_value=132.0, value=120.0, step=1.0
)
opt_objective = st.sidebar.selectbox(
    "Objective Function",
    [
        "Minimize Total Plant Energy (kW)",
        "Minimize Exergy-Weighted Power (kWe)",
        "Minimize Specific Energy (kWh/kg)"
    ]
)

st.sidebar.markdown("---")
with st.sidebar.expander("Utility Exergy / Quality Factors", expanded=False):
    w_comp = st.number_input("Compressor Shaft Factor (kWe/kW)", value=1.00, step=0.05)
    w_deeth_c = st.number_input("Deethanizer Condenser Factor", value=0.33, step=0.02)
    w_debut_c = st.number_input("Debutanizer Condenser Factor", value=0.08, step=0.01)
    w_reb = st.number_input("Reboiler Steam Exergy Factor", value=0.35, step=0.02)

grid_df, grid_eval_ms = evaluate_dense_grid(w_comp, w_deeth_c, w_debut_c, w_reb)

pt_arr = np.array([[input_temp, input_press]])
is_feasible = bool(clf_feas.predict(pt_arr)[0] == 1)

with torch.no_grad():
    pt_s = scaler_X.transform(pt_arr)
    pt_pred = scaler_y.inverse_transform(model(torch.tensor(pt_s, dtype=torch.float32)).numpy())[0]

pt_comp, pt_cool, pt_dc, pt_dr, pt_bc, pt_br, pt_lpg = pt_pred
pt_tot_kw = pt_comp + pt_cool + pt_dc + pt_dr + pt_bc + pt_br
pt_sec = pt_tot_kw / max(pt_lpg, 1e-3)
pt_cop = np.clip(273.15 + input_temp, 280, 320) / (45.0 - input_temp + 15.0) * 0.45
pt_kwe = w_comp * pt_comp + (1.0 / pt_cop) * pt_cool + w_deeth_c * pt_dc + w_debut_c * pt_bc + w_reb * (pt_dr + pt_br)

# Finding optimal point under user constraint
feasible_sub = grid_df[grid_df["LPG_Recovery_kgh"] >= target_min_lpg]
if opt_objective == "Minimize Total Plant Energy (kW)":
    opt_row = feasible_sub.loc[feasible_sub["Total_Thermal_kW"].idxmin()]
elif opt_objective == "Minimize Exergy-Weighted Power (kWe)":
    opt_row = feasible_sub.loc[feasible_sub["Equiv_Elec_kWe"].idxmin()]
else:
    opt_row = feasible_sub.loc[feasible_sub["SEC_Thermal_kWh_kg"].idxmin()]

st.title("Associated Gas LPG Recovery: Surrogate Optimization Console")
st.caption(
    f"Thermodynamic Baseline: Peng-Robinson EoS (DWSIM) | Surrogate Architecture: 7-Output SiLU Neural Network | "
    f"40,000-State Grid Evaluated in {grid_eval_ms:.1f} ms"
)

if not is_feasible:
    st.error(
        f"THERMODYNAMIC CONVERGENCE WARNING: Operating state ({input_temp:.2f} °C, {input_press:.2f} bar) lies inside "
        f"the Deethanizer (SCOL-1) distillate dry-out zone. Reduce cooler temperature or increase compressor discharge pressure."
    )

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Thermodynamic Status", "Converged" if is_feasible else "Column Dry-Out")
col2.metric("Pure LPG Yield", f"{pt_lpg:.2f} kg/h" if is_feasible else "N/A")
col3.metric("Total Plant Load", f"{pt_tot_kw:.2f} kW" if is_feasible else "N/A")
col4.metric("Equivalent Electrical", f"{pt_kwe:.2f} kWe" if is_feasible else "N/A")
col5.metric("Specific Energy (SEC)", f"{pt_sec:.4f} kWh/kg" if is_feasible else "N/A")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "1. Unit Duty Breakdown & Setpoint Comparison",
    "2. Pareto Frontier & Operating Envelope",
    "3. Ground-Truth DWSIM Validation & Benchmarks"
])

with tab1:
    left_col, right_col = st.columns([1.15, 1.0])

    with left_col:
        st.subheader("Current Manual Setpoint vs. Constrained Optimum")
        comp_table = pd.DataFrame({
            "Process Parameter": [
                "Cooler Outlet Temperature (°C)",
                "Compressor Discharge Pressure (bar)",
                "Pure LPG Recovery Rate (kg/h)",
                "Total Plant Energy Demand (kW)",
                "Equivalent Electrical Work (kWe)",
                "Specific Energy Consumption (kWh/kg)",
                "Compressor Shaft Power - E1 (kW)",
                "Feed Chiller Duty - E2 (kW)",
                "Deethanizer Total Duty - SCOL-1 (kW)",
                "Debutanizer Total Duty - SCOL-2 (kW)"
            ],
            "Manual Setpoint": [
                f"{input_temp:.2f}",
                f"{input_press:.2f}",
                f"{pt_lpg:.2f}" if is_feasible else "Infeasible",
                f"{pt_tot_kw:.2f}" if is_feasible else "Infeasible",
                f"{pt_kwe:.2f}" if is_feasible else "Infeasible",
                f"{pt_sec:.4f}" if is_feasible else "Infeasible",
                f"{pt_comp:.2f}" if is_feasible else "-",
                f"{pt_cool:.2f}" if is_feasible else "-",
                f"{(pt_dc + pt_dr):.2f}" if is_feasible else "-",
                f"{(pt_bc + pt_br):.2f}" if is_feasible else "-"
            ],
            f"Recommended Optimum (>= {target_min_lpg:.0f} kg/h)": [
                f"{opt_row['Temp_C']:.2f}",
                f"{opt_row['Press_bar']:.2f}",
                f"{opt_row['LPG_Recovery_kgh']:.2f}",
                f"{opt_row['Total_Thermal_kW']:.2f}",
                f"{opt_row['Equiv_Elec_kWe']:.2f}",
                f"{opt_row['SEC_Thermal_kWh_kg']:.4f}",
                f"{opt_row['Compressor_kW']:.2f}",
                f"{opt_row['Cooler_kW']:.2f}",
                f"{(opt_row['Deeth_Cond_kW'] + opt_row['Deeth_Reb_kW']):.2f}",
                f"{(opt_row['Debut_Cond_kW'] + opt_row['Debut_Reb_kW']):.2f}"
            ]
        })
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Unit-Operation Energy Distribution")
        fig_bar, ax_bar = plt.subplots(figsize=(6.5, 4.2))
        units = [
            "Compressor (E1)", "Feed Cooler (E2)",
            "Deeth Cond (QC)", "Deeth Reb (QR)",
            "Debut Cond (E5)", "Debut Reb (E6)"
        ]
        opt_duties = [
            opt_row["Compressor_kW"], opt_row["Cooler_kW"],
            opt_row["Deeth_Cond_kW"], opt_row["Deeth_Reb_kW"],
            opt_row["Debut_Cond_kW"], opt_row["Debut_Reb_kW"]
        ]
        y_pos = np.arange(len(units))

        if is_feasible:
            man_duties = [pt_comp, pt_cool, pt_dc, pt_dr, pt_bc, pt_br]
            ax_bar.barh(y_pos - 0.18, man_duties, height=0.35, label="Manual Setpoint", color="#4a6fe3")
            ax_bar.barh(y_pos + 0.18, opt_duties, height=0.35, label="Optimal Setpoint", color="#1b9e77")
        else:
            ax_bar.barh(y_pos, opt_duties, height=0.45, label="Optimal Setpoint", color="#1b9e77")

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(units, fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Unit Duty (kW)", fontsize=9)
        ax_bar.set_title("Sub-System Duty Breakdown (kW)", fontsize=10, fontweight="bold")
        ax_bar.legend(fontsize=8.5)
        ax_bar.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_bar)

with tab2:
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.subheader("Multi-Objective Pareto Frontier")
        bins = np.linspace(40, 132.5, 75)
        grid_df["LPG_Bin"] = pd.cut(grid_df["LPG_Recovery_kgh"], bins)
        pareto_df = grid_df.loc[grid_df.groupby("LPG_Bin", observed=True)["Total_Thermal_kW"].idxmin()]

        fig_p, ax_p = plt.subplots(figsize=(6.5, 4.6))
        sc_p = ax_p.scatter(
            grid_df["LPG_Recovery_kgh"], grid_df["Total_Thermal_kW"],
            c=grid_df["Temp_C"], cmap="viridis", s=4, alpha=0.25
        )
        ax_p.plot(
            pareto_df["LPG_Recovery_kgh"], pareto_df["Total_Thermal_kW"],
            color="#d95f02", lw=2.2, label="Pareto Frontier (Min Energy)"
        )
        if is_feasible:
            ax_p.scatter(
                pt_lpg, pt_tot_kw, color="#e7298a", edgecolors="black",
                s=90, zorder=5, label="Manual Setpoint"
            )
        ax_p.scatter(
            opt_row["LPG_Recovery_kgh"], opt_row["Total_Thermal_kW"],
            color="#66a61e", edgecolors="black", marker="D",
            s=90, zorder=6, label="Constrained Optimum"
        )
        plt.colorbar(sc_p, ax=ax_p, label="Cooler Outlet Temp (°C)")
        ax_p.set_xlabel("Pure LPG Recovery Yield (kg/h)")
        ax_p.set_ylabel("Total Plant Energy Consumption (kW)")
        ax_p.set_title("Plant Energy vs. LPG Recovery Trade-Off", fontsize=10, fontweight="bold")
        ax_p.legend(fontsize=8)
        ax_p.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_p)

    with p_col2:
        st.subheader("Specific Energy Contour & Failure Boundary")
        fig_c, ax_c = plt.subplots(figsize=(6.5, 4.6))

        failed_pts = df_all[df_all["Solve_Status"] != "Success"]
        ax_c.scatter(
            failed_pts["Cooler_Outlet_Temperature_C"],
            failed_pts["Compressor_Outlet_Pressure_bar"],
            color="#cccccc", s=12, label="SCOL-1 Non-Convergence Zone"
        )
        sc_c = ax_c.scatter(
            df_clean["Cooler_Outlet_Temperature_C"],
            df_clean["Compressor_Outlet_Pressure_bar"],
            c=np.clip(df_clean["Specific_Energy_kWh_per_kg"], 1.74, 4.5),
            cmap="YlGnBu_r", s=16
        )
        ax_c.scatter(
            input_temp, input_press, color="#e7298a", edgecolors="black",
            s=90, zorder=5, label="Manual Setpoint"
        )
        ax_c.scatter(
            opt_row["Temp_C"], opt_row["Press_bar"], color="#66a61e",
            edgecolors="black", marker="D", s=90, zorder=6, label="Constrained Optimum"
        )
        plt.colorbar(sc_c, ax=ax_c, label="Specific Energy (kWh/kg, clipped at 4.5)")
        ax_c.set_xlabel("Cooler Outlet Temperature (°C)")
        ax_c.set_ylabel("Compressor Outlet Pressure (bar)")
        ax_c.set_title("Operating Envelope & Specific Energy Valley", fontsize=10, fontweight="bold")
        ax_c.legend(loc="lower left", fontsize=8)
        ax_c.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_c)

with tab3:
    st.subheader("Closed-Loop Verification Against Rigorous DWSIM Solver")
    val_summary = pd.DataFrame({
        "Production Constraint": [">= 100 kg/h", ">= 110 kg/h", ">= 115 kg/h", ">= 120 kg/h", ">= 125 kg/h", ">= 130 kg/h"],
        "Optimal Temp (°C)": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        "Optimal Pressure (bar)": [25.65, 28.07, 29.47, 30.88, 32.39, 33.99],
        "Surrogate LPG (kg/h)": [100.20, 110.03, 115.31, 120.25, 125.19, 130.07],
        "DWSIM True LPG (kg/h)": [100.22, 110.03, 115.25, 120.19, 125.14, 130.05],
        "LPG Relative Error (%)": ["0.024%", "0.003%", "0.052%", "0.052%", "0.036%", "0.018%"],
        "Surrogate Energy (kW)": [175.16, 192.29, 201.81, 211.00, 220.49, 230.23],
        "DWSIM True Energy (kW)": [175.16, 192.32, 201.76, 210.95, 220.46, 230.18],
        "Energy Relative Error (%)": ["0.002%", "0.017%", "0.024%", "0.021%", "0.015%", "0.024%"]
    })
    st.dataframe(val_summary, use_container_width=True, hide_index=True)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### Computational Latency Benchmark")
        bench_df = pd.DataFrame({
            "Solver Method": ["DWSIM Rigorous EoS Solver", "PyTorch Surrogate Digital Twin", "Net Speedup Factor"],
            "Single State Latency": ["271.26 ms", "0.01588 ms (15.88 µs)", "17,085x faster"],
            "40,000-Point Grid Search": ["~3.01 hours", "583.40 ms", "Real-Time Capable"]
        })
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    with b_col2:
        st.markdown("#### Off-Grid Generalization Summary (20 Continuous Points)")
        gen_df = pd.DataFrame({
            "Evaluation Domain": ["Interior Envelope (n = 12)", "Near Failure Wedge (n = 8)"],
            "Mean LPG Error (%)": ["0.256%", "1.029% (0.14 kg/h abs at pinch)"],
            "Mean Energy Error (%)": ["0.025%", "0.032%"],
            "Max Energy Error (%)": ["0.112%", "0.055%"]
        })
        st.dataframe(gen_df, use_container_width=True, hide_index=True)
