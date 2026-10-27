"""
Baseline comparison and feasibility-classifier check for the LPG recovery surrogate.

Run from the repo root:  python 03_baselines_and_feasibility_check.py
Uses the same 70/15/15 split (random_state=42) as notebook 02, so results
compare directly with the PyTorch surrogate's test metrics.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

FEATURES = ["Cooler_Outlet_Temperature_C", "Compressor_Outlet_Pressure_bar"]
TARGETS = [
    "Compressor_Power_kW", "Cooler_Duty_kW",
    "Deethanizer_Condenser_kW", "Deethanizer_Reboiler_kW",
    "Debutanizer_Condenser_kW", "Debutanizer_Reboiler_kW",
    "Pure_LPG_Recovery_kgh",
]


def regression_baselines(clean_csv="data/LPG_Energy_Data_Clean.csv"):
    df = pd.read_csv(clean_csv)
    X, y = df[FEATURES].values, df[TARGETS].values
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
    _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
    near_edge = (X_te[:, 0] >= 32) & (X_te[:, 1] <= 26.5)

    models = {f"Polynomial deg {d}": make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())
              for d in range(2, 7)}
    rows = []
    for name, m in models.items():
        rows.append(score(name, m.fit(X_tr, y_tr).predict(X_te), y_te, near_edge))

    sx, sy = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr)
    gp = GaussianProcessRegressor(C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-6), random_state=0)
    gp.fit(sx.transform(X_tr), sy.transform(y_tr))
    rows.append(score("Gaussian process (RBF)", sy.inverse_transform(gp.predict(sx.transform(X_te))), y_te, near_edge))

    print(f"Test points: {len(X_te)} (near failure edge: {near_edge.sum()})")
    print(pd.DataFrame(rows).to_string(index=False))
    print("PyTorch surrogate (notebook 02): worst-target R2 0.999937, LPG R2 0.999998, LPG RMSE 0.045 kg/h")


def score(name, pred, y_te, near_edge):
    r2 = [r2_score(y_te[:, i], pred[:, i]) for i in range(len(TARGETS))]
    e_true, e_pred = y_te[:, :6].sum(1), pred[:, :6].sum(1)
    e_err = np.abs(e_pred - e_true) / e_true * 100
    l_err = np.abs(pred[:, 6] - y_te[:, 6]) / np.maximum(y_te[:, 6], 1e-3) * 100
    return {
        "Model": name,
        "Worst-target R2": round(min(r2), 6),
        "LPG RMSE (kg/h)": round(np.sqrt(mean_squared_error(y_te[:, 6], pred[:, 6])), 3),
        "Energy err max (%)": round(e_err.max(), 2),
        "Near-edge LPG err max (%)": round(l_err[near_edge].max(), 1),
    }


def feasibility_check(raw_csv="data/LPG_Energy_Data.csv"):
    df = pd.read_csv(raw_csv)
    X = df[FEATURES].values
    fail = (df["Solve_Status"] != "Success").values
    p_fail = cross_val_predict(
        RandomForestClassifier(n_estimators=100, random_state=42), X, (~fail).astype(int),
        cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba",
    )[:, 0]
    print(f"\nFeasibility classifier, 5-fold cross-validation ({fail.sum()} failures, {(~fail).sum()} converged)")
    print("Threshold P(fail)  Failures caught  False alarms")
    for t in [0.5, 0.3, 0.2, 0.1]:
        flag = p_fail >= t
        print(f"{t:>17}  {(flag & fail).sum():>4} ({(flag & fail).sum() / fail.sum():.1%})"
              f"  {(flag & ~fail).sum():>4} ({(flag & ~fail).sum() / (~fail).sum():.1%})")


if __name__ == "__main__":
    regression_baselines()
    feasibility_check()
