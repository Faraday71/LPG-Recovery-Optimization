"""
Core logic of the LPG recovery digital twin.

Used by the dashboard (app.py). It mirrors notebook 03 so the dashboard shows the
same results: same sensors, same noise seeds, same detection and alarm settings.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Loading the twin
# ----------------------------------------------------------------------------

class Twin:
    """The trained models plus the energy and limit calculations built on them."""

    def __init__(self, model_dir, prefer="nn"):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.inputs, self.targets = self.cfg["inputs"], self.cfg["targets"]
        self.energy = self.cfg["energy"]
        self.margin_limit = self.cfg.get("margin_bar", 1.5)
        self.feas = joblib.load(os.path.join(model_dir, "feasibility_boundary.pkl"))
        self.kind = None
        if prefer == "nn":
            try:
                self._load_nn(model_dir)
            except Exception:
                self.kind = None
        if self.kind is None:
            self._load_gp(model_dir)
        self._build_boundary()

    def _load_nn(self, model_dir):
        import torch
        import torch.nn as nn
        ck = torch.load(os.path.join(model_dir, "nn_surrogate.pt"), map_location="cpu")
        scalers = joblib.load(os.path.join(model_dir, "nn_scalers.pkl"))
        h = ck["hidden"]
        net = nn.Sequential(nn.Linear(ck["n_in"], h[0]), nn.SiLU(), nn.Linear(h[0], h[1]), nn.SiLU(),
                            nn.Linear(h[1], h[2]), nn.SiLU(), nn.Linear(h[2], ck["n_out"]))
        net.load_state_dict({k.replace("net.", "", 1): v for k, v in ck["state_dict"].items()})
        net.eval()
        self._torch, self._net, self._sc = torch, net, scalers
        self.kind = "Neural network (PyTorch)"

    def _load_gp(self, model_dir):
        self._gp = joblib.load(os.path.join(model_dir, "gp_surrogate.pkl"))
        self.kind = "Gaussian process"

    def predict(self, X):
        """X columns: chiller outlet T (C), compressor pressure (bar), feed flow (kg/h), feed T (C)."""
        X = np.asarray(X, dtype=float)
        if self.kind.startswith("Neural"):
            with self._torch.no_grad():
                y = self._net(self._torch.tensor(self._sc["x_scaler"].transform(X), dtype=self._torch.float32)).numpy()
            y = self._sc["y_scaler"].inverse_transform(y)
        else:
            g = self._gp
            Xs = g["x_scaler"].transform(X)
            y = np.vstack([g["gp"].predict(Xs[i:i + 500]) for i in range(0, len(Xs), 500)])
            y = g["y_scaler"].inverse_transform(y)
        return pd.DataFrame(y, columns=self.targets)

    # --- operating limit -------------------------------------------------
    def _build_boundary(self):
        P = np.linspace(15, 35, 801)
        self.bound_t = np.round(np.arange(15.0, 45.0001, 0.05), 3)
        pmin = []
        for t in self.bound_t:
            pf = self.feas.predict_proba(np.c_[np.full_like(P, t), P])[:, 1]
            pmin.append(P[np.argmax(pf < 0.5)] if pf[0] >= 0.5 else np.nan)
        self.bound_p = np.array(pmin)

    def p_min(self, t_c):
        """Lowest safe compressor pressure at this chiller temperature (NaN = no limit in 15-35 bar)."""
        t = np.clip(np.asarray(t_c, dtype=float), 15, 45)
        has = ~np.isnan(self.bound_p)
        first = self.bound_t[has][0] if has.any() else 99
        out = np.interp(t, self.bound_t[has], self.bound_p[has]) if has.any() else np.full_like(t, np.nan)
        return np.where(t < first, np.nan, out)

    def margin(self, t_c, p_bar):
        """Distance above the operating limit in bar. 99 means there is no limit at this temperature."""
        pm = self.p_min(t_c)
        return np.where(np.isnan(pm), 99.0, np.asarray(p_bar, dtype=float) - pm)

    # --- energy ----------------------------------------------------------
    def cop(self, t_process_c):
        E = self.energy
        te = np.asarray(t_process_c, dtype=float) - E["APPROACH_C"] + 273.15
        tc = E["T_COND_REF_C"] + 273.15
        return E["CARNOT_FRACTION"] * te / (tc - te)

    def fuel(self, d, chiller_t):
        """Electricity, heat and fuel (all kW) from a table of unit duties."""
        E = self.energy
        elec = (d["Compressor_Power_kW"]
                + d["Chiller_Duty_kW"] / self.cop(chiller_t)
                + d["Deethanizer_Condenser_kW"] / self.cop(d["Deethanizer_Condenser_T_C"])
                + E["FAN_FRACTION"] * (d["Air_Cooler_Duty_kW"] + d["Debutanizer_Condenser_kW"]))
        heat = d["Deethanizer_Reboiler_kW"] + d["Debutanizer_Reboiler_kW"]
        return elec, heat, elec / E["ETA_POWER"] + heat / E["ETA_HEATER"]

    def best_setpoint(self, feed_kgh, feed_t, lpg_target):
        """Lowest-fuel safe setpoint that still makes the target, and the usual 35 bar way of doing it."""
        T, P = np.meshgrid(np.arange(15, 45.001, 0.25), np.arange(15, 35.001, 0.125))
        Q = np.c_[T.ravel(), P.ravel(), np.full(T.size, feed_kgh), np.full(T.size, feed_t)]
        pr = self.predict(Q)
        _, _, fuel = self.fuel(pr, Q[:, 0])
        g = pd.DataFrame({"T": Q[:, 0], "P": Q[:, 1], "LPG": pr["LPG_kgh"].values, "Fuel": np.asarray(fuel)})
        g["Safe"] = self.margin(g["T"].values, g["P"].values) >= self.margin_limit
        ok = g[g.Safe & (g.LPG >= lpg_target)]
        if ok.empty:
            return None, None
        best = ok.loc[ok.Fuel.idxmin()]
        usual = ok[np.isclose(ok.P, 35.0)]
        usual = usual.loc[usual["T"].idxmax()] if not usual.empty else None
        return best, usual


# ----------------------------------------------------------------------------
# Sensors (same as notebook 03)
# ----------------------------------------------------------------------------

SENSORS = {
    "FT-001": ("feed_kgh", "rel", 0.010, "kg/h", "Feed gas flow"),
    "TT-001": ("feed_t_c", "abs", 0.50, "C", "Feed gas temperature"),
    "PT-101": ("sp_pressure_bar", "abs", 0.05, "bar", "Compressor discharge pressure"),
    "TT-101": ("Compressor_Discharge_T_C", "abs", 0.50, "C", "Compressor discharge temperature"),
    "JI-101": ("Compressor_Power_kW", "rel", 0.005, "kW", "Compressor power"),
    "TT-103": ("Air_Cooled_T_C", "abs", 0.50, "C", "Air cooler outlet temperature"),
    "TT-102": ("chiller_actual_c", "abs", 0.30, "C", "Chiller outlet temperature"),
    "JI-102": ("Chiller_Duty_kW", "rel", 0.020, "kW", "Chiller duty"),
    "FT-301": ("LPG_kgh", "rel", 0.010, "kg/h", "LPG product flow"),
}


def measure(truth, seed, noise_scale=1.0):
    r = np.random.default_rng(seed)
    m = pd.DataFrame({"time_min": truth.time_min.values, "hour": truth.hour.values})
    for tag, (col, kind, size, _, _) in SENSORS.items():
        x = truth[col].astype(float).values
        sd = size * noise_scale * (np.abs(x) if kind == "rel" else 1.0)
        m[tag] = x + r.normal(0, 1, len(x)) * sd
    m["PT-101"] += truth["pt_bias_bar"].values
    m["SP-TT102"] = truth["sp_chiller_c"].values
    m["SP-PT101"] = truth["sp_pressure_bar"].values
    m["plant_ok"] = truth["status"].values == "ok"
    return m


# ----------------------------------------------------------------------------
# Detection and alarms (same settings as notebook 03)
# ----------------------------------------------------------------------------

RESIDUAL_TAGS = {"JI-101": "Compressor_Power_kW", "TT-101": "Compressor_Discharge_T_C",
                 "JI-102": "Chiller_Duty_kW", "FT-301": "LPG_kgh"}
EWMA_LAMBDA, EWMA_LIMIT, CALIB_MIN, STEP_MIN = 0.05, 4.0, 120, 1
EWMA_SD = np.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA))

# Alarm rationalization (ISA-18.2): what it means and what to do
ALARMS = {
    "FEAS-HI":        dict(priority="High", unit="Deethanizer",
                           text="Very close to the operating limit",
                           cause="Chiller too warm or pressure too low for the deethanizer to get enough liquid.",
                           action="Act now: raise compressor pressure or lower the chiller setpoint."),
    "FEAS-ADV":       dict(priority="Low", unit="Deethanizer",
                           text="Getting close to the operating limit",
                           cause="Chiller too warm or pressure too low for the deethanizer to get enough liquid.",
                           action="Lower the chiller setpoint or raise pressure before the high alarm."),
    "TT-102 DEV":     dict(priority="Medium", unit="Chiller",
                           text="Chiller cannot hold its setpoint",
                           cause="Low refrigerant, condenser problem or chiller overload.",
                           action="Check the chiller. LPG output will drop while this lasts."),
    "JI-101 PERF":    dict(priority="Medium", unit="Compressor",
                           text="Compressor using more power than expected",
                           cause="Wear or fouling is reducing compressor efficiency.",
                           action="Plan an inspection or wash. Fuel use per kg of LPG is rising."),
    "JI-102 PERF":    dict(priority="Low", unit="Air cooler",
                           text="Chiller working harder than expected",
                           cause="Gas reaching the chiller is warmer than normal. Air cooler fouling or very hot weather.",
                           action="Inspect and clean air cooler AC-1."),
    "FT-301 LOW":     dict(priority="Medium", unit="Debutanizer",
                           text="LPG output lower than expected",
                           cause="Recovery is below what the twin expects for these conditions.",
                           action="Check the separator and the two columns."),
    "PT-101 SUSPECT": dict(priority="Medium", unit="Compressor",
                           text="Pressure reading looks wrong",
                           cause="Power, discharge temperature and LPG all read lower than PT-101 says they should.",
                           action="Check or recalibrate transmitter PT-101 before trusting its reading."),
}
ALARM_ORDER = ["FEAS-HI", "TT-102 DEV", "JI-101 PERF", "FT-301 LOW", "PT-101 SUSPECT", "FEAS-ADV", "JI-102 PERF"]
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}


def _ewma(x, lam=EWMA_LAMBDA):
    out, z = np.empty(len(x)), 0.0
    for i, v in enumerate(x):
        z = z if np.isnan(v) else lam * v + (1 - lam) * z
        out[i] = z
    return out


def _delayed(flag, minutes):
    out, run = np.zeros(len(flag), bool), 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        out[i] = run >= minutes / STEP_MIN
    return out


def _hysteresis(value, on, off):
    out, state = np.zeros(len(value), bool), False
    for i, v in enumerate(value):
        state = (v > on) if not state else (v > off)
        out[i] = state
    return out


def predictions(twin, meas):
    X = np.c_[meas["TT-102"], meas["PT-101"], meas["FT-001"], meas["TT-001"]]
    return twin.predict(X)


def commission(twin, truth0):
    """Learn the normal size of each residual from the first 2 hours of the healthy day."""
    m = measure(truth0, seed=100)
    pr = predictions(twin, m)
    keep = m.time_min.values < CALIB_MIN
    calib = {}
    for tag, col in RESIDUAL_TAGS.items():
        r = m[tag].values[keep] - pr[col].values[keep]
        calib[tag] = (float(np.nanmean(r)), float(np.nanstd(r)))
    return calib


def run_scenario(twin, truth, seed, calib):
    """Measurements, twin predictions, alarm states and plant KPIs for one scenario."""
    meas = measure(truth, seed)
    pred = predictions(twin, meas)
    ok = meas["plant_ok"].values
    z = {}
    for tag, col in RESIDUAL_TAGS.items():
        r = meas[tag].values - pred[col].values
        r[~ok] = np.nan
        bias, spread = calib[tag]
        z[tag] = _ewma((r - bias) / spread) / (EWMA_LIMIT * EWMA_SD)
    z = pd.DataFrame(z).fillna(0)

    margin = twin.margin(meas["TT-102"].values, meas["PT-101"].values)
    A = pd.DataFrame({"time_min": meas.time_min})
    A["FEAS-ADV"] = _delayed(_hysteresis(-margin, -1.5, -2.0), 2)
    A["FEAS-HI"] = _delayed(_hysteresis(-margin, -0.5, -1.0), 1)
    A["TT-102 DEV"] = _delayed(_hysteresis((meas["TT-102"] - meas["SP-TT102"]).values, 2.0, 1.5), 5)
    pt_suspect = (z["JI-101"] < -1) & (z["TT-101"] < -1) & (z["FT-301"] < -1)
    A["PT-101 SUSPECT"] = _delayed(pt_suspect.values, 10)
    A["JI-101 PERF"] = _delayed(((z["JI-101"] > 1) & (z["TT-101"] > 0)).values, 10)
    A["JI-102 PERF"] = _delayed((z["JI-102"] > 1).values, 10)
    A["FT-301 LOW"] = _delayed(((z["FT-301"] < -1) & ~pt_suspect).values, 10) & ~A["TT-102 DEV"].values
    A["PLANT TRIP"] = ~ok
    A["margin_bar"] = margin

    # KPIs: measured where the plant has a meter, twin estimate for the rest
    d = pred.copy()
    d["Compressor_Power_kW"] = meas["JI-101"].values
    d["Chiller_Duty_kW"] = meas["JI-102"].values
    elec, heat, fuel = twin.fuel(d, meas["TT-102"].values)
    lpg = meas["FT-301"].values
    kpi = pd.DataFrame({"fuel_kw": np.where(ok, fuel, np.nan), "elec_kw": np.where(ok, elec, np.nan),
                        "lpg_kgh": np.where(ok, lpg, np.nan)})
    kpi["fuel_per_kg"] = kpi.fuel_kw / kpi.lpg_kgh
    # what the same feed would cost if the plant were healthy and at its setpoints
    return dict(meas=meas, pred=pred, z=z, alarms=A, kpi=kpi)


def alarm_events(A):
    """Turn alarm on/off columns into a log of raised and cleared events."""
    rows = []
    for c in list(ALARMS) + ["PLANT TRIP"]:
        on = A[c].values.astype(bool)
        change = np.flatnonzero(np.diff(np.r_[False, on]).astype(bool))
        for i in change:
            rows.append({"time_min": int(A.time_min.iloc[i]), "alarm": c, "event": "Raised" if on[i] else "Cleared"})
    return pd.DataFrame(rows, columns=["time_min", "alarm", "event"]).sort_values("time_min").reset_index(drop=True)


def fmt_time(minutes):
    minutes = int(minutes)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
