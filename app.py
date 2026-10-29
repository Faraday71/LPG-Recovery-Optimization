"""
LPG Recovery Digital Twin - operator dashboard.

Run from the project folder:   streamlit run app.py

Replays plant scenarios generated in DWSIM (data/scenarios) and runs the twin on them
live: it predicts what every instrument should read, raises alarms when readings drift,
and lets you test setpoints before using them on the plant.

Screen layout follows ISA-101 ideas: grey background, no colour unless something is
abnormal, overview first and detail on request.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import twin_core as tc

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "v2"
SCEN_DIR = BASE / "data" / "scenarios"

SCENARIOS = {
    "S0_healthy": "Normal day",
    "S1_compressor_wear": "Compressor wear",
    "S2_aircooler_fouling": "Air cooler fouling",
    "S3_chiller_capacity": "Chiller losing capacity",
    "S4_pt101_drift": "Pressure sensor drift",
    "S5_unsafe_move": "Unsafe setpoint change",
}
SCREENS = ["Overview", "Alarms", "Trends", "Operating limits", "Try a setpoint", "About"]
SPEEDS = {"1 minute per step": 1, "5 minutes per step": 5, "15 minutes per step": 15}

# ISA-101 style palette: greys for normal, colour only for abnormal
INK, MUTED, LINE, PANEL, PAGE = "#1F1F1F", "#5F6368", "#9E9E9E", "#F3F3F3", "#E6E6E6"
EXPECT = "#33475B"                     # twin's expected value
PRIORITY_COLOUR = {"High": "#B3261E", "Medium": "#C77700", "Low": "#8C7A00"}

st.set_page_config(page_title="LPG Recovery Twin", layout="wide")

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.5rem; padding-bottom: 2rem;}}
  h1, h2, h3 {{font-weight: 600; color: {INK};}}
  .strip {{display:flex; gap:18px; align-items:center; padding:8px 12px; background:{PANEL};
           border:1px solid #CFCFCF; border-radius:4px; font-size:0.95rem; color:{INK};}}
  .strip b {{font-weight:600;}}
  .tag-abn {{display:inline-block; padding:1px 8px; border-radius:3px; color:white; font-weight:600;}}
  .cards {{display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; margin:10px 0 14px 0;}}
  .card {{background:{PANEL}; border:1px solid #CFCFCF; border-radius:4px; padding:10px 12px;}}
  .card .label {{font-size:0.78rem; color:{MUTED};}}
  .card .value {{font-size:1.45rem; font-weight:600; color:{INK}; margin-top:2px;}}
  .card .u {{font-size:0.8rem; color:{MUTED}; font-weight:400;}}
  .card .note {{font-size:0.75rem; color:{MUTED}; margin-top:2px;}}
  .flow {{display:flex; align-items:stretch; gap:0; overflow-x:auto; padding:6px 0 4px 0;}}
  .unit {{background:{PANEL}; border:1px solid #BDBDBD; border-radius:4px; padding:8px 8px; flex:1 1 0; min-width:0;}}
  .unit .name {{font-weight:600; font-size:0.9rem; color:{INK}; margin-bottom:4px;}}
  .unit .row {{font-size:0.8rem; color:{INK}; display:flex; justify-content:space-between; gap:8px;
               font-variant-numeric: tabular-nums;}}
  .unit .row span:first-child {{color:{MUTED};}}
  .unit .msg {{font-size:0.75rem; font-weight:600; margin-top:5px;}}
  .arrow {{align-self:center; color:{LINE}; font-size:1.2rem; padding:0 3px;}}
  .quiet {{color:{MUTED}; font-size:0.85rem;}}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": "#F8F8F8", "axes.edgecolor": "#B0B0B0",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#DADADA",
    "axes.grid": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "600",
    "legend.frameon": False, "legend.fontsize": 8,
})


# ----------------------------------------------------------------------------
# Data and models
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the twin models...")
def load_twin():
    return tc.Twin(str(MODEL_DIR), prefer="nn")


@st.cache_resource(show_spinner="Running the twin on this scenario...")
def load_scenario(name):
    twin = load_twin()
    truth0 = pd.read_csv(SCEN_DIR / "S0_healthy.csv")
    calib = tc.commission(twin, truth0)
    truth = pd.read_csv(SCEN_DIR / f"{name}.csv")
    seed = 200 + list(SCENARIOS).index(name)          # same seeds as notebook 03
    run = tc.run_scenario(twin, truth, seed, calib)
    run["truth"] = truth
    run["events"] = tc.alarm_events(run["alarms"])
    return run


missing = [p for p in [MODEL_DIR / "config.json", MODEL_DIR / "feasibility_boundary.pkl", SCEN_DIR / "S0_healthy.csv"]
           if not p.exists()]
if missing:
    st.error("Some files the dashboard needs are missing:\n\n" + "\n".join(f"- {p.relative_to(BASE)}" for p in missing)
             + "\n\nRun notebooks 02b and 03 first, then start the dashboard again from the project folder.")
    st.stop()

twin = load_twin()


# ----------------------------------------------------------------------------
# Sidebar: scenario, screen, time and playback
# ----------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("minute", 0)
ss.setdefault("playing", False)
ss.setdefault("speed", "5 minutes per step")

if ss.playing:                                     # move time forward before the slider is drawn
    ss.minute = min(ss.minute + SPEEDS[ss.speed], 719)
    if ss.minute >= 719:
        ss.playing = False

st.sidebar.markdown("### LPG Recovery Twin")
scen = st.sidebar.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k])
screen = st.sidebar.radio("Screen", SCREENS)
st.sidebar.select_slider("Time of day", options=list(range(720)), key="minute", format_func=tc.fmt_time)
c1, c2 = st.sidebar.columns(2)
c1.button("Pause" if ss.playing else "Play", on_click=lambda: ss.update(playing=not ss.playing))
c2.button("Restart", on_click=lambda: ss.update(minute=0, playing=False))
st.sidebar.selectbox("Playback speed", list(SPEEDS), key="speed")
st.sidebar.markdown(f"<div class='quiet'>Model in use: {twin.kind}<br>"
                    f"All plant data on this screen is simulated in DWSIM.</div>", unsafe_allow_html=True)

run = load_scenario(scen)
i = int(ss.minute)
meas, pred, A, kpi, z, events = run["meas"], run["pred"], run["alarms"], run["kpi"], run["z"], run["events"]
now = meas.iloc[i]
plant_ok = bool(now["plant_ok"])
active = [a for a in tc.ALARM_ORDER if bool(A[a].iloc[i])]
active.sort(key=lambda a: tc.PRIORITY_RANK[tc.ALARMS[a]["priority"]])


def since(alarm):
    """Time the alarm last came on, as hh:mm."""
    col = A[alarm].values[: i + 1].astype(bool)
    start = i
    while start > 0 and col[start - 1]:
        start -= 1
    return tc.fmt_time(meas.time_min.iloc[start])


def badge(text, priority):
    return f"<span class='tag-abn' style='background:{PRIORITY_COLOUR[priority]}'>{text}</span>"


# ----------------------------------------------------------------------------
# Status strip (all screens)
# ----------------------------------------------------------------------------

if not plant_ok:
    status = badge("Deethanizer not running", "High")
elif active:
    worst = tc.ALARMS[active[0]]["priority"]
    status = badge(f"{len(active)} active alarm{'s' if len(active) > 1 else ''}", worst)
else:
    status = "<span>No active alarms</span>"
st.markdown(f"<div class='strip'><span><b>Time</b> {tc.fmt_time(now.time_min)}</span>"
            f"<span><b>Scenario</b> {SCENARIOS[scen]}</span><span>{status}</span></div>", unsafe_allow_html=True)


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


# ----------------------------------------------------------------------------
# Screens
# ----------------------------------------------------------------------------

def screen_overview():
    st.markdown("## Plant overview")
    k = kpi.iloc[i]
    margin = float(A["margin_bar"].iloc[i])
    margin_txt = "No limit near" if margin > 50 else f"{margin:.1f}<span class='u'> bar</span>"
    margin_style = ""
    if margin < 0.5:
        margin_style = f"style='color:{PRIORITY_COLOUR['High']}'"
    elif margin < 1.5:
        margin_style = f"style='color:{PRIORITY_COLOUR['Low']}'"
    n_act = len(active)
    act_style = f"style='color:{PRIORITY_COLOUR[tc.ALARMS[active[0]]['priority']]}'" if n_act else ""
    cards = [
        ("LPG product", f"{fmt(k.lpg_kgh)}<span class='u'> kg/h</span>", "FT-301, measured"),
        ("Fuel use", f"{fmt(k.fuel_kw, 0)}<span class='u'> kW</span>", "electricity and heat, as fuel gas"),
        ("Fuel per kg of LPG", f"{fmt(k.fuel_per_kg, 2)}<span class='u'> kWh/kg</span>", "lower is better"),
        ("Distance to operating limit", f"<span {margin_style}>{margin_txt}</span>",
         "below the limit" if margin < 0 else "pressure above the limit"),
        ("Active alarms", f"<span {act_style}>{n_act}</span>", "see the Alarms screen"),
    ]
    st.markdown("<div class='cards'>" + "".join(
        f"<div class='card'><div class='label'>{a}</div><div class='value'>{b}</div><div class='note'>{c}</div></div>"
        for a, b, c in cards) + "</div>", unsafe_allow_html=True)

    unit_alarm = {}
    for a in active:
        unit_alarm.setdefault(tc.ALARMS[a]["unit"], a)

    def box(name, rows, extra_msg=None, extra_priority=None):
        a = unit_alarm.get(name)
        style, msg = "", ""
        if a:
            col = PRIORITY_COLOUR[tc.ALARMS[a]["priority"]]
            style = f"style='border:2px solid {col}'"
            msg = f"<div class='msg' style='color:{col}'>{tc.ALARMS[a]['text']}</div>"
        elif extra_msg:
            col = PRIORITY_COLOUR[extra_priority]
            style = f"style='border:2px solid {col}'"
            msg = f"<div class='msg' style='color:{col}'>{extra_msg}</div>"
        body = "".join(f"<div class='row'><span>{t}</span><span>{v}</span></div>" for t, v in rows)
        return f"<div class='unit' {style}><div class='name'>{name}</div>{body}{msg}</div>"

    m, p = now, pred.iloc[i]
    arrow = "<div class='arrow'>&#8594;</div>"
    down = not plant_ok

    def v(x, nd, unit):
        """Reading with its unit, or a dash when there is no reading."""
        return "-" if x is None or np.isnan(float(x)) else f"{float(x):.{nd}f} {unit}"

    lpg_txt = "no product" if down else v(m["FT-301"], 1, "kg/h")
    if down:                                   # a stopped deethanizer outranks any warning about it
        unit_alarm.pop("Deethanizer", None)
    html = arrow.join([
        box("Feed gas", [("FT-001", v(m["FT-001"], 0, "kg/h")), ("TT-001", v(m["TT-001"], 1, "C"))]),
        box("Compressor", [("PT-101", v(m["PT-101"], 2, "bar")), ("TT-101", v(m["TT-101"], 1, "C")),
                           ("JI-101", v(m["JI-101"], 1, "kW"))]),
        box("Air cooler", [("TT-103", v(m["TT-103"], 1, "C")),
                           ("Duty", "-" if down else v(p["Air_Cooler_Duty_kW"], 1, "kW"))]),
        box("Chiller", [("TT-102", v(m["TT-102"], 1, "C")), ("Setpoint", v(m["SP-TT102"], 1, "C")),
                        ("JI-102", v(m["JI-102"], 1, "kW"))]),
        box("Separator", [("Liquid to columns", "too little" if down else "normal")]),
        box("Deethanizer", [("Condenser", "-" if down else v(p["Deethanizer_Condenser_T_C"], 0, "C")),
                            ("Reboiler", "-" if down else v(p["Deethanizer_Reboiler_kW"], 1, "kW"))],
            extra_msg="Not running" if down else None, extra_priority="High"),
        box("Debutanizer", [("FT-301", lpg_txt),
                            ("Reboiler", "-" if down else v(p["Debutanizer_Reboiler_kW"], 1, "kW"))]),
    ])
    st.markdown(f"<div class='flow'>{html}</div>", unsafe_allow_html=True)
    note = ("Values are instrument readings. Duties without a tag are the twin's estimate. "
            "A coloured box means that unit has an alarm.")
    if down:
        note += " A dash means the simulation has no reading while the deethanizer is down."
    st.markdown(f"<div class='quiet'>{note}</div>", unsafe_allow_html=True)

    if active:
        st.markdown("### What needs attention")
        for a in active:
            info = tc.ALARMS[a]
            st.markdown(f"{badge(info['priority'], info['priority'])} &nbsp;**{info['text']}** "
                        f"(since {since(a)}). {info['action']}", unsafe_allow_html=True)


def screen_alarms():
    st.markdown("## Alarms")
    st.markdown("### Active now")
    if active:
        rows = [{"Priority": tc.ALARMS[a]["priority"], "Alarm": a, "What it means": tc.ALARMS[a]["text"],
                 "Since": since(a), "What to do": tc.ALARMS[a]["action"]} for a in active]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.markdown("<div class='quiet'>No active alarms.</div>", unsafe_allow_html=True)

    st.markdown("### Alarm log")
    log = events[events.time_min <= meas.time_min.iloc[i]].copy()
    if log.empty:
        st.markdown("<div class='quiet'>Nothing logged so far today.</div>", unsafe_allow_html=True)
    else:
        log["Time"] = log.time_min.map(tc.fmt_time)
        log["Priority"] = log.alarm.map(lambda a: tc.ALARMS[a]["priority"] if a in tc.ALARMS else "High")
        log["Message"] = log.alarm.map(lambda a: tc.ALARMS[a]["text"] if a in tc.ALARMS
                                       else "Deethanizer stopped: not enough liquid")
        log = log.rename(columns={"alarm": "Alarm", "event": "Event"})
        st.dataframe(log[["Time", "Priority", "Alarm", "Event", "Message"]].iloc[::-1],
                     hide_index=True)
    n_new = int((events.event == "Raised").sum())
    st.markdown(f"<div class='quiet'>Alarms raised in this 12-hour scenario: {n_new} "
                f"({n_new / 12:.2f} per hour). ISA-18.2 suggests an average of about 6 per hour or fewer "
                f"for one operator.</div>", unsafe_allow_html=True)

    with st.expander("Alarm reference: cause, action and settings for every alarm"):
        settings = {"FEAS-HI": "Margin below 0.5 bar, clears above 1.0 bar, 1 min delay",
                    "FEAS-ADV": "Margin below 1.5 bar, clears above 2.0 bar, 2 min delay",
                    "TT-102 DEV": "More than 2 C above setpoint, clears below 1.5 C, 5 min delay",
                    "JI-101 PERF": "Power and discharge temperature above expected, 10 min delay",
                    "JI-102 PERF": "Chiller duty above expected, 10 min delay",
                    "FT-301 LOW": "LPG below expected, 10 min delay",
                    "PT-101 SUSPECT": "Power, discharge temperature and LPG all below expected, 10 min delay"}
        ref = pd.DataFrame([{"Alarm": a, "Priority": v["priority"], "Unit": v["unit"], "Likely cause": v["cause"],
                             "What to do": v["action"], "Setting": settings[a]} for a, v in tc.ALARMS.items()])
        st.dataframe(ref, hide_index=True)


DEFAULT_TRENDS = {"S0_healthy": ["JI-101", "FT-301"], "S1_compressor_wear": ["JI-101", "TT-101"],
                  "S2_aircooler_fouling": ["JI-102", "TT-103"], "S3_chiller_capacity": ["TT-102", "FT-301"],
                  "S4_pt101_drift": ["JI-101", "FT-301"], "S5_unsafe_move": ["TT-102", "PT-101"]}


def screen_trends():
    st.markdown("## Trends")
    tags = st.multiselect("Instruments", list(tc.SENSORS), default=DEFAULT_TRENDS[scen],
                          format_func=lambda t: f"{t}  {tc.SENSORS[t][4]}")
    window = st.radio("Show", ["Last 3 hours", "Whole day so far"], horizontal=True)
    lo = max(0, i - 180) if window == "Last 3 hours" else 0
    sl = slice(lo, i + 1)
    h = meas.hour.values[sl]
    for t in tags:
        fig, ax = plt.subplots(figsize=(10, 2.4))
        ax.plot(h, meas[t].values[sl], ".", ms=2.5, color=LINE, label="Measured")
        col = tc.RESIDUAL_TAGS.get(t)
        if col:
            expected = pd.Series(pred[col].values).rolling(15, center=True, min_periods=1).mean().values[sl]
            ax.plot(h, expected, "-", color=EXPECT, lw=1.6, label="Twin expects (15-min average)")
        if t == "TT-102":
            ax.plot(h, meas["SP-TT102"].values[sl], "--", color=EXPECT, lw=1.2, label="Setpoint")
        ax.set_ylabel(f"{t} ({tc.SENSORS[t][3]})"); ax.set_xlabel("Hour")
        ax.set_title(tc.SENSORS[t][4], loc="left")
        ax.set_xlim(h[0], max(h[-1], h[0] + 0.5))
        ax.legend(loc="upper left", ncol=3)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### How far readings are from what the twin expects")
    st.markdown("<div class='quiet'>Each line is smoothed and scaled so that 1 is the alarm limit. "
                "Readings that stay near 0 match the twin.</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    styles = {"JI-101": "-", "TT-101": "--", "JI-102": "-.", "FT-301": ":"}
    for t, ls in styles.items():
        ax.plot(h, z[t].values[sl], ls, color=INK, lw=1.2, label=t)
    ax.axhspan(1, 3, color="#E7DCC7", alpha=0.6, lw=0, label="Beyond alarm limit")
    ax.axhspan(-3, -1, color="#E7DCC7", alpha=0.6, lw=0)
    ax.set_ylim(-3, 3); ax.set_ylabel("Scaled difference"); ax.set_xlabel("Hour")
    ax.set_xlim(h[0], max(h[-1], h[0] + 0.5)); ax.legend(loc="upper left", ncol=4)
    st.pyplot(fig)
    plt.close(fig)


def limit_chart(ax, path_t=None, path_p=None, point=None, best=None):
    t = twin.bound_t
    ax.fill_between(t, 15, np.nan_to_num(twin.bound_p, nan=15), color="#D5D5D5", lw=0, label="Unsafe area")
    ax.plot(t, twin.bound_p, "-", color=INK, lw=1.6, label="Operating limit")
    ax.plot(t, np.minimum(twin.bound_p + twin.margin_limit, 35), "--", color=INK, lw=1, label="Advisory line")
    if path_t is not None:
        ax.plot(path_t, path_p, ".", ms=2, color=LINE, label="Path so far")
    if best is not None:
        ax.plot(best[0], best[1], marker="*", ms=13, mfc="white", mec=INK, ls="none", label="Lowest-fuel setpoint")
    if point is not None:
        ax.plot(point[0], point[1], marker="o", ms=9, color=point[2], ls="none", label="Now")
    ax.set_xlim(14.5, 45.5); ax.set_ylim(14.5, 35.5)
    ax.set_xlabel("Chiller outlet temperature (C)"); ax.set_ylabel("Compressor pressure (bar)")
    ax.legend(loc="upper right")


def screen_limits():
    st.markdown("## Operating limits")
    st.markdown("<div class='quiet'>Below the solid line the deethanizer does not get enough liquid and stops "
                "working. The dashed line is the advisory line, 1.5 bar above the limit.</div>", unsafe_allow_html=True)
    margin = float(A["margin_bar"].iloc[i])
    colour = INK if margin >= 1.5 else (PRIORITY_COLOUR["Low"] if margin >= 0.5 else PRIORITY_COLOUR["High"])
    lpg_now = float(np.nan_to_num(kpi.lpg_kgh.iloc[i], nan=0))
    best = None
    if lpg_now > 5:
        b, _ = cached_best(round(float(now["FT-001"]), -1), round(float(now["TT-001"])), round(lpg_now, -1))
        best = (b["T"], b["P"]) if b is not None else None
    fig, ax = plt.subplots(figsize=(7.5, 5))
    limit_chart(ax, meas["TT-102"].values[: i + 1], meas["PT-101"].values[: i + 1],
                (float(now["TT-102"]), float(now["PT-101"]), colour), best)
    left, right = st.columns([3, 2])
    left.pyplot(fig)
    plt.close(fig)
    with right:
        st.markdown("### Right now")
        st.markdown(f"Chiller outlet: **{now['TT-102']:.1f} C**  \nCompressor pressure: **{now['PT-101']:.2f} bar**")
        if margin > 50:
            st.markdown("There is no limit at this chiller temperature. The plant is safe at any pressure in range.")
        else:
            st.markdown(f"Distance to the limit: **{margin:.1f} bar**")
            if margin < 0.5:
                st.markdown(f"<span style='color:{PRIORITY_COLOUR['High']}; font-weight:600'>Very close to the limit. "
                            f"Raise pressure or lower the chiller setpoint now.</span>", unsafe_allow_html=True)
            elif margin < 1.5:
                st.markdown(f"<span style='color:{PRIORITY_COLOUR['Low']}; font-weight:600'>Inside the advisory band.</span>",
                            unsafe_allow_html=True)
        if best:
            st.markdown(f"For the current output of about {round(lpg_now, -1):.0f} kg/h, the lowest-fuel safe setpoint is "
                        f"**{best[0]:.1f} C** and **{best[1]:.2f} bar** (star on the chart).")


@st.cache_data(show_spinner="Searching for the best setpoint...")
def cached_best(feed_kgh, feed_t, target):
    b, u = load_twin().best_setpoint(feed_kgh, feed_t, target)
    return (None if b is None else b.to_dict()), (None if u is None else u.to_dict())


def screen_try():
    st.markdown("## Try a setpoint")
    st.markdown("<div class='quiet'>Change the values and the twin shows what the plant would do, in a fraction of a "
                "second. Nothing here touches the plant.</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        t_set = st.slider("Chiller outlet temperature (C)", 15.0, 45.0, float(round(now["TT-102"] * 2) / 2), 0.5)
        p_set = st.slider("Compressor pressure (bar)", 15.0, 35.0, float(round(now["SP-PT101"] * 4) / 4), 0.25)
    with b:
        f_set = st.slider("Feed gas flow (kg/h)", 800.0, 1200.0, float(round(now["FT-001"], -1)), 10.0)
        ft_set = st.slider("Feed gas temperature (C)", 25.0, 45.0, float(round(now["TT-001"])), 0.5)
    X = np.array([[t_set, p_set, f_set, ft_set]])
    pr = twin.predict(X)
    elec, heat, fuel = twin.fuel(pr, np.array([t_set]))
    lpg = float(pr["LPG_kgh"].iloc[0]); fuel = float(np.asarray(fuel)[0])
    margin = float(twin.margin(np.array([t_set]), np.array([p_set]))[0])

    if margin < 0:
        st.markdown(f"<div class='strip' style='border-color:{PRIORITY_COLOUR['High']}'>"
                    f"{badge('Not possible', 'High')} This setpoint is below the operating limit. "
                    f"The deethanizer would stop. Raise the pressure or lower the chiller temperature.</div>",
                    unsafe_allow_html=True)
        return
    cards = [("LPG product", f"{lpg:.1f}<span class='u'> kg/h</span>", "twin prediction"),
             ("Fuel use", f"{fuel:.0f}<span class='u'> kW</span>", "electricity and heat, as fuel gas"),
             ("Fuel per kg of LPG", f"{fuel / max(lpg, 1e-3):.2f}<span class='u'> kWh/kg</span>", "lower is better"),
             ("Distance to operating limit", "No limit near" if margin > 50 else f"{margin:.1f}<span class='u'> bar</span>",
              "advisory below 1.5 bar"),
             ("Compressor power", f"{pr['Compressor_Power_kW'].iloc[0]:.1f}<span class='u'> kW</span>", "biggest energy user")]
    st.markdown("<div class='cards'>" + "".join(
        f"<div class='card'><div class='label'>{x}</div><div class='value'>{y}</div><div class='note'>{n}</div></div>"
        for x, y, n in cards) + "</div>", unsafe_allow_html=True)
    if margin < 1.5:
        st.markdown(f"<span style='color:{PRIORITY_COLOUR['Low']}; font-weight:600'>This is inside the advisory band. "
                    f"It works on paper but leaves little room for error.</span>", unsafe_allow_html=True)

    E = twin.energy
    parts = {"Compressor": pr["Compressor_Power_kW"].iloc[0] / E["ETA_POWER"],
             "Chiller": pr["Chiller_Duty_kW"].iloc[0] / twin.cop(t_set) / E["ETA_POWER"],
             "Deethanizer cooling": pr["Deethanizer_Condenser_kW"].iloc[0] / twin.cop(pr["Deethanizer_Condenser_T_C"].iloc[0]) / E["ETA_POWER"],
             "Air cooler fans": E["FAN_FRACTION"] * (pr["Air_Cooler_Duty_kW"].iloc[0] + pr["Debutanizer_Condenser_kW"].iloc[0]) / E["ETA_POWER"],
             "Reboiler heat": (pr["Deethanizer_Reboiler_kW"].iloc[0] + pr["Debutanizer_Reboiler_kW"].iloc[0]) / E["ETA_HEATER"]}
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        names = list(parts)[::-1]
        ax.barh(names, [parts[n] for n in names], color="#8A8A8A")
        for y, n in enumerate(names):
            ax.text(parts[n] + 1, y, f"{parts[n]:.0f} kW", va="center", fontsize=8, color=INK)
        ax.set_xlabel("Fuel gas used (kW)"); ax.set_title("Where the fuel goes", loc="left")
        ax.set_xlim(0, max(parts.values()) * 1.2); ax.grid(axis="y", visible=False)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("### Best way to make this much LPG")
        target = st.number_input("LPG needed (kg/h)", 20.0, 150.0, float(max(20, round(lpg, -1))), 5.0)
        b, u = cached_best(f_set, ft_set, target)
        if b is None:
            st.markdown("The plant cannot make that much LPG safely at this feed.")
        else:
            sec_b = b["Fuel"] / b["LPG"]
            st.markdown(f"Lowest fuel: **{b['T']:.1f} C** and **{b['P']:.2f} bar**, about **{sec_b:.2f} kWh/kg**.")
            if u is not None:
                sec_u = u["Fuel"] / u["LPG"]
                st.markdown(f"The usual way, compressor at 35 bar and chiller at {u['T']:.1f} C, uses about "
                            f"{sec_u:.2f} kWh/kg. The best setpoint saves **{(1 - sec_b / sec_u) * 100:.0f}%** of the fuel per kg.")
            here = fuel / max(lpg, 1e-3)
            if lpg >= target and here > sec_b * 1.01:
                st.markdown(f"Your setpoint above uses {here:.2f} kWh/kg, so it could save "
                            f"{(1 - sec_b / here) * 100:.0f}% by moving to the best setpoint.")


def screen_about():
    st.markdown("## About this twin")
    st.markdown(f"""
This dashboard is the operator screen for a digital twin of a small LPG recovery unit. The unit takes
associated gas from oil wells, compresses and cools it, and separates propane and butane in two columns.

**What it does**

- Watches the plant every minute and predicts what each instrument should read for the current feed and setpoints.
- Raises an alarm when readings drift from what it expects, and says which unit is the likely cause.
- Warns operators before they reach the point where the deethanizer stops working.
- Lets operators test a setpoint and see the effect on LPG output and fuel use before they change anything.

**What it is built from**

- A rigorous model of the plant in DWSIM (Peng-Robinson), with the columns set at 1.3 times minimum reflux.
- 3,000 simulated operating points covering chiller temperature, compressor pressure, feed flow and feed temperature.
- A fast model of the plant trained on those points ({twin.kind}). It predicts in microseconds what DWSIM takes about
  0.1 s to calculate.
- A fitted operating limit that flagged all 236 failed points in testing, with one false alarm.

**Simulated, not field data**

Every number on this dashboard comes from DWSIM, with realistic sensor noise added. The faults in the scenarios were
also created in DWSIM. Before use on a real unit, the twin would be retrained and checked on the plant's own data.

**Standards it follows**

- ISA-5.1 for instrument tags (FT, TT, PT, JI).
- ISA-18.2 for alarms: every alarm has a cause, a consequence, an operator action, a priority, a delay and a deadband.
- ISA-101 for this screen: grey background, colour only for abnormal states, overview first and detail on request.
- ISA-95: the twin sits at Level 3, above the control system, and only reads plant data.
- ISA/IEC 62443: data flows one way from the control system to the twin. The twin never writes setpoints.
- ISA-84: the operating-limit warning is advice to the operator. It does not replace the plant's safety systems.

**Limits of this version**

- The model is steady state. It shows where the plant settles, not how it moves between states.
- Feed composition is fixed.
- The columns use shortcut models, not tray-by-tray models.
""")


{"Overview": screen_overview, "Alarms": screen_alarms, "Trends": screen_trends,
 "Operating limits": screen_limits, "Try a setpoint": screen_try, "About": screen_about}[screen]()

if ss.playing:
    time.sleep(0.6)
    st.rerun()
