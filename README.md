# LPG Recovery Digital Twin

A digital twin of a small unit that recovers LPG (propane and butane) from associated gas. It watches the plant minute by minute, warns operators before the deethanizer stops working, spots equipment problems early, and finds setpoints that use less fuel for the same LPG output.

Built by Yusuf Barika-Bodunrin (Electrical and Electronics Engineering) and Azeezat Salawu (Chemical Engineering), University of Lagos.

**All results here come from simulation (DWSIM). None of it is field data yet.**

## What it does

- **Predicts what every instrument should read** for the current feed and setpoints, and compares that with what the plant actually reads.
- **Raises alarms with a likely cause.** For example, if compressor power and discharge temperature are both higher than expected, the compressor is probably wearing.
- **Warns before infeasible operation.** At high chiller temperature and low pressure, too little liquid condenses and the deethanizer stops. The twin knows where that limit is and warns operators as they approach it.
- **Finds the lowest-fuel setpoint** for a required LPG output, and lets operators test any setpoint before using it.

## Results

| What | Result |
|---|---|
| Rigorous model | DWSIM, Peng-Robinson, corrected flowsheet. 3,000 random operating points over chiller temperature, compressor pressure, feed flow and feed temperature. 92.1% converged; all failures are the real low-liquid limit. Median solve time 97 ms. |
| Fast model (surrogate) | PyTorch neural network, 4 inputs, 10 outputs. Plant energy within 0.6% of DWSIM on unseen test points. About 13.5 µs per prediction, roughly 7,000 times faster than DWSIM. |
| Model comparison | Polynomial, Gaussian process and neural network all predicted plant energy within 1% with full data. The neural network was chosen because it stayed most consistent outside its training range and is fast enough for live use. |
| Operating limit | Smooth boundary fitted on chiller temperature and pressure. Flagged all 236 failed points in 5-fold cross-validation, with 1 false alarm in 2,764 safe points. |
| Fuel saving | Against conventional operation (compressor at 35 bar, chiller adjusted to meet the target), the best setpoint uses 33.7% less fuel per kg at 60 kg/h, 22.0% at 80 kg/h, 12.2% at 100 kg/h and 4.6% at 120 kg/h. These numbers were rerun and confirmed in DWSIM. |
| Fault detection | Compressor efficiency loss caught after 58 min (at 1 point below normal). Air cooler fouling after 41 min. Chiller capacity loss after 30 min. Pressure transmitter drift after 71 min, correctly flagged as an instrument fault. Unsafe setpoint change: warning 28 min before the deethanizer failed. No false alarms on the healthy day. |

Fuel is counted as the fuel gas burned on site: electricity from a gas-turbine generator (33% efficient), refrigeration at a realistic COP, and reboiler heat from a fired heater (85% efficient). The assumptions are listed in notebook 02b.

## The dashboard

An operator screen styled after ISA-101: grey background, colour only when something is abnormal, overview first and detail on request. It replays six 12-hour scenarios generated in DWSIM and runs the twin on them live.

Screens: plant overview, alarms (with cause and operator action for each), trends, operating limits, a setpoint tester, and notes on the model and standards.

To run it from the project folder:

```
pip install -r requirements.txt
py -m streamlit run app.py
```

## Standards

- **ISA-5.1:** instrument tags (FT, TT, PT, JI).
- **ISA-18.2:** every alarm has a cause, consequence, operator action, priority, delay and deadband. Worst case in the demo: 0.25 alarms per hour.
- **ISA-101:** dashboard design.
- **ISA-95:** the twin sits at Level 3, above the control system.
- **ISA/IEC 62443:** data flows one way into the twin. The twin never writes setpoints.
- **ISA-84:** the operating-limit warning advises the operator. It does not replace the plant's safety systems.

## Repository layout

```
app.py, twin_core.py          Operator dashboard and the twin's core logic
.streamlit/config.toml        Dashboard theme
simulation/
  LPG_Recovery_v2.dwxmz       Corrected flowsheet (used for all current results)
  LPG_Recovery_Base.dwxmz     Original flowsheet (kept for reference)
notebooks/
  01b_data_generation_v2      DWSIM sweep: 3,000 points, 4 inputs
  02b_models_and_optimization_v2   Energy weighting, operating limit, model comparison, optimization, DWSIM check
  03_fault_and_alarm_demo     Six fault scenarios, detection and alarms
  01_data_generation, 02_surrogate_and_optimization   Original version (kept for reference)
data/
  LPG_v2_Data.csv             Current dataset
  scenarios/                  Fault scenarios and detection results
models/
  v2/                         Current models (neural network, Gaussian process, operating limit)
```

## Changes from the first version

The first version had a deethanizer at atmospheric pressure (condenser at -154 C), no valves between pressure levels, one cooler doing all the cooling, and a fixed reflux ratio. The corrected flowsheet runs the deethanizer at 14 bar with a partial condenser, the debutanizer at 10 bar, adds an air cooler before the chiller and valves before both columns, and sets each column at 1.3 times its minimum reflux. All current results use the corrected flowsheet.

## Limitations

- Steady-state model: it shows where the plant settles, not how it moves between states.
- Feed composition is fixed.
- The columns use shortcut (FUG) models, not tray-by-tray models.
- The best setpoint sits at the coldest chiller setting studied (15 C), so deeper chilling may save more.
- All data is simulated. A real deployment would need retraining and checks on plant data.
