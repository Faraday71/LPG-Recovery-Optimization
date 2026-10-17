# Deep Learning-Assisted LPG Recovery Optimization

An end-to-end digital twin and surrogate optimization framework for Liquefied Petroleum Gas (LPG) recovery from associated petroleum gas. This project links Peng-Robinson thermodynamic simulations in **DWSIM** with a custom **PyTorch multi-output neural network** to optimize plant operations in real-time.

## Key Engineering Achievements
* **17,085x Computational Speedup:** Replaced a 3-hour rigorous DWSIM grid search with a 583-millisecond PyTorch surrogate evaluation ($15.88\text{ }\mu\text{s}$ per state).
* **0% Column Crash Rate:** Implemented a Random Forest feasibility classifier to automatically map and avoid the Deethanizer thermodynamic dry-out wedge.
* **$52.4\text{ kW}$ Energy Savings:** Discovered a nonlinear efficiency valley at $15.0\text{ }^\circ\text{C}$ and $26.84\text{ bar}$ yielding a global minimum specific energy of $1.747\text{ kWh/kg LPG}$.
* **Closed-Loop Validation:** Validated neural network Pareto setpoints back in DWSIM with $<0.06\%$ relative error.

## Repository Structure
* `/data/`: Contains the automated 2,500-state DWSIM sweep (`LPG_Energy_Data.csv`) and cleaned datasets.
* `/models/`: Contains the trained `SiLU` PyTorch model, feasibility classifier, and scaler artifacts.
* `/notebooks/`: 
  * `01_data_generation.ipynb`: Python.NET automation of the DWSIM flowsheet.
  * `02_surrogate_and_optimization.ipynb`: Neural network training, Pareto extraction, and exergy-weighted optimization.
* `/simulation/`: DWSIM flowsheets (`.dwxmz`).
* `app.py`: Interactive Streamlit Digital Twin Dashboard.

## Running the Digital Twin Locally
```bash
pip install -r requirements.txt
streamlit run app.py
