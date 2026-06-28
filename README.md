# ⚙️ Digital Twin of a 3-Phase Induction Motor

A living virtual copy of an industrial induction motor. It continuously
simulates the motor's electromagnetic, mechanical and thermal behaviour as
operating conditions change, and surfaces engineering guidance + (later)
ML-based failure prediction.

> **Status:** Milestones 1 & 3 done — physics + thermal engine, live dashboard,
> and ML bearing prognostics (RUL + failure + anomaly) on the NASA-IMS feature
> schema. See [Roadmap](#roadmap).

## 🌐 Live demo

**[▶ Open the live dashboard](https://YOUR_USERNAME.github.io/digital-motor-twin/)**
— a fully in-browser version (no install) hosted on GitHub Pages.

[`index.html`](index.html) is a self-contained port of the twin: the same
equivalent-circuit + Thévenin electromagnetics, 2-node thermal model and rotor
dynamics, plus a lightweight surrogate of the trained ML models. It needs no
Python — just open the page.

**Enable it on your fork:** Settings ▸ Pages ▸ Source: *Deploy from a branch* ▸
`main` / `/root` ▸ Save. After ~1 min the dashboard is live at
`https://<username>.github.io/digital-motor-twin/`. (Replace `YOUR_USERNAME`
in the link above.) To preview locally: `python -m http.server 8000` then open
<http://localhost:8000>.

> The hosted page uses an in-browser **surrogate** of the scikit-learn models
> (a static site can't run sklearn). The full Streamlit app below uses the real
> trained models in `models/`.

## What it models

| Domain | Model | File |
|--------|-------|------|
| Electromagnetics | Per-phase equivalent circuit + Thévenin reduction (torque, current, power factor, losses, efficiency vs. slip) | `motor_twin/induction.py` |
| Mechanics | Rotor equation of motion `J·dω/dt = Te − Tload − Bω` | `motor_twin/simulation.py` |
| Thermal | 2-node lumped RC network (winding ↔ housing ↔ ambient, cooling-dependent) | `motor_twin/thermal.py` |
| Health | Empirical bearing-wear / vibration / RUL accumulator | `motor_twin/simulation.py` |
| Prognostics (ML) | Vibration features → RUL regressor + failure classifier + anomaly detector | `motor_twin/ml/` |
| Guidance | Rule-based engineering recommendations | `motor_twin/recommendations.py` |

## ML bearing prognostics (Milestone 3)

A single vibration feature extractor (`motor_twin/ml/features.py`) turns a raw
20 kHz accelerometer window into time- and frequency-domain features, including
energy at the four bearing **defect frequencies** (BPFO/BPFI/BSF/FTF) derived
from the IMS rig's bearing geometry. Three scikit-learn models share this schema:

| Model | Algorithm | Output |
|-------|-----------|--------|
| RUL regressor | GradientBoosting | remaining useful life (files → hours) |
| Failure classifier | RandomForest | P(bearing in failure stage) |
| Anomaly detector | IsolationForest | unsupervised novelty score |

**Train (synthetic, runs offline now):**
```bash
python -m motor_twin.ml.train          # writes models/*.joblib + metrics.json
```
The training data comes from a physics-grounded run-to-failure generator
(`synthetic.py`) that produces the *same feature schema* as real IMS files, so
models transfer directly.

**Retrain on the real NASA IMS dataset:** download the "Bearing Data Set" from
the NASA Prognostics Center of Excellence, unpack a test set into `data/raw/`,
then:
```bash
python -m motor_twin.ml.train --source ims --path data/raw/2nd_test
```
`ims_loader.py` reads the timestamped 20 kHz files and feeds the identical
extractor — no other code changes needed.

The dashboard's **bearing-condition** and **lubrication** sliders drive the
prognostics live via `twin_bridge.py`: lowering bearing health raises the ML
failure probability, shortens predicted RUL, and trips the anomaly flag.

The default parameters describe a ~7.5 kW, 400 V, 50 Hz, 4-pole machine.
At rated slip (≈3 %) the twin reproduces ~1455 RPM, ~86 % efficiency and
~7.3 kW shaft output — consistent with the nameplate.

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to redesign the motor (voltage, poles, inertia) and change
live operating conditions (load, ambient temperature, cooling airflow,
bearing condition, lubrication). Gauges, trends and recommendations update
in real time.

## Use the engine directly

```python
from motor_twin import InductionMotor, DigitalTwin, Operating, recommend

twin = DigitalTwin(motor=InductionMotor(v_line=400, poles=4))
twin.reset(t_ambient=25)
states = twin.run(Operating(load_torque=45, airflow_m3h=80), duration=10)
print(states[-1])          # full state snapshot
print(recommend(states[-1]))
```

## Google Colab

Open `notebooks/digital_twin_colab.ipynb` in Colab — it clones the repo,
runs the physics engine and plots torque–speed curves, thermal response and
a full transient, all inline (no Streamlit needed).

## Deploy the dashboard (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set the entrypoint to `app.py`.

## Roadmap

- **M1 (done):** induction-motor physics + thermal + live dashboard
- **M3 (done):** ML failure prediction, RUL & anomaly detection on the NASA-IMS
  feature schema (synthetic training now; one-command retrain on real IMS data)
- **M2:** mechanical stress detail (shaft/bearing forces, torsional), thermal map
- **M4:** efficiency-map visualisation, 3D exploded model (PyVista), time-series
  forecasting (LSTM) for degradation trends

## Project layout

```
digital-motor-twin/
├── index.html                   # standalone in-browser twin (GitHub Pages)
├── app.py                       # Streamlit live dashboard
├── motor_twin/
│   ├── induction.py             # equivalent-circuit electromagnetics
│   ├── thermal.py               # lumped thermal network
│   ├── simulation.py            # coupled dynamic twin
│   ├── recommendations.py       # rule-based guidance
│   └── ml/                      # bearing prognostics
│       ├── features.py          # vibration feature extractor (defect freqs)
│       ├── synthetic.py         # run-to-failure data generator
│       ├── ims_loader.py        # real NASA IMS dataset loader
│       ├── train.py             # train RUL / failure / anomaly models
│       ├── predict.py           # inference wrapper
│       └── twin_bridge.py       # physics-state → ML prediction
├── models/                      # trained .joblib artifacts + metrics.json
├── notebooks/
│   └── digital_twin_colab.ipynb # Colab demo
└── requirements.txt
```
