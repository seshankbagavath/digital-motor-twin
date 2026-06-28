"""Live digital-twin dashboard for a 3-phase induction motor.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io -> point at app.py

The live loop uses an ``st.fragment(run_every=...)`` so only the dashboard
re-renders on each tick — the sidebar controls are not re-executed — instead of
the old full-script ``time.sleep`` + ``st.rerun`` busy-loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from motor_twin import InductionMotor, DigitalTwin, Operating, recommend
from motor_twin.ml import models_available
from motor_twin.ml.twin_bridge import predict_for_state

st.set_page_config(page_title="Motor Digital Twin", page_icon="⚙️", layout="wide")

HISTORY = 600          # samples kept in the rolling time-series
SIM_DT = 0.01          # integrator timestep [s]
STEPS_PER_TICK = 20    # sim steps advanced per UI tick (~0.2 s of sim)
TICK = "0.2s"          # wall-clock cadence of the live fragment
ML_EVERY = 5           # recompute ML prognostics every N ticks (~1 s)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def init_state() -> None:
    if "twin" not in st.session_state:
        st.session_state.motor = InductionMotor()
        twin = DigitalTwin(motor=st.session_state.motor)
        twin.reset(t_ambient=25.0)
        st.session_state.twin = twin
        st.session_state.history = []
        st.session_state.ml_tick = 0
        st.session_state.ml_pred = None


init_state()


@st.cache_data(show_spinner=False)
def torque_speed_curve(v_line: float, freq: float, poles: int,
                       inertia: float, n: int = 150):
    """Cached torque-speed curve — only recomputed when the design changes."""
    m = InductionMotor(v_line=v_line, frequency=freq, poles=poles, inertia=inertia)
    return m.torque_speed_curve(n)


def gauge(value, title, vmin, vmax, suffix="", good=None, warn=None):
    steps = []
    if good is not None and warn is not None:
        steps = [
            {"range": [vmin, good], "color": "rgba(46,204,113,0.35)"},
            {"range": [good, warn], "color": "rgba(241,196,15,0.35)"},
            {"range": [warn, vmax], "color": "rgba(231,76,60,0.35)"},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix},
        title={"text": title, "font": {"size": 16}},
        gauge={"axis": {"range": [vmin, vmax]},
               "bar": {"color": "#2c3e50"},
               "steps": steps},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: motor design + live operating inputs (full-script scope)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Motor Designer")
    v_line = st.number_input("Rated line voltage [V]", 100.0, 690.0,
                             st.session_state.motor.v_line, 10.0)
    freq = st.selectbox("Supply frequency [Hz]", [50.0, 60.0],
                        index=0 if st.session_state.motor.frequency == 50 else 1)
    poles = st.selectbox("Poles", [2, 4, 6, 8],
                         index=[2, 4, 6, 8].index(st.session_state.motor.poles))
    inertia = st.slider("Rotor + load inertia J [kg·m²]", 0.01, 0.5,
                        st.session_state.motor.inertia, 0.01)

    # Rebuild the motor if the design changed, and reset the twin so the
    # transient restarts cleanly instead of carrying over stale speed/temp.
    m = st.session_state.motor
    if (v_line, freq, poles, inertia) != (m.v_line, m.frequency, m.poles, m.inertia):
        st.session_state.motor = InductionMotor(
            v_line=v_line, frequency=freq, poles=poles, inertia=inertia)
        st.session_state.twin.motor = st.session_state.motor
        st.session_state.twin.reset(t_ambient=st.session_state.get("t_ambient", 25.0))
        st.session_state.history = []
        st.session_state.ml_pred = None

    st.divider()
    st.header("🎛️ Operating Conditions")
    load_torque = st.slider("Load torque [N·m]", 0.0, 160.0, 45.0, 1.0)
    t_ambient = st.slider("Ambient temp [°C]", -10.0, 50.0, 25.0, 1.0)
    airflow = st.slider("Cooling airflow [m³/h]", 0.0, 300.0, 80.0, 5.0)
    bearing = st.slider("Bearing condition (1=healthy)", 0.0, 1.0, 1.0, 0.05)
    lube = st.slider("Lubrication quality (1=good)", 0.0, 1.0, 1.0, 0.05)

    st.divider()
    running = st.toggle("▶️ Run simulation", value=True)
    if st.button("⟲ Reset twin"):
        st.session_state.twin.reset(t_ambient=t_ambient)
        st.session_state.history = []
        st.session_state.ml_pred = None

# Stash live controls so the fragment (which doesn't re-run the sidebar) sees
# the latest values.
st.session_state.t_ambient = t_ambient
st.session_state.controls = dict(
    load_torque=load_torque, t_ambient=t_ambient, airflow=airflow,
    bearing=bearing, lube=lube, running=running)


# --------------------------------------------------------------------------- #
# Live dashboard fragment — re-runs every TICK without re-executing the sidebar
# --------------------------------------------------------------------------- #
st.title("Digital Twin — 3-Phase Induction Motor")
motor = st.session_state.motor
sync = motor.sync_speed_rpm
rated_torque = motor.rated_power / motor.sync_speed_rad  # for gauge scaling

# Disable the auto-tick when paused so the fragment goes idle.
_interval = TICK if running else None


@st.fragment(run_every=_interval)
def live_dashboard():
    ctrl = st.session_state.controls
    op = Operating(load_torque=ctrl["load_torque"], t_ambient=ctrl["t_ambient"],
                   airflow_m3h=ctrl["airflow"], bearing_condition=ctrl["bearing"],
                   lubrication=ctrl["lube"])
    twin: DigitalTwin = st.session_state.twin

    # Advance the simulation when running.
    if ctrl["running"]:
        for _ in range(STEPS_PER_TICK):
            snap = twin.step(op, dt=SIM_DT)
        st.session_state.history.append(snap)
        st.session_state.history = st.session_state.history[-HISTORY:]
    elif st.session_state.history:
        snap = st.session_state.history[-1]
    else:
        snap = twin.step(op, dt=SIM_DT)

    state = st.session_state.history[-1] if st.session_state.history else snap

    # --- top gauges (thresholds scaled to the motor's rated torque) ---------
    c1, c2, c3, c4 = st.columns(4)
    c1.plotly_chart(gauge(state["speed_rpm"], "Speed [RPM]", 0, sync * 1.05, "",
                          good=sync * 0.9, warn=sync), width="stretch")
    c2.plotly_chart(gauge(state["torque_em"], "Torque [N·m]", 0,
                          max(3 * rated_torque, 60),
                          good=1.5 * rated_torque, warn=2.2 * rated_torque),
                    width="stretch")
    c3.plotly_chart(gauge(state["t_winding"], "Winding Temp [°C]", 0, 180, "",
                          good=100, warn=130), width="stretch")
    c4.plotly_chart(gauge(state["efficiency"] * 100, "Efficiency [%]", 0, 100, "%"),
                    width="stretch")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Input power", f"{state['p_in']/1000:.2f} kW")
    c6.metric("Output power", f"{state['p_out']/1000:.2f} kW")
    c7.metric("Stator current", f"{state['stator_current']:.1f} A")
    c8.metric("Power factor", f"{state['power_factor']:.2f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Slip", f"{state['slip']*100:.1f} %")
    c10.metric("Vibration", f"{state['vibration']:.2f} mm/s")
    c11.metric("Bearing wear", f"{state['bearing_wear']*100:.1f} %")
    rul = state["rul_hours"]
    c12.metric("Est. remaining life", "∞" if rul == float("inf") else f"{rul:,.0f} h")

    # --- trends + torque-speed + ML -----------------------------------------
    left, right = st.columns([3, 2])

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        with left:
            st.subheader("Live trends")
            f = go.Figure()
            f.add_scatter(x=df["time"], y=df["speed_rpm"], name="Speed [RPM]", yaxis="y1")
            f.add_scatter(x=df["time"], y=df["t_winding"], name="Winding °C", yaxis="y2")
            f.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="time [s]",
                yaxis=dict(title="RPM"),
                yaxis2=dict(title="°C", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(f, width="stretch")

            p = go.Figure()
            p.add_scatter(x=df["time"], y=df["p_in"] / 1000, name="P in [kW]")
            p.add_scatter(x=df["time"], y=df["p_out"] / 1000, name="P out [kW]")
            p.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_title="time [s]", yaxis_title="kW",
                            legend=dict(orientation="h", y=1.2))
            st.plotly_chart(p, width="stretch")

    with right:
        st.subheader("Torque–speed characteristic")
        sp, tq = torque_speed_curve(motor.v_line, motor.frequency,
                                    motor.poles, motor.inertia, 150)
        tsc = go.Figure()
        tsc.add_scatter(x=sp, y=tq, name="EM torque", line=dict(color="#2980b9"))
        tsc.add_scatter(x=[state["speed_rpm"]], y=[state["torque_em"]],
                        mode="markers", name="operating point",
                        marker=dict(size=14, color="#e74c3c"))
        tsc.add_hline(y=ctrl["load_torque"], line_dash="dash", line_color="gray",
                      annotation_text="load")
        tsc.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="speed [RPM]", yaxis_title="torque [N·m]",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(tsc, width="stretch")

        st.subheader("🤖 ML bearing prognostics")
        if models_available():
            # Throttle + fixed seed: recompute every ML_EVERY ticks so the
            # stochastic vibration window doesn't make the readouts flicker.
            st.session_state.ml_tick += 1
            if (st.session_state.ml_pred is None
                    or st.session_state.ml_tick % ML_EVERY == 0):
                st.session_state.ml_pred = predict_for_state(
                    state, op_bearing_condition=ctrl["bearing"],
                    op_lubrication=ctrl["lube"], seed=0)
            pred = st.session_state.ml_pred
            mc1, mc2 = st.columns(2)
            mc1.plotly_chart(
                gauge(pred["failure_prob"] * 100, "Failure prob [%]", 0, 100, "%",
                      good=30, warn=70), width="stretch")
            mc2.metric("ML remaining life", f"{pred['rul_hours']:.1f} h",
                       help="From the RUL regressor (NASA-IMS feature schema).")
            mc2.metric("Anomaly", "🚨 Anomalous" if pred["is_anomaly"] else "✅ Normal",
                       help=f"IsolationForest score {pred['anomaly_score']:+.2f}")
        else:
            st.info("No trained models found. Run `python -m motor_twin.ml.train` "
                    "to enable ML prognostics.")

        st.subheader("🧠 Recommendations")
        icons = {"info": "✅", "warning": "⚠️", "critical": "🚨"}
        for r in recommend(state):
            st.markdown(f"{icons[r['severity']]} {r['message']}")


live_dashboard()
