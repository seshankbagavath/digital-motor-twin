"""Rule-based engineering recommendations derived from the twin's state.

These are deterministic guardrail rules. The ML stage adds learned anomaly /
failure scoring on top; the two are intended to be shown side by side.
"""
from __future__ import annotations


def recommend(state: dict) -> list[dict]:
    """Return a list of {severity, message} recommendations for a state snapshot.

    severity in {"info", "warning", "critical"}.
    """
    recs: list[dict] = []

    tw = state["t_winding"]
    if tw > 130:
        recs.append({"severity": "critical",
                     "message": f"Winding at {tw:.0f}°C exceeds insulation limit — reduce load or increase cooling immediately."})
    elif tw > 100:
        recs.append({"severity": "warning",
                     "message": f"Winding temperature high ({tw:.0f}°C). Increase cooling airflow or lower load."})

    eff = state["efficiency"]
    if 0 < eff < 0.75:
        recs.append({"severity": "warning",
                     "message": f"Efficiency low ({eff*100:.0f}%). Motor is likely lightly loaded or oversized for the duty."})

    vib = state["vibration"]
    if vib > 4.0:
        recs.append({"severity": "critical",
                     "message": f"Excessive vibration ({vib:.1f} mm/s). Inspect/balance rotor and check bearings."})
    elif vib > 2.8:
        recs.append({"severity": "warning",
                     "message": f"Elevated vibration ({vib:.1f} mm/s). Schedule a balancing check."})

    wear = state["bearing_wear"]
    if wear > 0.8:
        recs.append({"severity": "critical",
                     "message": "Bearing wear critical — schedule bearing replacement now."})
    elif wear > 0.5:
        recs.append({"severity": "warning",
                     "message": "Bearing wear past half-life — plan a replacement at the next maintenance window."})

    rul = state.get("rul_hours", float("inf"))
    if rul != float("inf") and rul < 500:
        recs.append({"severity": "warning",
                     "message": f"Estimated remaining useful life ≈ {rul:.0f} h. Order spares."})

    if not recs:
        recs.append({"severity": "info", "message": "All monitored parameters within normal limits."})
    return recs
