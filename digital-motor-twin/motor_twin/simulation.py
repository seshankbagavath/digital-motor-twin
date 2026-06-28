"""Dynamic digital-twin simulation: couples electromagnetics + mechanics + thermal.

The twin integrates the rotor equation of motion under a (possibly time-varying)
load, while the induction-motor equivalent circuit supplies the electromagnetic
torque at the instantaneous slip. Losses feed the lumped thermal model. A simple
empirical bearing-wear / vibration accumulator is included as a placeholder that
the ML failure-prediction stage will later replace or calibrate against the NASA
bearing dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .induction import InductionMotor
from .thermal import ThermalModel


@dataclass
class Operating:
    """Time-varying operating conditions the user can change live."""
    load_torque: float = 40.0     # mechanical load torque [N*m]
    t_ambient: float = 25.0       # ambient temperature [degC]
    airflow_m3h: float = 80.0     # cooling airflow [m^3/h]
    bearing_condition: float = 1.0  # 1.0 = healthy, 0.0 = failed
    lubrication: float = 1.0      # 1.0 = good lubrication, 0.0 = none


@dataclass
class DigitalTwin:
    motor: InductionMotor = field(default_factory=InductionMotor)
    thermal: ThermalModel = field(default_factory=ThermalModel)

    # State
    speed_rpm: float = 0.0
    t_winding: float = 25.0
    t_housing: float = 25.0
    bearing_wear: float = 0.0     # 0 -> 1 accumulated damage
    time: float = 0.0

    def reset(self, t_ambient: float = 25.0) -> None:
        self.speed_rpm = 0.0
        self.t_winding = t_ambient
        self.t_housing = t_ambient
        self.bearing_wear = 0.0
        self.time = 0.0

    def step(self, op: Operating, dt: float = 0.01) -> dict:
        """Advance the twin by dt seconds and return a full state snapshot."""
        m = self.motor

        # Electromagnetic solve at current speed
        em_torque = m.torque_at_speed(self.speed_rpm)
        slip = (m.sync_speed_rpm - self.speed_rpm) / m.sync_speed_rpm
        op_pt = m.operating_point(max(slip, 1e-6))

        # Rotor equation of motion: J dw/dt = Te - Tload - B*w
        w = 2.0 * math.pi * self.speed_rpm / 60.0
        net_torque = em_torque - op.load_torque - m.friction_coeff * w
        domega = net_torque / m.inertia
        w_new = max(w + domega * dt, 0.0)
        self.speed_rpm = 60.0 * w_new / (2.0 * math.pi)

        # Thermal: total loss heats the winding
        q_loss = (
            op_pt["loss_stator_cu"]
            + op_pt["loss_rotor_cu"]
            + op_pt["loss_core"]
            + op_pt["loss_mech"]
        )
        self.t_winding, self.t_housing = self.thermal.step(
            self.t_winding, self.t_housing, q_loss,
            op.t_ambient, op.airflow_m3h, dt,
        )

        # Empirical bearing wear: accelerated by heat, load, poor lube & bad bearing.
        # Placeholder degradation law -- ML stage will refine against real data.
        load_factor = abs(op.load_torque) / max(m.rated_power / m.sync_speed_rad, 1.0)
        heat_factor = max(self.t_winding - 60.0, 0.0) / 60.0
        lube_factor = (2.0 - op.lubrication)
        wear_rate = 1e-6 * (1.0 + load_factor) * (1.0 + heat_factor) * lube_factor
        wear_rate *= (2.0 - op.bearing_condition)
        self.bearing_wear = min(self.bearing_wear + wear_rate * dt, 1.0)

        # Vibration proxy: grows with wear, imbalance and speed
        vibration = (
            0.5
            + 4.0 * self.bearing_wear
            + 0.002 * self.speed_rpm * (1.0 - op.bearing_condition)
        )

        # Remaining useful life (very rough, physics-based placeholder in hours)
        rul_hours = float("inf")
        if wear_rate > 0:
            rul_hours = (1.0 - self.bearing_wear) / (wear_rate * 3600.0)

        self.time += dt

        return {
            "time": self.time,
            "speed_rpm": self.speed_rpm,
            "slip": op_pt["slip"],
            "torque_em": em_torque,
            "shaft_torque": op_pt["shaft_torque"],
            "load_torque": op.load_torque,
            "stator_current": op_pt["stator_current"],
            "power_factor": op_pt["power_factor"],
            "p_in": op_pt["p_in"],
            "p_out": op_pt["p_out"],
            "efficiency": op_pt["efficiency"],
            "q_loss": q_loss,
            "t_winding": self.t_winding,
            "t_housing": self.t_housing,
            "bearing_wear": self.bearing_wear,
            "vibration": vibration,
            "rul_hours": rul_hours,
        }

    def run(self, op: Operating, duration: float = 10.0, dt: float = 0.01) -> list[dict]:
        """Run a fixed-duration simulation, returning a list of state snapshots."""
        steps = int(duration / dt)
        return [self.step(op, dt) for _ in range(steps)]
