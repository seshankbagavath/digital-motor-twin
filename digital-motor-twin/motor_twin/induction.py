"""Steady-state electromagnetic model of a 3-phase induction motor.

Uses the classic per-phase equivalent circuit with a Thevenin reduction to
compute developed torque, currents, power flow and losses as a function of
slip. All quantities are SI unless noted. Per-phase voltage is line-to-line
voltage / sqrt(3) (star/wye assumption).

References: Fitzgerald, "Electric Machinery"; Chapman, "Electric Machinery
Fundamentals", ch. 7 (induction motor equivalent circuit & torque-speed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class InductionMotor:
    """Nameplate + equivalent-circuit parameters for a 3-phase induction motor.

    Defaults model a ~7.5 kW / 10 hp, 400 V, 50 Hz, 4-pole industrial machine.
    Equivalent-circuit resistances/reactances are per-phase, referred to the
    stator. They can be estimated from no-load + locked-rotor tests; the
    defaults here are representative textbook values.
    """

    # Nameplate
    v_line: float = 400.0        # line-to-line RMS voltage [V]
    frequency: float = 50.0      # supply frequency [Hz]
    poles: int = 4               # number of poles
    rated_power: float = 7500.0  # rated mechanical output [W]

    # Per-phase equivalent circuit (referred to stator) [ohm]
    r1: float = 0.50   # stator resistance
    x1: float = 1.20   # stator leakage reactance @ rated freq
    r2: float = 0.55   # rotor resistance (referred)
    x2: float = 1.20   # rotor leakage reactance (referred)
    xm: float = 35.0   # magnetizing reactance
    rc: float = 360.0  # core-loss resistance (shunt)

    # Mechanical
    inertia: float = 0.05            # rotor + load inertia J [kg*m^2]
    friction_coeff: float = 0.002    # viscous damping B [N*m*s/rad]
    mech_loss: float = 80.0          # constant friction+windage loss [W]

    # Derived, filled in __post_init__
    sync_speed_rpm: float = field(init=False)
    sync_speed_rad: float = field(init=False)

    def __post_init__(self) -> None:
        # Synchronous speed: ns = 120 f / P (rpm)
        self.sync_speed_rpm = 120.0 * self.frequency / self.poles
        self.sync_speed_rad = 2.0 * math.pi * self.sync_speed_rpm / 60.0

    # -- helpers -----------------------------------------------------------
    @property
    def v_phase(self) -> float:
        """Per-phase RMS voltage (wye)."""
        return self.v_line / math.sqrt(3.0)

    def _thevenin(self) -> tuple[float, float, float]:
        """Thevenin equivalent (Vth, Rth, Xth) seen by the rotor branch."""
        v1 = self.v_phase
        # Vth = V1 * jXm / (R1 + j(X1 + Xm))
        denom = complex(self.r1, self.x1 + self.xm)
        vth = abs(v1 * complex(0, self.xm) / denom)
        # Zth = (jXm)(R1 + jX1) / (R1 + j(X1+Xm))
        zth = (complex(0, self.xm) * complex(self.r1, self.x1)) / denom
        return vth, zth.real, zth.imag

    # -- core solve --------------------------------------------------------
    def operating_point(self, slip: float) -> dict:
        """Solve the equivalent circuit at a given slip.

        Returns developed torque, currents, power flow, losses and efficiency.
        Slip s = (ns - n) / ns. s=1 at standstill, s->0 near synchronous.
        """
        s = max(slip, 1e-6)  # avoid divide-by-zero at exact synchronism
        vth, rth, xth = self._thevenin()
        ws = self.sync_speed_rad

        # Developed (air-gap) torque via Thevenin form
        r2_s = self.r2 / s
        z_sq = (rth + r2_s) ** 2 + (xth + self.x2) ** 2
        torque_em = (3.0 * vth ** 2 * r2_s) / (ws * z_sq)

        # Rotor current (referred), from Thevenin circuit
        i2 = vth / math.sqrt(z_sq)

        # Power flow
        p_airgap = torque_em * ws                      # air-gap power
        p_rotor_cu = 3.0 * i2 ** 2 * self.r2           # = s * p_airgap
        p_mech_dev = p_airgap - p_rotor_cu             # developed mech power
        p_out = p_mech_dev - self.mech_loss            # shaft output

        # Stator current: rotor branch + magnetizing/core branch
        v1 = complex(self.v_phase, 0.0)
        z_rotor = complex(self.r1 + r2_s, self.x1 + self.x2)
        i_rotor = v1 / z_rotor
        # shunt magnetizing + core loss branch across the air gap
        e = v1 - i_rotor * complex(self.r1, self.x1)
        i_core = e / self.rc
        i_mag = e / complex(0, self.xm)
        i_stator = i_rotor + i_core + i_mag
        i1 = abs(i_stator)

        p_stator_cu = 3.0 * i1 ** 2 * self.r1
        p_core = 3.0 * (abs(e) ** 2) / self.rc

        p_in = 3.0 * (v1 * i_stator.conjugate()).real  # real input power
        pf = p_in / (3.0 * self.v_phase * i1) if i1 > 0 else 0.0
        efficiency = p_out / p_in if p_in > 0 else 0.0

        speed_rad = ws * (1.0 - s)
        speed_rpm = self.sync_speed_rpm * (1.0 - s)
        shaft_torque = p_out / speed_rad if speed_rad > 0 else 0.0

        return {
            "slip": s,
            "speed_rpm": speed_rpm,
            "speed_rad": speed_rad,
            "torque_em": torque_em,         # developed electromagnetic torque
            "shaft_torque": shaft_torque,   # net torque at shaft
            "stator_current": i1,
            "rotor_current": i2,
            "power_factor": pf,
            "p_in": p_in,
            "p_out": max(p_out, 0.0),
            "p_airgap": p_airgap,
            "loss_stator_cu": p_stator_cu,
            "loss_rotor_cu": p_rotor_cu,
            "loss_core": p_core,
            "loss_mech": self.mech_loss,
            "efficiency": max(min(efficiency, 1.0), 0.0),
        }

    def torque_at_speed(self, speed_rpm: float) -> float:
        """Electromagnetic torque developed at a given mechanical speed."""
        slip = (self.sync_speed_rpm - speed_rpm) / self.sync_speed_rpm
        return self.operating_point(slip)["torque_em"]

    def torque_speed_curve(self, n: int = 200) -> tuple[list[float], list[float]]:
        """Return (speed_rpm[], torque[]) sweeping slip from 1 -> ~0."""
        speeds, torques = [], []
        for i in range(1, n + 1):
            s = 1.0 - (i - 1) / n  # 1 down to ~0
            op = self.operating_point(s)
            speeds.append(op["speed_rpm"])
            torques.append(op["torque_em"])
        return speeds, torques
