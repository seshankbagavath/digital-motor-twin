"""Lumped-capacitance thermal model of the motor.

Two thermal nodes:
  * winding (copper) -- where I^2 R and most loss is dissipated
  * housing/frame    -- coupled to ambient via the cooling system

Energy balance per node:
    C_node * dT/dt = Q_in - Q_conducted_out

This is a first-order RC thermal network, the standard lumped approach for
motor temperature estimation when a full FE thermal map is not required.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThermalModel:
    # Thermal capacitances [J/K]
    c_winding: float = 1500.0      # copper + slot mass
    c_housing: float = 9000.0      # frame/housing mass

    # Thermal conductances [W/K]
    g_wind_house: float = 12.0     # winding -> housing path
    g_house_amb_base: float = 6.0  # housing -> ambient (still air)

    # Cooling: forced-air increases housing->ambient conductance
    cooling_gain: float = 0.10     # extra W/K per (m^3/h) of airflow

    def housing_to_ambient_G(self, airflow_m3h: float) -> float:
        """Effective housing->ambient conductance given cooling airflow."""
        return self.g_house_amb_base + self.cooling_gain * max(airflow_m3h, 0.0)

    def step(
        self,
        t_winding: float,
        t_housing: float,
        q_loss: float,
        t_ambient: float,
        airflow_m3h: float,
        dt: float,
    ) -> tuple[float, float]:
        """Advance winding & housing temperatures by dt seconds (explicit Euler).

        q_loss: total electrical + mechanical loss power [W], deposited in the
        winding node (a conservative simplification -- all heat enters copper).
        Returns (t_winding_new, t_housing_new) in the same units as inputs (degC).
        """
        g_wh = self.g_wind_house
        g_ha = self.housing_to_ambient_G(airflow_m3h)

        # Heat flows
        q_wind_to_house = g_wh * (t_winding - t_housing)
        q_house_to_amb = g_ha * (t_housing - t_ambient)

        dT_wind = (q_loss - q_wind_to_house) / self.c_winding
        dT_house = (q_wind_to_house - q_house_to_amb) / self.c_housing

        return t_winding + dT_wind * dt, t_housing + dT_house * dt

    def steady_state(
        self, q_loss: float, t_ambient: float, airflow_m3h: float
    ) -> tuple[float, float]:
        """Analytical steady-state winding & housing temperatures."""
        g_wh = self.g_wind_house
        g_ha = self.housing_to_ambient_G(airflow_m3h)
        t_house = t_ambient + q_loss / g_ha
        t_wind = t_house + q_loss / g_wh
        return t_wind, t_house
