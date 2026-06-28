"""Digital Twin of a 3-phase induction motor.

Public API:
    InductionMotor  -- steady-state equivalent-circuit electromagnetics
    ThermalModel    -- lumped-capacitance thermal network
    DigitalTwin     -- dynamic coupled simulation (the "twin")
    Operating       -- live operating conditions
    recommend       -- rule-based engineering guidance
"""
from .induction import InductionMotor
from .thermal import ThermalModel
from .simulation import DigitalTwin, Operating
from .recommendations import recommend

__all__ = [
    "InductionMotor",
    "ThermalModel",
    "DigitalTwin",
    "Operating",
    "recommend",
]
__version__ = "0.1.0"
