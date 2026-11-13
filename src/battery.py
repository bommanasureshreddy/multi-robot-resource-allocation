"""
battery.py
----------
Battery-aware state tracking for each robot: State of Charge (SOC), Depth
of Discharge (DoD), and State of Health (SOH / battery degradation).

This is explicitly a SIMULATION-LEVEL degradation model, not an
electrochemical one (per spec Section 8: "Do not claim a highly accurate
electrochemical battery model"). It captures the qualitatively correct
relationship -- deeper, more frequent discharges degrade health faster --
using a simple linear DoD-weighted throughput model, and documents that
simplification openly rather than hiding it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from robots import RobotFleet, Robot


@dataclass
class BatteryParams:
    battery_capacity: float      # Wh
    initial_soc: float           # %
    initial_soh: float           # 0-1
    initial_dod: float           # %
    min_soc: float                # % — safety floor
    energy_per_distance: float   # Wh per unit distance travelled
    task_energy_rate: float      # Wh per unit workload
    degradation_factor: float    # SOH loss per unit (DoD% x energy Wh) throughput


class BatteryModel:
    """
    Loads per-robot battery parameters and applies them to Robot objects
    (robots.py) so allocation code can read robot.soc / robot.dod /
    robot.soh directly.
    """

    def __init__(self, fleet: RobotFleet, dataset_root: str = "data/dataset"):
        self.fleet = fleet
        self.params: Dict[str, BatteryParams] = {}
        self._load(dataset_root)
        self._apply_initial_state()

    def _load(self, dataset_root: str):
        df = pd.read_csv(os.path.join(dataset_root, "robots", "robot_battery.csv"))
        for _, r in df.iterrows():
            self.params[r["robot_id"]] = BatteryParams(
                battery_capacity=float(r["battery_capacity"]),
                initial_soc=float(r["initial_soc"]),
                initial_soh=float(r["initial_soh"]),
                initial_dod=float(r["initial_dod"]),
                min_soc=float(r["min_soc"]),
                energy_per_distance=float(r["energy_per_distance"]),
                task_energy_rate=float(r["task_energy_rate"]),
                degradation_factor=float(r["degradation_factor"]),
            )

    def _apply_initial_state(self):
        for rid, p in self.params.items():
            robot = self.fleet.robots.get(rid)
            if robot is None:
                continue
            robot.soc = p.initial_soc
            robot.dod = p.initial_dod
            robot.soh = p.initial_soh
            robot.min_soc = p.min_soc

    # ------------------------------------------------------------------
    # Predictive (used by auction bidding, before committing to a task)
    # ------------------------------------------------------------------
    def predicted_energy_wh(self, robot: Robot, distance: float, workload: float) -> float:
        p = self.params[robot.robot_id]
        return distance * p.energy_per_distance + workload * p.task_energy_rate

    def predicted_soc_after(self, robot: Robot, distance: float, workload: float) -> float:
        p = self.params[robot.robot_id]
        energy_wh = self.predicted_energy_wh(robot, distance, workload)
        soc_drop_pct = 100.0 * energy_wh / p.battery_capacity
        return max(0.0, robot.soc - soc_drop_pct)

    def is_feasible(self, robot: Robot, distance: float, workload: float, safety_margin: float = 2.0) -> bool:
        """A task is battery-feasible if predicted post-task SOC stays
        above the robot's minimum safe SOC (+ a small safety margin)."""
        predicted = self.predicted_soc_after(robot, distance, workload)
        return predicted >= (robot.min_soc + safety_margin)

    # ------------------------------------------------------------------
    # Actual state update (applied once a task is committed/executed)
    # ------------------------------------------------------------------
    def consume(self, robot: Robot, distance: float, workload: float):
        p = self.params[robot.robot_id]
        energy_wh = self.predicted_energy_wh(robot, distance, workload)
        soc_drop_pct = 100.0 * energy_wh / p.battery_capacity

        prev_soc = robot.soc
        robot.soc = max(0.0, robot.soc - soc_drop_pct)
        # DoD tracks how deep the *current* discharge cycle has gone from
        # the last "full" reference point; here we accumulate it simply as
        # (100 - soc) with a floor, sufficient for a simulation-level model.
        robot.dod = max(robot.dod, 100.0 - robot.soc)

        # Degradation: proportional to (DoD-weighted) energy throughput.
        # Deeper discharges (higher DoD) degrade health faster per Wh.
        dod_weight = 1.0 + (robot.dod / 100.0)
        soh_loss = p.degradation_factor * energy_wh * dod_weight / p.battery_capacity
        robot.soh = max(0.0, robot.soh - soh_loss)

        robot.total_distance_travelled += distance
        return prev_soc - robot.soc  # actual SOC consumed

    def recharge(self, robot: Robot, rate: float, dt: float = 1.0):
        """Charging increases SOC and resets DoD reference for a fresh
        cycle; it does not restore SOH (degradation is permanent, matching
        real battery behaviour at the level of fidelity this project
        targets)."""
        robot.soc = min(100.0, robot.soc + rate * dt)
        if robot.soc > 95.0:
            robot.dod = 0.0

    def summary(self) -> str:
        rows = []
        for rid, robot in self.fleet.robots.items():
            rows.append(f"{rid:6s} SOC={robot.soc:5.1f}%  DoD={robot.dod:5.1f}%  SOH={robot.soh:.3f}")
        return "\n".join(rows)


if __name__ == "__main__":
    fleet = RobotFleet()
    batt = BatteryModel(fleet)
    print(batt.summary())

    r = fleet.robots["UGV01"]
    print("\n--- Simulating a task for UGV01 ---")
    print("Before:", r)
    consumed = batt.consume(r, distance=15.0, workload=6.3)
    print(f"SOC consumed: {consumed:.2f}%")
    print("After: ", r)
    feasible = batt.is_feasible(r, distance=15.0, workload=6.3)
    print("Feasible for another similar task now?", feasible)
