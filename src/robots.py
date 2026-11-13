"""
robots.py
---------
Heterogeneous robot fleet model (UGV / USV / UAV).

IMPORTANT (Section 7 of the spec): the CSVs in data/dataset/robots/ are the
*initial* state only. During a simulation run, robot state (position, SOC,
DoD, SOH, status, current task) is mutated **in memory** on the Robot
object. The CSV is never rewritten mid-run; a separate history log
(robot_state_history.csv) records every state snapshot -- see
logging_utils.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Robot:
    robot_id: str
    robot_type: str  # UGV / USV / UAV
    x: float
    y: float
    speed: float
    payload_capacity: float
    communication_range: float
    status: str  # IDLE / BUSY / CHARGING / FAILED

    capabilities: Dict[str, int] = field(default_factory=dict)

    # Battery state (populated from battery.py's BatteryModel, referenced
    # here for convenience so allocation code has one object to query)
    soc: float = 100.0
    dod: float = 0.0
    soh: float = 1.0
    min_soc: float = 15.0

    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    total_distance_travelled: float = 0.0

    def has_capability(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, 0))

    def is_available(self) -> bool:
        return self.status == "IDLE"

    def is_operational(self) -> bool:
        return self.status != "FAILED"

    def domain_ok(self, required_zone_type: str) -> bool:
        """True if this robot's operating domain matches a zone type
        (LAND -> UGV, WATER -> USV, AIR -> UAV). NOTE: dataset.csv's
        task_type column is the task CATEGORY (SEARCH/RESCUE/...), not the
        zone domain -- use task.required_zone / environment.zones_at() for
        domain checks, and tasks.is_feasible_edge() for the full
        robot-task compatibility check (type + capability + payload)."""
        mapping = {"LAND": "UGV", "WATER": "USV", "AIR": "UAV"}
        return mapping.get(required_zone_type) == self.robot_type

    def __repr__(self):
        return (
            f"Robot({self.robot_id}, {self.robot_type}, pos=({self.x:.1f},{self.y:.1f}), "
            f"SOC={self.soc:.1f}%, status={self.status}, task={self.current_task})"
        )


class RobotFleet:
    """Loads the robot fleet from CSV and exposes it as mutable Robot
    objects, keyed by robot_id."""

    CAPABILITY_COLUMNS = [
        "search", "rescue", "transport", "surveillance", "inspection",
        "medical", "land_operation", "water_operation", "air_operation",
    ]

    def __init__(self, dataset_root: str = "data/dataset"):
        self.dataset_root = dataset_root
        self.robots_dir = os.path.join(dataset_root, "robots")
        self.robots: Dict[str, Robot] = {}
        self._load()

    def _load(self):
        robots_df = pd.read_csv(os.path.join(self.robots_dir, "robots.csv"))
        caps_df = pd.read_csv(os.path.join(self.robots_dir, "robot_capabilities.csv"))
        caps_by_id = {row["robot_id"]: row for _, row in caps_df.iterrows()}

        for _, r in robots_df.iterrows():
            rid = r["robot_id"]
            cap_row = caps_by_id.get(rid, {})
            capabilities = {
                c: int(cap_row[c]) for c in self.CAPABILITY_COLUMNS if c in cap_row
            }
            self.robots[rid] = Robot(
                robot_id=rid,
                robot_type=r["robot_type"],
                x=float(r["x"]),
                y=float(r["y"]),
                speed=float(r["speed"]),
                payload_capacity=float(r["payload_capacity"]),
                communication_range=float(r["communication_range"]),
                status=r["status"],
                capabilities=capabilities,
            )

    def by_type(self, robot_type: str) -> List[Robot]:
        return [r for r in self.robots.values() if r.robot_type == robot_type]

    def available(self) -> List[Robot]:
        return [r for r in self.robots.values() if r.is_available()]

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.robots.values():
            rows.append({
                "robot_id": r.robot_id, "robot_type": r.robot_type,
                "x": r.x, "y": r.y, "status": r.status,
                "soc": r.soc, "dod": r.dod, "soh": r.soh,
                "current_task": r.current_task,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        counts = {}
        for r in self.robots.values():
            counts[r.robot_type] = counts.get(r.robot_type, 0) + 1
        return f"Fleet: {len(self.robots)} robots -> {counts}"


if __name__ == "__main__":
    fleet = RobotFleet()
    print(fleet.summary())
    for r in list(fleet.robots.values())[:3]:
        print(r, "capabilities:", r.capabilities)
    assert len(fleet.robots) == 12, "Expected 12 robots per spec Section 2"
    assert len(fleet.by_type("UGV")) == 4
    assert len(fleet.by_type("USV")) == 4
    assert len(fleet.by_type("UAV")) == 4
    print("OK: 4 UGV / 4 USV / 4 UAV confirmed.")
