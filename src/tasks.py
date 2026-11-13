"""
tasks.py
--------
Task model and the robot-task FEASIBILITY LAYER (spec Section 4).

A robot-task edge is only allowed into the allocation graph when the robot
satisfies: robot type/zone-domain compatibility, required capability,
payload, and (checked separately in battery.py) battery feasibility.
This module answers "is this edge even allowed to exist", not "how good is
this edge" (that scoring is auction.py's job).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

from robots import Robot


@dataclass
class Task:
    task_id: str
    x: float
    y: float
    zone_id: str
    task_type: str            # LAND / WATER / AIR (matches zone domain)
    required_robot_type: str  # primary required type, e.g. "UGV"
    workload: float
    priority: int              # 1..5
    deadline: float
    release_time: float
    emergency: bool
    status: str                # PENDING / ASSIGNED / IN_PROGRESS / COMPLETED / CANCELLED / DELAYED

    # From task_requirements.csv
    required_zone: str = ""
    required_capability: str = ""
    allowed_robot_types: Set[str] = field(default_factory=set)
    minimum_payload: float = 0.0

    assigned_robot: Optional[str] = None

    def is_active(self) -> bool:
        return self.status in ("PENDING", "ASSIGNED", "IN_PROGRESS", "DELAYED")

    def priority_label(self) -> str:
        return {1: "Low", 2: "Normal", 3: "High", 4: "Critical", 5: "Emergency"}.get(self.priority, "Unknown")


class TaskManager:
    def __init__(self, dataset_root: str = "data/dataset"):
        self.dataset_root = dataset_root
        self.tasks_dir = os.path.join(dataset_root, "tasks")
        self.tasks: dict[str, Task] = {}
        self._load()

    def _load(self):
        tasks_df = pd.read_csv(os.path.join(self.tasks_dir, "tasks.csv"))
        req_df = pd.read_csv(os.path.join(self.tasks_dir, "task_requirements.csv"))
        req_by_id = {row["task_id"]: row for _, row in req_df.iterrows()}

        for _, r in tasks_df.iterrows():
            tid = r["task_id"]
            req = req_by_id.get(tid, {})
            allowed_types = set()
            if "allowed_robot_types" in req and isinstance(req["allowed_robot_types"], str):
                allowed_types = set(req["allowed_robot_types"].split("|"))
            self.tasks[tid] = Task(
                task_id=tid,
                x=float(r["x"]),
                y=float(r["y"]),
                zone_id=r["zone_id"],
                task_type=r["task_type"] if "task_type" not in ("",) else r["task_type"],
                required_robot_type=r["required_robot_type"],
                workload=float(r["workload"]),
                priority=int(r["priority"]),
                deadline=float(r["deadline"]),
                release_time=float(r["release_time"]),
                emergency=bool(r["emergency"]),
                status=r["status"],
                required_zone=req.get("required_zone", r["zone_id"]),
                required_capability=req.get("required_capability", ""),
                allowed_robot_types=allowed_types or {r["required_robot_type"]},
                minimum_payload=float(req.get("minimum_payload", 0.0)),
            )

    def active_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.is_active()]

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for t in self.tasks.values():
            rows.append({
                "task_id": t.task_id, "task_type": t.task_type, "priority": t.priority,
                "priority_label": t.priority_label(), "emergency": t.emergency,
                "status": t.status, "assigned_robot": t.assigned_robot,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        counts = {}
        for t in self.tasks.values():
            counts[t.task_type] = counts.get(t.task_type, 0) + 1
        emergencies = sum(1 for t in self.tasks.values() if t.emergency)
        return f"Tasks: {len(self.tasks)} -> {counts}, emergencies={emergencies}"


# ------------------------------------------------------------------------
# Feasibility layer (Section 4): decides whether a robot-task EDGE may
# exist at all, before any cost/bid is computed.
# ------------------------------------------------------------------------
def is_feasible_edge(robot: Robot, task: Task) -> bool:
    if not robot.is_operational():
        return False
    if task.status not in ("PENDING", "DELAYED"):
        return False

    # 1. Robot type must be in the task's allowed set (handles special
    #    tasks executable by more than one robot type, e.g. UGV|UAV).
    if robot.robot_type not in task.allowed_robot_types:
        return False

    # 2. Capability requirement.
    if task.required_capability and not robot.has_capability(task.required_capability):
        return False

    # 3. Payload requirement.
    if robot.payload_capacity < task.minimum_payload:
        return False

    # (Battery feasibility and zone/obstacle traversability are checked
    #  separately by battery.py and environment.py / astar.py, since they
    #  require a computed path and predicted energy cost.)
    return True


def feasible_robots_for_task(robots: List[Robot], task: Task) -> List[Robot]:
    return [r for r in robots if is_feasible_edge(r, task)]


if __name__ == "__main__":
    from robots import RobotFleet

    tm = TaskManager()
    print(tm.summary())

    fleet = RobotFleet()
    t3 = tm.tasks["T03"]  # INSPECTION, allowed UGV|UAV per requirements.csv
    print(f"\nTask T03: type={t3.task_type}, capability={t3.required_capability}, "
          f"allowed_types={t3.allowed_robot_types}, emergency={t3.emergency}")

    feasible = feasible_robots_for_task(list(fleet.robots.values()), t3)
    print("Feasible robots for T03:", [r.robot_id for r in feasible])

    assert len(tm.tasks) == 30, "Expected 30 tasks per spec Section 3"
    counts = {t.task_type: 0 for t in tm.tasks.values()}
    for t in tm.tasks.values():
        counts[t.task_type] += 1
    print("Counts by type:", counts)
