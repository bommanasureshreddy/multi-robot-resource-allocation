"""
events.py
---------
Section 11 (Emergency Handling) + Section 12 (Dynamic Reallocation
triggers): loads data/dataset/events/events.csv and dispatches each event
to the correct effect on the live simulation state (robots, tasks,
environment). This module is the "Check events" step in the Section 12 /
36 simulation loop; reallocation.py calls apply_event() for each due event.

Event types actually present in the dataset (verified against
events.csv): NEW_TASK, NEW_EMERGENCY, TASK_CANCELLED, ROBOT_FAILURE,
LOW_BATTERY, ENVIRONMENT_CHANGE. TASK_DELAY does not occur in this
dataset instance; the handler exists but will simply never fire here --
documented rather than fabricated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from robots import RobotFleet
from tasks import TaskManager
from environment import Environment


@dataclass
class Event:
    event_id: str
    time: float
    event_type: str
    task_id: Optional[str]
    robot_id: Optional[str]
    zone_id: Optional[str]
    severity: float


class EventManager:
    def __init__(self, dataset_root: str = "data/dataset"):
        path = os.path.join(dataset_root, "events", "events.csv")
        df = pd.read_csv(path)
        self.events: List[Event] = []
        for _, r in df.iterrows():
            self.events.append(Event(
                event_id=r["event_id"],
                time=float(r["time"]),
                event_type=r["event_type"],
                task_id=(r["task_id"] if isinstance(r["task_id"], str) else None),
                robot_id=(r["robot_id"] if isinstance(r["robot_id"], str) else None),
                zone_id=(r["zone_id"] if isinstance(r["zone_id"], str) else None),
                severity=float(r["severity"]),
            ))
        self.events.sort(key=lambda e: e.time)

    def event_times(self) -> List[float]:
        return sorted(set(e.time for e in self.events))

    def events_at(self, t: float) -> List[Event]:
        return [e for e in self.events if e.time == t]

    def events_up_to(self, t: float, applied_ids: set) -> List[Event]:
        return [e for e in self.events if e.time <= t and e.event_id not in applied_ids]


def apply_event(event: Event, fleet: RobotFleet, tm: TaskManager, env: Environment, log=print) -> dict:
    """
    Applies one event's effect to the live (in-memory) simulation state.
    Returns a small dict describing what changed, for allocation_history /
    explainability logging.
    """
    effect = {"event_id": event.event_id, "event_type": event.event_type, "detail": ""}

    if event.event_type in ("NEW_TASK", "NEW_EMERGENCY"):
        task = tm.tasks.get(event.task_id)
        if task is None:
            effect["detail"] = f"unknown task {event.task_id}"
            return effect
        if event.event_type == "NEW_EMERGENCY":
            task.emergency = True
            task.priority = max(task.priority, 5)
        effect["detail"] = (f"task {task.task_id} released (priority={task.priority}, "
                             f"emergency={task.emergency})")

    elif event.event_type == "TASK_CANCELLED":
        task = tm.tasks.get(event.task_id)
        if task is None:
            effect["detail"] = f"unknown task {event.task_id}"
            return effect
        if task.assigned_robot:
            robot = fleet.robots.get(task.assigned_robot)
            if robot is not None and robot.current_task == task.task_id:
                robot.current_task = None
                robot.status = "IDLE"
        task.status = "CANCELLED"
        task.assigned_robot = None
        effect["detail"] = f"task {task.task_id} cancelled and removed from active graph"

    elif event.event_type == "ROBOT_FAILURE":
        robot = fleet.robots.get(event.robot_id)
        if robot is None:
            effect["detail"] = f"unknown robot {event.robot_id}"
            return effect
        # If it was mid-task, release the task back to PENDING so it can
        # be picked up by another feasible robot on the next allocation
        # cycle (Scenario 4: robot failure -> reallocate its task).
        if robot.current_task:
            task = tm.tasks.get(robot.current_task)
            if task is not None and task.status in ("ASSIGNED", "IN_PROGRESS"):
                task.status = "PENDING"
                task.assigned_robot = None
                effect["detail"] += f"; released task {task.task_id} back to PENDING"
        robot.status = "FAILED"
        robot.current_task = None
        effect["detail"] = f"robot {robot.robot_id} FAILED, removed from graph" + effect["detail"]

    elif event.event_type == "LOW_BATTERY":
        robot = fleet.robots.get(event.robot_id)
        if robot is None:
            effect["detail"] = f"unknown robot {event.robot_id}"
            return effect
        # We do NOT fabricate a specific SOC value the dataset doesn't
        # give us; the robot's actual SOC already reflects its true
        # simulated consumption trajectory (battery.py). This event is
        # logged as a flagged warning; feasibility/auction scoring already
        # naturally penalizes/excludes low-SOC robots via battery.py.
        effect["detail"] = (f"LOW_BATTERY warning for {robot.robot_id} "
                             f"(current SOC={robot.soc:.1f}%, min_soc={robot.min_soc:.1f}%)")

    elif event.event_type == "ENVIRONMENT_CHANGE":
        zone = next((z for z in env.zones if z.zone_id == event.zone_id), None)
        if zone is None:
            effect["detail"] = f"unknown zone {event.zone_id}"
            return effect
        old_risk = zone.risk_level
        zone.risk_level = min(5.0, zone.risk_level + max(1.0, event.severity - old_risk))
        effect["detail"] = f"zone {zone.zone_id} risk_level {old_risk:.1f} -> {zone.risk_level:.1f}"

    elif event.event_type == "TASK_DELAY":
        task = tm.tasks.get(event.task_id)
        if task is not None:
            task.status = "DELAYED"
            effect["detail"] = f"task {task.task_id} marked DELAYED"

    else:
        effect["detail"] = f"unhandled event type {event.event_type}"

    return effect


if __name__ == "__main__":
    em = EventManager()
    print(f"Loaded {len(em.events)} events, spanning t={em.event_times()[0]} to t={em.event_times()[-1]}")
    from collections import Counter
    print("Event type counts:", Counter(e.event_type for e in em.events))

    fleet = RobotFleet()
    tm = TaskManager()
    env = Environment()

    # Apply a ROBOT_FAILURE event and verify the robot is marked FAILED
    failure_event = next(e for e in em.events if e.event_type == "ROBOT_FAILURE")
    fleet.robots[failure_event.robot_id].status = "BUSY"
    fleet.robots[failure_event.robot_id].current_task = "T05"
    tm.tasks["T05"].status = "ASSIGNED"
    tm.tasks["T05"].assigned_robot = failure_event.robot_id

    effect = apply_event(failure_event, fleet, tm, env)
    print("\nApplied:", effect)
    assert fleet.robots[failure_event.robot_id].status == "FAILED"
    assert tm.tasks["T05"].status == "PENDING"
    print("OK: robot failure correctly frees its task back to PENDING and marks robot FAILED.")
