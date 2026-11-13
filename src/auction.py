"""
auction.py
----------
Sections 15/16: auction-based allocation. Max-flow (network_flow.py) tells
us the *feasible* robot-task pairing structure; the auction decides *which*
feasible assignment is actually best, using a transparent, normalized,
multi-factor bid.

Bid components (spec Section 15), each normalized to [0, 1] where LOWER is
better-for-the-robot's-cost side and HIGHER is better for benefit side; the
final bid score is a weighted sum in [0, 1] where **higher = more
desirable pairing** (see `score_bid`). Weights come from configs/config.yaml
so they are not implicit/hidden.

This module implements a **simplified, clearly-documented auction /
CBBA-style mechanism** (spec Section 16: "do not falsely claim full CBBA
if only a simplified auction is implemented"):

    1. Every feasible robot computes a bid for every feasible task.
    2. Tasks are processed in priority order (Emergency first).
    3. For each task, the highest bidder among *still-available* robots
       wins (sequential greedy auction with priority ordering).
    4. Winning robot is removed from the pool; conflicts are resolved by
       this ordering rather than an iterative consensus protocol.

This is NOT full multi-round CBBA consensus/bundle building; it is a
single-pass, priority-ordered greedy auction, which is what is actually
implemented and tested here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from robots import Robot
from tasks import Task, is_feasible_edge
from battery import BatteryModel
from astar import astar, PathResult

DEFAULT_WEIGHTS = {
    "priority": 0.20,
    "distance_cost": 0.15,
    "path_cost": 0.10,
    "predicted_energy": 0.15,
    "soc": 0.12,
    "dod": 0.05,
    "soh": 0.05,
    "risk": 0.08,
    "deadline": 0.05,
    "workload_fairness": 0.05,
}


@dataclass
class Bid:
    robot_id: str
    task_id: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)  # raw + normalized, for explainability
    path: Optional[PathResult] = None

    def explanation(self) -> str:
        parts = [f"{k}={v:.3f}" for k, v in self.components.items()]
        return f"{self.robot_id} -> {self.task_id} | score={self.score:.3f} | " + ", ".join(parts)


def _norm(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Min-max normalize to [0,1], clamped. invert=True means lower raw
    value -> higher normalized score (used for cost-like terms)."""
    if hi <= lo:
        return 0.5
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    return (1.0 - x) if invert else x


def compute_bid(robot: Robot, task: Task, env, battery: BatteryModel,
                 weights: Dict[str, float] = None,
                 workload_counts: Dict[str, int] = None,
                 astar_step: int = 2,
                 predictor=None) -> Optional[Bid]:
    """Returns None if the robot-task pair is infeasible (type/capability/
    payload) or battery-infeasible (predicted SOC would drop below the
    robot's safety floor).

    If `predictor` (ml_prediction.Predictor) is supplied, the ML-predicted
    energy and risk (Section 19) are blended 50/50 with the existing
    physics-based/zone-based estimates for the `predicted_energy` and
    `risk` SCORE components (same weight keys, so DEFAULT_WEIGHTS still
    applies unchanged) -- this is "ML as a prediction component feeding
    the auction", not a replacement allocator. The hard battery-safety
    GATE (is_feasible below) always uses the physics-based battery.py
    model, never the ML estimate, since that is a safety constraint, not
    a scoring preference. The raw ML predictions are also stored on the
    Bid for explainability."""
    weights = weights or DEFAULT_WEIGHTS
    if not is_feasible_edge(robot, task):
        return None

    path = astar(env, (robot.x, robot.y), (task.x, task.y), robot.robot_type, step=astar_step)
    if not path.found:
        return None

    distance = path.path_length
    predicted_energy = battery.predicted_energy_wh(robot, distance, task.workload)
    if not battery.is_feasible(robot, distance, task.workload):
        return None

    # --- normalize each component (ranges chosen from domain knowledge of
    #     the generated dataset scale; documented, not hidden) ---
    comp = {}
    comp["priority"] = task.priority / 5.0                                  # higher priority -> higher score
    comp["distance_cost"] = _norm(distance, 0, 120, invert=True)            # shorter distance -> higher score
    comp["path_cost"] = _norm(path.travel_cost, 0, 180, invert=True)        # lower A* cost -> higher score
    comp["predicted_energy"] = _norm(predicted_energy, 0, 400, invert=True) # less energy -> higher score
    comp["soc"] = robot.soc / 100.0                                          # higher SOC -> higher score
    comp["dod"] = _norm(robot.dod, 0, 100, invert=True)                     # lower DoD -> higher score
    comp["soh"] = robot.soh                                                  # higher SOH -> higher score
    zone_risk = max((z.risk_level for z in env.zones_at(task.x, task.y)), default=1.0)
    comp["risk"] = _norm(zone_risk, 1, 5, invert=True)                      # lower risk -> higher score

    if predictor is not None:
        ml_features = {
            "soc": robot.soc, "dod": robot.dod, "soh": robot.soh,
            "distance": distance, "path_length": distance, "speed": robot.speed,
            "task_workload": task.workload, "task_priority": task.priority,
            "zone_risk": zone_risk,
            "robot_load": min(1.0, task.workload / max(robot.payload_capacity, 1e-6)),
            "temperature_or_environment_factor": 25.0,  # not tracked by environment.py; documented default
            "robot_type": robot.robot_type, "zone_type": task.task_type,
        }
        ml_pred = predictor.predict(ml_features)
        ml_energy_norm = _norm(ml_pred["predicted_energy"], 0, 500, invert=True)
        ml_risk_norm = _norm(ml_pred["predicted_risk"], 0, 10, invert=True)
        # Blend 50/50 with the physics/zone-based estimates already computed
        # above -- same weight keys, so DEFAULT_WEIGHTS is unaffected.
        comp["predicted_energy"] = 0.5 * comp["predicted_energy"] + 0.5 * ml_energy_norm
        comp["risk"] = 0.5 * comp["risk"] + 0.5 * ml_risk_norm
        comp["ml_predicted_energy_wh"] = ml_pred["predicted_energy"]
        comp["ml_predicted_time"] = ml_pred["predicted_time"]
        comp["ml_predicted_risk"] = ml_pred["predicted_risk"]

    slack = task.deadline - path.estimated_time(robot.speed)
    comp["deadline"] = 1.0 if slack >= 0 else max(0.0, 1.0 + slack / max(task.deadline, 1))
    # Fairness term: robots with fewer completed tasks so far get a small boost.
    wc = workload_counts or {}
    max_load = max(wc.values(), default=0) or 1
    comp["workload_fairness"] = 1.0 - (wc.get(robot.robot_id, 0) / max_load) if max_load else 1.0

    score = sum(weights[k] * comp[k] for k in weights)
    comp["predicted_energy_wh"] = predicted_energy
    comp["distance_raw"] = distance

    return Bid(robot_id=robot.robot_id, task_id=task.task_id, score=score, components=comp, path=path)


def run_auction(robots: List[Robot], tasks: List[Task], env, battery: BatteryModel,
                 weights: Dict[str, float] = None, astar_step: int = 2,
                 predictor=None
                 ) -> Dict[str, Bid]:
    """
    Single-pass, priority-ordered greedy auction (see module docstring).
    Returns robot_id -> winning Bid.
    """
    weights = weights or DEFAULT_WEIGHTS
    available_robots = {r.robot_id: r for r in robots if r.is_operational() and r.is_available()}
    workload_counts = {rid: len(robots_by_id.completed_tasks) for rid, robots_by_id in available_robots.items()}

    # Emergency + priority ordering: emergencies first, then priority desc,
    # then earliest deadline.
    ordered_tasks = sorted(
        [t for t in tasks if t.is_active() and t.status == "PENDING"],
        key=lambda t: (not t.emergency, -t.priority, t.deadline),
    )

    assignment: Dict[str, Bid] = {}
    for task in ordered_tasks:
        best_bid = None
        for rid, robot in available_robots.items():
            if rid in assignment:
                continue
            bid = compute_bid(robot, task, env, battery, weights, workload_counts, astar_step,
                               predictor=predictor)
            if bid is None:
                continue
            if best_bid is None or bid.score > best_bid.score:
                best_bid = bid
        if best_bid is not None:
            assignment[best_bid.robot_id] = best_bid

    return assignment


if __name__ == "__main__":
    from environment import Environment
    from robots import RobotFleet
    from tasks import TaskManager

    env = Environment()
    fleet = RobotFleet()
    batt = BatteryModel(fleet)
    tm = TaskManager()

    assignment = run_auction(list(fleet.robots.values()), tm.active_tasks(), env, batt)
    print(f"Auction assigned {len(assignment)} / {len(fleet.robots)} robots\n")
    for rid, bid in sorted(assignment.items(), key=lambda kv: -kv[1].score):
        print(bid.explanation())

    # Emergency tasks should be strongly represented among early/high-score wins
    emergency_ids = {t.task_id for t in tm.active_tasks() if t.emergency}
    won_emergency = [b for b in assignment.values() if b.task_id in emergency_ids]
    print(f"\nEmergency tasks won: {len(won_emergency)} / {len(emergency_ids)} total emergencies")
    assert len(assignment) > 0
    print("OK: auction module produced a valid, explainable assignment.")
