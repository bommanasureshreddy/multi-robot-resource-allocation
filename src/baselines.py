"""
baselines.py
------------
Section 26: baseline allocation strategies for research evaluation. Each
function has the SAME signature as auction.run_auction so it can be
plugged into reallocation.Simulator(allocation_fn=...) -- meaning every
baseline runs through the identical dynamic-event loop (battery
consumption, events, fairness/logging), and only the allocation POLICY
differs. This is what makes the comparison in notebook 12 fair.

Baseline 1 -- Nearest Robot / Greedy:
    Tasks processed in arbitrary (task_id) order, not priority-sorted.
    Each task goes to whichever available, type/capability-feasible
    robot is physically NEAREST -- no battery, priority, or fairness
    weighting at all (a robot can be driven to 0% SOC by this baseline;
    that is the intended, honest contrast with the battery-aware methods).

Baseline 2 -- Network Flow only:
    Pure Edmonds-Karp max-flow over TYPE/CAPABILITY-feasible edges only
    (battery is NOT checked before adding an edge). Whichever max
    matching Edmonds-Karp finds is used, unweighted by cost or priority.

Baseline 3 -- Network Flow + battery awareness:
    Same Edmonds-Karp max-flow, but edges are additionally filtered by
    battery.is_feasible() before the graph is built -- still no cost/
    priority weighting, just a battery-safe feasible matching.

Baseline 4 -- Auction without ML/RL:
    Exactly auction.run_auction() with predictor=None (the Day 2-3
    physics-only auction). Provided here as a thin wrapper for a
    consistent baselines.* import surface.

Proposed system:
    auction.run_auction() with an ml_prediction.Predictor supplied,
    running inside the full dynamic reallocation loop (events, fairness,
    explainability) -- see experiments.py's `run_proposed`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from robots import Robot
from tasks import Task, is_feasible_edge
from battery import BatteryModel
from astar import astar, PathResult
from auction import Bid, run_auction, DEFAULT_WEIGHTS
from network_flow import build_flow_graph, edmonds_karp_max_flow, flow_to_assignment


def _make_bid(robot: Robot, task: Task, path: PathResult, score: float, components: Dict) -> Bid:
    return Bid(robot_id=robot.robot_id, task_id=task.task_id, score=score, components=components, path=path)


def greedy_nearest_allocation(robots: List[Robot], tasks: List[Task], env, battery: BatteryModel,
                                astar_step: int = 2, predictor=None) -> Dict[str, Bid]:
    """Baseline 1: nearest feasible robot wins, tasks in id order, no
    other weighting. Still respects the hard battery-safety gate (a
    robot physically cannot be sent below its minimum SOC), since
    allowing that would break the simulation rather than reveal a
    meaningful research contrast."""
    available = {r.robot_id: r for r in robots}
    assignment: Dict[str, Bid] = {}
    ordered_tasks = sorted(tasks, key=lambda t: t.task_id)  # arbitrary order, NOT priority-sorted

    for task in ordered_tasks:
        best = None  # (distance, robot, path)
        for rid, robot in available.items():
            if rid in assignment or not is_feasible_edge(robot, task):
                continue
            path = astar(env, (robot.x, robot.y), (task.x, task.y), robot.robot_type, step=astar_step)
            if not path.found or not battery.is_feasible(robot, path.path_length, task.workload):
                continue
            if best is None or path.path_length < best[0]:
                best = (path.path_length, robot, path)
        if best is not None:
            distance, robot, path = best
            score = 1.0 / (1.0 + distance)  # informational only; not used for selection (already nearest-chosen)
            assignment[robot.robot_id] = _make_bid(robot, task, path, score, {"distance_cost": distance})
    return assignment


def network_flow_only_allocation(robots: List[Robot], tasks: List[Task], env, battery: BatteryModel,
                                   astar_step: int = 2, predictor=None) -> Dict[str, Bid]:
    """Baseline 2: pure max-flow over type/capability-feasible edges,
    battery NOT considered when building the graph."""
    def euclid(r, t):
        return math.hypot(r.x - t.x, r.y - t.y)

    G = build_flow_graph(robots, tasks, distance_fn=euclid)
    _, flow_dict = edmonds_karp_max_flow(G)
    pairs = flow_to_assignment(flow_dict)  # robot_id -> task_id

    robots_by_id = {r.robot_id: r for r in robots}
    tasks_by_id = {t.task_id: t for t in tasks}
    assignment: Dict[str, Bid] = {}
    for rid, tid in pairs.items():
        robot, task = robots_by_id[rid], tasks_by_id[tid]
        path = astar(env, (robot.x, robot.y), (task.x, task.y), robot.robot_type, step=astar_step)
        if not path.found:
            continue  # max-flow ignored geometry; a handful of pairs may be unreachable, reported honestly
        assignment[rid] = _make_bid(robot, task, path, score=1.0, components={"distance_cost": path.path_length})
    return assignment


def network_flow_battery_allocation(robots: List[Robot], tasks: List[Task], env, battery: BatteryModel,
                                      astar_step: int = 2, predictor=None) -> Dict[str, Bid]:
    """Baseline 3: max-flow, but edges are pre-filtered by
    battery.is_feasible() (requires a path to be computed up front to
    check feasibility, unlike Baseline 2 which is battery-blind)."""
    feasible_robots, feasible_tasks = [], []
    edge_paths: Dict[tuple, PathResult] = {}
    robots_by_id = {r.robot_id: r for r in robots}
    tasks_by_id = {t.task_id: t for t in tasks}

    import networkx as nx
    from network_flow import SOURCE, SINK

    G = nx.DiGraph()
    G.add_node(SOURCE); G.add_node(SINK)
    for r in robots:
        G.add_node(f"R:{r.robot_id}"); G.add_edge(SOURCE, f"R:{r.robot_id}", capacity=1)
    for t in tasks:
        G.add_node(f"T:{t.task_id}"); G.add_edge(f"T:{t.task_id}", SINK, capacity=1)

    for r in robots:
        for t in tasks:
            if not is_feasible_edge(r, t):
                continue
            path = astar(env, (r.x, r.y), (t.x, t.y), r.robot_type, step=astar_step)
            if path.found and battery.is_feasible(r, path.path_length, t.workload):
                G.add_edge(f"R:{r.robot_id}", f"T:{t.task_id}", capacity=1)
                edge_paths[(r.robot_id, t.task_id)] = path

    _, flow_dict = edmonds_karp_max_flow(G)
    pairs = flow_to_assignment(flow_dict)

    assignment: Dict[str, Bid] = {}
    for rid, tid in pairs.items():
        robot, task, path = robots_by_id[rid], tasks_by_id[tid], edge_paths[(rid, tid)]
        assignment[rid] = _make_bid(robot, task, path, score=1.0, components={"distance_cost": path.path_length})
    return assignment


def auction_no_ml_allocation(robots: List[Robot], tasks: List[Task], env, battery: BatteryModel,
                               astar_step: int = 2, predictor=None) -> Dict[str, Bid]:
    """Baseline 4: the Day 2-3 physics-only auction (predictor forced to
    None regardless of what's passed in, so this baseline is always
    ML-free even if called with a predictor by mistake)."""
    return run_auction(robots, tasks, env, battery, astar_step=astar_step, predictor=None)


BASELINE_REGISTRY = {
    "B1_greedy_nearest": greedy_nearest_allocation,
    "B2_network_flow_only": network_flow_only_allocation,
    "B3_network_flow_battery": network_flow_battery_allocation,
    "B4_auction_no_ml": auction_no_ml_allocation,
}


if __name__ == "__main__":
    from environment import Environment
    from robots import RobotFleet
    from tasks import TaskManager
    from battery import BatteryModel as BM

    env = Environment()
    fleet = RobotFleet()
    batt = BM(fleet)
    tm = TaskManager()
    robots = list(fleet.robots.values())
    tasks = tm.active_tasks()

    for name, fn in BASELINE_REGISTRY.items():
        assignment = fn(robots, tasks, env, batt, astar_step=3)
        print(f"{name:28s} -> {len(assignment)} / {len(robots)} robots assigned")
        assert isinstance(assignment, dict)

    print("\nOK: all 4 baseline allocation strategies run successfully on the real dataset.")
