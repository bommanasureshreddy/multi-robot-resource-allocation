"""
network_flow.py
----------------
Section 13/14 of the spec: the dynamic Network Flow graph is the backbone
of the whole allocation pipeline.

Graph shape:

    SOURCE -> robot_i -> task_j -> SINK

- SOURCE->robot edges have capacity 1 (each robot can hold one active task
  at a time in this simplified capacity model).
- robot->task edges exist ONLY for feasible edges (tasks.is_feasible_edge)
  and carry capacity 1, plus a `cost` attribute (distance/energy/etc, used
  by auction.py for bidding -- max-flow itself is capacity/feasibility
  only, per spec Section 14: "Do NOT incorrectly call a weighted auction
  mechanism pure max-flow").
- task->SINK edges have capacity 1.

Edmonds-Karp (Ford-Fulkerson using BFS to find augmenting paths) computes
the maximum feasible number of robot-task pairings satisfiable by the
capacity/feasibility structure alone -- this is a BASELINE allocation
component, not the final decision (the auction refines *which* feasible
pairing is chosen using weighted bids).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import networkx as nx

from robots import Robot
from tasks import Task, is_feasible_edge

SOURCE = "SOURCE"
SINK = "SINK"


def build_flow_graph(robots: List[Robot], tasks: List[Task],
                      distance_fn=None) -> nx.DiGraph:
    """
    distance_fn(robot, task) -> float, used only to annotate edge `cost`
    for downstream auction use; does not affect max-flow capacities.
    """
    G = nx.DiGraph()
    G.add_node(SOURCE)
    G.add_node(SINK)

    for r in robots:
        if not r.is_operational():
            continue
        node = f"R:{r.robot_id}"
        G.add_node(node, kind="robot", robot_type=r.robot_type)
        G.add_edge(SOURCE, node, capacity=1)

    for t in tasks:
        if not t.is_active():
            continue
        node = f"T:{t.task_id}"
        G.add_node(node, kind="task", priority=t.priority, emergency=t.emergency)
        G.add_edge(node, SINK, capacity=1)

    for r in robots:
        if not r.is_operational():
            continue
        for t in tasks:
            if not t.is_active():
                continue
            if is_feasible_edge(r, t):
                cost = distance_fn(r, t) if distance_fn else 1.0
                G.add_edge(f"R:{r.robot_id}", f"T:{t.task_id}", capacity=1, cost=cost)

    return G


def bfs_augmenting_path(residual: nx.DiGraph, source: str, sink: str):
    """BFS over the residual graph to find an augmenting path (the 'Karp'
    half of Edmonds-Karp). Returns the path as a list of nodes, or None."""
    visited = {source}
    parent = {}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == sink:
            break
        for v in residual.successors(u):
            if v in visited:
                continue
            if residual[u][v].get("capacity", 0) > 0:
                visited.add(v)
                parent[v] = u
                queue.append(v)
    if sink not in visited:
        return None
    path = [sink]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def edmonds_karp_max_flow(G: nx.DiGraph, source: str = SOURCE, sink: str = SINK
                           ) -> Tuple[int, Dict[Tuple[str, str], int]]:
    """
    Explicit Edmonds-Karp implementation (Ford-Fulkerson using BFS to find
    augmenting paths), rather than relying on a black-box library solver,
    since the spec asks for the algorithm to be shown, not just invoked.

    Returns (max_flow_value, flow_dict) where flow_dict maps (u, v) -> 1
    for every saturated robot->task edge actually used in the flow (i.e.
    the max-flow-selected robot-task pairing).
    """
    residual = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", 0)
        residual.add_edge(u, v, capacity=cap)
        if not residual.has_edge(v, u):
            residual.add_edge(v, u, capacity=0)

    max_flow = 0
    while True:
        path = bfs_augmenting_path(residual, source, sink)
        if path is None:
            break
        # bottleneck capacity along the path
        bottleneck = min(residual[u][v]["capacity"] for u, v in zip(path, path[1:]))
        for u, v in zip(path, path[1:]):
            residual[u][v]["capacity"] -= bottleneck
            residual[v][u]["capacity"] += bottleneck
        max_flow += bottleneck

    # Extract the used robot->task edges: an original forward edge (u,v)
    # with capacity 1 is "used" if its residual capacity dropped to 0.
    flow_dict: Dict[Tuple[str, str], int] = {}
    for u, v, data in G.edges(data=True):
        if u.startswith("R:") and v.startswith("T:"):
            orig_cap = data.get("capacity", 0)
            resid_cap = residual[u][v]["capacity"] if residual.has_edge(u, v) else orig_cap
            used = orig_cap - resid_cap
            if used > 0:
                flow_dict[(u, v)] = used

    return max_flow, flow_dict


def flow_to_assignment(flow_dict: Dict[Tuple[str, str], int]) -> Dict[str, str]:
    """robot_id -> task_id mapping from a flow_dict produced above."""
    assignment = {}
    for (u, v), amount in flow_dict.items():
        if amount > 0:
            robot_id = u.split("R:", 1)[1]
            task_id = v.split("T:", 1)[1]
            assignment[robot_id] = task_id
    return assignment


if __name__ == "__main__":
    from robots import RobotFleet
    from tasks import TaskManager
    import math

    fleet = RobotFleet()
    tm = TaskManager()

    def euclid(r, t):
        return math.hypot(r.x - t.x, r.y - t.y)

    G = build_flow_graph(list(fleet.robots.values()), tm.active_tasks(), distance_fn=euclid)
    n_robot_nodes = sum(1 for n, d in G.nodes(data=True) if d.get("kind") == "robot")
    n_task_nodes = sum(1 for n, d in G.nodes(data=True) if d.get("kind") == "task")
    n_feasible_edges = sum(1 for u, v in G.edges() if u.startswith("R:") and v.startswith("T:"))
    print(f"Graph: {n_robot_nodes} robot nodes, {n_task_nodes} task nodes, "
          f"{n_feasible_edges} feasible robot-task edges")

    max_flow, flow_dict = edmonds_karp_max_flow(G)
    print(f"Edmonds-Karp max flow (feasibility-only baseline allocation): {max_flow}")

    assignment = flow_to_assignment(flow_dict)
    print(f"Assigned {len(assignment)} / {len(fleet.robots)} robots:")
    for rid, tid in sorted(assignment.items()):
        print(f"  {rid} -> {tid}")

    # Sanity check against networkx's own max-flow-value as an independent
    # cross-check of our from-scratch Edmonds-Karp implementation.
    nx_flow_value = nx.maximum_flow_value(G, SOURCE, SINK, capacity="capacity")
    assert max_flow == nx_flow_value, f"Mismatch: ours={max_flow} vs networkx={nx_flow_value}"
    print(f"OK: matches networkx.maximum_flow_value ({nx_flow_value}) — implementation verified.")
