"""
astar.py
--------
Section 18: A* path planning per robot domain (UGV on LAND cells, USV on
WATER cells, UAV over AIR/LAND/WATER). Operates on an integer grid over
the Environment, respecting obstacles, FORBIDDEN zones, and domain
traversability (environment.is_traversable), and using
environment.movement_cost() (RESTRICTED-zone / risk surcharge) as the
edge-cost modifier.

Returns path, path length (cell count), travel cost (cost-weighted), and
estimated travel time/energy -- these feed directly into the auction bid
(spec: "The A* result must influence the allocation bid/cost. Do not use
A* merely for visualization.").
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from environment import Environment

Cell = Tuple[int, int]


@dataclass
class PathResult:
    path: List[Cell]
    path_length: float          # number of steps (Euclidean-weighted)
    travel_cost: float          # cost-weighted (accounts for RESTRICTED/risk)
    found: bool

    def estimated_time(self, speed: float) -> float:
        return self.path_length / speed if speed > 0 else math.inf

    def estimated_energy(self, energy_per_distance: float) -> float:
        return self.path_length * energy_per_distance


def _heuristic(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(env: Environment, start: Cell, goal: Cell, robot_type: str,
          step: int = 2) -> PathResult:
    """
    Grid A* with 8-connected movement. `step` controls grid resolution
    (coarser step = faster search over the 100x100 map; step=2 gives a
    50x50 effective grid, adequate for allocation-time cost estimation
    without the cost of a full-resolution planner).
    """
    start = (round(start[0] / step) * step, round(start[1] / step) * step)
    goal = (round(goal[0] / step) * step, round(goal[1] / step) * step)

    def neighbors(cell: Cell):
        x, y = cell
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if env.is_traversable(nx_, ny_, robot_type):
                    yield (nx_, ny_)

    def _start_ok(x, y, robot_type) -> bool:
        """A robot's OWN current cell is, by definition, a cell it already
        legitimately occupies. If it happens to sit exactly on an
        obstacle's bounding-box edge (a dataset placement quirk -- the
        robot didn't teleport into a building, it's just adjacent to one
        at integer-coordinate resolution), that should not permanently
        strand it. So the start cell is checked for domain/forbidden-zone
        validity only, not obstacle overlap; every other cell (goal and
        all intermediate steps) still enforces the full obstacle check."""
        if not (0 <= x <= env.width and 0 <= y <= env.height):
            return False
        zs = env.zones_at(x, y)
        if any(z.zone_type == "FORBIDDEN" for z in zs):
            return False
        if not zs:
            return False
        domain = env.DOMAIN_ZONES.get(robot_type, set())
        return any(z.zone_type in domain for z in zs)

    if not _start_ok(*start, robot_type) or not env.is_traversable(*goal, robot_type):
        return PathResult(path=[], path_length=math.inf, travel_cost=math.inf, found=False)
    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    max_expansions = 6000  # safety bound so a pathological case can't hang

    expansions = 0
    while open_heap and expansions < max_expansions:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        expansions += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            path_length = sum(_heuristic(path[i], path[i + 1]) for i in range(len(path) - 1))
            travel_cost = g_score[path[-1] if False else goal]
            return PathResult(path=path, path_length=path_length, travel_cost=travel_cost, found=True)

        for nb in neighbors(current):
            step_dist = _heuristic(current, nb)
            step_cost = step_dist * env.movement_cost(*nb)
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(nb, math.inf):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f = tentative_g + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    return PathResult(path=[], path_length=math.inf, travel_cost=math.inf, found=False)


if __name__ == "__main__":
    env = Environment()

    print("--- UGV path across LAND (avoiding obstacles/restricted) ---")
    r = astar(env, (20, 55), (32, 88), "UGV", step=2)
    print(f"found={r.found}, path_length={r.path_length:.1f}, travel_cost={r.travel_cost:.1f}, "
          f"steps={len(r.path)}")
    assert r.found

    print("\n--- USV path across WATER ---")
    r2 = astar(env, (40, 15), (55, 80), "USV", step=2)
    print(f"found={r2.found}, path_length={r2.path_length:.1f}, travel_cost={r2.travel_cost:.1f}, "
          f"steps={len(r2.path)}")
    assert r2.found

    print("\n--- UGV cannot reach a WATER-only goal ---")
    r3 = astar(env, (20, 55), (50, 10), "UGV", step=2)
    print(f"found={r3.found} (expected False: goal cell is WATER, not traversable by UGV)")
    assert not r3.found

    print("\nOK: A* module passes basic domain-feasibility sanity checks.")
