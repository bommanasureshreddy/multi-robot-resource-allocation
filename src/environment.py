"""
environment.py
---------------
Loads and represents the 2D simulation environment: the map bounds, zone
polygons (LAND / WATER / AIR / SAFE / RESTRICTED / FORBIDDEN / EMERGENCY /
CHARGING), obstacles, and charging stations.

This module is intentionally simple: it is a lookup/query layer over the
dataset CSVs in data/dataset/environment/. It does not do any allocation
logic -- that lives in network_flow.py, auction.py, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class Zone:
    zone_id: str
    zone_type: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    risk_level: float
    entry_allowed: bool

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class Obstacle:
    obstacle_id: str
    x: float
    y: float
    width: float
    height: float
    obstacle_type: str

    def blocks(self, x: float, y: float) -> bool:
        return (self.x <= x <= self.x + self.width) and (
            self.y <= y <= self.y + self.height
        )


@dataclass
class ChargingStation:
    station_id: str
    zone_id: str
    x: float
    y: float
    robot_type: str
    charging_rate: float


class Environment:
    """
    Container + query layer for the simulation environment.

    Grid convention: (x, y) in [0, width) x [0, height), integer resolution
    cells unless `resolution` in map.csv says otherwise.
    """

    def __init__(self, dataset_root: str = "data/dataset"):
        self.dataset_root = dataset_root
        self.env_dir = os.path.join(dataset_root, "environment")

        self.map_id: str = ""
        self.width: int = 0
        self.height: int = 0
        self.resolution: float = 1.0

        self.zones: List[Zone] = []
        self.obstacles: List[Obstacle] = []
        self.charging_stations: List[ChargingStation] = []

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        map_df = pd.read_csv(os.path.join(self.env_dir, "map.csv"))
        row = map_df.iloc[0]
        self.map_id = row["map_id"]
        self.width = int(row["width"])
        self.height = int(row["height"])
        self.resolution = float(row["resolution"])

        zones_df = pd.read_csv(os.path.join(self.env_dir, "zones.csv"))
        for _, r in zones_df.iterrows():
            self.zones.append(
                Zone(
                    zone_id=r["zone_id"],
                    zone_type=r["zone_type"],
                    x_min=float(r["x_min"]),
                    x_max=float(r["x_max"]),
                    y_min=float(r["y_min"]),
                    y_max=float(r["y_max"]),
                    risk_level=float(r["risk_level"]),
                    entry_allowed=bool(r["entry_allowed"]),
                )
            )

        obs_df = pd.read_csv(os.path.join(self.env_dir, "obstacles.csv"))
        for _, r in obs_df.iterrows():
            self.obstacles.append(
                Obstacle(
                    obstacle_id=r["obstacle_id"],
                    x=float(r["x"]),
                    y=float(r["y"]),
                    width=float(r["width"]),
                    height=float(r["height"]),
                    obstacle_type=r["obstacle_type"],
                )
            )

        cs_df = pd.read_csv(os.path.join(self.env_dir, "charging_stations.csv"))
        for _, r in cs_df.iterrows():
            self.charging_stations.append(
                ChargingStation(
                    station_id=r["station_id"],
                    zone_id=r["zone_id"],
                    x=float(r["x"]),
                    y=float(r["y"]),
                    robot_type=r["robot_type"],
                    charging_rate=float(r["charging_rate"]),
                )
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def zone_at(self, x: float, y: float) -> Optional[Zone]:
        """Return the first matching zone containing (x, y). A cell can
        belong to multiple overlapping zones (e.g. LAND + SAFE); callers
        that need all matches should use zones_at()."""
        matches = self.zones_at(x, y)
        return matches[0] if matches else None

    def zones_at(self, x: float, y: float) -> List[Zone]:
        return [z for z in self.zones if z.contains(x, y)]

    def is_blocked(self, x: float, y: float) -> bool:
        return any(o.blocks(x, y) for o in self.obstacles)

    def is_forbidden(self, x: float, y: float) -> bool:
        return any(z.zone_type == "FORBIDDEN" for z in self.zones_at(x, y))

    # Domain zones actually grant traversal rights. SAFE / RESTRICTED /
    # EMERGENCY are modifier zones that overlay a domain zone (they change
    # cost/urgency) but do not by themselves make a cell traversable.
    DOMAIN_ZONES = {
        "UGV": {"LAND", "CHARGING"},
        "USV": {"WATER", "CHARGING"},
        "UAV": {"AIR", "LAND", "WATER", "CHARGING"},  # flies over land & water
    }

    def traversable_zone_types(self, robot_type: str) -> List[str]:
        return sorted(self.DOMAIN_ZONES.get(robot_type, set()))

    def is_traversable(self, x: float, y: float, robot_type: str) -> bool:
        """A cell is traversable by robot_type if: in bounds, not blocked
        by an obstacle, not FORBIDDEN, and at least one overlapping zone is
        a DOMAIN zone for that robot type (e.g. USV needs a WATER zone
        present, not merely an overlapping SAFE zone)."""
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False
        if self.is_blocked(x, y):
            return False
        zs = self.zones_at(x, y)
        if any(z.zone_type == "FORBIDDEN" for z in zs):
            return False
        if not zs:
            return False
        domain = self.DOMAIN_ZONES.get(robot_type, set())
        return any(z.zone_type in domain for z in zs)

    def movement_cost(self, x: float, y: float) -> float:
        """Extra cost multiplier from RESTRICTED zones / risk level.
        Used by A* (astar.py) as an edge-cost modifier."""
        zs = self.zones_at(x, y)
        if not zs:
            return 1.0
        risk = max(z.risk_level for z in zs)
        restricted = any(z.zone_type == "RESTRICTED" for z in zs)
        cost = 1.0 + 0.15 * risk
        if restricted:
            cost *= 1.5
        return cost

    def nearest_charging_station(self, x: float, y: float, robot_type: str) -> Optional[ChargingStation]:
        candidates = [c for c in self.charging_stations if c.robot_type == robot_type]
        if not candidates:
            return None
        return min(candidates, key=lambda c: (c.x - x) ** 2 + (c.y - y) ** 2)

    def summary(self) -> str:
        zone_counts = {}
        for z in self.zones:
            zone_counts[z.zone_type] = zone_counts.get(z.zone_type, 0) + 1
        lines = [
            f"Map: {self.map_id}  ({self.width} x {self.height}, resolution={self.resolution})",
            f"Zones: {len(self.zones)}  -> {zone_counts}",
            f"Obstacles: {len(self.obstacles)}",
            f"Charging stations: {len(self.charging_stations)}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    env = Environment()
    print(env.summary())
    print("Zone at (10,10):", [z.zone_id for z in env.zones_at(10, 10)])
    print("Is (18,40) blocked (should be True, OBS01):", env.is_blocked(18, 40))
    print("UGV traversable at (10,10):", env.is_traversable(10, 10, "UGV"))
    print("USV traversable at (10,10):", env.is_traversable(10, 10, "USV"))
