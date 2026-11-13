"""
data_generator.py
------------------
Section 35 (Day 1) lists "Synthetic dataset generator" as a foundation
deliverable. The original dataset.zip already provided a complete,
realistic 12-robot/30-task dataset, so Days 1-4 built directly on that
real data rather than a generator -- this module fills that gap and
additionally supplies exactly what Section 28 Scenario H (scalability:
6/12, 12/30, 18/45, 30/60, 50/100 robots/tasks) needs, since the fixed
dataset only covers one size.

Design choice (documented): the ENVIRONMENT (map, zones, obstacles,
charging stations) is reused as-is from the real dataset -- scaling the
*operating environment* itself isn't what Section 28H asks for, only the
robot/task counts. Robots and tasks are randomly generated within their
domain zones (respecting the same LAND/WATER/AIR constraints astar.py and
environment.py already enforce), using parameter ranges drawn from the
real dataset's observed statistics (battery capacity, speed, workload,
priority mix, etc.) so generated fleets are realistic, not arbitrary.
"""

from __future__ import annotations

import os
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CAPABILITY_COLUMNS = ["search", "rescue", "transport", "surveillance",
                      "inspection", "medical", "land_operation", "water_operation", "air_operation"]
TASK_CATEGORY_CAPABILITY = {
    "SEARCH": "search", "RESCUE": "rescue", "TRANSPORT": "transport",
    "SURVEILLANCE": "surveillance", "INSPECTION": "inspection", "MEDICAL": "medical",
    "MONITORING": "surveillance",
}
TASK_CATEGORIES = list(TASK_CATEGORY_CAPABILITY.keys())

DOMAIN_ZONES = {"UGV": ["Z_LAND_W", "Z_LAND_E"], "USV": ["Z_WATER_M"], "UAV": ["Z_AIR_ALL"]}
ZONE_TASK_TYPE = {"Z_LAND_W": "LAND", "Z_LAND_E": "LAND", "Z_WATER_M": "WATER", "Z_AIR_ALL": "AIR"}


def _load_zone_bounds(source_env_dir: str) -> Dict[str, Tuple[float, float, float, float]]:
    zones = pd.read_csv(os.path.join(source_env_dir, "zones.csv"))
    bounds = {}
    for _, z in zones.iterrows():
        bounds[z["zone_id"]] = (z["x_min"], z["x_max"], z["y_min"], z["y_max"])
    return bounds


def _load_obstacles(source_env_dir: str) -> List[Tuple[float, float, float, float]]:
    obs = pd.read_csv(os.path.join(source_env_dir, "obstacles.csv"))
    return [(r.x, r.y, r.width, r.height) for r in obs.itertuples()]


def _random_point_in_zone(rng: random.Random, zone_id: str, bounds, obstacles, max_tries=50):
    x_min, x_max, y_min, y_max = bounds[zone_id]
    for _ in range(max_tries):
        x = rng.uniform(x_min + 1, x_max - 1)
        y = rng.uniform(y_min + 1, y_max - 1)
        if not any(ox <= x <= ox + ow and oy <= y <= oy + oh for ox, oy, ow, oh in obstacles):
            return round(x, 1), round(y, 1)
    return round((x_min + x_max) / 2, 1), round((y_min + y_max) / 2, 1)


def generate_dataset(n_robots: int, n_tasks: int, output_dir: str, seed: int = 42,
                      source_dataset_root: str = "data/dataset") -> str:
    """Generates a full synthetic dataset of the given scale under
    output_dir, reusing the real environment layout. Returns output_dir."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    os.makedirs(output_dir, exist_ok=True)
    src_env_dir = os.path.join(source_dataset_root, "environment")
    dst_env_dir = os.path.join(output_dir, "environment")
    os.makedirs(dst_env_dir, exist_ok=True)
    for fname in ["map.csv", "zones.csv", "obstacles.csv", "charging_stations.csv"]:
        shutil.copy(os.path.join(src_env_dir, fname), os.path.join(dst_env_dir, fname))

    bounds = _load_zone_bounds(src_env_dir)
    obstacles = _load_obstacles(src_env_dir)

    # --- Robots: split as evenly as possible across UGV/USV/UAV ---
    types_cycle = (["UGV", "USV", "UAV"] * (n_robots // 3 + 1))[:n_robots]
    rng.shuffle(types_cycle)

    robots_rows, caps_rows, battery_rows = [], [], []
    for i, rtype in enumerate(types_cycle, start=1):
        rid = f"{rtype}{i:02d}"
        zone_id = rng.choice(DOMAIN_ZONES[rtype])
        x, y = _random_point_in_zone(rng, zone_id, bounds, obstacles)
        speed = round(np_rng.uniform(2.0, 9.0), 2)
        payload = round(np_rng.uniform(10.0, 60.0), 1)
        comm_range = round(np_rng.uniform(16.0, 46.0), 1)
        robots_rows.append({
            "robot_id": rid, "robot_type": rtype, "x": x, "y": y, "speed": speed,
            "payload_capacity": payload, "communication_range": comm_range, "status": "IDLE",
        })

        caps = {c: 0 for c in CAPABILITY_COLUMNS}
        n_caps = rng.randint(3, 5)
        for c in rng.sample(CAPABILITY_COLUMNS[:6], k=min(n_caps, 6)):
            caps[c] = 1
        caps[f"{ {'UGV': 'land', 'USV': 'water', 'UAV': 'air'}[rtype] }_operation"] = 1
        caps_rows.append({"robot_id": rid, "robot_type": rtype, **caps})

        battery_capacity = round(np_rng.uniform(340.0, 1150.0), 1)
        initial_soc = round(np_rng.uniform(65.0, 100.0), 1)
        initial_soh = round(np_rng.uniform(0.85, 1.0), 3)
        battery_rows.append({
            "robot_id": rid,
            "battery_capacity": battery_capacity,
            "initial_soc": initial_soc,
            "initial_soh": initial_soh,
            "initial_dod": round(100.0 - initial_soc, 1),
            "min_soc": round(np_rng.uniform(10.0, 18.0), 1),
            "energy_per_distance": round(np_rng.uniform(1.5, 5.5), 3),
            "task_energy_rate": round(np_rng.uniform(0.9, 2.2), 3),
            "degradation_factor": round(np_rng.uniform(0.0005, 0.0017), 6),
        })

    # --- Tasks: proportional category/domain mix, ~10% emergency ---
    domain_cycle = (["UGV", "USV", "UAV"] * (n_tasks // 3 + 1))[:n_tasks]
    rng.shuffle(domain_cycle)
    horizon = max(60.0, n_tasks * 6.0)

    tasks_rows, req_rows = [], []
    for i, req_type in enumerate(domain_cycle, start=1):
        tid = f"T{i:02d}"
        category = rng.choice(TASK_CATEGORIES)
        zone_id = rng.choice(DOMAIN_ZONES[req_type])
        x, y = _random_point_in_zone(rng, zone_id, bounds, obstacles)
        emergency = rng.random() < 0.10
        priority = 5 if emergency else rng.choice([1, 2, 2, 3, 3, 4])
        workload = round(np_rng.uniform(3.0, 40.0), 1)
        release_time = 0.0 if i <= max(1, n_tasks // 5) else round(rng.uniform(0, horizon), 1)
        deadline = round(release_time + np_rng.uniform(15.0, 130.0), 1)

        allowed_types = {req_type}
        if rng.random() < 0.15 and req_type != "USV":
            allowed_types.add(rng.choice(["UGV", "UAV"]))

        tasks_rows.append({
            "task_id": tid, "x": x, "y": y, "zone_id": zone_id, "task_type": category,
            "required_robot_type": req_type, "workload": workload, "priority": priority,
            "deadline": deadline, "release_time": release_time,
            "emergency": int(emergency), "status": "PENDING",
        })
        req_rows.append({
            "task_id": tid, "required_zone": zone_id,
            "required_capability": TASK_CATEGORY_CAPABILITY[category],
            "allowed_robot_types": "|".join(sorted(allowed_types)),
            "minimum_payload": round(np_rng.uniform(0.0, 15.0), 1),
            "estimated_workload": workload,
        })

    # --- Write everything out ---
    robots_dir = os.path.join(output_dir, "robots")
    tasks_dir = os.path.join(output_dir, "tasks")
    events_dir = os.path.join(output_dir, "events")
    for d in (robots_dir, tasks_dir, events_dir):
        os.makedirs(d, exist_ok=True)

    pd.DataFrame(robots_rows).to_csv(os.path.join(robots_dir, "robots.csv"), index=False)
    pd.DataFrame(caps_rows).to_csv(os.path.join(robots_dir, "robot_capabilities.csv"), index=False)
    pd.DataFrame(battery_rows).to_csv(os.path.join(robots_dir, "robot_battery.csv"), index=False)
    pd.DataFrame(tasks_rows).to_csv(os.path.join(tasks_dir, "tasks.csv"), index=False)
    pd.DataFrame(req_rows).to_csv(os.path.join(tasks_dir, "task_requirements.csv"), index=False)

    # A small proportional set of dynamic events (new tasks matching release
    # times already assigned above, plus a couple of failures/emergencies)
    # so the generated dataset can also drive the full dynamic Simulator,
    # not just one-shot allocation.
    events_rows = []
    eid = 1
    for row in tasks_rows:
        if row["release_time"] > 0:
            events_rows.append({"event_id": f"E{eid:03d}", "time": row["release_time"],
                                 "event_type": "NEW_TASK", "task_id": row["task_id"],
                                 "robot_id": None, "zone_id": None, "severity": 2})
            eid += 1
        if row["emergency"]:
            events_rows.append({"event_id": f"E{eid:03d}", "time": row["release_time"],
                                 "event_type": "NEW_EMERGENCY", "task_id": row["task_id"],
                                 "robot_id": None, "zone_id": None, "severity": 5})
            eid += 1
    n_failures = max(1, n_robots // 8)
    for robot in rng.sample(robots_rows, k=min(n_failures, len(robots_rows))):
        events_rows.append({"event_id": f"E{eid:03d}", "time": round(rng.uniform(horizon * 0.2, horizon * 0.8), 1),
                             "event_type": "ROBOT_FAILURE", "task_id": None,
                             "robot_id": robot["robot_id"], "zone_id": None, "severity": 4})
        eid += 1
    pd.DataFrame(events_rows).sort_values("time").to_csv(os.path.join(events_dir, "events.csv"), index=False)

    return output_dir


if __name__ == "__main__":
    import tempfile

    tmp = tempfile.mkdtemp()
    out = generate_dataset(n_robots=18, n_tasks=45, output_dir=os.path.join(tmp, "synthetic_18_45"), seed=123)
    print("Generated dataset at:", out)

    # Validate it loads cleanly through every existing Day 1-3 module.
    from environment import Environment
    from robots import RobotFleet
    from tasks import TaskManager
    from battery import BatteryModel
    from events import EventManager

    env = Environment(out)
    fleet = RobotFleet(out)
    tm = TaskManager(out)
    batt = BatteryModel(fleet, out)
    em = EventManager(out)

    print(env.summary())
    print(fleet.summary())
    print(tm.summary())
    print(f"Events: {len(em.events)}")

    assert len(fleet.robots) == 18
    assert len(tm.tasks) == 45
    assert abs(len(fleet.by_type("UGV")) - 6) <= 1
    print("\nOK: data_generator.py produces a valid 18-robot/45-task dataset "
          "that loads through Environment/RobotFleet/TaskManager/BatteryModel/EventManager.")

    shutil.rmtree(tmp)
