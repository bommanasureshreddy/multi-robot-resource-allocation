"""
logging_utils.py
-----------------
Appends simulation state to the history CSVs described in spec Section 7
(dataset/simulation/*.csv) and Section 31 (Data Logging). Each row is
tagged with a run_id so multiple runs/seeds never overwrite each other
(spec Section 32: reproducibility across seeds).
"""

from __future__ import annotations

import os
import uuid
import datetime as dt
from typing import Any, Dict, List

import pandas as pd


def new_run_id() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:6]}"


class HistoryLogger:
    """
    Buffers rows in memory during a run and flushes (append-mode) to the
    corresponding CSV under dataset/simulation/. Nothing here ever
    truncates or overwrites an existing file -- only appends.
    """

    FILES = {
        "robot_state": "robot_state_history.csv",
        "task_state": "task_state_history.csv",
        "allocation": "allocation_history.csv",
        "trajectory": "trajectory_history.csv",
        "graph": "graph_history.csv",
    }

    def __init__(self, dataset_root: str = "data/dataset", run_id: str | None = None):
        self.sim_dir = os.path.join(dataset_root, "simulation")
        os.makedirs(self.sim_dir, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self._buffers: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.FILES}

    def log(self, kind: str, **fields):
        if kind not in self.FILES:
            raise ValueError(f"Unknown history kind: {kind}. Valid: {list(self.FILES)}")
        row = {"run_id": self.run_id, **fields}
        self._buffers[kind].append(row)

    def flush(self):
        for kind, rows in self._buffers.items():
            if not rows:
                continue
            path = os.path.join(self.sim_dir, self.FILES[kind])
            df = pd.DataFrame(rows)
            header = not os.path.exists(path)
            df.to_csv(path, mode="a", header=header, index=False)
            self._buffers[kind] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


if __name__ == "__main__":
    with HistoryLogger() as log:
        log.log("robot_state", t=0, robot_id="UGV01", x=10, y=20, soc=79.3, status="IDLE")
        log.log("task_state", t=0, task_id="T01", status="PENDING")
    print("run_id:", log.run_id)
    print("Wrote:", os.path.join(log.sim_dir, "robot_state_history.csv"))
