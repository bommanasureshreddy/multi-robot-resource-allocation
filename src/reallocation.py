"""
reallocation.py
----------------
Section 12 (Dynamic Reallocation) + Section 36 (final integrated loop),
implemented at Day-3 scope: a discrete-event simulation driver.

Loop (per spec Section 12):

    State t -> Build graph -> Allocate -> Execute -> Update battery
            -> Update position -> Update task state -> Check events
            -> Modify graph -> Reallocate -> State t+1

Modeling choices made explicit here (documented, not hidden):
- Time advances over the union of {event times} and {scheduled task
  completion times} -- a discrete-event simulation, not fixed-dt stepping.
- A task's duration once assigned = A*-estimated travel time
  + a workload-processing term (workload * WORKLOAD_TIME_FACTOR). This is
  a simplification (no continuous-motion animation yet -- that is Day 4's
  live panel); position is updated by jumping the robot to the task
  location upon completion, and battery is consumed for the full
  distance+workload at that point.
- Robots below (min_soc + CHARGE_TRIGGER_MARGIN) while IDLE are
  automatically routed to their nearest charging station (Section 9:
  "potentially go to charging") and recharge each simulated interval
  using that station's charging_rate until SOC >= CHARGE_TARGET_SOC.
- LOW_BATTERY events are logged, not used to fabricate a SOC value the
  dataset doesn't provide (see events.py docstring).
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from environment import Environment
from robots import RobotFleet, Robot
from tasks import TaskManager, Task
from battery import BatteryModel
from auction import run_auction, Bid
from events import EventManager, apply_event
from fairness import fairness_report
from logging_utils import HistoryLogger

WORKLOAD_TIME_FACTOR = 0.6      # simulated time units per workload unit while executing a task
CHARGE_TRIGGER_MARGIN = 15.0    # SOC margin above min_soc that triggers auto-charging
CHARGE_TARGET_SOC = 90.0


@dataclass
class PendingCompletion:
    complete_time: float
    robot_id: str
    task_id: str
    distance: float
    workload: float

    def __lt__(self, other):
        return self.complete_time < other.complete_time


class Simulator:
    def __init__(self, dataset_root: str = "data/dataset", astar_step: int = 2, run_id: str = None,
                 predictor=None, allocation_fn=None, seed: int = 42):
        """`predictor` (ml_prediction.Predictor, optional): when supplied,
        every allocation cycle's auction blends ML-predicted energy/risk
        into its bid scoring (Section 19), matching the "proposed system"
        described in Section 26. Left as None to reproduce the Day 2-3
        physics-only auction baseline (Section 26 Baseline 4).

        `allocation_fn` (callable, optional): swaps out the allocation
        POLICY while keeping the identical dynamic-event loop (battery
        consumption, events, fairness, logging) -- this is what lets
        baselines.py implement Section 26's 4 baselines + proposed system,
        and experiments.py implement Section 27's ablations, as true
        apples-to-apples comparisons (same environment, same events, same
        seeds; only the allocation mechanism differs). Signature must
        match auction.run_auction: (robots, tasks, env, battery,
        astar_step, predictor) -> Dict[robot_id, Bid-like]. Defaults to
        run_auction (the full proposed auction) when None.

        `seed` (int): this Simulator's environment/robots/tasks/events are
        all loaded deterministically from a fixed dataset -- there is no
        inherent randomness to seed. What `seed` DOES control is the
        iteration order of available robots at each allocation cycle
        (shuffled per-seed via random.Random(seed)), which affects
        tie-breaking whenever two robots produce equal or near-equal bids
        (or, for the flow-based baselines, which edges Edmonds-Karp
        considers first among equal-cost options). This is what makes
        Section 32's "run multiple seeds, report mean/std" meaningful for
        this deterministic simulator rather than trivially producing
        std=0 for every metric regardless of method."""
        self.dataset_root = dataset_root
        self.env = Environment(dataset_root)
        self.fleet = RobotFleet(dataset_root)
        self.battery = BatteryModel(self.fleet, dataset_root)
        self.tm = TaskManager(dataset_root)
        self.events = EventManager(dataset_root)
        self.logger = HistoryLogger(dataset_root, run_id=run_id)
        self.astar_step = astar_step
        self.predictor = predictor
        self.allocation_fn = allocation_fn
        self.seed = seed
        self._rng = random.Random(seed)

        self.t: float = 0.0
        self.applied_event_ids: set = set()
        self.pending_completions: List[PendingCompletion] = []
        self.completed_counts: Dict[str, int] = {r: 0 for r in self.fleet.robots}
        self.energy_used: Dict[str, float] = {r: 0.0 for r in self.fleet.robots}
        self.reallocation_count: int = 0
        self.event_log: List[dict] = []

    # ------------------------------------------------------------------
    def _due_event_times(self, horizon: float) -> List[float]:
        return sorted(t for t in self.events.event_times() if t <= horizon and
                       any(e.event_id not in self.applied_event_ids for e in self.events.events_at(t)))

    def _apply_due_events(self, t: float):
        for e in self.events.events_at(t):
            if e.event_id in self.applied_event_ids:
                continue
            effect = apply_event(e, self.fleet, self.tm, self.env)
            self.applied_event_ids.add(e.event_id)
            self.event_log.append({"t": t, **effect})
            self.logger.log("graph", t=t, event="EVENT_APPLIED", **effect)

    def _complete_due_tasks(self, t: float):
        while self.pending_completions and self.pending_completions[0].complete_time <= t:
            pc = heapq.heappop(self.pending_completions)
            robot = self.fleet.robots.get(pc.robot_id)
            task = self.tm.tasks.get(pc.task_id)
            if robot is None or task is None:
                continue
            # A robot that FAILED or a task that was CANCELLED mid-flight
            # should not silently "complete" -- skip actual completion,
            # state already handled by the failure/cancel event.
            if robot.status == "FAILED" or task.status == "CANCELLED":
                continue

            consumed = self.battery.consume(robot, pc.distance, pc.workload)
            robot.x, robot.y = task.x, task.y
            robot.status = "IDLE"
            robot.current_task = None
            robot.completed_tasks.append(task.task_id)

            task.status = "COMPLETED"
            self.completed_counts[robot.robot_id] += 1
            self.energy_used[robot.robot_id] += consumed

            self.logger.log("task_state", t=t, task_id=task.task_id, status="COMPLETED",
                             completed_by=robot.robot_id)
            self.logger.log("robot_state", t=t, robot_id=robot.robot_id, x=robot.x, y=robot.y,
                             soc=robot.soc, dod=robot.dod, soh=robot.soh, status=robot.status)
            self.logger.log("trajectory", t=t, robot_id=robot.robot_id, x=robot.x, y=robot.y)

    def _auto_charge_pass(self, t: float, dt: float):
        for robot in self.fleet.robots.values():
            if robot.status == "CHARGING":
                station = self.env.nearest_charging_station(robot.x, robot.y, robot.robot_type)
                rate = station.charging_rate if station else 5.0
                self.battery.recharge(robot, rate=rate, dt=dt)
                if robot.soc >= CHARGE_TARGET_SOC:
                    robot.status = "IDLE"
                    self.logger.log("robot_state", t=t, robot_id=robot.robot_id, x=robot.x, y=robot.y,
                                     soc=robot.soc, dod=robot.dod, soh=robot.soh, status="IDLE",
                                     note="charge complete")
            elif robot.status == "IDLE" and robot.soc < (robot.min_soc + CHARGE_TRIGGER_MARGIN):
                station = self.env.nearest_charging_station(robot.x, robot.y, robot.robot_type)
                if station is not None:
                    robot.x, robot.y = station.x, station.y  # simplified: instant relocation to station
                    robot.status = "CHARGING"
                    self.logger.log("robot_state", t=t, robot_id=robot.robot_id, x=robot.x, y=robot.y,
                                     soc=robot.soc, dod=robot.dod, soh=robot.soh, status="CHARGING",
                                     note=f"routed to {station.station_id} (low SOC)")

    def _allocate(self, t: float):
        released_tasks = [tsk for tsk in self.tm.tasks.values()
                           if tsk.status == "PENDING" and tsk.release_time <= t]
        available_robots = [r for r in self.fleet.robots.values() if r.is_operational() and r.is_available()]
        # Shuffle robot order per-seed so ties in bid score (or in which
        # edge Edmonds-Karp explores first for the flow-based baselines)
        # break differently across seeds -- see the `seed` docstring above.
        self._rng.shuffle(available_robots)
        if not released_tasks or not available_robots:
            self.logger.log("graph", t=t, event="ALLOCATE_SKIPPED",
                             n_available_robots=len(available_robots), n_pending_tasks=len(released_tasks))
            return

        if self.allocation_fn is not None:
            assignment: Dict[str, Bid] = self.allocation_fn(
                available_robots, released_tasks, self.env, self.battery,
                astar_step=self.astar_step, predictor=self.predictor)
        else:
            assignment: Dict[str, Bid] = run_auction(available_robots, released_tasks, self.env, self.battery,
                                                        astar_step=self.astar_step, predictor=self.predictor)
        self.logger.log("graph", t=t, event="ALLOCATE", n_available_robots=len(available_robots),
                         n_pending_tasks=len(released_tasks), n_assigned=len(assignment))

        for robot_id, bid in assignment.items():
            robot = self.fleet.robots[robot_id]
            task = self.tm.tasks[bid.task_id]

            robot.status = "BUSY"
            robot.current_task = task.task_id
            task.status = "ASSIGNED"
            task.assigned_robot = robot_id

            duration = bid.path.estimated_time(robot.speed) + task.workload * WORKLOAD_TIME_FACTOR
            heapq.heappush(self.pending_completions, PendingCompletion(
                complete_time=t + duration, robot_id=robot_id, task_id=task.task_id,
                distance=bid.path.path_length, workload=task.workload,
            ))
            self.reallocation_count += 1

            self.logger.log("allocation", t=t, robot_id=robot_id, task_id=task.task_id,
                             score=bid.score, priority=task.priority, emergency=task.emergency,
                             explanation=bid.explanation())

    def _log_fairness(self, t: float):
        stress = {rid: 100.0 - r.soc for rid, r in self.fleet.robots.items()}
        report = fairness_report(list(self.fleet.robots.keys()), self.completed_counts,
                                  self.energy_used, stress)
        self.logger.log("graph", t=t, event="FAIRNESS", **report)
        return report

    # ------------------------------------------------------------------
    def run(self, until: Optional[float] = None, verbose: bool = True) -> dict:
        horizon = until if until is not None else max(self.events.event_times())
        last_t = 0.0

        checkpoints = sorted(set(self.events.event_times()) | {0.0, horizon})
        for t in checkpoints:
            if t > horizon:
                break
            dt = t - last_t
            self._complete_due_tasks(t)
            self._auto_charge_pass(t, dt if dt > 0 else 1.0)
            self._apply_due_events(t)
            self._complete_due_tasks(t)  # a just-applied event (e.g. cancellation) may free things
            self._allocate(t)
            fairness = self._log_fairness(t)
            self.t = t
            last_t = t
            if verbose:
                n_active = sum(1 for tsk in self.tm.tasks.values() if tsk.is_active())
                n_completed = sum(1 for tsk in self.tm.tasks.values() if tsk.status == "COMPLETED")
                print(f"t={t:6.1f}  active_tasks={n_active:2d}  completed={n_completed:2d}  "
                      f"reallocations_so_far={self.reallocation_count:2d}  "
                      f"workload_fairness={fairness['workload_fairness']:.3f}")

        # Drain any completions scheduled beyond the last checkpoint but at/under horizon
        self._complete_due_tasks(horizon)
        self.logger.flush()

        return {
            "run_id": self.logger.run_id,
            "final_t": horizon,
            "completed_tasks": sum(1 for tsk in self.tm.tasks.values() if tsk.status == "COMPLETED"),
            "cancelled_tasks": sum(1 for tsk in self.tm.tasks.values() if tsk.status == "CANCELLED"),
            "unassigned_tasks": sum(1 for tsk in self.tm.tasks.values() if tsk.status == "PENDING"),
            "failed_robots": sum(1 for r in self.fleet.robots.values() if r.status == "FAILED"),
            "reallocation_count": self.reallocation_count,
            "fairness": self._log_fairness(horizon),
        }


if __name__ == "__main__":
    sim = Simulator(run_id="day3_selftest")
    result = sim.run()
    print("\n--- Final summary ---")
    for k, v in result.items():
        print(f"{k}: {v}")

    assert result["completed_tasks"] + result["cancelled_tasks"] + result["unassigned_tasks"] <= 30
    assert result["failed_robots"] == 2, "Dataset has exactly 2 ROBOT_FAILURE events"
    assert result["reallocation_count"] > 12, "Should exceed the 12 initial assignments due to dynamic events"
    print("\nOK: reallocation engine ran end-to-end with dynamic events applied.")
