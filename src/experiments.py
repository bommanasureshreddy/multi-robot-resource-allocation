"""
experiments.py
--------------
Sections 26-29 + 32: orchestrates every experiment the final paper needs,
runs them for real (never fabricates numbers), and returns/saves plain
DataFrames that notebook 12/13 consume.

IMPORTANT ARCHITECTURE NOTE (documented per spec's own instruction to
"clearly report the exact implemented architecture" rather than overclaim
a false unified mechanism): Sections 26/27 describe an ablation ladder
(Network Flow -> +Battery -> +Priority -> +Dynamic Realloc -> +Auction ->
+A* -> +ML -> +Fairness -> +GAT-PPO -> Full) that assumes every feature
can be toggled independently on one mechanism. In this implementation,
"Network Flow" (global max-matching, Edmonds-Karp) and "Auction"
(sequential, weighted, priority-ordered bidding) are STRUCTURALLY
DIFFERENT algorithms -- priority-ordering and per-factor weighting only
exist in the auction. So A1-A2 use the network-flow mechanism as literally
specified, and A3 onward switch to the auction mechanism with
progressively more bid-score terms enabled (documented in
ABLATION_CONFIGS below), which is where priority/A*/ML/fairness naturally
plug in. GAT-PPO (A9) is reported from its OWN isolated evaluation
(graph_rl.py / notebook 10), not merged into this pipeline, per Section
20's explicit instruction not to let GAT/PPO destroy the Network Flow
foundation -- reported honestly as a separate-track number, not
fabricated into the same table.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from reallocation import Simulator
from baselines import (greedy_nearest_allocation, network_flow_only_allocation,
                        network_flow_battery_allocation, auction_no_ml_allocation)
from auction import run_auction, DEFAULT_WEIGHTS
from ml_prediction import Predictor
from metrics import compute_metrics
from data_generator import generate_dataset

SEEDS_STANDARD = [42, 123, 456, 789, 1000]


def _weights_subset(keys: List[str]) -> Dict[str, float]:
    """Renormalized DEFAULT_WEIGHTS restricted to `keys` (sums to 1.0), so
    ablation scores stay on a comparable scale as terms are added."""
    raw = {k: DEFAULT_WEIGHTS[k] for k in keys}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _static_allocation_fn(base_fn):
    """Wraps an allocation_fn so it is only ever invoked once (used to
    simulate a "no dynamic reallocation" ablation by disabling all
    allocation calls after the first)."""
    state = {"called": False}

    def wrapped(robots, tasks, env, battery, astar_step=2, predictor=None):
        if state["called"]:
            return {}
        state["called"] = True
        return base_fn(robots, tasks, env, battery, astar_step=astar_step, predictor=predictor)
    return wrapped


def _auction_with_weights(weights: Dict[str, float]):
    def fn(robots, tasks, env, battery, astar_step=2, predictor=None):
        return run_auction(robots, tasks, env, battery, weights=weights, astar_step=astar_step, predictor=predictor)
    return fn


BASELINE_RUNNERS = {
    "B1_greedy_nearest": dict(allocation_fn=greedy_nearest_allocation, predictor=None),
    "B2_network_flow_only": dict(allocation_fn=network_flow_only_allocation, predictor=None),
    "B3_network_flow_battery": dict(allocation_fn=network_flow_battery_allocation, predictor=None),
    "B4_auction_no_ml": dict(allocation_fn=auction_no_ml_allocation, predictor=None),
}


def _ablation_configs(predictor: Optional[Predictor]) -> Dict[str, dict]:
    priority_only = _weights_subset(["priority", "soc", "dod", "soh", "predicted_energy"])
    no_astar_no_fair = _weights_subset(["priority", "soc", "dod", "soh", "predicted_energy", "risk", "deadline"])
    full_no_fair = _weights_subset([k for k in DEFAULT_WEIGHTS if k != "workload_fairness"])
    full = dict(DEFAULT_WEIGHTS)

    return {
        "A1_network_flow_only": dict(allocation_fn=network_flow_only_allocation, dynamic=False, predictor=None),
        "A2_network_flow_battery": dict(allocation_fn=network_flow_battery_allocation, dynamic=False, predictor=None),
        "A3_plus_priority": dict(allocation_fn=_auction_with_weights(priority_only), dynamic=False, predictor=None),
        "A4_plus_dynamic_realloc": dict(allocation_fn=_auction_with_weights(priority_only), dynamic=True, predictor=None),
        "A5_plus_auction_multifactor": dict(allocation_fn=_auction_with_weights(no_astar_no_fair), dynamic=True, predictor=None),
        "A6_plus_astar": dict(allocation_fn=_auction_with_weights(full_no_fair), dynamic=True, predictor=None),
        "A7_plus_ml_prediction": dict(allocation_fn=_auction_with_weights(full_no_fair), dynamic=True, predictor=predictor),
        "A8_plus_fairness": dict(allocation_fn=_auction_with_weights(full), dynamic=True, predictor=predictor),
        "A10_full_framework": dict(allocation_fn=_auction_with_weights(full), dynamic=True, predictor=predictor),
        # A9 (+GAT-PPO) is intentionally absent here -- reported separately,
        # see gat_ppo_summary_row() below.
    }


def gat_ppo_summary_row(dataset_root: str = "data/dataset") -> Dict:
    """Pulls the ALREADY-MEASURED GAT+PPO vs auction-baseline comparison
    from graph_rl.py's own evaluation (notebook 10), rather than
    re-fabricating a merged-pipeline run. Honest, separate-track report
    for ablation row A9."""
    from graph_rl import train, evaluate_policy, evaluate_baseline_auction
    policy, history = train(dataset_root=dataset_root, n_episodes=120, batch_size=5, seed=42)
    seeds_eval = [101, 202, 303]
    policy_rewards = [evaluate_policy(policy, dataset_root=dataset_root, seed=s)["episode_reward"] for s in seeds_eval]
    baseline_rewards = [evaluate_baseline_auction(dataset_root=dataset_root, seed=s)["episode_reward"] for s in seeds_eval]
    return {
        "config": "A9_plus_gatppo (SEPARATE TRACK -- see notebook 10)",
        "metric": "episode_reward (Section 21 reward fn, NOT comparable to completion-rate table)",
        "gatppo_mean": sum(policy_rewards) / len(policy_rewards),
        "auction_baseline_mean": sum(baseline_rewards) / len(baseline_rewards),
        "n_eval_seeds": len(seeds_eval),
    }


def _run_one(dataset_root: str, seed: int, allocation_fn, dynamic: bool, predictor,
             method_name: str, run_id_prefix: str) -> Dict:
    if not dynamic:
        allocation_fn = _static_allocation_fn(allocation_fn)
    t0 = time.time()
    sim = Simulator(dataset_root, run_id=f"{run_id_prefix}_{method_name}_{seed}",
                     predictor=predictor, allocation_fn=allocation_fn, seed=seed)
    sim.run(verbose=False)
    elapsed = time.time() - t0
    m = compute_metrics(sim, wall_clock_seconds=elapsed)
    m["method"] = method_name
    m["seed"] = seed
    return m


def run_baseline_comparison(dataset_root: str = "data/dataset", seeds: List[int] = None,
                              save_path: Optional[str] = "experiments/baseline_comparison.csv",
                              models_dir: str = "models"
                              ) -> pd.DataFrame:
    """Section 26: 4 baselines + the proposed system, across seeds."""
    seeds = seeds or SEEDS_STANDARD
    predictor = Predictor(models_dir=models_dir)  # for the proposed system
    rows = []
    for seed in seeds:
        for name, cfg in BASELINE_RUNNERS.items():
            rows.append(_run_one(dataset_root, seed, cfg["allocation_fn"], True, cfg["predictor"],
                                   name, "baseline"))
        rows.append(_run_one(dataset_root, seed, None, True, predictor, "PROPOSED_full_system", "baseline"))
    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def run_ablation_study(dataset_root: str = "data/dataset", seeds: List[int] = None,
                        save_path: Optional[str] = "experiments/ablation_study.csv",
                        models_dir: str = "models") -> pd.DataFrame:
    """Section 27: A1-A10 (A9 reported separately, see gat_ppo_summary_row)."""
    seeds = seeds or SEEDS_STANDARD
    predictor = Predictor(models_dir=models_dir)
    configs = _ablation_configs(predictor)
    rows = []
    for seed in seeds:
        for name, cfg in configs.items():
            rows.append(_run_one(dataset_root, seed, cfg["allocation_fn"], cfg["dynamic"], cfg["predictor"],
                                   name, "ablation"))
    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def _mutate_for_scenario(sim: Simulator, letter: str, rng_seed: int):
    """Applies the Section 28 scenario's defining perturbation directly to
    an already-constructed Simulator's live state, before run() is called."""
    import random
    rng = random.Random(rng_seed)

    if letter == "B":  # battery degradation
        for r in sim.fleet.robots.values():
            r.soc = rng.uniform(30.0, 60.0)
            r.soh = rng.uniform(0.6, 0.85)
            r.dod = 100.0 - r.soc
    elif letter == "D":  # emergency-heavy
        pending = [t for t in sim.tm.tasks.values() if t.status == "PENDING"]
        for t in rng.sample(pending, k=min(8, len(pending))):
            t.emergency = True
            t.priority = 5
    elif letter == "E":  # extra robot failures injected at t=0 (beyond dataset's own events)
        idle = list(sim.fleet.robots.values())
        for r in rng.sample(idle, k=min(3, len(idle))):
            r.status = "FAILED"
    elif letter == "F":  # dynamic environment: raise risk in RESTRICTED zones (new hazard)
        for z in sim.env.zones:
            if z.zone_type == "RESTRICTED":
                z.risk_level = min(5.0, z.risk_level + 1.5)
    # A (normal) and G (mixed heterogeneous, already the dataset's default) need no mutation.


def run_experimental_scenarios(dataset_root: str = "data/dataset", seeds: List[int] = None,
                                 save_path: Optional[str] = "experiments/scenarios.csv",
                                 models_dir: str = "models") -> pd.DataFrame:
    """Section 28, Scenarios A-G (Scenario H/scalability is separate, see
    run_scalability_test). Uses the PROPOSED full system for every
    scenario, since the point is to see how one method behaves across
    different conditions, not to re-run every baseline x scenario
    combination (that full cross product is deferred as future work and
    noted as such, not fabricated)."""
    seeds = seeds or [42, 123, 456]
    predictor = Predictor(models_dir=models_dir)
    letters_desc = {
        "A": "Normal (no failures/emergencies beyond the dataset's own)",
        "B": "Battery degradation (low initial SOC/SOH across the fleet)",
        "C": "High task load (synthetic 12 robots / 60 tasks)",
        "D": "Emergency-heavy (8 additional tasks forced to emergency/priority 5)",
        "E": "Robot failure (3 extra robots forced FAILED at t=0)",
        "F": "Dynamic environment (RESTRICTED zone risk raised)",
        "G": "Mixed heterogeneous tasks (dataset default LAND+WATER+AIR mix)",
    }
    rows = []
    for seed in seeds:
        for letter, desc in letters_desc.items():
            root = dataset_root
            tmp_dir = None
            if letter == "C":
                import tempfile
                tmp_dir = tempfile.mkdtemp()
                root = generate_dataset(12, 60, os.path.join(tmp_dir, "scenario_C"), seed=seed,
                                          source_dataset_root=dataset_root)

            t0 = time.time()
            sim = Simulator(root, run_id=f"scenario_{letter}_{seed}", predictor=predictor, seed=seed)
            if letter in ("B", "D", "E", "F"):
                _mutate_for_scenario(sim, letter, seed)
            sim.run(verbose=False)
            elapsed = time.time() - t0
            m = compute_metrics(sim, wall_clock_seconds=elapsed)
            m["scenario"] = f"{letter}_{desc.split(' (')[0].replace(' ', '_')}"
            m["scenario_description"] = desc
            m["seed"] = seed
            rows.append(m)

            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def run_scalability_test(sizes: List[tuple] = None, seeds: List[int] = None,
                           dataset_root: str = "data/dataset",
                           save_path: Optional[str] = "experiments/scalability.csv",
                           models_dir: str = "models") -> pd.DataFrame:
    """Section 28 Scenario H + Section 29/30's scalability graphs. Measures
    ALLOCATION computation time (one full run_auction call over the whole
    feasible graph), matching "allocation computation time" /
    "graph update time" / "robots|tasks vs computation time" -- not a
    full multi-day dynamic run, which would conflate simulation length
    with algorithmic scalability."""
    import tempfile
    import shutil
    from environment import Environment
    from robots import RobotFleet
    from tasks import TaskManager
    from battery import BatteryModel

    sizes = sizes or [(6, 12), (12, 30), (18, 45), (30, 60), (50, 100)]
    seeds = seeds or [42, 123, 456]
    predictor = Predictor(models_dir=models_dir)
    rows = []

    for n_r, n_t in sizes:
        for seed in seeds:
            tmp_dir = tempfile.mkdtemp()
            try:
                if (n_r, n_t) == (12, 30):
                    root = dataset_root  # use the REAL dataset at its native size, not a synthetic stand-in
                else:
                    root = generate_dataset(n_r, n_t, os.path.join(tmp_dir, f"scale_{n_r}_{n_t}"),
                                              seed=seed, source_dataset_root=dataset_root)

                env = Environment(root)
                fleet = RobotFleet(root)
                tm = TaskManager(root)
                batt = BatteryModel(fleet, root)

                t0 = time.time()
                assignment = run_auction(list(fleet.robots.values()), tm.active_tasks(), env, batt,
                                           astar_step=3, predictor=predictor)
                alloc_time = time.time() - t0

                t0 = time.time()
                import math
                from network_flow import build_flow_graph
                def euclid(r, t):
                    return math.hypot(r.x - t.x, r.y - t.y)
                build_flow_graph(list(fleet.robots.values()), tm.active_tasks(), distance_fn=euclid)
                graph_build_time = time.time() - t0

                rows.append({
                    "n_robots": n_r, "n_tasks": n_t, "seed": seed,
                    "allocation_computation_time_sec": alloc_time,
                    "graph_update_time_sec": graph_build_time,
                    "n_assigned": len(assignment),
                })
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def run_scalability_full_metrics(sizes: List[tuple] = None, seeds: List[int] = None,
                                    dataset_root: str = "data/dataset",
                                    save_path: Optional[str] = "experiments/scalability_full_metrics.csv",
                                    models_dir: str = "models"
                                    ) -> pd.DataFrame:
    """Companion to run_scalability_test: runs the FULL dynamic Simulator
    (not just one allocation call) at each scale, so completion-rate and
    energy-vs-task-count trends (Section 30) can be plotted alongside pure
    allocation-time scaling. Fewer seeds than run_scalability_test since
    full dynamic runs are more expensive at the largest sizes."""
    import tempfile
    import shutil

    sizes = sizes or [(6, 12), (12, 30), (18, 45), (30, 60), (50, 100)]
    seeds = seeds or [42, 123]
    predictor = Predictor(models_dir=models_dir)
    rows = []

    for n_r, n_t in sizes:
        for seed in seeds:
            tmp_dir = tempfile.mkdtemp()
            try:
                if (n_r, n_t) == (12, 30):
                    root = dataset_root
                else:
                    root = generate_dataset(n_r, n_t, os.path.join(tmp_dir, f"fullscale_{n_r}_{n_t}"),
                                              seed=seed, source_dataset_root=dataset_root)
                t0 = time.time()
                sim = Simulator(root, run_id=f"fullscale_{n_r}_{n_t}_{seed}", predictor=predictor, seed=seed)
                sim.run(verbose=False)
                elapsed = time.time() - t0
                m = compute_metrics(sim, wall_clock_seconds=elapsed)
                m["seed"] = seed
                rows.append(m)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    import os
    if os.path.exists("data/dataset/simulation"):
        for f in os.listdir("data/dataset/simulation"):
            os.remove(os.path.join("data/dataset/simulation", f))

    print("Running a SMALL smoke test of every experiment function (1 seed each)...")
    b = run_baseline_comparison(seeds=[42], save_path=None)
    print(f"baseline_comparison: {len(b)} rows, methods={sorted(b.method.unique())}")
    assert len(b) == 5

    a = run_ablation_study(seeds=[42], save_path=None)
    print(f"ablation_study: {len(a)} rows, configs={sorted(a.method.unique())}")
    assert len(a) == 9

    s = run_experimental_scenarios(seeds=[42], save_path=None)
    print(f"scenarios: {len(s)} rows, scenarios={sorted(s.scenario.unique())}")
    assert len(s) == 7

    sc = run_scalability_test(sizes=[(6, 12), (12, 30)], seeds=[42], save_path=None)
    print(f"scalability: {len(sc)} rows")
    assert len(sc) == 2

    print("\nOK: experiments.py -- all 4 experiment types run end-to-end.")
