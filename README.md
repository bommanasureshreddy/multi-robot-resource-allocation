# Dynamic Network-Flow Resource Allocation for Heterogeneous Multi-Robot Systems

B.Tech research project: battery-aware, priority-aware, explainable task
allocation for a 12-robot heterogeneous fleet (UGV/USV/UAV), built around a
**dynamic Network Flow graph** as the core allocation backbone, extended
with A* path costs, an auction/CBBA-style bidding mechanism, ML prediction,
Graph RL (GAT+PPO), Jain's fairness, full explainability, and a complete
experimental evaluation (baselines, ablations, scenarios, scalability).

## Status — PROJECT COMPLETE

| Day | Scope | Status |
|---|---|---|
| 1 | Foundation: environment, robots, tasks, battery, viz, data generator | ✅ Complete & tested |
| 2 | Network flow, A*, auction | ✅ Complete & tested |
| 3 | Dynamic reallocation, fairness, explainability | ✅ Complete & tested |
| 4 | ML prediction, GAT-PPO, full simulation | ✅ Complete & tested |
| 5 | Baselines, ablations, scenarios, scalability, result graphs | ✅ Complete & tested |

**All 14 notebooks (01–14), 84 code cells total, execute end-to-end with
zero errors.** All 18 `src/` modules pass their standalone self-tests.

**Post-Day-4 polish:** professional Matplotlib/Seaborn explainability
charts added throughout (correlation heatmap, prediction diagnostics,
bid-breakdown, SOC timelines, task-status donuts, workload/energy bars,
reward box-plots) — see `src/visualization.py`.

## Project structure

```
project/
├── data/dataset/           # environment/, robots/, tasks/, events/, ml/,
│                            # simulation/ (generated history), rl/
├── notebooks/               # 01..14, thin — import from src/
├── src/                     # environment.py, robots.py, tasks.py, battery.py,
│                            # logging_utils.py, visualization.py, network_flow.py,
│                            # astar.py, auction.py, fairness.py, events.py,
│                            # reallocation.py, ml_prediction.py, graph_rl.py,
│                            # simulation.py, data_generator.py, metrics.py,
│                            # baselines.py, experiments.py
├── models/                  # saved ML prediction models (.joblib) + metrics.json
├── configs/config.yaml      # all tunable parameters — nothing hard-coded
├── experiments/              # baseline_comparison.csv, ablation_study.csv,
│                            # scenarios.csv, scalability.csv,
│                            # scalability_full_metrics.csv, gat_ppo_summary.json
├── results/                  # all generated figures (day1_*.png ... day5_*.png)
└── requirements.txt
```

## Running

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_environment.ipynb
```

All `src/` modules are also directly runnable/testable standalone, e.g.:
`python3 src/environment.py`, `python3 src/experiments.py`.

## Day 5 — Experiments (Sections 26-30, 32)

- **Baselines (Section 26):** 4 baselines (greedy-nearest, network-flow-only,
  network-flow+battery, auction-without-ML) + the proposed full system, all
  run through the *identical* dynamic-event Simulator loop via a pluggable
  `allocation_fn` — only the allocation policy differs, making the
  comparison fair. 5 seeds each, saved to `experiments/baseline_comparison.csv`.
- **Ablation study (Section 27):** A1→A10, a progressively-enabled
  mechanism/weight ladder (Network Flow → +Battery → +Priority → +Dynamic
  Reallocation → +Auction → +A* → +ML → +Fairness → Full). A9 (+GAT-PPO) is
  reported as a **separate track** (its own reward-based metric, not merged
  into the completion-rate table) per Section 20's instruction not to let
  GAT/PPO destroy the Network Flow foundation — see
  `experiments/gat_ppo_summary.json`. Documented in `experiments.py`'s
  module docstring exactly why Network-Flow and Auction are structurally
  different mechanisms in this implementation, rather than pretending a
  false unified architecture.
- **Scenarios A-G (Section 28):** normal, battery degradation, high task
  load (synthetic 12/60), emergency-heavy, robot failure, dynamic
  environment, mixed heterogeneous — 3 seeds each.
- **Scalability / Scenario H:** 6/12 → 50/100 robots/tasks, using the new
  `src/data_generator.py` (see below), both pure allocation-time scaling
  and full-dynamic-run completion/energy trends.
- **Metrics (Section 29):** `src/metrics.py` computes the full metric set
  (allocation, time, energy, fairness, path, risk, computational) from a
  real completed `Simulator` run.
- **Results (Section 30):** all 19+ required graphs in
  `notebooks/13_results_visualization.ipynb`.

## Design notes / documented limitations (per project rules)

- The battery degradation model is an explicitly **simulation-level**
  linear DoD-weighted model, not an electrochemical one.
- `task_type` in the dataset is the task category (SEARCH/RESCUE/...),
  not the LAND/WATER/AIR zone domain — domain compatibility is enforced
  via `allowed_robot_types` / zone lookups, not the category string.
- The WATER zone spans the full height of the map (x: 35–65), completely
  separating the west and east LAND zones for UGVs — confirmed as a real
  topological property of this dataset (no bridge/crossing exists), not a
  pathfinding bug. UAVs are unaffected since they fly over both. This also
  explains why aggregate task-completion RATE is capacity-bound and
  saturates near the same value across allocation methods on this
  dataset — the methods still produce genuinely different underlying
  assignments (verified) and differ meaningfully on fairness/energy.
- The auction (`auction.py`) is a **single-pass, priority-ordered greedy
  auction**, explicitly documented as a simplification of full multi-round
  CBBA consensus/bundle-building — not claimed to be the latter.
- The dynamic reallocation engine (`reallocation.py`) is a discrete-event
  simulation (time advances over event/completion times), not fixed-dt
  continuous animation; a task's duration = A*-estimated travel time +
  workload-based processing time, and robot position updates by jumping
  to the task location on completion.
- LOW_BATTERY events are logged, not used to fabricate an SOC value the
  dataset doesn't provide — the robot's real SOC (from `battery.py`)
  already reflects its true simulated consumption trajectory.
- **On seeds and determinism:** the real 12-robot/30-task dataset is
  loaded deterministically from fixed CSVs with no inherent randomness, so
  `baseline_comparison.csv`/`ablation_study.csv` (run on that fixed
  dataset) correctly show **std=0 across seeds** — this was directly
  investigated and confirmed mathematically correct, not a bug (see
  `Simulator.seed`'s docstring: seed controls tie-breaking order, and this
  dataset's bid scores essentially never tie exactly). Experiments that DO
  involve real randomness (scenario mutations, synthetic scalability
  datasets via `data_generator.py`) correctly show genuine seed-to-seed
  variance — e.g. Scenario E's completion rate ranges 0.40–0.53 across
  seeds. Both behaviors are verified and expected.
- No results are fabricated. Anything not yet measured is labelled
  Expected/Illustrative/Placeholder until an actual run produces it.

## Bugs found & fixed during verification

- **A\* start-cell obstacle bug (fixed):** UGV01's spawn coordinate sits
  exactly on an obstacle's bounding-box edge, which made A* treat its own
  starting cell as blocked — permanently stranding it. Fixed by checking
  obstacle-overlap on the goal and intermediate path cells only, not the
  robot's own already-occupied starting cell.
- **Network graph never showed battery-blocked edges (fixed):**
  `simulation.py`'s dashboard graph only checked type/capability
  feasibility when drawing edges, never battery feasibility — so a
  robot's edges never visibly shrank when its SOC dropped, even though
  the auction correctly excluded those pairings. Fixed by running the
  same A* + `battery.is_feasible()` check `auction.compute_bid()` uses,
  and drawing battery-blocked edges as dashed red.
- **ML prediction never reached the dynamic Simulator (fixed):** Day 4
  built `ml_prediction.py` and wired it into `auction.py`, but
  `reallocation.Simulator` never actually passed a `Predictor` through —
  so the "proposed system" in all of Day 3-4's demos was silently running
  ML-free. Added a `predictor` parameter to `Simulator` and threaded it
  into every allocation cycle; verified the ML-blended auction now runs
  inside the full dynamic loop.
- **`Predictor` broke when run from a different working directory
  (fixed):** `ml_prediction_metrics.json` stores model file paths
  relative to wherever `train_all()` was originally run from, so loading
  a `Predictor` from `notebooks/` (cwd ≠ project root) raised
  `FileNotFoundError`. Fixed by reconstructing paths from the caller's own
  `models_dir` + filename instead of trusting the stored path string.
- **Missing synthetic data generator (Day 1 gap, closed):** the original
  Day 1 plan called for a synthetic dataset generator, but since a real
  dataset was already provided, this was skipped — until Day 5's
  scalability tests needed datasets at sizes the fixed real dataset
  doesn't have. Added `src/data_generator.py`, validated it loads cleanly
  through every existing module at 4 different scales (6/12 → 50/100).

