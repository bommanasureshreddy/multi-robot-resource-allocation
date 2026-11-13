"""
fairness.py
-----------
Section 17: Jain's Fairness Index, plus battery-aware / priority-aware
fairness reporting built on TOP of the standard (undistorted) Jain formula
-- the spec explicitly forbids distorting the mathematical formula itself,
so battery/priority awareness here means choosing WHAT quantity xi we
measure fairness over, not changing the formula.

Standard Jain's Fairness Index:

    J(x1..xn) = (sum(xi))^2 / (n * sum(xi^2))

J = 1 means perfectly fair (all xi equal); J = 1/n means maximally unfair
(all resource concentrated on one agent).
"""

from __future__ import annotations

from typing import Dict, List


def jains_fairness_index(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 1.0
    sum_x = sum(values)
    sum_x2 = sum(x * x for x in values)
    if sum_x2 == 0:
        # No agent received anything -- define as perfectly fair (nobody
        # is worse off than anybody else) rather than dividing by zero.
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


def fairness_report(robot_ids: List[str],
                     completed_counts: Dict[str, int],
                     energy_used: Dict[str, float],
                     battery_stress: Dict[str, float] = None,
                     ) -> Dict[str, float]:
    """
    Computes the standard Jain index over several MEANINGFUL, NORMALIZED
    allocation quantities (spec: "use the standard Jain index on
    meaningful normalized allocation quantities"):

    - workload_fairness: over completed-task counts per robot
    - energy_fairness: over cumulative energy consumed per robot
    - battery_stress_fairness: over (100 - SOC) per robot, i.e. how evenly
      "battery pain" is distributed -- this is the battery-aware fairness
      metric requested in Section 17.
    """
    workload_vals = [completed_counts.get(rid, 0) for rid in robot_ids]
    energy_vals = [energy_used.get(rid, 0.0) for rid in robot_ids]

    report = {
        "workload_fairness": jains_fairness_index(workload_vals),
        "energy_fairness": jains_fairness_index(energy_vals),
    }
    if battery_stress is not None:
        stress_vals = [battery_stress.get(rid, 0.0) for rid in robot_ids]
        report["battery_stress_fairness"] = jains_fairness_index(stress_vals)
    return report


if __name__ == "__main__":
    # Sanity checks against known Jain-index properties.
    perfectly_fair = jains_fairness_index([5, 5, 5, 5])
    assert abs(perfectly_fair - 1.0) < 1e-9, perfectly_fair
    print(f"Equal allocation [5,5,5,5] -> J = {perfectly_fair:.4f} (expected 1.0)")

    maximally_unfair = jains_fairness_index([10, 0, 0, 0])
    expected = 1.0 / 4
    assert abs(maximally_unfair - expected) < 1e-9
    print(f"All-to-one allocation [10,0,0,0] -> J = {maximally_unfair:.4f} (expected {expected:.4f})")

    mixed = jains_fairness_index([3, 4, 5, 2])
    print(f"Mixed allocation [3,4,5,2] -> J = {mixed:.4f}")
    assert 0 < mixed < 1

    empty = jains_fairness_index([])
    assert empty == 1.0
    print("Edge cases (empty list, all-zero) handled without division errors.")

    report = fairness_report(
        robot_ids=["R1", "R2", "R3"],
        completed_counts={"R1": 2, "R2": 2, "R3": 2},
        energy_used={"R1": 50.0, "R2": 55.0, "R3": 48.0},
        battery_stress={"R1": 20.0, "R2": 22.0, "R3": 19.0},
    )
    print("\nFairness report (near-equal allocation):", report)
    assert all(v > 0.95 for v in report.values())
    print("OK: fairness.py verified.")
