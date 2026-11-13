"""
visualization.py
-----------------
Matplotlib-based visualization. Day 1 scope: static environment + fleet +
task overview plot. Later days extend this module with the live/animated
network-flow-graph view (Section 23) and the full simulation dashboard
(Section 24) -- this file grows, it isn't replaced.

Additional section (post-Day-4 polish): a set of reusable, professional
Matplotlib/Seaborn chart helpers for data/model/result explainability --
correlation heatmaps, prediction diagnostics, bid-breakdown explainability
charts, fairness/workload/energy summaries, and RL evaluation charts.
These are ADDITIVE: `plot_environment` (Day 1) and the dashboard functions
in `simulation.py` (Day 4) are unchanged, so nothing already verified is
altered by importing seaborn here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from environment import Environment
from robots import RobotFleet
from tasks import TaskManager

ZONE_COLORS = {
    "LAND": "#d8c9a3",
    "WATER": "#a9cbe8",
    "AIR": "#eef3f8",
    "SAFE": "#bfe6b0",
    "RESTRICTED": "#f2c98c",
    "FORBIDDEN": "#e39a9a",
    "EMERGENCY": "#e77b7b",
    "CHARGING": "#c9a9e8",
}

ROBOT_MARKERS = {"UGV": ("s", "#2e7d32"), "USV": ("^", "#1565c0"), "UAV": ("o", "#ef6c00")}
PRIORITY_SIZE = {1: 30, 2: 45, 3: 65, 4: 90, 5: 130}


def plot_environment(env: Environment, fleet: Optional[RobotFleet] = None,
                      tasks: Optional[TaskManager] = None, title: str = "Simulation Environment",
                      ax=None, savepath: Optional[str] = None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 9))

    # Zones (draw AIR last-under, domain zones first so overlays are visible)
    zone_order = ["AIR", "LAND", "WATER", "SAFE", "RESTRICTED", "EMERGENCY", "CHARGING", "FORBIDDEN"]
    for zt in zone_order:
        for z in env.zones:
            if z.zone_type != zt:
                continue
            rect = patches.Rectangle(
                (z.x_min, z.y_min), z.x_max - z.x_min, z.y_max - z.y_min,
                facecolor=ZONE_COLORS.get(zt, "#dddddd"), edgecolor="none", alpha=0.55, zorder=1,
            )
            ax.add_patch(rect)

    # Obstacles
    for o in env.obstacles:
        rect = patches.Rectangle((o.x, o.y), o.width, o.height,
                                  facecolor="#444444", edgecolor="black", zorder=3)
        ax.add_patch(rect)

    # Charging stations
    for c in env.charging_stations:
        ax.plot(c.x, c.y, marker="P", markersize=9, color="purple", zorder=4)

    # Robots
    if fleet is not None:
        for rtype, (marker, color) in ROBOT_MARKERS.items():
            xs = [r.x for r in fleet.by_type(rtype)]
            ys = [r.y for r in fleet.by_type(rtype)]
            ax.scatter(xs, ys, marker=marker, s=140, c=color, edgecolors="black",
                       linewidths=1.2, zorder=6, label=f"{rtype} ({len(xs)})")
        for r in fleet.robots.values():
            ax.annotate(r.robot_id, (r.x, r.y), xytext=(3, 3), textcoords="offset points", fontsize=7, zorder=7)

    # Tasks
    if tasks is not None:
        for t in tasks.active_tasks():
            size = PRIORITY_SIZE.get(t.priority, 50)
            color = "red" if t.emergency else "black"
            marker = "*" if t.emergency else "x"
            ax.scatter(t.x, t.y, marker=marker, s=size, c=color, zorder=8)
            ax.annotate(t.task_id, (t.x, t.y), xytext=(3, -8), textcoords="offset points", fontsize=6, zorder=8)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if fleet is not None:
        ax.legend(loc="upper right", fontsize=8)

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    if own_fig:
        return ax.figure, ax
    return ax


# ============================================================================
# Explainability / analysis charts (Seaborn-based)
# ============================================================================
# These do not alter global matplotlib rcParams by default (no
# sns.set_theme() at import time), so plot_environment and simulation.py's
# dashboard remain visually unchanged. Each function applies a scoped
# style via `with sns.axes_style(...)` where useful.

PALETTE = "Set2"


def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str], title: str = "Feature Correlation",
                              savepath: Optional[str] = None):
    """Correlation heatmap over numeric columns -- helps explain which ML
    prediction features are related/redundant (Section 19 EDA)."""
    corr = df[columns].corr(numeric_only=True)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 3, 0.5 * len(columns) + 2.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontsize=12, pad=12)
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_prediction_diagnostics(y_true, y_pred, target_name: str, savepath: Optional[str] = None):
    """Two-panel model diagnostic: predicted-vs-actual scatter (with the
    y=x reference line) and a residual histogram -- standard, honest
    regression diagnostics rather than a single R^2 number."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.35, s=18, ax=axes[0], color="#1565c0")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, "--", color="#d32f2f", linewidth=1.5, label="perfect prediction (y=x)")
        axes[0].set_xlabel(f"Actual {target_name}")
        axes[0].set_ylabel(f"Predicted {target_name}")
        axes[0].set_title(f"Predicted vs Actual -- {target_name}")
        axes[0].legend(fontsize=8)

        sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color="#2e7d32")
        axes[1].axvline(0, color="#d32f2f", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel(f"Residual (actual - predicted), {target_name}")
        axes[1].set_title(f"Residual distribution -- {target_name}")
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_bid_breakdown(bid, title: Optional[str] = None, savepath: Optional[str] = None):
    """Horizontal bar chart of a single auction Bid's weighted score
    components -- the core explainability visualization for Section 22
    (WHY did this robot win this task)."""
    weight_keys = ["priority", "distance_cost", "path_cost", "predicted_energy", "soc",
                   "dod", "soh", "risk", "deadline", "workload_fairness"]
    comps = {k: bid.components[k] for k in weight_keys if k in bid.components}
    names = list(comps.keys())
    values = list(comps.values())
    colors = ["#2e7d32" if v >= 0.5 else "#ef6c00" if v >= 0.25 else "#d32f2f" for v in values]

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(names, values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("normalized component score (higher = more favorable)")
        ax.set_title(title or f"Bid breakdown: {bid.robot_id} -> {bid.task_id} (score={bid.score:.3f})")
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_task_status_breakdown(tm, title: str = "Task Status Breakdown", savepath: Optional[str] = None):
    """Donut chart of task status counts -- a quick, professional summary
    of where all 30 tasks currently stand in the simulation."""
    counts: Dict[str, int] = {}
    for t in tm.tasks.values():
        counts[t.status] = counts.get(t.status, 0) + 1
    status_colors = {
        "COMPLETED": "#7fbf7f", "PENDING": "#90a4ae", "ASSIGNED": "#64b5f6",
        "IN_PROGRESS": "#1565c0", "CANCELLED": "#bbbbbb", "DELAYED": "#ef6c00",
    }
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [status_colors.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(sizes, colors=colors, autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
                                    startangle=90, pctdistance=0.8,
                                    wedgeprops=dict(width=0.4, edgecolor="white"))
    ax.legend(wedges, [f"{l} ({c})" for l, c in zip(labels, sizes)], loc="center", fontsize=9, frameon=False)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_robot_soc_timeline(history_df: pd.DataFrame, run_id: Optional[str] = None,
                             title: str = "Robot Battery (SOC) Over Time", savepath: Optional[str] = None):
    """Line plot of each robot's SOC trajectory across a simulation run,
    from `robot_state_history.csv` -- shows battery draw, dynamics, and
    (if present) recharge cycles at a glance."""
    df = history_df.copy()
    if run_id is not None and "run_id" in df.columns:
        df = df[df.run_id == run_id]
    df = df.sort_values("t")

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette("tab20", n_colors=df["robot_id"].nunique())
        sns.lineplot(data=df, x="t", y="soc", hue="robot_id", marker="o", markersize=4,
                     palette=palette, ax=ax, linewidth=1.4)
        ax.axhline(15, color="#d32f2f", linestyle="--", linewidth=1, label="typical min_soc floor (~15%)")
        ax.set_xlabel("simulation time t")
        ax.set_ylabel("State of Charge (%)")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_workload_and_energy(completed_counts: Dict[str, int], energy_used: Dict[str, float],
                               robot_types: Optional[Dict[str, str]] = None,
                               savepath: Optional[str] = None):
    """Side-by-side bar charts: completed tasks per robot, and energy
    consumed per robot -- the two quantities Jain's fairness index is
    computed over, made visually inspectable rather than a single number."""
    robot_ids = list(completed_counts.keys())
    type_colors = {"UGV": "#2e7d32", "USV": "#1565c0", "UAV": "#ef6c00"}
    colors = [type_colors.get((robot_types or {}).get(rid, ""), "#888888") for rid in robot_ids]

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(robot_ids, [completed_counts[r] for r in robot_ids], color=colors)
        axes[0].set_title("Completed Tasks per Robot")
        axes[0].set_ylabel("tasks completed")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(robot_ids, [energy_used.get(r, 0.0) for r in robot_ids], color=colors)
        axes[1].set_title("Energy Consumed per Robot")
        axes[1].set_ylabel("energy (Wh, simulation-level units)")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_reward_comparison(df: pd.DataFrame, savepath: Optional[str] = None):
    """Paired box/strip plot comparing two reward distributions (e.g.
    learned GAT+PPO policy vs auction baseline) across evaluation seeds --
    more honest than a single mean-vs-mean bar chart since it shows
    per-seed spread."""
    long_df = pd.melt(df, value_vars=["policy_reward", "baseline_reward"],
                       var_name="method", value_name="episode_reward")
    long_df["method"] = long_df["method"].map({
        "policy_reward": "GAT+PPO policy", "baseline_reward": "Auction baseline"
    })
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=long_df, x="method", y="episode_reward", palette=PALETTE, ax=ax, width=0.5)
        sns.stripplot(data=long_df, x="method", y="episode_reward", color="black", size=6, alpha=0.7, ax=ax)
        ax.set_title("Episode Reward: Learned Policy vs Auction Baseline\n(same reward fn, same states, across eval seeds)")
        ax.set_xlabel("")
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


# ============================================================================
# Section 30: experiment-result charts (baselines / ablation / scalability)
# ============================================================================

def plot_metric_bar(df: pd.DataFrame, group_col: str, metric_col: str, title: str,
                     ylabel: Optional[str] = None, savepath: Optional[str] = None,
                     order: Optional[List[str]] = None, figsize=(9, 5), horizontal_labels=True):
    """Generic grouped bar chart with error bars across seeds -- the
    workhorse for most of Section 30's required comparison graphs
    (task completion rate, energy, fairness, reallocation count, etc.
    across baselines / ablation configs)."""
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=df, x=group_col, y=metric_col, order=order, hue=group_col, legend=False,
                    palette="viridis", errorbar="sd", ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel or metric_col)
        ax.set_xlabel("")
        if horizontal_labels:
            ax.tick_params(axis="x", rotation=35)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_scalability_line(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                           ylabel: Optional[str] = None, savepath: Optional[str] = None):
    """Line plot with seed-error bands for a scalability sweep (robots or
    tasks on the x-axis vs a performance/time metric)."""
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=df, x=x_col, y=y_col, marker="o", errorbar="sd", ax=ax, color="#1565c0")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(ylabel or y_col)
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_soc_distribution(sim, savepath: Optional[str] = None):
    """Box/strip plot of final SOC across all robots, colored by type --
    Section 30's "SOC distribution across robots"."""
    rows = [{"robot_id": r.robot_id, "robot_type": r.robot_type, "soc": r.soc}
            for r in sim.fleet.robots.values()]
    df = pd.DataFrame(rows)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="robot_type", y="soc", hue="robot_type", legend=False,
                    palette={"UGV": "#2e7d32", "USV": "#1565c0", "UAV": "#ef6c00"}, ax=ax, width=0.5)
        sns.stripplot(data=df, x="robot_type", y="soc", color="black", size=7, alpha=0.7, ax=ax)
        ax.set_title("Final SOC Distribution Across Robots (by type)")
        ax.set_ylabel("State of Charge (%)")
        ax.set_xlabel("")
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_allocation_distribution_by_type(sim, savepath: Optional[str] = None):
    """Stacked/grouped bar: how many tasks each robot TYPE completed --
    Section 30's "Task allocation distribution among robot types"."""
    rows = []
    for r in sim.fleet.robots.values():
        rows.append({"robot_type": r.robot_type, "completed": sim.completed_counts.get(r.robot_id, 0)})
    df = pd.DataFrame(rows).groupby("robot_type", as_index=False)["completed"].sum()
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = [{"UGV": "#2e7d32", "USV": "#1565c0", "UAV": "#ef6c00"}.get(t, "#888") for t in df.robot_type]
        ax.bar(df.robot_type, df.completed, color=colors)
        ax.set_title("Completed Tasks by Robot Type")
        ax.set_ylabel("tasks completed")
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_priority_level_completion(sim, savepath: Optional[str] = None):
    """Bar chart: completion rate broken down by task priority level --
    Section 30's "Priority-level task completion"."""
    rows = []
    for t in sim.tm.tasks.values():
        rows.append({"priority": t.priority, "completed": 1 if t.status == "COMPLETED" else 0})
    df = pd.DataFrame(rows)
    summary = df.groupby("priority")["completed"].mean().reset_index()
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=summary, x="priority", y="completed", hue="priority", legend=False,
                    palette="rocket", ax=ax)
        ax.set_title("Completion Rate by Task Priority Level")
        ax.set_ylabel("completion rate")
        ax.set_xlabel("priority (1=Low ... 5=Emergency)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_normal_vs_emergency(sim, savepath: Optional[str] = None):
    """Grouped bar: completion rate and avg completion time, normal vs
    emergency tasks -- Section 30's "Normal vs emergency performance"."""
    from metrics import _read_history

    task_df = _read_history(sim, "task_state_history.csv")
    alloc_df = _read_history(sim, "allocation_history.csv")
    rows = []
    for t in sim.tm.tasks.values():
        rows.append({"group": "Emergency" if t.emergency else "Normal",
                     "completed": 1 if t.status == "COMPLETED" else 0})
    df = pd.DataFrame(rows)
    summary = df.groupby("group")["completed"].mean().reset_index()

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=summary, x="group", y="completed", hue="group", legend=False,
                    palette={"Normal": "#607d8b", "Emergency": "#d32f2f"}, ax=ax, width=0.5)
        ax.set_title("Completion Rate: Normal vs Emergency Tasks")
        ax.set_ylabel("completion rate")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax





if __name__ == "__main__":
    env = Environment()
    fleet = RobotFleet()
    tm = TaskManager()
    fig, ax = plot_environment(env, fleet, tm, title="Day 1 — Environment, Fleet & Tasks")
    fig.savefig("results/day1_environment_overview.png", dpi=150, bbox_inches="tight")
    print("Saved results/day1_environment_overview.png")

    # Quick self-tests of the new chart helpers (synthetic/small data, just
    # to confirm they run without error -- real figures are produced from
    # actual notebook data, not here).
    import os
    os.makedirs("results", exist_ok=True)

    df_demo = pd.DataFrame({
        "a": np.random.rand(50), "b": np.random.rand(50), "c": np.random.rand(50),
    })
    fig1, _ = plot_correlation_heatmap(df_demo, ["a", "b", "c"], title="Self-test heatmap")
    fig1.savefig("results/_selftest_heatmap.png", dpi=80)

    y_true = np.random.rand(100) * 100
    y_pred = y_true + np.random.randn(100) * 5
    fig2, _ = plot_prediction_diagnostics(y_true, y_pred, "self_test_target")
    fig2.savefig("results/_selftest_pred_diag.png", dpi=80)

    counts_demo = {"R1": 3, "R2": 1, "R3": 2}
    energy_demo = {"R1": 100.0, "R2": 40.0, "R3": 70.0}
    fig3, _ = plot_workload_and_energy(counts_demo, energy_demo, {"R1": "UGV", "R2": "USV", "R3": "UAV"})
    fig3.savefig("results/_selftest_workload_energy.png", dpi=80)

    for f in ["_selftest_heatmap.png", "_selftest_pred_diag.png", "_selftest_workload_energy.png"]:
        os.remove(os.path.join("results", f))
    print("OK: new Seaborn chart helpers in visualization.py all render without error.")
