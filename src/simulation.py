"""
simulation.py
--------------
Section 23 (Dynamic Network Graph Visualization) + Section 24 (Live 2D
Simulation): the full integrated dashboard, built on top of the Day 3
`reallocation.Simulator` engine (state is real, not staged for the demo).

Delivered as:
    - `render_frame()`: one multi-panel dashboard snapshot (map + network
      graph + text side panel) for the CURRENT simulator state.
    - `render_before_after()`: the exact Section 23 example -- two graph
      snapshots side by side showing edges appear/disappear when the
      state changes (battery drop, emergency, robot failure).
    - `make_animation()`: steps a Simulator through checkpoints and saves
      an animated GIF, the closest a static notebook can get to "live".

This module does not reimplement allocation logic -- it only visualizes
`reallocation.Simulator` state, so what you see is exactly what the Day 3
engine actually computed, not a separate scripted demo.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from environment import Environment
from robots import Robot
from tasks import Task
from network_flow import build_flow_graph, SOURCE, SINK
from visualization import plot_environment, ZONE_COLORS, ROBOT_MARKERS
from astar import astar
from fairness import jains_fairness_index

TYPE_COLOR = {"UGV": "#2e7d32", "USV": "#1565c0", "UAV": "#ef6c00"}


def _task_color(task: Task) -> str:
    if task.status == "COMPLETED":
        return "#7fbf7f"
    if task.status == "CANCELLED":
        return "#bbbbbb"
    if task.emergency:
        return "#d32f2f"
    if task.priority >= 4:
        return "#ef6c00"
    return "#455a64"


def draw_network_graph(ax, sim, title: str = "Dynamic Network Flow Graph", astar_step: int = 3):
    """SOURCE -> Robots -> Tasks -> SINK, per Section 23's exact layout.

    Draws THREE edge states, not two:
      - structurally infeasible (type/capability/payload mismatch): not
        drawn at all (as before)
      - battery-infeasible (type-feasible but the robot's predicted SOC
        would drop below its safety floor): drawn dashed red, so a
        low-battery robot's edges visibly shrink/disappear (Section 23's
        required "R1 loses an edge when SOC drops" example) -- this was
        previously MISSING (the graph only checked type/capability, never
        battery, so low-SOC never visibly changed the graph; fixed here
        by running the same astar + battery.is_feasible check
        auction.compute_bid() uses, not a separate/looser check).
      - feasible (passes both checks): thin gray, as before.
      - selected/allocated: bold, robot-type colored, as before.
    """
    robots = list(sim.fleet.robots.values())
    tasks = [t for t in sim.tm.tasks.values() if t.is_active() or t.status == "COMPLETED"]
    tasks = tasks[:30]

    def euclid(r, t):
        return math.hypot(r.x - t.x, r.y - t.y)

    G = build_flow_graph([r for r in robots if r.is_operational()],
                          [t for t in tasks if t.is_active()], distance_fn=euclid)

    n_r, n_t = len(robots), len(tasks)
    pos = {SOURCE: (0, n_r / 2), SINK: (3, n_t / 2)}
    for i, r in enumerate(robots):
        pos[f"R:{r.robot_id}"] = (1, n_r - i)
    for i, t in enumerate(tasks):
        pos[f"T:{t.task_id}"] = (2, n_t - i)

    # Classify each type-feasible edge as battery-feasible or battery-blocked
    # using the SAME check auction.compute_bid() uses (astar path + battery
    # safety margin), not a cheaper/looser proxy.
    battery_ok, battery_blocked = [], []
    for u, v in G.edges():
        if u == SOURCE or v == SINK:
            continue
        rid, tid = u.split("R:", 1)[1], v.split("T:", 1)[1]
        robot, task = sim.fleet.robots[rid], sim.tm.tasks[tid]
        path = astar(sim.env, (robot.x, robot.y), (task.x, task.y), robot.robot_type, step=astar_step)
        if path.found and sim.battery.is_feasible(robot, path.path_length, task.workload):
            battery_ok.append((u, v))
        else:
            battery_blocked.append((u, v))

    for u, v in battery_blocked:
        if u in pos and v in pos:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#e57373", linewidth=0.7, linestyle="--",
                    alpha=0.6, zorder=1)
    for u, v in battery_ok:
        if u in pos and v in pos:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#dddddd", linewidth=0.6, zorder=1)

    # Selected/allocated edges (bold, robot-type colored)
    for r in robots:
        if r.current_task and f"R:{r.robot_id}" in pos and f"T:{r.current_task}" in pos:
            x1, y1 = pos[f"R:{r.robot_id}"]; x2, y2 = pos[f"T:{r.current_task}"]
            ax.plot([x1, x2], [y1, y2], color=TYPE_COLOR[r.robot_type], linewidth=2.5, zorder=3)

    # SOURCE/SINK
    for node in (SOURCE, SINK):
        x, y = pos[node]
        ax.scatter([x], [y], marker="s", s=220, c="black", zorder=4)
        ax.annotate(node, (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")

    # Robot nodes
    for r in robots:
        x, y = pos[f"R:{r.robot_id}"]
        face = TYPE_COLOR[r.robot_type] if r.status != "FAILED" else "#999999"
        marker = "X" if r.status == "FAILED" else {"UGV": "s", "USV": "^", "UAV": "o"}[r.robot_type]
        ax.scatter([x], [y], marker=marker, s=140, c=face, edgecolors="black", zorder=5)
        ax.annotate(f"{r.robot_id}\n{r.soc:.0f}%", (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=6, va="center")

    # Task nodes
    for t in tasks:
        x, y = pos[f"T:{t.task_id}"]
        marker = "*" if t.emergency else ("o" if t.status == "COMPLETED" else "x")
        ax.scatter([x], [y], marker=marker, s=(160 if t.emergency else 70),
                   c=_task_color(t), zorder=5)
        ax.annotate(t.task_id, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=6, va="center")

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1, max(n_r, n_t) + 2)
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    legend_elems = [
        Line2D([0], [0], color="#dddddd", lw=1.5, label="feasible edge"),
        Line2D([0], [0], color="#e57373", lw=1.5, linestyle="--", label="battery-blocked edge"),
        Line2D([0], [0], color=TYPE_COLOR["UGV"], lw=2.5, label="selected edge (color = robot type)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#d32f2f", markersize=12, label="emergency task"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7fbf7f", markersize=9, label="completed task"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#999999", markersize=9, label="failed robot"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=7, frameon=False)
    return G


def side_panel_text(sim) -> str:
    active = sum(1 for t in sim.tm.tasks.values() if t.is_active())
    completed = sum(1 for t in sim.tm.tasks.values() if t.status == "COMPLETED")
    fairness = jains_fairness_index(list(sim.completed_counts.values()))
    emergency_active = [t.task_id for t in sim.tm.tasks.values() if t.emergency and t.is_active()]
    last_event = sim.event_log[-1] if sim.event_log else None
    last_event_str = "none"
    if last_event:
        detail = last_event["detail"]
        if len(detail) > 45:
            detail = detail[:42] + "..."
        last_event_str = f"{last_event['event_type']} - {detail}"

    lines = [
        f"Simulation Time: t = {sim.t:.1f}",
        f"Active Tasks: {active}    Completed: {completed}",
        "",
        "--- Fleet ---",
    ]
    for r in sim.fleet.robots.values():
        tag = f"{r.robot_id} — {r.robot_type} — SOC {r.soc:5.1f}% — {r.status}"
        if r.current_task:
            tag += f" -> {r.current_task}"
        lines.append(tag)
    lines += [
        "",
        f"Jain Fairness (workload): {fairness:.3f}",
        f"Emergency active: {emergency_active if emergency_active else 'none'}",
        f"Last event: {last_event_str}",
        f"Reallocation count: {sim.reallocation_count}",
    ]
    return "\n".join(lines)


def render_frame(sim, title_suffix: str = "", figsize=(16, 9)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.7])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])
    ax_text.axis("off")

    plot_environment(sim.env, sim.fleet, sim.tm, title=f"Live Map (t={sim.t:.1f}) {title_suffix}", ax=ax_map)

    for r in sim.fleet.robots.values():
        if r.current_task and r.status == "BUSY":
            task = sim.tm.tasks.get(r.current_task)
            if task:
                path = astar(sim.env, (r.x, r.y), (task.x, task.y), r.robot_type, step=2)
                if path.found:
                    xs = [p[0] for p in path.path]; ys = [p[1] for p in path.path]
                    ax_map.plot(xs, ys, "--", linewidth=1.2, color=TYPE_COLOR[r.robot_type], alpha=0.7, zorder=9)

    draw_network_graph(ax_graph, sim, title=f"Network Flow Graph {title_suffix}")
    ax_text.text(0, 1, side_panel_text(sim), fontsize=8.5, family="monospace", va="top", transform=ax_text.transAxes)

    fig.tight_layout()
    return fig


def render_before_after(sim_before, sim_after, event_desc: str, savepath: Optional[str] = None):
    """Section 23's exact required example: two graph snapshots side by
    side showing edges that appeared/disappeared due to a state change."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    draw_network_graph(axes[0], sim_before, title="BEFORE")
    draw_network_graph(axes[1], sim_after, title="AFTER")
    fig.suptitle(event_desc, fontsize=12)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def make_animation(dataset_root: str = "data/dataset", savepath: str = "results/day4_live_simulation.gif",
                    max_frames: int = 12, fps: float = 1.0, run_id: str = "animation_run"):
    """Steps a fresh Simulator through the real event timeline (Section
    12's actual loop, not a scripted fake sequence) and saves one dashboard
    frame per checkpoint (capped at max_frames, evenly sampled across the
    run) as an animated GIF -- the closest a static notebook artifact can
    get to Section 24's "live 2D simulation"."""
    import matplotlib.animation as animation
    from reallocation import Simulator

    sim = Simulator(dataset_root, run_id=run_id)
    horizon = max(sim.events.event_times())
    checkpoints = sorted(set(sim.events.event_times()) | {0.0, horizon})
    if len(checkpoints) > max_frames:
        idx = np.linspace(0, len(checkpoints) - 1, max_frames).astype(int)
        sample_times = sorted(set(checkpoints[i] for i in idx))
    else:
        sample_times = checkpoints

    frames_state = []  # list of (t, dashboard-relevant snapshot info) captured as we step the real sim
    last_t = 0.0
    for t in checkpoints:
        dt = t - last_t
        sim._complete_due_tasks(t)
        sim._auto_charge_pass(t, dt if dt > 0 else 1.0)
        sim._apply_due_events(t)
        sim._complete_due_tasks(t)
        sim._allocate(t)
        sim._log_fairness(t)
        sim.t = t
        last_t = t
        if t in sample_times:
            import copy as _copy
            frames_state.append(_copy.deepcopy(sim))
    sim.logger.flush()

    fig = plt.figure(figsize=(16, 9))

    def draw(i):
        fig.clf()
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.7])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_graph = fig.add_subplot(gs[0, 1])
        ax_text = fig.add_subplot(gs[0, 2])
        ax_text.axis("off")
        s = frames_state[i]
        plot_environment(s.env, s.fleet, s.tm, title=f"Live Map (t={s.t:.1f})", ax=ax_map)
        draw_network_graph(ax_graph, s, title=f"Network Flow Graph (t={s.t:.1f})")
        ax_text.text(0, 1, side_panel_text(s), fontsize=8, family="monospace", va="top",
                     transform=ax_text.transAxes)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames_state), interval=1000 / fps)
    anim.save(savepath, writer="pillow", fps=fps)
    plt.close(fig)
    return savepath, [s.t for s in frames_state]


if __name__ == "__main__":
    from reallocation import Simulator

    sim = Simulator(run_id="simulation_selftest")
    sim._allocate(t=0.0)
    fig = render_frame(sim, title_suffix="(self-test)")
    fig.savefig("results/day4_dashboard_selftest.png", dpi=120, bbox_inches="tight")
    print("Saved results/day4_dashboard_selftest.png")

    n_robot_nodes = sum(1 for r in sim.fleet.robots.values())
    assert n_robot_nodes == 12
    print("OK: simulation.py dashboard renders without error.")
