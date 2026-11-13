"""
graph_rl.py
-----------
Section 20 (Graph Reinforcement Learning) + Section 21 (Reward Function).

Architecture (matches the spec's diagram):

    Dynamic Graph -> GAT -> Graph Embedding -> PPO Policy -> Action
                  -> Allocation/decision -> Environment -> Reward -> Updated Graph

SCOPE / DOCUMENTED LIMITATION (per spec: "If full GAT-PPO training becomes
computationally excessive... implement a clean simplified version and
clearly document the limitation rather than producing non-working code."):

- A single-head Graph Attention (GAT) layer encodes robot and task nodes
  (not a multi-layer/multi-head production GAT).
- The MDP is ONE ALLOCATION ROUND: robots act in sequence (one action
  each: pick a feasible pending task, or NO-OP), not a full multi-day
  continuous-time simulation -- this keeps state/action spaces small
  enough to train in seconds-to-minutes on CPU, matching this fleet's
  actual scale (<=12 robots, <=30 tasks).
- PPO is implemented directly (not via an RL framework) with a single
  actor-critic network, GAE advantages, and the standard clipped
  surrogate objective -- a real, working, from-scratch implementation,
  kept intentionally small.
- The action space is masked to only currently-feasible tasks (type,
  capability, payload, AND battery-safety), so unsafe-SOC / forbidden
  robot-task pairs are structurally prevented rather than merely
  penalized -- forbidden-zone and unsafe-SOC violations literally cannot
  be selected as actions.

The GAT/PPO component is explicitly a COMPLEMENT to, not a replacement
for, the Network-Flow + auction pipeline (Section 20: "Do not allow
GAT/PPO to destroy the Network Flow foundation") -- it is trained and
evaluated as an alternative allocation policy, compared against the
auction baseline in notebook 10, not substituted into the main pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from robots import RobotFleet, Robot
from tasks import TaskManager, Task, is_feasible_edge
from battery import BatteryModel
from astar import astar
from fairness import jains_fairness_index

ROBOT_TYPES = ["UGV", "USV", "UAV"]
ROBOT_FEAT_DIM = 6 + len(ROBOT_TYPES)   # soc,dod,soh,x,y,speed + one-hot type
TASK_FEAT_DIM = 7                        # priority,workload,deadline,emergency,x,y,risk
HIDDEN_DIM = 16
MAX_TASKS = 30  # padding size for the action space (dataset has 30 tasks)


# ------------------------------------------------------------------------
# Reward function (Section 21) -- every term traces to a spec bullet.
# ------------------------------------------------------------------------
REWARD_WEIGHTS = {
    "completion": 1.0,          # + task completion
    "priority": 1.5,            # + high-priority completion
    "emergency": 2.0,           # + emergency response
    "energy": -1.0,             # - energy consumption (низкий energy => small penalty)
    "battery_stress": -1.0,     # - battery stress (post-task SOC drop)
    "distance": -0.5,           # - unnecessary travel
    "deadline_bonus": 0.5,      # + deadline satisfaction
    "deadline_penalty": -2.0,   # - missed deadline
    "risk": -0.5,               # - risk
    "noop_penalty": -0.2,       # small penalty for skipping when work exists (discourage idling)
}


def reward_for_assignment(robot: Robot, task: Task, distance: float, workload: float,
                           battery: BatteryModel, env) -> float:
    w = REWARD_WEIGHTS
    r = w["completion"]
    r += w["priority"] * (task.priority / 5.0)
    if task.emergency:
        r += w["emergency"]

    energy = battery.predicted_energy_wh(robot, distance, workload)
    energy_norm = min(1.0, energy / 400.0)
    r += w["energy"] * energy_norm

    predicted_soc = battery.predicted_soc_after(robot, distance, workload)
    stress_norm = min(1.0, (robot.soc - predicted_soc) / 100.0)
    r += w["battery_stress"] * stress_norm

    dist_norm = min(1.0, distance / 120.0)
    r += w["distance"] * dist_norm

    est_time = distance / max(robot.speed, 1e-6)
    if est_time <= task.deadline:
        r += w["deadline_bonus"]
    else:
        r += w["deadline_penalty"]

    zone_risk = max((z.risk_level for z in env.zones_at(task.x, task.y)), default=1.0)
    r += w["risk"] * (zone_risk / 5.0)

    return r


# ------------------------------------------------------------------------
# Feature encoding
# ------------------------------------------------------------------------
def robot_features(robot: Robot) -> np.ndarray:
    type_onehot = [1.0 if robot.robot_type == t else 0.0 for t in ROBOT_TYPES]
    return np.array([
        robot.soc / 100.0, robot.dod / 100.0, robot.soh,
        robot.x / 100.0, robot.y / 100.0, robot.speed / 10.0,
        *type_onehot,
    ], dtype=np.float32)


def task_features(task: Task, env) -> np.ndarray:
    zone_risk = max((z.risk_level for z in env.zones_at(task.x, task.y)), default=1.0)
    return np.array([
        task.priority / 5.0, task.workload / 40.0, min(task.deadline, 200.0) / 200.0,
        1.0 if task.emergency else 0.0, task.x / 100.0, task.y / 100.0, zone_risk / 5.0,
    ], dtype=np.float32)


# ------------------------------------------------------------------------
# Single-head GAT layer (from scratch, small and explicit)
# ------------------------------------------------------------------------
class GATLayer(nn.Module):
    """One robot node attends over its currently-feasible task nodes
    (a bipartite, dynamically-masked attention -- exactly the
    "robot-task relationship" edge set the spec asks the GAT to
    represent), producing an updated robot embedding."""

    def __init__(self, robot_dim: int, task_dim: int, hidden: int):
        super().__init__()
        self.Wr = nn.Linear(robot_dim, hidden, bias=False)
        self.Wt = nn.Linear(task_dim, hidden, bias=False)
        self.attn = nn.Linear(2 * hidden, 1, bias=False)

    def forward(self, robot_vec: torch.Tensor, task_mat: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """robot_vec: (hidden,) after projection is computed inside.
        task_mat: (n_tasks, task_dim). mask: (n_tasks,) bool, True = valid."""
        h_r = self.Wr(robot_vec)                     # (hidden,)
        h_t = self.Wt(task_mat)                       # (n_tasks, hidden)
        h_r_expand = h_r.unsqueeze(0).expand(h_t.size(0), -1)
        e = self.attn(torch.cat([h_r_expand, h_t], dim=-1)).squeeze(-1)  # (n_tasks,)
        e = e.masked_fill(~mask, float("-inf"))
        if mask.sum() == 0:
            alpha = torch.zeros_like(e)
        else:
            alpha = F.softmax(e, dim=0)
            alpha = torch.nan_to_num(alpha, nan=0.0)
        h_agg = (alpha.unsqueeze(-1) * h_t).sum(dim=0)   # (hidden,)
        return F.elu(h_r + h_agg), h_t                   # updated robot embedding, task embeddings


# ------------------------------------------------------------------------
# Actor-Critic policy
# ------------------------------------------------------------------------
class GATPolicy(nn.Module):
    def __init__(self, robot_dim=ROBOT_FEAT_DIM, task_dim=TASK_FEAT_DIM, hidden=HIDDEN_DIM,
                 max_tasks=MAX_TASKS):
        super().__init__()
        self.robot_proj = nn.Linear(robot_dim, hidden)
        self.gat = GATLayer(hidden, task_dim, hidden)
        self.score_head = nn.Linear(2 * hidden, 1)   # score(robot_emb, task_emb) -> logit
        self.noop_bias = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.max_tasks = max_tasks

    def forward(self, robot_feat: torch.Tensor, task_feats: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits [n_tasks+1 incl. NO-OP], value)."""
        r0 = self.robot_proj(robot_feat)
        h_r, h_t = self.gat(r0, task_feats, mask)
        h_r_expand = h_r.unsqueeze(0).expand(h_t.size(0), -1)
        task_logits = self.score_head(torch.cat([h_r_expand, h_t], dim=-1)).squeeze(-1)
        task_logits = task_logits.masked_fill(~mask, float("-inf"))
        noop_logit = self.noop_bias.expand(1)
        logits = torch.cat([task_logits, noop_logit], dim=0)  # last index = NO-OP, always valid
        value = self.value_head(h_r).squeeze(-1)
        return logits, value


# ------------------------------------------------------------------------
# Environment: one allocation round as an MDP (robots act in sequence)
# ------------------------------------------------------------------------
class AllocationEnv:
    def __init__(self, dataset_root: str = "data/dataset", seed: Optional[int] = None):
        self.dataset_root = dataset_root
        self.seed = seed
        self.reset()

    def reset(self):
        self.env = Environment(self.dataset_root)
        self.fleet = RobotFleet(self.dataset_root)
        self.battery = BatteryModel(self.fleet, self.dataset_root)
        self.tm = TaskManager(self.dataset_root)
        self.robot_order = list(self.fleet.robots.keys())
        if self.seed is not None:
            random.Random(self.seed).shuffle(self.robot_order)
        self.ptr = 0
        self.completed_counts = {rid: 0 for rid in self.fleet.robots}
        self.episode_reward = 0.0
        return self._obs()

    def _pending_tasks(self) -> List[Task]:
        return [t for t in self.tm.tasks.values() if t.status == "PENDING" and t.release_time <= 0]

    def _current_robot(self) -> Optional[Robot]:
        while self.ptr < len(self.robot_order):
            r = self.fleet.robots[self.robot_order[self.ptr]]
            if r.is_available():
                return r
            self.ptr += 1
        return None

    def _obs(self):
        robot = self._current_robot()
        tasks = self._pending_tasks()[: MAX_TASKS]
        if robot is None:
            return None
        rf = torch.tensor(robot_features(robot))
        n_pad = MAX_TASKS - len(tasks)
        if tasks:
            tf = torch.tensor(np.stack([task_features(t, self.env) for t in tasks]))
        else:
            tf = torch.zeros((0, TASK_FEAT_DIM))
        mask = torch.zeros(MAX_TASKS, dtype=torch.bool)
        distances = {}
        for i, t in enumerate(tasks):
            feasible = is_feasible_edge(robot, t)
            if feasible:
                path = astar(self.env, (robot.x, robot.y), (t.x, t.y), robot.robot_type, step=3)
                if path.found and self.battery.is_feasible(robot, path.path_length, t.workload):
                    mask[i] = True
                    distances[i] = path.path_length
        if n_pad > 0:
            tf = torch.cat([tf, torch.zeros((n_pad, TASK_FEAT_DIM))], dim=0)
        return {"robot": robot, "tasks": tasks, "robot_feat": rf.float(), "task_feats": tf.float(),
                "mask": mask, "distances": distances}

    def step(self, action: int, obs: dict) -> Tuple[Optional[dict], float, bool]:
        robot = obs["robot"]
        tasks = obs["tasks"]
        is_noop = (action == MAX_TASKS) or (action >= len(tasks))

        if is_noop:
            reward = REWARD_WEIGHTS["noop_penalty"] if obs["mask"].any() else 0.0
        else:
            task = tasks[action]
            distance = obs["distances"][action]
            reward = reward_for_assignment(robot, task, distance, task.workload, self.battery, self.env)
            self.battery.consume(robot, distance, task.workload)
            robot.status, robot.current_task = "BUSY", task.task_id
            task.status, task.assigned_robot = "ASSIGNED", robot.robot_id
            self.completed_counts[robot.robot_id] += 1

        self.episode_reward += reward
        self.ptr += 1
        next_obs = self._obs()
        done = next_obs is None
        if done:
            fairness = jains_fairness_index(list(self.completed_counts.values()))
            reward += 1.0 * fairness  # terminal fairness bonus (Section 21: "fairness")
            self.episode_reward += 1.0 * fairness
        return next_obs, reward, done


# ------------------------------------------------------------------------
# PPO training (from-scratch, single actor-critic, clipped surrogate)
# ------------------------------------------------------------------------
@dataclass
class Rollout:
    robot_feats: List[torch.Tensor] = field(default_factory=list)
    task_feats: List[torch.Tensor] = field(default_factory=list)
    masks: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)


def collect_episode(env: AllocationEnv, policy: GATPolicy) -> Tuple[Rollout, float]:
    obs = env.reset()
    rollout = Rollout()
    while obs is not None:
        logits, value = policy(obs["robot_feat"], obs["task_feats"], obs["mask"])
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)

        rollout.robot_feats.append(obs["robot_feat"])
        rollout.task_feats.append(obs["task_feats"])
        rollout.masks.append(obs["mask"])
        rollout.actions.append(int(action.item()))
        rollout.logprobs.append(logprob.detach())
        rollout.values.append(value.detach())

        next_obs, reward, done = env.step(int(action.item()), obs)
        rollout.rewards.append(reward)
        obs = next_obs
    return rollout, env.episode_reward


def compute_gae(rewards: List[float], values: List[torch.Tensor], gamma=0.95, lam=0.9):
    values = [v.item() for v in values] + [0.0]
    advantages, gae = [], 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(policy: GATPolicy, optimizer, rollouts: List[Rollout],
                clip_eps=0.2, epochs=4, value_coef=0.5, entropy_coef=0.01):
    all_adv, all_ret = [], []
    for ro in rollouts:
        adv, ret = compute_gae(ro.rewards, ro.values)
        all_adv.extend(adv)
        all_ret.extend(ret)
    adv_t = torch.tensor(all_adv, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(all_ret, dtype=torch.float32)

    flat = [(rf, tf, m, a, lp) for ro in rollouts
            for rf, tf, m, a, lp in zip(ro.robot_feats, ro.task_feats, ro.masks, ro.actions, ro.logprobs)]

    total_loss_val = 0.0
    for _ in range(epochs):
        for i, (rf, tf, m, a, old_lp) in enumerate(flat):
            logits, value = policy(rf, tf, m)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(torch.tensor(a))
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * adv_t[i]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[i]
            policy_loss = -torch.min(surr1, surr2)
            value_loss = F.mse_loss(value, ret_t[i])
            entropy = dist.entropy()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_val += loss.item()
    return total_loss_val / max(1, len(flat) * epochs)


def train(dataset_root: str = "data/dataset", n_episodes: int = 200, batch_size: int = 4,
          lr: float = 3e-3, seed: int = 42) -> Tuple[GATPolicy, List[float]]:
    torch.manual_seed(seed)
    policy = GATPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    reward_history = []

    batch = []
    for ep in range(n_episodes):
        env = AllocationEnv(dataset_root, seed=seed + ep)
        rollout, ep_reward = collect_episode(env, policy)
        reward_history.append(ep_reward)
        batch.append(rollout)
        if len(batch) >= batch_size:
            ppo_update(policy, optimizer, batch)
            batch = []
    if batch:
        ppo_update(policy, optimizer, batch)

    return policy, reward_history


def evaluate_policy(policy: GATPolicy, dataset_root: str = "data/dataset", seed: int = 999
                     ) -> Dict[str, float]:
    env = AllocationEnv(dataset_root, seed=seed)
    obs = env.reset()
    n_assigned = 0
    with torch.no_grad():
        while obs is not None:
            logits, _ = policy(obs["robot_feat"], obs["task_feats"], obs["mask"])
            action = int(torch.argmax(logits).item())
            if action < len(obs["tasks"]):
                n_assigned += 1
            obs, _, _ = env.step(action, obs)
    fairness = jains_fairness_index(list(env.completed_counts.values()))
    return {"episode_reward": env.episode_reward, "n_assigned": n_assigned, "fairness": fairness}


def evaluate_baseline_auction(dataset_root: str = "data/dataset", seed: int = 999) -> Dict[str, float]:
    """Runs the EXISTING auction (auction.py, Days 2-3's mechanism) over
    the same one-round allocation setting and scores it with the IDENTICAL
    reward function used for the RL policy, so the comparison in
    notebook 10 is apples-to-apples (same reward, same initial state,
    same feasibility rules) rather than comparing different metrics."""
    from auction import run_auction

    env = AllocationEnv(dataset_root, seed=seed)
    obs = env.reset()  # sets up env.env/env.fleet/env.battery/env.tm fresh

    pending = env._pending_tasks()
    robots = [env.fleet.robots[rid] for rid in env.robot_order]
    assignment = run_auction(robots, pending, env.env, env.battery, astar_step=3)

    total_reward = 0.0
    n_assigned = 0
    for robot_id, bid in assignment.items():
        robot = env.fleet.robots[robot_id]
        task = env.tm.tasks[bid.task_id]
        distance = bid.path.path_length
        total_reward += reward_for_assignment(robot, task, distance, task.workload, env.battery, env.env)
        env.completed_counts[robot_id] += 1
        n_assigned += 1
    fairness = jains_fairness_index(list(env.completed_counts.values()))
    total_reward += 1.0 * fairness
    return {"episode_reward": total_reward, "n_assigned": n_assigned, "fairness": fairness}


if __name__ == "__main__":
    print("Training a small GAT+PPO policy (this is the documented simplified version)...")
    policy, history = train(n_episodes=60, batch_size=4)
    print(f"Episode reward: first={history[0]:.2f}  last={history[-1]:.2f}  "
          f"mean(last 10)={np.mean(history[-10:]):.2f}")
    assert len(history) == 60

    result = evaluate_policy(policy)
    print("Greedy evaluation of trained policy:", result)
    assert result["n_assigned"] > 0

    baseline = evaluate_baseline_auction()
    print("Baseline auction (same reward function):", baseline)
    print("OK: graph_rl.py GAT+PPO trains and produces a usable, comparable evaluation policy.")
