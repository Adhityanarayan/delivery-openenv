#!/usr/bin/env python3
"""
Reproducible greedy baseline scores (in-process environment, no HTTP).

Usage:
  PYTHONPATH=. python scripts/baseline.py
  PYTHONPATH=. python scripts/baseline.py --seeds 0,1,2 --tiers easy,medium,hard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.environment import DeliveryOptimizationEnvironment


def parse_list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_episode(seed: int, task_tier: str) -> dict:
    env = DeliveryOptimizationEnvironment()
    obs = env.reset(seed=seed, task_tier=task_tier)
    total_reward = 0.0
    steps = 0
    while not obs.done and steps < 5000:
        steps += 1
        a = env.suggest_greedy_action()
        obs = env.step(a)
        total_reward += float(obs.reward or 0.0)
    return {
        "seed": seed,
        "task_tier": task_tier,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "grader_score": obs.grader_score,
        "delivered": env.state.orders_delivered_count,
        "orders_total": env.state.orders_total,
        "illegal_actions": env.state.illegal_actions,
        "travel_minutes": round(env.state.total_travel_minutes, 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Greedy baseline for Delivery OpenEnv")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,42,100,123")
    p.add_argument("--tiers", type=str, default="easy,medium,hard")
    p.add_argument("--json", action="store_true", help="print one JSON array")
    args = p.parse_args()
    seeds = [int(x) for x in parse_list(args.seeds)]
    tiers = parse_list(args.tiers)
    rows = []
    for tier in tiers:
        scores = []
        for seed in seeds:
            row = run_episode(seed, tier)
            rows.append(row)
            if row.get("grader_score") is not None:
                scores.append(row["grader_score"])
        if scores and not args.json:
            mean = sum(scores) / len(scores)
            print(f"\n{tier.upper()}  mean_grader={mean:.4f}  (n={len(scores)} episodes)")
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
