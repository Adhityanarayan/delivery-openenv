#!/usr/bin/env python3
"""
WebSocket smoke test: local env + remote env stay in lockstep with identical actions.

Run server first: uvicorn server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delivery_env.client import DeliveryEnvClient
from server.environment import DeliveryOptimizationEnvironment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--task-tier", default="easy")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    local = DeliveryOptimizationEnvironment()
    local.reset(seed=args.seed, task_tier=args.task_tier)

    client = DeliveryEnvClient(base_url=args.base_url).sync()
    with client:
        r = client.reset(seed=args.seed, task_tier=args.task_tier)
        n = 0
        while not r.done and n < 500:
            n += 1
            a = local.suggest_greedy_action()
            local.step(a)
            r = client.step(a)
        obs = r.observation
        print("steps", n, "done", r.done, "grader", obs.grader_score, "info", obs.info_message)


if __name__ == "__main__":
    main()
