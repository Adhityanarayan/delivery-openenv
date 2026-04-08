from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Optional, Tuple

from delivery_env.models import DeliveryAction
from delivery_env.simulator import CourierWorld, Scenario, build_scenario, shortest_path_travel


def grader_score_open_interval(raw: float) -> float:
    """Strict (0, 1) — some validators reject exactly 0.0 or 1.0."""
    if raw != raw or not math.isfinite(raw):  # NaN / inf
        return 0.5
    return float(max(0.01, min(0.99, raw)))


def grade_from_counts(
    task_tier: str,
    total_travel: float,
    baseline_travel: float,
    orders_total: int,
    on_time_deliveries: int,
    late_deliveries: int,
    all_delivered: bool,
) -> float:
    """Map episode outcomes to (0, 1) open interval for platform validators."""
    if orders_total == 0:
        return grader_score_open_interval(0.0)
    on_time_ratio = on_time_deliveries / max(1, on_time_deliveries + late_deliveries)
    if on_time_deliveries + late_deliveries == 0:
        on_time_ratio = 0.0
    if baseline_travel <= 1e-6:
        travel_eff = 1.0 if total_travel <= 1e-6 else 0.5
    else:
        travel_eff = min(1.0, baseline_travel / max(total_travel, baseline_travel * 0.5))
    completion = 1.0 if all_delivered else (0.25 * (on_time_deliveries + late_deliveries) / orders_total)
    tier = task_tier.lower()
    w_time = 0.45 if tier == "easy" else 0.5
    w_travel = 0.35 if tier == "easy" else 0.35
    w_comp = 0.20 if tier == "easy" else 0.15
    raw = w_time * on_time_ratio + w_travel * travel_eff + w_comp * completion
    if tier == "hard" and not all_delivered:
        raw *= 0.85
    closed = float(max(0.0, min(1.0, raw)))
    return grader_score_open_interval(closed)


def _primary_target_node(world: CourierWorld) -> Optional[int]:
    """Single goal node (pickup or drop) with best myopic cost — avoids ring oscillation."""
    scenario = world.scenario
    dist = world.dist
    cur = world.courier_node
    best_node: Optional[int] = None
    best_key: Optional[Tuple[float, int]] = None
    for i, o in enumerate(scenario.orders):
        if world.delivered[i]:
            continue
        if not world.picked[i]:
            c = shortest_path_travel(dist, cur, o.pickup_node) + shortest_path_travel(
                dist, o.pickup_node, o.drop_node
            )
            node = o.pickup_node
        elif world.in_vehicle[i]:
            c = shortest_path_travel(dist, cur, o.drop_node)
            node = o.drop_node
        else:
            continue
        key = (c, node)
        if best_key is None or key < best_key:
            best_key = key
            best_node = node
    return best_node


def next_greedy_action(world: CourierWorld) -> DeliveryAction:
    """One greedy decision matching `greedy_baseline_travel` (for env.step rollouts)."""
    scenario = world.scenario
    dist = world.dist
    _, legal_pickup, legal_deliver = world.legal_masks()
    deliver_candidates = [i for i, ok in enumerate(legal_deliver) if ok]
    if deliver_candidates:
        deliver_candidates.sort()
        return DeliveryAction(opcode=2, order_index=deliver_candidates[0], edge_index=0)
    pickup_candidates = [i for i, ok in enumerate(legal_pickup) if ok]
    if pickup_candidates:
        best_i = None
        best_key = None
        cur = world.courier_node
        for i in pickup_candidates:
            o = scenario.orders[i]
            c = shortest_path_travel(dist, cur, o.pickup_node) + shortest_path_travel(
                dist, o.pickup_node, o.drop_node
            )
            key = (c, i)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        if best_i is not None:
            return DeliveryAction(opcode=1, order_index=best_i, edge_index=0)
    best_node = _primary_target_node(world)
    nbrs, _ = world.neighbors()
    if not nbrs:
        return DeliveryAction(opcode=0, edge_index=0, order_index=0)
    if best_node is None:
        return DeliveryAction(opcode=0, edge_index=0, order_index=0)
    best_e = 0
    best_tuple: Optional[Tuple[float, int, int]] = None
    for e, v in enumerate(nbrs):
        d2 = shortest_path_travel(dist, v, best_node)
        t = (d2, v, e)
        if best_tuple is None or t < best_tuple:
            best_tuple = t
            best_e = e
    return DeliveryAction(opcode=0, edge_index=best_e, order_index=0)


def greedy_baseline_travel(scenario: Scenario) -> float:
    """Deterministic greedy: nearest feasible goal (pickup before drop, capacity)."""
    world = CourierWorld(scenario=scenario)
    max_steps = scenario.max_env_steps * 2
    steps = 0
    while not world.all_delivered() and steps < max_steps:
        steps += 1
        act = next_greedy_action(world)
        if act.opcode == 2:
            world.step_deliver(act.order_index)
        elif act.opcode == 1:
            world.step_pickup(act.order_index)
        else:
            world.step_move(act.edge_index)
    return world.total_travel


def baseline_travel_for_seed(task_tier: str, seed: int) -> float:
    sc = build_scenario(task_tier, seed)
    return greedy_baseline_travel(sc)
