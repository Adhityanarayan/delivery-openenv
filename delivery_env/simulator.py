from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OrderSpec:
    pickup_node: int
    drop_node: int
    ready_time: float
    deadline: float


@dataclass
class Scenario:
    task_tier: str
    seed: int
    depot_node: int
    capacity: int
    num_nodes: int
    edges: Dict[int, List[Tuple[int, float]]]
    orders: List[OrderSpec]
    time_limit_minutes: float
    max_env_steps: int


def _add_edge(edges: Dict[int, List[Tuple[int, float]]], u: int, v: int, w: float) -> None:
    edges.setdefault(u, []).append((v, w))
    edges.setdefault(v, []).append((u, w))


def all_pairs_shortest_paths(
    n: int, edges: Dict[int, List[Tuple[int, float]]]
) -> Tuple[List[List[float]], List[List[int]]]:
    """Floyd–Warshall; returns dist and next hop for path reconstruction."""
    dist = [[math.inf] * n for _ in range(n)]
    nxt = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    for u, nbrs in edges.items():
        for v, w in nbrs:
            dist[u][v] = min(dist[u][v], w)
            nxt[u][v] = v
    for k in range(n):
        for i in range(n):
            if dist[i][k] == math.inf:
                continue
            for j in range(n):
                if dist[k][j] == math.inf:
                    continue
                nd = dist[i][k] + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd
                    nxt[i][j] = nxt[i][k]
    return dist, nxt


def shortest_path_travel(dist: List[List[float]], a: int, b: int) -> float:
    d = dist[a][b]
    return d if d != math.inf else 0.0


def build_scenario(task_tier: str, seed: int) -> Scenario:
    rng = __import__("random").Random(seed)
    tier = task_tier.lower()
    if tier not in ("easy", "medium", "hard"):
        tier = "easy"

    edges: Dict[int, List[Tuple[int, float]]] = {}
    depot = 0

    if tier == "easy":
        # Small line + branch; single order
        n = 6
        _add_edge(edges, 0, 1, 4.0)
        _add_edge(edges, 1, 2, 3.0)
        _add_edge(edges, 1, 3, 5.0)
        _add_edge(edges, 3, 4, 2.0)
        _add_edge(edges, 2, 5, 6.0)
        pu, dr = 4, 5
        orders = [
            OrderSpec(
                pickup_node=pu,
                drop_node=dr,
                ready_time=0.0,
                deadline=200.0,
            )
        ]
        capacity = 1
        time_limit = 240.0
        max_steps = 120
    elif tier == "medium":
        # Ring + chords; 4 orders, soft deadlines
        n = 8
        for i in range(n):
            _add_edge(edges, i, (i + 1) % n, 3.0 + rng.random() * 2.0)
        _add_edge(edges, 0, 4, 5.0)
        _add_edge(edges, 2, 6, 4.0)
        orders = []
        for _ in range(4):
            pu = rng.randrange(1, n)
            dr = rng.randrange(1, n)
            while dr == pu:
                dr = rng.randrange(1, n)
            ready = 0.0
            slack = float(rng.randint(40, 90))
            orders.append(
                OrderSpec(
                    pickup_node=pu,
                    drop_node=dr,
                    ready_time=ready,
                    deadline=ready + slack + 35.0,
                )
            )
        capacity = 1
        time_limit = 520.0
        max_steps = 320
    else:
        # Hard: batching, 6 orders, tighter windows
        n = 10
        for i in range(n):
            _add_edge(edges, i, (i + 1) % n, 2.5 + rng.random())
        _add_edge(edges, 0, 5, 6.0)
        _add_edge(edges, 2, 7, 5.0)
        _add_edge(edges, 3, 8, 4.0)
        orders = []
        for _ in range(6):
            pu = rng.randrange(1, n)
            dr = rng.randrange(1, n)
            while dr == pu:
                dr = rng.randrange(1, n)
            ready = 0.0
            slack = float(rng.randint(25, 55))
            orders.append(
                OrderSpec(
                    pickup_node=pu,
                    drop_node=dr,
                    ready_time=ready,
                    deadline=ready + slack + 25.0,
                )
            )
        capacity = 3
        time_limit = 480.0
        max_steps = 400

    return Scenario(
        task_tier=tier,
        seed=seed,
        depot_node=depot,
        capacity=capacity,
        num_nodes=n,
        edges=edges,
        orders=orders,
        time_limit_minutes=time_limit,
        max_env_steps=max_steps,
    )


@dataclass
class CourierWorld:
    scenario: Scenario
    dist: List[List[float]] = field(init=False)
    courier_node: int = 0
    time_minutes: float = 0.0
    picked: List[bool] = field(default_factory=list)
    delivered: List[bool] = field(default_factory=list)
    in_vehicle: List[bool] = field(default_factory=list)
    total_travel: float = 0.0

    def __post_init__(self) -> None:
        self.dist, _ = all_pairs_shortest_paths(self.scenario.num_nodes, self.scenario.edges)
        k = len(self.scenario.orders)
        self.picked = [False] * k
        self.delivered = [False] * k
        self.in_vehicle = [False] * k
        self.courier_node = self.scenario.depot_node

    def load_count(self) -> int:
        return sum(1 for i, v in enumerate(self.in_vehicle) if v and not self.delivered[i])

    def all_delivered(self) -> bool:
        return all(self.delivered)

    def neighbors(self) -> Tuple[List[int], List[float]]:
        nbrs = self.scenario.edges.get(self.courier_node, [])
        nodes = [v for v, _ in nbrs]
        times = [w for _, w in nbrs]
        return nodes, times

    def legal_masks(self) -> Tuple[List[bool], List[bool], List[bool]]:
        nodes, _ = self.neighbors()
        legal_move = [True] * len(nodes)
        k = len(self.scenario.orders)
        legal_pickup = [False] * k
        legal_deliver = [False] * k
        for i, o in enumerate(self.scenario.orders):
            if self.delivered[i]:
                continue
            if (
                not self.picked[i]
                and self.courier_node == o.pickup_node
                and self.time_minutes + 1e-6 >= o.ready_time
                and self.load_count() < self.scenario.capacity
            ):
                legal_pickup[i] = True
            if self.picked[i] and self.in_vehicle[i] and self.courier_node == o.drop_node:
                legal_deliver[i] = True
        if not nodes:
            legal_move = []
        return legal_move, legal_pickup, legal_deliver

    def step_move(self, edge_index: int) -> Tuple[bool, str]:
        nodes, times = self.neighbors()
        if edge_index < 0 or edge_index >= len(nodes):
            return False, "invalid edge_index"
        target = nodes[edge_index]
        travel = times[edge_index]
        self.courier_node = target
        self.time_minutes += travel
        self.total_travel += travel
        return True, f"moved to node {target} (+{travel:.1f} min)"

    def step_pickup(self, order_index: int) -> Tuple[bool, str]:
        o = self.scenario.orders
        if order_index < 0 or order_index >= len(o):
            return False, "invalid order_index"
        if self.delivered[order_index] or self.picked[order_index]:
            return False, "order not available for pickup"
        spec = o[order_index]
        if self.courier_node != spec.pickup_node:
            return False, "not at pickup location"
        if self.time_minutes + 1e-6 < spec.ready_time:
            return False, "order not ready"
        if self.load_count() >= self.scenario.capacity:
            return False, "vehicle at capacity"
        self.picked[order_index] = True
        self.in_vehicle[order_index] = True
        return True, f"picked order {order_index}"

    def step_deliver(self, order_index: int) -> Tuple[bool, str, float]:
        """Returns (ok, message, step_reward_component)."""
        o = self.scenario.orders
        if order_index < 0 or order_index >= len(o):
            return False, "invalid order_index", 0.0
        if not self.picked[order_index] or not self.in_vehicle[order_index]:
            return False, "order not in vehicle", 0.0
        if self.courier_node != o[order_index].drop_node:
            return False, "not at dropoff", 0.0
        t = self.time_minutes
        dl = o[order_index].deadline
        self.delivered[order_index] = True
        self.in_vehicle[order_index] = False
        on_time = t <= dl + 1e-6
        # partial progress: strong signal on delivery
        bonus = 0.35 if on_time else -0.15
        return True, f"delivered order {order_index} (on_time={on_time})", bonus
