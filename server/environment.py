from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from delivery_env.grading import baseline_travel_for_seed, grade_from_counts, next_greedy_action
from delivery_env.models import DeliveryAction, DeliveryObservation, DeliveryState
from delivery_env.simulator import CourierWorld, build_scenario


class DeliveryOptimizationEnvironment(
    Environment[DeliveryAction, DeliveryObservation, DeliveryState]
):
    """Courier routing: next stop, pickups, batching (capacity), deadlines."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__(transform=None, rubric=None)
        self._world: Optional[CourierWorld] = None
        self._state = DeliveryState()
        self._task_tier: str = "easy"
        self._seed: int = 0
        self._baseline_travel: float = 1.0
        self._on_time: int = 0
        self._late: int = 0

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="delivery_optimization",
            description=(
                "Real-world delivery dispatch simulation: graph routing, order batching, "
                "and time windows (easy / medium / hard)."
            ),
            version="1.0.0",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DeliveryObservation:
        task_tier = kwargs.get("task_tier", "easy")
        if seed is None:
            seed = uuid.uuid4().int % (2**31)
        self._seed = int(seed)
        self._task_tier = str(task_tier).lower()
        if self._task_tier not in ("easy", "medium", "hard"):
            self._task_tier = "easy"
        scenario = build_scenario(self._task_tier, self._seed)
        self._world = CourierWorld(scenario=scenario)
        self._baseline_travel = baseline_travel_for_seed(self._task_tier, self._seed)
        self._on_time = 0
        self._late = 0
        self._state = DeliveryState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_tier=self._task_tier,
            seed=self._seed,
            total_travel_minutes=0.0,
            orders_total=len(scenario.orders),
            orders_delivered_count=0,
            late_deliveries=0,
            on_time_deliveries=0,
            illegal_actions=0,
            max_steps=scenario.max_env_steps,
        )
        return self._observe(done=False, reward=None, info="episode started", grader=None)

    def step(
        self,
        action: DeliveryAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DeliveryObservation:
        assert self._world is not None
        w = self._world
        sc = w.scenario
        self._state.step_count += 1
        reward = -0.03
        info = ""
        done = False
        grader: Optional[float] = None
        n_orders = len(sc.orders)

        if self._state.step_count > sc.max_env_steps:
            done = True
            info = "max steps exceeded"
            reward = -1.0
            grader = self._final_grader(done_deliveries=sum(w.delivered) == n_orders)
        elif w.time_minutes > sc.time_limit_minutes:
            done = True
            info = "time limit exceeded"
            reward = -1.0
            grader = self._final_grader(done_deliveries=sum(w.delivered) == n_orders)
        elif action.opcode == 0:
            travel_before = w.total_travel
            ok, msg = w.step_move(action.edge_index)
            if ok:
                reward += -0.015 * max(0.0, w.total_travel - travel_before)
                info = msg
            else:
                self._state.illegal_actions += 1
                reward = -0.35
                info = f"illegal move: {msg}"
        elif action.opcode == 1:
            ok, msg = w.step_pickup(action.order_index)
            if ok:
                reward += 0.08
                info = msg
            else:
                self._state.illegal_actions += 1
                reward = -0.35
                info = f"illegal pickup: {msg}"
        elif action.opcode == 2:
            ok, msg, del_bonus = w.step_deliver(action.order_index)
            if ok:
                t = w.time_minutes
                dl = sc.orders[action.order_index].deadline
                if t <= dl + 1e-6:
                    self._on_time += 1
                    self._state.on_time_deliveries += 1
                else:
                    self._late += 1
                    self._state.late_deliveries += 1
                self._state.orders_delivered_count += 1
                reward += del_bonus + 0.12
                info = msg
            else:
                self._state.illegal_actions += 1
                reward = -0.35
                info = f"illegal deliver: {msg}"
        else:
            self._state.illegal_actions += 1
            reward = -0.4
            info = "unknown opcode"

        self._state.total_travel_minutes = w.total_travel

        if w.all_delivered():
            done = True
            reward += 0.5
            info = "all orders delivered"
            grader = self._final_grader(done_deliveries=True)

        if done and grader is None:
            grader = self._final_grader(done_deliveries=w.all_delivered())

        return self._observe(done=done, reward=reward, info=info, grader=grader)

    def _final_grader(self, done_deliveries: bool) -> float:
        assert self._world is not None
        w = self._world
        return grade_from_counts(
            self._task_tier,
            total_travel=w.total_travel,
            baseline_travel=self._baseline_travel,
            orders_total=len(w.scenario.orders),
            on_time_deliveries=self._on_time,
            late_deliveries=self._late,
            all_delivered=done_deliveries,
        )

    def _observe(
        self,
        done: bool,
        reward: Optional[float],
        info: str,
        grader: Optional[float],
    ) -> DeliveryObservation:
        assert self._world is not None
        w = self._world
        sc = w.scenario
        nodes, times = w.neighbors()
        le, lp, ld = w.legal_masks()
        return DeliveryObservation(
            done=done,
            reward=reward,
            task_tier=self._task_tier,
            courier_node=w.courier_node,
            current_time_minutes=w.time_minutes,
            capacity=sc.capacity,
            load_count=w.load_count(),
            graph_num_nodes=sc.num_nodes,
            neighbor_nodes=list(nodes),
            travel_minutes=list(times),
            order_pickup_node=[o.pickup_node for o in sc.orders],
            order_drop_node=[o.drop_node for o in sc.orders],
            order_ready_time=[o.ready_time for o in sc.orders],
            order_deadline=[o.deadline for o in sc.orders],
            order_picked=list(w.picked),
            order_delivered=list(w.delivered),
            legal_edge_mask=list(le),
            legal_pickup_mask=list(lp),
            legal_deliver_mask=list(ld),
            info_message=info,
            grader_score=grader,
        )

    @property
    def state(self) -> DeliveryState:
        return self._state

    def suggest_greedy_action(self) -> DeliveryAction:
        assert self._world is not None
        return next_greedy_action(self._world)
