"""WebSocket EnvClient for Delivery Optimization (use with a running OpenEnv server)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from delivery_env.models import DeliveryAction, DeliveryObservation, DeliveryState


class DeliveryEnvClient(EnvClient[DeliveryAction, DeliveryObservation, DeliveryState]):
    """
    Connects to ``ws://<host>/ws`` (HTTP base URL is converted automatically).

    Example::

        with DeliveryEnvClient(base_url=\"http://localhost:8000\").sync() as env:
            r = env.reset(seed=42, task_tier=\"hard\")
            while not r.done:
                r = env.step(DeliveryAction(opcode=0, edge_index=0))
    """

    def _step_payload(self, action: DeliveryAction) -> Dict[str, Any]:
        return {
            "opcode": action.opcode,
            "edge_index": action.edge_index,
            "order_index": action.order_index,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DeliveryObservation]:
        obs_data = payload.get("observation", {})
        observation = DeliveryObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_tier=obs_data.get("task_tier", "easy"),
            courier_node=obs_data.get("courier_node", 0),
            current_time_minutes=float(obs_data.get("current_time_minutes", 0.0)),
            capacity=int(obs_data.get("capacity", 1)),
            load_count=int(obs_data.get("load_count", 0)),
            graph_num_nodes=int(obs_data.get("graph_num_nodes", 1)),
            neighbor_nodes=list(obs_data.get("neighbor_nodes", [])),
            travel_minutes=[float(x) for x in obs_data.get("travel_minutes", [])],
            order_pickup_node=list(obs_data.get("order_pickup_node", [])),
            order_drop_node=list(obs_data.get("order_drop_node", [])),
            order_ready_time=[float(x) for x in obs_data.get("order_ready_time", [])],
            order_deadline=[float(x) for x in obs_data.get("order_deadline", [])],
            order_picked=list(obs_data.get("order_picked", [])),
            order_delivered=list(obs_data.get("order_delivered", [])),
            legal_edge_mask=list(obs_data.get("legal_edge_mask", [])),
            legal_pickup_mask=list(obs_data.get("legal_pickup_mask", [])),
            legal_deliver_mask=list(obs_data.get("legal_deliver_mask", [])),
            info_message=obs_data.get("info_message", ""),
            grader_score=obs_data.get("grader_score"),
            metadata=obs_data.get("metadata", {}) or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DeliveryState:
        return DeliveryState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            task_tier=str(payload.get("task_tier", "easy")),
            seed=payload.get("seed"),
            total_travel_minutes=float(payload.get("total_travel_minutes", 0.0)),
            orders_total=int(payload.get("orders_total", 0)),
            orders_delivered_count=int(payload.get("orders_delivered_count", 0)),
            late_deliveries=int(payload.get("late_deliveries", 0)),
            on_time_deliveries=int(payload.get("on_time_deliveries", 0)),
            illegal_actions=int(payload.get("illegal_actions", 0)),
            max_steps=int(payload.get("max_steps", 0)),
        )
