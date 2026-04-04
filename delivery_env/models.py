"""Typed Action / Observation / State (Pydantic) for Delivery Optimization OpenEnv."""

from __future__ import annotations

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class DeliveryAction(Action):
    """MOVE (edge index), PICKUP (order index), or DELIVER (order index)."""

    opcode: int = Field(
        0,
        ge=0,
        le=2,
        description="0=MOVE along neighbor edge index, 1=PICKUP, 2=DELIVER",
    )
    edge_index: int = Field(0, ge=0, description="Neighbor index when opcode==0")
    order_index: int = Field(0, ge=0, description="Order index when opcode==1 or 2")


class DeliveryObservation(Observation):
    """Observation after reset or step (lists are JSON-friendly for LLM agents)."""

    task_tier: str = Field(default="easy", description="easy | medium | hard")
    courier_node: int = Field(default=0)
    current_time_minutes: float = Field(default=0.0)
    capacity: int = Field(default=1)
    load_count: int = Field(default=0)
    graph_num_nodes: int = Field(default=1)
    neighbor_nodes: List[int] = Field(default_factory=list)
    travel_minutes: List[float] = Field(default_factory=list)
    order_pickup_node: List[int] = Field(default_factory=list)
    order_drop_node: List[int] = Field(default_factory=list)
    order_ready_time: List[float] = Field(default_factory=list)
    order_deadline: List[float] = Field(default_factory=list)
    order_picked: List[bool] = Field(default_factory=list)
    order_delivered: List[bool] = Field(default_factory=list)
    legal_edge_mask: List[bool] = Field(default_factory=list)
    legal_pickup_mask: List[bool] = Field(default_factory=list)
    legal_deliver_mask: List[bool] = Field(default_factory=list)
    info_message: str = Field(default="", description="Human-readable feedback")
    grader_score: Optional[float] = Field(
        default=None,
        description="Task score in [0,1] when episode is done",
    )


class DeliveryState(State):
    """Episode metadata from GET /state (WebSocket session) or server introspection."""

    task_tier: str = Field(default="easy")
    seed: Optional[int] = Field(default=None)
    total_travel_minutes: float = Field(default=0.0)
    orders_total: int = Field(default=0)
    orders_delivered_count: int = Field(default=0)
    late_deliveries: int = Field(default=0)
    on_time_deliveries: int = Field(default=0)
    illegal_actions: int = Field(default=0)
    max_steps: int = Field(default=0)
