"""
LLM-driven inference for Delivery Optimization OpenEnv.

Uses the OpenAI-compatible client (Hugging Face Inference Providers / vLLM / OpenAI)
against a running OpenEnv server (local or Hugging Face Space).

Required stdout format (hackathon-style):
  [START] task=... env=... model=...
  [STEP]  step=n action=... reward=... done=true|false error=...
  [END]   success=true|false steps=n score=... rewards=r1,r2,...
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from delivery_env.client import DeliveryEnvClient
from delivery_env.models import DeliveryAction, DeliveryObservation

# --- Env (OpenEnv HTTP base → WebSocket client) ---
DELIVERY_ENV_BASE_URL = os.getenv(
    "DELIVERY_ENV_BASE_URL",
    os.getenv("ENV_URL", os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")),
)

# --- LLM (OpenAI-compatible API) ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Episode ---
TASK_NAME = os.getenv("DELIVERY_TASK_NAME", "delivery-optimization")
BENCHMARK = os.getenv("DELIVERY_BENCHMARK", "delivery_openenv")
TASK_TIER = os.getenv("DELIVERY_TASK_TIER", "easy")
SEED_RAW = os.getenv("DELIVERY_SEED", "42")
MAX_STEPS = int(os.getenv("DELIVERY_MAX_STEPS", "250"))
TEMPERATURE = float(os.getenv("DELIVERY_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("DELIVERY_MAX_TOKENS", "256"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("DELIVERY_SUCCESS_THRESHOLD", "0.5"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _error_token(error: Optional[str]) -> str:
    """Single-token error for strict [STEP] line parsing (no raw spaces/newlines)."""
    if not error:
        return "null"
    cleaned = " ".join(error.replace("\n", " ").split())
    cleaned = cleaned.replace(" ", "_")
    return cleaned


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    # Two spaces after [STEP] per hackathon sample format
    err = _error_token(error)
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Three spaces after [END] per hackathon sample; score two decimals like sample
    print(
        f"[END]   success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def observation_payload(obs: DeliveryObservation) -> Dict[str, Any]:
    """Compact dict for the model (no huge duplication)."""
    return {
        "task_tier": obs.task_tier,
        "courier_node": obs.courier_node,
        "current_time_minutes": obs.current_time_minutes,
        "capacity": obs.capacity,
        "load_count": obs.load_count,
        "neighbor_nodes": obs.neighbor_nodes,
        "travel_minutes": obs.travel_minutes,
        "legal_edge_mask": obs.legal_edge_mask,
        "legal_pickup_mask": obs.legal_pickup_mask,
        "legal_deliver_mask": obs.legal_deliver_mask,
        "orders": [
            {
                "i": i,
                "pickup": obs.order_pickup_node[i],
                "drop": obs.order_drop_node[i],
                "ready": obs.order_ready_time[i],
                "deadline": obs.order_deadline[i],
                "picked": obs.order_picked[i],
                "delivered": obs.order_delivered[i],
            }
            for i in range(len(obs.order_pickup_node))
        ],
        "info_message": obs.info_message,
        "done": obs.done,
    }


SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a delivery courier on a graph.

    Reply with ONE JSON object only, no markdown fences, no extra text:
    {"opcode": <0|1|2>, "edge_index": <int>, "order_index": <int>}

    Semantics:
    - opcode 0 = MOVE one step along an edge from the current node.
      edge_index indexes neighbor_nodes / travel_minutes (must be legal per legal_edge_mask).
    - opcode 1 = PICKUP order order_index (must be legal per legal_pickup_mask).
    - opcode 2 = DELIVER order order_index (must be legal per legal_deliver_mask).

    Prefer delivering when legal, then picking up when legal and capacity allows,
    otherwise move toward serving the next order. Respect deadlines.
    """
).strip()


def build_user_prompt(obs: DeliveryObservation) -> str:
    return (
        "Current observation (JSON):\n"
        + json.dumps(observation_payload(obs), indent=2)
        + "\nChoose the next action as a single JSON object."
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    out = json.loads(s[start : i + 1])
                    return out if isinstance(out, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def dict_to_action(d: Dict[str, Any]) -> DeliveryAction:
    op = int(d.get("opcode", d.get("op", 0)))
    ei = int(d.get("edge_index", d.get("edge", 0)))
    oi = int(d.get("order_index", d.get("order", 0)))
    return DeliveryAction(opcode=op, edge_index=ei, order_index=oi)


def is_legal(obs: DeliveryObservation, action: DeliveryAction) -> bool:
    if action.opcode == 2:
        m = obs.legal_deliver_mask
        return (
            0 <= action.order_index < len(m)
            and bool(m[action.order_index])
        )
    if action.opcode == 1:
        m = obs.legal_pickup_mask
        return (
            0 <= action.order_index < len(m)
            and bool(m[action.order_index])
        )
    if action.opcode == 0:
        m = obs.legal_edge_mask
        return 0 <= action.edge_index < len(m) and bool(m[action.edge_index])
    return False


def fallback_action(obs: DeliveryObservation) -> DeliveryAction:
    for i, ok in enumerate(obs.legal_deliver_mask):
        if ok:
            return DeliveryAction(opcode=2, order_index=i, edge_index=0)
    for i, ok in enumerate(obs.legal_pickup_mask):
        if ok:
            return DeliveryAction(opcode=1, order_index=i, edge_index=0)
    for e, ok in enumerate(obs.legal_edge_mask):
        if ok:
            return DeliveryAction(opcode=0, edge_index=e, order_index=0)
    return DeliveryAction(opcode=0, edge_index=0, order_index=0)


def get_model_action(
    client: OpenAI,
    obs: DeliveryObservation,
) -> Tuple[DeliveryAction, Optional[str]]:
    """Returns (action, parse_or_validation_error_message)."""
    user = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        return fallback_action(obs), f"llm_request_failed:{exc}"

    data = extract_json_object(raw)
    if data is None:
        return fallback_action(obs), f"json_parse_failed:{raw[:200]!r}"

    try:
        action = dict_to_action(data)
    except (TypeError, ValueError) as exc:
        return fallback_action(obs), f"invalid_fields:{exc}"

    if not is_legal(obs, action):
        return fallback_action(obs), f"illegal_action:{data}"
    return action, None


def action_log_string(action: DeliveryAction) -> str:
    return json.dumps(
        {
            "opcode": action.opcode,
            "edge_index": action.edge_index,
            "order_index": action.order_index,
        },
        separators=(",", ":"),
    )


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def main() -> None:
    try:
        seed = int(SEED_RAW)
    except ValueError:
        seed = 42

    # If the platform injected API_BASE_URL, it expects ALL LLM traffic to go there.
    # Fail fast if API_KEY is missing in that case to avoid silently bypassing the proxy.
    if os.getenv("API_BASE_URL") and not os.getenv("API_KEY"):
        raise RuntimeError("API_KEY is required when API_BASE_URL is provided")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        with DeliveryEnvClient(base_url=DELIVERY_ENV_BASE_URL).sync() as env:
            r = env.reset(seed=seed, task_tier=TASK_TIER)
            obs = r.observation

            if obs.done:
                final_score = _clamp01(
                    float(obs.grader_score) if obs.grader_score is not None else 0.0
                )
                success = final_score >= SUCCESS_SCORE_THRESHOLD
            else:
                for step in range(1, MAX_STEPS + 1):
                    action, err = get_model_action(client, obs)
                    r = env.step(action)
                    obs = r.observation
                    reward = float(r.reward or 0.0)
                    done = bool(r.done)
                    rewards.append(reward)
                    steps_taken = step

                    log_step(
                        step=step,
                        action=action_log_string(action),
                        reward=reward,
                        done=done,
                        error=err,
                    )

                    if done:
                        if obs.grader_score is not None:
                            final_score = _clamp01(float(obs.grader_score))
                        else:
                            avg = sum(rewards) / max(len(rewards), 1)
                            final_score = _clamp01(avg)
                        success = final_score >= SUCCESS_SCORE_THRESHOLD
                        break
                else:
                    if obs.grader_score is not None:
                        final_score = _clamp01(float(obs.grader_score))
                    success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] episode_failed: {exc}", flush=True)
        final_score = 0.0
        success = False
    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


if __name__ == "__main__":
    main()
