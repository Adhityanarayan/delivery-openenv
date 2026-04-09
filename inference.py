"""
LLM-driven inference for Delivery Optimization OpenEnv.

Uses the OpenAI-compatible client (Hugging Face Inference Providers / vLLM / OpenAI)
against a running OpenEnv server (local or Hugging Face Space).

Required stdout format (Phase 2 / Odyssey-style):
  One [START]…[END] pair per task id (easy, medium, hard by default).
  [START] task=<task_id> env=<benchmark> model=<model_name>
  [STEP] step=n action=... reward=... done=true|false error=...
  [END] task=<task_id> success=true|false steps=n score=... rewards=r1,r2,...
  Scores must satisfy 0 < score < 1.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from delivery_env.client import DeliveryEnvClient
from delivery_env.grading import grader_score_open_interval
from delivery_env.models import DeliveryAction, DeliveryObservation

# --- Env (OpenEnv HTTP base → WebSocket client) ---
DELIVERY_ENV_BASE_URL = os.getenv(
    "DELIVERY_ENV_BASE_URL",
    os.getenv("ENV_URL", os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")),
)

# --- LLM (OpenAI-compatible API) ---
# Competition / LiteLLM: when both are injected, use them exactly — do not prefer HF_TOKEN,
# or the proxy never sees traffic on the provided API_KEY (Phase 2 fail).
_INJECTED_LLM = "API_BASE_URL" in os.environ and "API_KEY" in os.environ
if _INJECTED_LLM:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY = os.environ["API_KEY"]
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or ""
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# --- Episode ---
# Phase 2: one `python inference.py` run must exercise every openenv.yaml task id (default easy/medium/hard).
BENCHMARK = os.getenv("DELIVERY_BENCHMARK", "delivery_openenv")
SEED_RAW = os.getenv("DELIVERY_SEED", "42")
# Comma-separated task ids; must match `tasks[].id` in openenv.yaml (default: all three).
_DEFAULT_TASK_IDS = "easy,medium,hard"
TASK_IDS = [
    x.strip()
    for x in (os.getenv("DELIVERY_TASK_IDS") or _DEFAULT_TASK_IDS).split(",")
    if x.strip()
]
MAX_STEPS = int(os.getenv("DELIVERY_MAX_STEPS", "250"))
TEMPERATURE = float(os.getenv("DELIVERY_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("DELIVERY_MAX_TOKENS", "256"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("DELIVERY_SUCCESS_THRESHOLD", "0.5"))

# Safety limits so validators (often ~300s) don't time out.
HTTP_TIMEOUT_S = float(os.getenv("DELIVERY_HTTP_TIMEOUT_S", "15"))
TASK_TIME_BUDGET_S = float(os.getenv("DELIVERY_TASK_TIME_BUDGET_S", "80"))


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
    # phase2_guide.md: `[STEP] step=<n> ...` (single space after [STEP])
    err = _error_token(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # phase2_guide.md: `[END] task=<task_id> success=... score=...` (strict 0<score<1; use 3 dp)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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

def _http_base(url: str) -> str:
    return url.rstrip("/")


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # requests is a dependency of openenv-core; import lazily.
    import requests

    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else {}


def _reset_http(base_url: str, *, seed: int, task_tier: str) -> Dict[str, Any]:
    return _post_json(
        f"{_http_base(base_url)}/reset",
        {"seed": seed, "task_tier": task_tier},
    )


def _step_http(base_url: str, action: DeliveryAction) -> Dict[str, Any]:
    return _post_json(
        f"{_http_base(base_url)}/step",
        {
            "action": {
                "opcode": action.opcode,
                "edge_index": action.edge_index,
                "order_index": action.order_index,
            }
        },
    )


def _obs_from_payload(payload: Dict[str, Any], *, fallback_tier: str) -> DeliveryObservation:
    obs_data = payload.get("observation", {}) if isinstance(payload, dict) else {}
    # Note: Observation schema is stable; default defensively.
    return DeliveryObservation(
        done=bool(payload.get("done", False)),
        reward=payload.get("reward"),
        task_tier=str(obs_data.get("task_tier", fallback_tier)),
        courier_node=int(obs_data.get("courier_node", 0)),
        current_time_minutes=float(obs_data.get("current_time_minutes", 0.0)),
        capacity=int(obs_data.get("capacity", 1)),
        load_count=int(obs_data.get("load_count", 0)),
        graph_num_nodes=int(obs_data.get("graph_num_nodes", 1)),
        neighbor_nodes=list(obs_data.get("neighbor_nodes", [])),
        travel_minutes=[float(x) for x in (obs_data.get("travel_minutes", []) or [])],
        order_pickup_node=list(obs_data.get("order_pickup_node", [])),
        order_drop_node=list(obs_data.get("order_drop_node", [])),
        order_ready_time=[float(x) for x in (obs_data.get("order_ready_time", []) or [])],
        order_deadline=[float(x) for x in (obs_data.get("order_deadline", []) or [])],
        order_picked=list(obs_data.get("order_picked", [])),
        order_delivered=list(obs_data.get("order_delivered", [])),
        legal_edge_mask=list(obs_data.get("legal_edge_mask", [])),
        legal_pickup_mask=list(obs_data.get("legal_pickup_mask", [])),
        legal_deliver_mask=list(obs_data.get("legal_deliver_mask", [])),
        info_message=str(obs_data.get("info_message", "")),
        grader_score=obs_data.get("grader_score"),
    )


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


def _llm_proxy_touch(client: OpenAI) -> None:
    """One minimal chat completion so validators that trace LiteLLM API_KEY see a request."""
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "."}],
        max_tokens=1,
        temperature=0.0,
        stream=False,
    )


def _run_one_task(
    *,
    client: OpenAI,
    env: Any,
    task_id: str,
    task_tier: str,
    seed: int,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    final_score = float(grader_score_open_interval(0.0))
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        t0 = time.time()
        # Preferred: persistent WebSocket session (OpenEnv canonical for multi-step episodes).
        # HF Space returned 500s on POST /step in some deployments; WS avoids that.
        try:
            with DeliveryEnvClient(base_url=DELIVERY_ENV_BASE_URL).sync() as ws_env:
                r = ws_env.reset(seed=seed, task_tier=task_tier)
                obs = r.observation

                if obs.done:
                    base = float(obs.grader_score) if obs.grader_score is not None else 0.0
                    final_score = float(
                        grader_score_open_interval(min(max(base, 0.0), 1.0))
                    )
                    success = final_score >= SUCCESS_SCORE_THRESHOLD
                    return

                for step in range(1, MAX_STEPS + 1):
                    if time.time() - t0 > TASK_TIME_BUDGET_S:
                        raise TimeoutError("task_time_budget_exceeded")
                    action, err = get_model_action(client, obs)
                    r = ws_env.step(action)
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
                            final_score = float(
                                grader_score_open_interval(
                                    min(max(float(obs.grader_score), 0.0), 1.0)
                                )
                            )
                        else:
                            avg = sum(rewards) / max(len(rewards), 1)
                            final_score = float(
                                grader_score_open_interval(min(max(avg, 0.0), 1.0))
                            )
                        success = final_score >= SUCCESS_SCORE_THRESHOLD
                        return

                # Max steps reached without done
                if obs.grader_score is not None:
                    final_score = float(
                        grader_score_open_interval(min(max(float(obs.grader_score), 0.0), 1.0))
                    )
                success = final_score >= SUCCESS_SCORE_THRESHOLD
                return
        except Exception as ws_exc:
            print(f"[DEBUG] ws_failed task={task_id} err={ws_exc}", flush=True)

        # Fallback: HTTP (best-effort). Keep strict timeouts.
        reset_payload = _reset_http(DELIVERY_ENV_BASE_URL, seed=seed, task_tier=task_tier)
        obs = _obs_from_payload(reset_payload, fallback_tier=task_tier)

        if obs.done:
            base = float(obs.grader_score) if obs.grader_score is not None else 0.0
            final_score = float(
                grader_score_open_interval(min(max(base, 0.0), 1.0))
            )
            success = final_score >= SUCCESS_SCORE_THRESHOLD
        else:
            for step in range(1, MAX_STEPS + 1):
                if time.time() - t0 > TASK_TIME_BUDGET_S:
                    raise TimeoutError("task_time_budget_exceeded")
                action, err = get_model_action(client, obs)
                step_payload = _step_http(DELIVERY_ENV_BASE_URL, action)
                reward = float(step_payload.get("reward") or 0.0)
                done = bool(step_payload.get("done", False))
                obs = _obs_from_payload(step_payload, fallback_tier=task_tier)
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
                        final_score = float(
                            grader_score_open_interval(
                                min(max(float(obs.grader_score), 0.0), 1.0)
                            )
                        )
                    else:
                        avg = sum(rewards) / max(len(rewards), 1)
                        final_score = float(
                            grader_score_open_interval(min(max(avg, 0.0), 1.0))
                        )
                    success = final_score >= SUCCESS_SCORE_THRESHOLD
                    break
            else:
                if obs.grader_score is not None:
                    final_score = float(
                        grader_score_open_interval(
                            min(max(float(obs.grader_score), 0.0), 1.0)
                        )
                    )
                success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        final_score = float(grader_score_open_interval(0.0))
        success = False
    finally:
        log_end(
            task=task_id,
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


def main() -> None:
    try:
        seed = int(SEED_RAW)
    except ValueError:
        seed = 42

    # Never raise at top-level; always emit END lines.
    try:
        if "API_BASE_URL" in os.environ and "API_KEY" not in os.environ:
            raise RuntimeError("API_KEY is required when API_BASE_URL is provided")

        client = OpenAI(
            base_url=os.environ["API_BASE_URL"] if _INJECTED_LLM else API_BASE_URL,
            api_key=os.environ["API_KEY"] if _INJECTED_LLM else API_KEY,
        )

        if _INJECTED_LLM:
            _llm_proxy_touch(client)
    except Exception as exc:
        print(f"[DEBUG] llm_init_failed: {exc}", flush=True)
        client = OpenAI(base_url="http://127.0.0.1:1", api_key="dummy")

    for task_id in TASK_IDS:
        tier = task_id.lower()
        if tier not in ("easy", "medium", "hard"):
            print(f"[DEBUG] skip unknown task id {task_id!r}", flush=True)
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(
                task=task_id,
                success=False,
                steps=0,
                score=float(grader_score_open_interval(0.0)),
                rewards=[],
            )
            continue
        _run_one_task(
            client=client,
            env=None,
            task_id=task_id,
            task_tier=tier,
            seed=seed,
        )


if __name__ == "__main__":
    main()
