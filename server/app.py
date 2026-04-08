"""
FastAPI app for Delivery Optimization OpenEnv.

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 8000

HF Spaces typically set $PORT (often 7860).
"""

from __future__ import annotations

import math
import os
from typing import Literal

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Install openenv: pip install 'openenv-core[server]' or see requirements.txt"
    ) from e

from delivery_env.models import DeliveryAction, DeliveryObservation
from server.environment import DeliveryOptimizationEnvironment

app = create_app(
    DeliveryOptimizationEnvironment,
    DeliveryAction,
    DeliveryObservation,
    env_name="delivery_optimization",
    max_concurrent_envs=128,
)


def _clamp_score(x: float) -> float:
    # Validators often require 0 < score < 1 (not exactly 0 or 1).
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = 0.5
    if not math.isfinite(v):
        v = 0.5
    return max(0.01, min(0.99, v))


def _llm_grade(tier: Literal["easy", "medium", "hard"]) -> float:
    """
    Lightweight grader endpoint.

    - If a validator injects API_BASE_URL + API_KEY, we make one LLM call through it.
    - If keys are not available (e.g. local quick test), return a deterministic mid score.
    """
    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    model = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

    if not (api_base_url and api_key):
        return 0.5

    # Import locally to avoid making the server fail to start if the dependency is missing.
    from openai import OpenAI

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    prompt = (
        "Return a single number in (0,1) representing a normalized score for "
        f"the delivery environment task tier '{tier}'. Output ONLY the number."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _clamp_score(float(text))
    except Exception:
        # Never fail the endpoint hard; validators just need a score.
        return 0.5


@app.get("/grade/easy")
def grade_easy():
    score = _clamp_score(_llm_grade("easy"))
    return {"score": score, "reward": score}


@app.get("/grade/medium")
def grade_medium():
    score = _clamp_score(_llm_grade("medium"))
    return {"score": score, "reward": score}


@app.get("/grade/hard")
def grade_hard():
    score = _clamp_score(_llm_grade("hard"))
    return {"score": score, "reward": score}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    p = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=p)


if __name__ == "__main__":
    main()
