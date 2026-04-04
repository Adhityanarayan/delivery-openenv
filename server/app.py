"""
FastAPI app for Delivery Optimization OpenEnv.

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 8000

HF Spaces typically set $PORT (often 7860).
"""

from __future__ import annotations

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


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import os

    import uvicorn

    p = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=p)


if __name__ == "__main__":
    main()
