---
title: Delivery OpenEnv
emoji: 🚚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv last-mile delivery dispatch API (FastAPI, WebSocket /ws, /docs).
---

# Delivery Optimization OpenEnv

OpenEnv environment for **last-mile delivery dispatch** (Swiggy / Uber–style): the agent chooses **which order to serve next**, **how to move on a road graph**, and **when to batch** multiple orders under a **vehicle capacity** constraint.

This repo targets the **OpenEnv** HTTP/WebSocket API: `reset()` · `step(action)` · `state()`, with **Pydantic** `Action` / `Observation` / `State`, plus `openenv.yaml`, Dockerfile, and a **reproducible greedy baseline**.

**Author on Hugging Face:** [adhiawesome](https://huggingface.co/adhiawesome) (Adhitya).

**Live Space:** [huggingface.co/spaces/adhiawesome/delivery-openenv](https://huggingface.co/spaces/adhiawesome/delivery-openenv) — after you push this repo (see below), use **`/health`**, **`/docs`**, and WebSocket **`/ws`**.

## Tasks (easy → hard) and grader

| Tier | Idea | Capacity | Orders | Grader |
|------|------|----------|--------|--------|
| **easy** | Single pickup → drop on a small graph | 1 | 1 | `0.0`–`1.0` |
| **medium** | Multiple stops, deadlines | 1 | 4 | `0.0`–`1.0` |
| **hard** | Batching (multi-pickup before drops) | 3 | 6 | `0.0`–`1.0` |

Select a task with **`reset(..., task_tier="easy"|"medium"|"hard")`** (WebSocket) or the same fields in the JSON body for **`POST /reset`**.

The **grader** (in `observation.grader_score` when `done`) combines:

- on-time vs late deliveries  
- travel efficiency vs a **greedy baseline** route for the same seed  
- completion (all orders delivered)

Step rewards add **partial signals**: small time/travel penalties, pickup bonus, delivery bonus (higher if on time), penalties for illegal actions.

## Action space

`DeliveryAction` (JSON keys):

| Field | Meaning |
|--------|---------|
| `opcode` | `0` = move along one edge, `1` = pickup order `order_index`, `2` = deliver order `order_index` |
| `edge_index` | Index into `neighbor_nodes` / `travel_minutes` at the courier’s current node (for `opcode==0`) |
| `order_index` | Index into the order list (for pickup/deliver) |

## Observation space

Rich JSON-friendly lists (good for LLM agents), including:

- `neighbor_nodes`, `travel_minutes`, `legal_*_mask` for feasible moves/pickups/deliveries  
- Per-order `order_pickup_node`, `order_drop_node`, `order_ready_time`, `order_deadline`, flags for picked/delivered  
- `info_message`, and `grader_score` when the episode ends  

See `delivery_env/models.py` for the full schema.

## Setup

Requires **Python 3.10+** (3.11 recommended).

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with **[uv](https://docs.astral.sh/uv/)** (uses `pyproject.toml`):

```bash
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Local server

```bash
export PYTHONPATH=.
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

- Health: `GET /health`  
- OpenAPI: `GET /docs`  
- **Persistent episodes:** use the **WebSocket** client (`/ws`), not stateless `POST /step` alone.

## Client (WebSocket)

```python
from delivery_env.client import DeliveryEnvClient
from delivery_env.models import DeliveryAction

with DeliveryEnvClient(base_url="http://localhost:8000").sync() as env:
    r = env.reset(seed=42, task_tier="medium")
    while not r.done:
        obs = r.observation
        # ... your policy ...
        r = env.step(DeliveryAction(opcode=0, edge_index=0))
    print("grader:", r.observation.grader_score)
```

## Baseline scores (greedy policy)

Deterministic **greedy** rollouts (same policy used for the grader’s baseline travel):

```bash
PYTHONPATH=. python scripts/baseline.py
PYTHONPATH=. python scripts/baseline.py --json  # machine-readable
```

Example run (seeds `0–4, 42, 100, 123`): greedy baseline mean grader **1.0** on easy and medium; hard **~0.97** on that seed set (use `scripts/baseline.py` to reproduce exactly).

## Pre-submission (hackathon checklist)

See **`checklist.txt`** for a line-by-line map to this repo.

**Official-style validator** (health + `/reset`, Docker build, `openenv validate`):

```bash
bash scripts/validate_presubmit.zsh https://adhiawesome-delivery-openenv.hf.space .
```

Requires **`uv.lock`** in the repo (`uv lock`) and **`[project.scripts] server`** in `pyproject.toml` so `openenv validate` passes.

Optional Python helper: `PYTHONPATH=. python scripts/validate_presubmit.py --env-url ...` (add `--skip-docker` if needed).

LLM run: **`inference.py`** at repo root uses **`OpenAI`**, **`API_BASE_URL`**, **`MODEL_NAME`**, **`HF_TOKEN`** (or `API_KEY`), and emits **`[START]` / `[STEP]`  / `[END]`** with the spacing expected by the sample spec.

## Docker (local)

```bash
docker build -t delivery-openenv .
docker run -p 7860:7860 -e PORT=7860 delivery-openenv
```

## Hugging Face Space

**Space:** [adhiawesome/delivery-openenv](https://huggingface.co/spaces/adhiawesome/delivery-openenv)

| What | URL |
|------|-----|
| Space page | [huggingface.co/spaces/adhiawesome/delivery-openenv](https://huggingface.co/spaces/adhiawesome/delivery-openenv) |
| API base (for clients) | `https://adhiawesome-delivery-openenv.hf.space` |
| Health check | `https://adhiawesome-delivery-openenv.hf.space/health` |
| OpenAPI UI | `https://adhiawesome-delivery-openenv.hf.space/docs` |

If the Space still shows **“No application file”**, the Docker app has not been pushed yet. From your **local project root** (`OpenEnvHack`):

```bash
# one-time: token with write access to Spaces — https://huggingface.co/settings/tokens

cd /path/to/OpenEnvHack
git init && git checkout -b main   # skip if you already use git here

git remote add hf https://huggingface.co/spaces/adhiawesome/delivery-openenv.git
# If `hf` exists: git remote set-url hf https://huggingface.co/spaces/adhiawesome/delivery-openenv.git

git add -A
git reset HEAD .venv .venv311 __pycache__ .DS_Store 2>/dev/null || true
# Do not commit Inference.py — use inference.py only
git commit -m "OpenEnv delivery optimization server"

# First push: HF often created an initial commit on the Space → histories differ.
git push hf main || {
  git pull hf main --allow-unrelated-histories --no-edit
  git push hf main
}
```

Use **`main`** unless the Space uses **`master`**. Password = **HF token**, not your login ([Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)).

**If `git push` fails with `ECONNREFUSED` … vscode-git-…sock** or **Invalid username or password**: Cursor/VS Code’s Git helper is not reachable from your terminal. Bypass it and use your token explicitly (create a token with **write** at [HF token settings](https://huggingface.co/settings/tokens)):

```bash
# Replace hf_xxx with your token. User name must be your HF username.
export HF_TOKEN="hf_xxxxxxxx"
git remote set-url hf "https://adhiawesome:${HF_TOKEN}@huggingface.co/spaces/adhiawesome/delivery-openenv.git"
git -c credential.helper= push hf main
```

After a successful push, remove the token from the URL so it is not stored in `.git/config`:

```bash
git remote set-url hf https://huggingface.co/spaces/adhiawesome/delivery-openenv.git
```

Or push once with the full URL:  
`git -c credential.helper= push "https://adhiawesome:${HF_TOKEN}@huggingface.co/spaces/adhiawesome/delivery-openenv.git" main`

**Alternative (fewer merge surprises):** `git clone` the Space repo empty, copy these project files into it, `git add -A`, `git push`.

After a successful push, the Space rebuilds automatically. Confirm with **`/health`** returning `{"status":"healthy"}`.

`Dockerfile` listens on **`${PORT:-7860}`**, which matches [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) expectations.

### Connect from Python (hosted Space)

```python
from delivery_env.client import DeliveryEnvClient
from delivery_env.models import DeliveryAction

BASE = "https://adhiawesome-delivery-openenv.hf.space"

with DeliveryEnvClient(base_url=BASE).sync() as env:
    r = env.reset(seed=0, task_tier="easy")
    # ... same WebSocket loop as local ...
```

If the Space is **private**, use a [read token](https://huggingface.co/settings/tokens) and pass headers per OpenEnv client docs (or keep the Space public for the hackathon demo).

### OpenEnv CLI (optional)

If you use `openenv push`, the repo id matches the Space: **`adhiawesome/delivery-openenv`**.

## Smoke test (WebSocket)

With the server running:

```bash
PYTHONPATH=. python scripts/smoke_ws_client.py --base-url http://127.0.0.1:8000 --seed 0 --task-tier easy
```

## Layout

```
delivery_env/
  models.py       # Pydantic Action, Observation, State
  simulator.py    # Graph, scenarios, physics
  grading.py      # Grader 0–1 + greedy baseline travel
  client.py       # EnvClient (WebSocket)
server/
  environment.py  # OpenEnv Environment
  app.py          # create_app(...)
scripts/
  baseline.py     # Reproducible greedy scores
  smoke_ws_client.py
openenv.yaml
Dockerfile
requirements.txt
```

## References

- [OpenEnv (Meta)](https://github.com/meta-pytorch/OpenEnv)  
- [OpenEnv course](https://github.com/raun/openenv-course)  
- [Environment Hub](https://huggingface.co/collections/openenv/environment-hub)
