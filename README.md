---
title: HeatShield Urban Heat Response
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - logistics
---

# HeatShield

HeatShield is an OpenEnv environment about urban heat emergency response.
The agent plays the role of an operations lead: open cooling centers, send
alerts, move limited field resources, and decide when to stop.

I built this as a practical benchmark rather than a toy game. The core idea is
simple: useful actions should help a little right away, bad sequencing should
hurt overall mission quality, and the hard task should force tradeoffs.

## What is in here

There are three tasks:

| Task ID | Difficulty | Scenario |
|---|---|---|
| `cooling_corridor_easy` | Easy | Recover a two-district corridor after a local grid wobble. |
| `blackout_triage_medium` | Medium | Balance a clinic blackout with waterfront and transit pressure. |
| `cascade_hard` | Hard | Handle four districts during a broader heatwave cascade. |

Each task has:
- a deterministic scenario definition
- a fixed step budget
- a grader that produces a score in `[0.0, 1.0]`

The environment follows the standard OpenEnv shape:
- typed action model
- typed observation model
- typed state model
- `reset()`
- `step()`
- `state()`

## Action model

The environment exposes one action type: `HeatShieldAction`.

Fields:
- `action_type`: `inspect`, `activate_center`, `dispatch_resource`, `broadcast_alert`, `finalize`
- `target_id`: district id, facility id, or `mission` for finalization
- `resource_type`: `cooling_bus`, `water_truck`, `medical_team`, `generator`
- `quantity`: units to dispatch
- `message`: alert copy or final summary

## Observation and state

`HeatShieldObservation` returns the public situation picture plus the immediate
result of the last action. It includes:
- task metadata and step budget
- district snapshots
- facility snapshots
- remaining resources
- action history and inspection log
- score breakdown
- recommendation hints
- transport metadata used by the baseline policy

`HeatShieldState` is the more compact internal progress view:
- active centers
- inspected targets
- resources already committed
- current score
- invalid action count
- finalized/not finalized

## Reward and grading

The mission score is normalized to `[0.0, 1.0]`. It combines:
- district relief coverage
- critical facility activation
- alert coverage
- priority intel coverage
- penalties for invalid actions

The per-step reward is the non-negative improvement in that mission score. That
keeps rewards bounded for validators while still giving dense feedback.

Invalid actions do not produce negative step rewards, but they still reduce the
underlying mission score through the grader penalty.

## Repository layout

```text
.
|-- heatshield_env/
|   |-- __init__.py
|   |-- client.py
|   |-- graders.py
|   |-- models.py
|   `-- scenario_data.py
|-- server/
|   |-- app.py
|   |-- Dockerfile
|   |-- heatshield_environment.py
|   `-- requirements.txt
|-- scripts/
|   `-- validate-submission.sh
|-- tests/
|   `-- test_heatshield_env.py
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
`-- README.md
```

## Local setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Run the server:

```bash
python -m server.app
```

Run tests:

```bash
pytest
```

Run OpenEnv validation:

```bash
openenv validate .
```

If you want to reset a client directly into a specific task while debugging:

```python
result = await env.reset(task_id="cascade_hard")
```

For plain HTTP debugging:
- call `POST /reset` first
- keep the returned `X-OpenEnv-Session-Id` header
- send that header on `/step` and `/state`
- `/step` accepts either raw `HeatShieldAction` JSON or `{"action": {...}}`

## Docker

Build locally with:

```bash
docker build -t heatshield_env-env:latest -f server/Dockerfile .
```

## Baseline inference

The submission script is `inference.py`.

It supports the required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:
- `LOCAL_IMAGE_NAME`
- `USE_LLM=true`

`.env.example` is included as a template.

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_xxx"
export LOCAL_IMAGE_NAME="heatshield_env-env:latest"
python inference.py
```

By default the baseline uses deterministic per-task playbooks so the run is
reproducible even without an API token. If `USE_LLM=true`, the script can ask
the model for actions using the OpenAI client.

Expected stdout format:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`

## Deployment

This repo is set up to deploy as a Docker Space:

```bash
openenv push
```

Useful endpoints after deployment:
- `/web`
- `/docs`
- `/health`
- `/ws`

## Notes

A few implementation choices are intentional:
- the baseline is deterministic first, model-assisted second
- HTTP routes are stateful for easier debugging, even though the main client path is WebSocket
- the hard task is tuned to be beatable, but not by random play

If you are reading this for the hackathon submission, the fastest sanity check is:
- run `pytest`
- run `openenv validate .`
- run `python inference.py` in an environment where Docker is available
