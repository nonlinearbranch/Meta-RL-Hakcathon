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

# HeatShield: Urban Heat Response Command

HeatShield is a real-world OpenEnv environment where an AI agent acts as a city heat-response commander during a cascading heatwave emergency. The agent must inspect operational intel, activate cooling centers, dispatch scarce field resources, send direct alerts, and finalize a response plan under a tight step budget.

The environment is designed for agentic RL rather than a toy benchmark:
- The state is multi-district and resource-constrained.
- Rewards provide partial progress signals after every useful action.
- Hidden intel changes which actions are actually high leverage.
- Three tasks scale from manageable corridor activation to city-wide cascading outages.

## Tasks

| Task ID | Difficulty | Goal |
|---|---|---|
| `cooling_corridor_easy` | Easy | Recover a two-district cooling corridor after a localized grid wobble. |
| `blackout_triage_medium` | Medium | Balance a clinic corridor blackout with waterfront and transit pressure. |
| `cascade_hard` | Hard | Coordinate four districts during simultaneous blackout, clinic, and commuter failures. |

Each task includes a deterministic grader that returns a normalized score in `[0.0, 1.0]`.

## Action Space

The environment exposes one typed action model: `HeatShieldAction`.

Fields:
- `action_type`: `inspect`, `activate_center`, `dispatch_resource`, `broadcast_alert`, `finalize`
- `target_id`: district id, facility id, or `mission` for finalization
- `resource_type`: `cooling_bus`, `water_truck`, `medical_team`, `generator`
- `quantity`: integer units for dispatch actions
- `message`: alert text or final summary

## Observation Space

`HeatShieldObservation` includes:
- mission metadata: task id, title, difficulty, mission brief, turns remaining
- district snapshots: population, vulnerability, unmet relief, outage flag, alert requirement
- facility snapshots: activation status, capacity, generator requirement
- resource inventory: remaining units and relief impact per unit
- inspection log and action history
- last event string
- recommended next actions
- grader summary and detailed score breakdown

## Reward Function

The reward is the non-negative improvement in normalized mission score after each action.

Mission score components:
- `0.55 * relief_score`
- `0.20 * facility_score`
- `0.15 * alert_score`
- `0.10 * intel_score`
- minus invalid-action penalties

This gives useful dense feedback while keeping transport-facing rewards in `[0.0, 1.0]`:
- inspecting priority intel helps
- activating the right centers helps more
- dispatching generators before blocked facilities matters
- targeted alerts close coverage gaps
- invalid or impossible actions reduce the underlying mission score and future upside

## Project Layout

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

## Local Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Run the server locally:

```bash
python -m server.app
```

Run the smoke tests:

```bash
pytest
```

Validate the repo structure:

```bash
openenv validate .
```

Select a task explicitly during local debugging:

```python
result = await env.reset(task_id="cascade_hard")
```

For plain HTTP debugging, call `POST /reset` with an optional `task_id` and
reuse the returned `X-OpenEnv-Session-Id` header on `/step` and `/state`.
`POST /step` accepts either a raw `HeatShieldAction` JSON body or
`{"action": {...}}`.

## Docker Build

```bash
docker build -t heatshield_env-env:latest -f server/Dockerfile .
```

## Baseline Inference

The baseline script is intentionally named `inference.py` and uses the OpenAI client with the required environment variables.

Required variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:
- `LOCAL_IMAGE_NAME` if you want to override the local Docker image name
- `USE_LLM=true` if you want to let the baseline query the model instead of using the deterministic task playbook

A template is included in `.env.example`.

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_xxx"
export LOCAL_IMAGE_NAME="heatshield_env-env:latest"
python inference.py
```

By default the baseline uses a deterministic task playbook so scores are reproducible even without an API token. Set `USE_LLM=true` to enable model-guided actions.

Structured stdout format:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`

## Hugging Face Spaces Deployment

You can deploy as a Docker Space:

```bash
openenv push
```

The resulting Space exposes:
- `/web` for the interactive UI
- `/docs` for the API schema
- `/health` for container health checks
- `/ws` for persistent OpenEnv sessions

## Why This Is Hackathon-Ready

- Real-world domain: heatwave emergency operations, not a game.
- Full OpenEnv spec: typed models plus `reset()`, `step()`, and `state()`.
- Three graded tasks with deterministic scoring.
- Baseline inference script with required structured logs.
- Dockerfile, README, local validator helper, and test coverage.
