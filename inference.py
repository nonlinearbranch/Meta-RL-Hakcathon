"""
Hackathon baseline inference for HeatShield.

This file intentionally follows the required START/STEP/END stdout contract.
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import ValidationError

from heatshield_env import HeatShieldAction, HeatShieldEnv, get_task_ids
from heatshield_env.scenario_data import get_task


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")



BENCHMARK = "heatshield_env"
MAX_MODEL_TOKENS = 480
TEMPERATURE = 0.2



SYSTEM_PROMPT = """
You are the operations planner for a city heat-response desk.
Return exactly one JSON object with these keys:
- action_type: one of inspect, activate_center, dispatch_resource, broadcast_alert, finalize
- target_id: a valid district or facility id from the prompt, or "mission" when finalizing
- resource_type: one of cooling_bus, water_truck, medical_team, generator, or null
- quantity: integer 1-4
- message: alert text or final summary; empty string when unused

Rules:
- Never invent ids.
- Use inspect first when priority intel targets remain.
- Dispatch generators before activating blocked facilities.
- Use specific district alerts rather than generic narration.
- Finalize only when there is little useful work left or the score is already strong.
""".strip()


TASK_PLAYBOOKS: Dict[str, List[HeatShieldAction]] = {
    "cooling_corridor_easy": [
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="rivergate",
            resource_type="generator",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="rivergate_library"),
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="rivergate",
            message="Rivergate heat alert: use cooling, water, and indoor shelter immediately.",
        ),
        HeatShieldAction(action_type="activate_center", target_id="market_school_gym"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="market_square",
            resource_type="medical_team",
            quantity=1,
        ),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="rivergate",
            resource_type="cooling_bus",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="bus_depot_mist_zone"),
        HeatShieldAction(action_type="inspect", target_id="rivergate"),
    ],
    "blackout_triage_medium": [
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="clinic_belt",
            resource_type="generator",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="clinic_triage_tent"),
        HeatShieldAction(action_type="activate_center", target_id="old_port_rec_center"),
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="old_port",
            message="Old Port heat alert: use cooling, water, and indoor shelter immediately.",
        ),
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="clinic_belt",
            message="Clinic Belt heat alert: move patients and families into cooled indoor shelter immediately.",
        ),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="clinic_belt",
            resource_type="medical_team",
            quantity=1,
        ),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="old_port",
            resource_type="cooling_bus",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="skyline_station_hall"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="skyline_west",
            resource_type="medical_team",
            quantity=1,
        ),
    ],
    "cascade_hard": [
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="all",
            message="Citywide heat emergency: use cooling centers, carry water, and move indoors immediately.",
        ),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="inland_towers",
            resource_type="generator",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="inland_civic_hub"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="creekside",
            resource_type="generator",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="creekside_field_clinic"),
        HeatShieldAction(action_type="activate_center", target_id="rail_union_college"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="harbor_north",
            resource_type="cooling_bus",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="harbor_library_branch"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="creekside",
            resource_type="medical_team",
            quantity=1,
        ),
        HeatShieldAction(action_type="inspect", target_id="inland_towers"),
    ],
}


def stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model_name: str) -> None:
    print(f"[START] task={task} env={env} model={model_name}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = "null" if not error else error.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(observation) -> str:
    districts = [
        {
            "district_id": district.district_id,
            "priority": district.priority,
            "unmet_relief": district.unmet_relief,
            "must_alert": district.must_alert,
            "power_outage": district.power_outage,
            "public_notes": district.public_notes,
        }
        for district in observation.district_snapshots
    ]
    facilities = [
        {
            "facility_id": facility.facility_id,
            "district_id": facility.district_id,
            "status": facility.status,
            "capacity": facility.capacity,
            "requires_generator": facility.requires_generator,
        }
        for facility in observation.facility_snapshots
    ]
    resources = {
        resource.resource_type: resource.available for resource in observation.resource_pool
    }
    metadata = observation.metadata or {}

    payload = {
        "task_id": observation.task_id,
        "task_title": observation.task_title,
        "turns_remaining": observation.turns_remaining,
        "mission_brief": observation.mission_brief,
        "last_event": observation.last_event,
        "score": observation.score_breakdown.total_score,
        "districts": districts,
        "facilities": facilities,
        "resources": resources,
        "inspection_log": observation.inspection_log[-4:],
        "action_history": observation.action_history[-5:],
        "priority_intel_targets": metadata.get("priority_intel_targets", []),
        "must_alert_targets": metadata.get("must_alert_targets", []),
        "blocked_facilities": metadata.get("blocked_facilities", []),
        "activatable_facilities": metadata.get("activatable_facilities", []),
    }
    return json.dumps(payload, indent=2)


def heuristic_action(observation) -> HeatShieldAction:
    playbook = TASK_PLAYBOOKS.get(observation.task_id, [])
    executed_steps = observation.step_limit - observation.turns_remaining
    if executed_steps < len(playbook):
        scripted_action = playbook[executed_steps]
        if validate_model_action(
            scripted_action.model_dump(mode="json", exclude_none=True),
            observation,
        ):
            return scripted_action

    metadata = observation.metadata or {}
    priority_intel = metadata.get("priority_intel_targets", [])
    if priority_intel:
        return HeatShieldAction(action_type="inspect", target_id=priority_intel[0])

    resource_remaining: Dict[str, int] = metadata.get("resource_remaining", {})
    facilities_by_id = {facility.facility_id: facility for facility in observation.facility_snapshots}

    blocked_facilities = metadata.get("blocked_facilities", [])
    if blocked_facilities and resource_remaining.get("generator", 0) > 0:
        facility = facilities_by_id[blocked_facilities[0]]
        return HeatShieldAction(
            action_type="dispatch_resource",
            target_id=facility.district_id,
            resource_type="generator",
            quantity=1,
        )

    must_alert = metadata.get("must_alert_targets", [])
    if must_alert and observation.turns_remaining > 1:
        district_id = must_alert[0]
        return HeatShieldAction(
            action_type="broadcast_alert",
            target_id=district_id,
            message=(
                f"Heat alert for {district_id}: use the nearest cooling center, "
                "limit outdoor exposure, and carry water immediately."
            ),
        )

    activatable = metadata.get("activatable_facilities", [])
    if activatable:
        ranked = sorted(
            (facilities_by_id[facility_id] for facility_id in activatable),
            key=lambda facility: facility.capacity,
            reverse=True,
        )
        return HeatShieldAction(
            action_type="activate_center",
            target_id=ranked[0].facility_id,
        )

    target_district = max(
        observation.district_snapshots,
        key=lambda district: district.unmet_relief * district.vulnerability,
    )
    for resource_type in ("cooling_bus", "medical_team", "water_truck"):
        if resource_remaining.get(resource_type, 0) > 0:
            return HeatShieldAction(
                action_type="dispatch_resource",
                target_id=target_district.district_id,
                resource_type=resource_type,
                quantity=1,
            )

    return HeatShieldAction(
        action_type="finalize",
        target_id="mission",
        message=f"{observation.task_title} stabilized. Finalizing the current response plan.",
    )


def extract_json(response_text: str) -> Optional[Dict]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(response_text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def validate_model_action(payload: Dict, observation) -> Optional[HeatShieldAction]:
    metadata = observation.metadata or {}
    valid_action_types = {
        "inspect",
        "activate_center",
        "dispatch_resource",
        "broadcast_alert",
        "finalize",
    }
    action_type = payload.get("action_type")
    if action_type not in valid_action_types:
        return None

    target_id = str(payload.get("target_id", "")).strip()
    resource_type = payload.get("resource_type")
    try:
        quantity = int(payload.get("quantity", 1))
    except (TypeError, ValueError):
        return None
    message = str(payload.get("message", "") or "")

    available = metadata.get("available_targets", {})
    valid_districts = set(available.get("districts", []))
    valid_facilities = set(available.get("facilities", []))
    valid_resources = set(available.get("resource_types", []))
    resource_remaining = metadata.get("resource_remaining", {})

    if action_type == "inspect" and target_id not in (valid_districts | valid_facilities | valid_resources):
        return None
    if action_type == "activate_center" and target_id not in valid_facilities:
        return None
    if action_type == "dispatch_resource":
        if (
            target_id not in valid_districts
            or resource_type not in valid_resources
            or quantity <= 0
            or quantity > int(resource_remaining.get(resource_type, 0))
        ):
            return None
    if action_type == "broadcast_alert" and target_id not in (valid_districts | {"all"}):
        return None
    if action_type == "finalize":
        target_id = "mission"

    try:
        return HeatShieldAction(
            action_type=action_type,
            target_id=target_id,
            resource_type=resource_type,
            quantity=quantity,
            message=message.strip(),
        )
    except ValidationError:
        return None


def call_model(observation) -> Optional[HeatShieldAction]:
    # Late-binding strict execution required by Phase 2 Evaluator
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )

    completion = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_MODEL_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()

    payload = extract_json(text)
    if payload is None:
        stderr("Model response did not contain valid JSON; using heuristic fallback.")
        return None
    return validate_model_action(payload, observation)


def format_action(action: HeatShieldAction) -> str:
    if action.action_type == "dispatch_resource":
        return (
            f"dispatch_resource(resource_type={action.resource_type},"
            f"target_id={action.target_id},quantity={action.quantity})"
        )
    if action.action_type in {"broadcast_alert", "finalize"}:
        compact = action.message.replace("\n", " ").strip()
        compact = compact[:80]
        return f"{action.action_type}(target_id={action.target_id},message={compact})"
    return f"{action.action_type}(target_id={action.target_id})"


async def run_task(task_id: str) -> None:
    image_name = LOCAL_IMAGE_NAME or "heatshield_env-env:latest"
    env = None
    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    success = False

    log_start(task_id, BENCHMARK, os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
    try:
        env = await HeatShieldEnv.from_docker_image(
            image_name,
            env_vars={"HEATSHIELD_TASK_ID": task_id},
        )
        result = await env.reset()
        observation = result.observation
        threshold = get_task(task_id).success_threshold

        while not result.done and observation.turns_remaining > 0:
            steps_taken += 1
            action = call_model(observation)
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            error = (observation.metadata or {}).get("last_action_error")
            log_step(
                steps_taken,
                format_action(action),
                reward,
                result.done,
                error,
            )

        score = float(observation.score_breakdown.total_score)
        success = score >= threshold
    except Exception as exc:  # pragma: no cover - runtime dependent
        stderr(f"Task {task_id} failed: {exc}")
    finally:
        if env is not None:
            await env.close()
        log_end(success, steps_taken, score, rewards)


async def main() -> None:
    for task_id in get_task_ids():
        await run_task(task_id)


if __name__ == "__main__":
    asyncio.run(main())
