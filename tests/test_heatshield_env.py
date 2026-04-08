"""Smoke tests for HeatShield."""

from fastapi.testclient import TestClient
from pydantic import ValidationError

from heatshield_env.models import HeatShieldAction
from heatshield_env.scenario_data import get_task, get_task_ids
from inference import TASK_PLAYBOOKS, extract_json, heuristic_action, validate_model_action
from server.app import HTTP_SESSION_HEADER, app
from server.heatshield_environment import HeatShieldEnvironment


def test_reset_returns_valid_observation():
    env = HeatShieldEnvironment(task_id="cooling_corridor_easy")
    observation = env.reset()

    assert observation.task_id == "cooling_corridor_easy"
    assert observation.score_breakdown.total_score == 0.0
    assert observation.turns_remaining == observation.step_limit
    assert len(observation.district_snapshots) >= 2


def test_easy_task_sequence_scores_well():
    env = HeatShieldEnvironment(task_id="cooling_corridor_easy")
    env.reset()

    actions = [
        HeatShieldAction(action_type="inspect", target_id="rivergate"),
        HeatShieldAction(action_type="inspect", target_id="rivergate_library"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="rivergate",
            resource_type="generator",
            quantity=1,
        ),
        HeatShieldAction(action_type="activate_center", target_id="rivergate_library"),
        HeatShieldAction(action_type="activate_center", target_id="market_school_gym"),
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="rivergate",
            resource_type="cooling_bus",
            quantity=1,
        ),
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="rivergate",
            message="Rivergate heat alert: use cooling centers immediately and carry water.",
        ),
        HeatShieldAction(
            action_type="finalize",
            target_id="mission",
            message="Core cooling sites are active and Rivergate has been alerted.",
        ),
    ]

    observation = None
    for action in actions:
        observation = env.step(action)

    assert observation is not None
    assert observation.done is True
    assert observation.score_breakdown.total_score > 0.75


def test_baseline_playbooks_clear_task_thresholds():
    for task_id in get_task_ids():
        env = HeatShieldEnvironment(task_id=task_id)
        observation = env.reset()

        for action in TASK_PLAYBOOKS[task_id]:
            observation = env.step(action)

        assert observation.done is True
        assert observation.score_breakdown.total_score >= get_task(task_id).success_threshold


def test_reward_stays_non_negative_after_invalid_action():
    env = HeatShieldEnvironment(task_id="blackout_triage_medium")
    env.reset()

    positive = env.step(HeatShieldAction(action_type="inspect", target_id="clinic_belt"))
    invalid = env.step(
        HeatShieldAction(action_type="activate_center", target_id="bad_facility")
    )

    assert positive.reward is not None
    assert positive.reward >= 0.0
    assert invalid.reward == 0.0
    assert invalid.score_breakdown.total_score < positive.score_breakdown.total_score
    assert invalid.metadata["last_action_error"] == "unknown_facility:bad_facility"


def test_extract_json_ignores_wrapped_noise_and_returns_first_valid_object():
    payload = extract_json(
        'Plan sketch {"example": true not_json} final action: '
        '{"action_type":"inspect","target_id":"rivergate","resource_type":null,"quantity":1,"message":""}'
        "\nThen extra notes."
    )

    assert payload == {
        "action_type": "inspect",
        "target_id": "rivergate",
        "resource_type": None,
        "quantity": 1,
        "message": "",
    }


def test_validate_model_action_rejects_resource_requests_above_inventory():
    env = HeatShieldEnvironment(task_id="cooling_corridor_easy")
    observation = env.reset()

    action = validate_model_action(
        {
            "action_type": "dispatch_resource",
            "target_id": "rivergate",
            "resource_type": "cooling_bus",
            "quantity": 2,
            "message": "",
        },
        observation,
    )

    assert action is None


def test_repeat_inspection_is_reported_as_invalid():
    env = HeatShieldEnvironment(task_id="cooling_corridor_easy")
    env.reset()

    first = env.step(HeatShieldAction(action_type="inspect", target_id="rivergate"))
    second = env.step(HeatShieldAction(action_type="inspect", target_id="rivergate"))

    assert first.metadata["last_action_error"] is None
    assert second.reward == 0.0
    assert second.metadata["last_action_error"] == "inspection_already_completed:rivergate"


def test_duplicate_alert_does_not_silently_overwrite_score():
    env = HeatShieldEnvironment(task_id="cooling_corridor_easy")
    env.reset()

    first = env.step(
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="rivergate",
            message="Rivergate heat alert: use cooling and water now.",
        )
    )
    second = env.step(
        HeatShieldAction(
            action_type="broadcast_alert",
            target_id="rivergate",
            message="Bad replacement alert without useful details.",
        )
    )

    assert first.metadata["last_action_error"] is None
    assert second.metadata["last_action_error"] == "alert_already_sent:rivergate"
    assert second.score_breakdown.total_score < first.score_breakdown.total_score


def test_district_power_outage_comes_from_scenario_data():
    env = HeatShieldEnvironment(task_id="cascade_hard")
    observation = env.reset()
    districts = {district.district_id: district for district in observation.district_snapshots}

    assert districts["inland_towers"].power_outage is True
    assert districts["creekside"].power_outage is True
    assert districts["rail_junction"].power_outage is False
    assert districts["harbor_north"].power_outage is False


def test_quantity_schema_rejects_zero_dispatch_units():
    try:
        HeatShieldAction(
            action_type="dispatch_resource",
            target_id="rivergate",
            resource_type="water_truck",
            quantity=0,
        )
    except ValidationError:
        return

    raise AssertionError("quantity=0 should fail validation")


def test_get_task_raises_helpful_error_for_unknown_id():
    try:
        get_task("not_a_real_task")
    except KeyError as exc:
        message = str(exc)
        assert "not_a_real_task" in message
        assert "cooling_corridor_easy" in message
        return

    raise AssertionError("Expected get_task() to raise KeyError for an unknown task id")


def test_playbook_selector_uses_turns_remaining_not_truncated_history():
    env = HeatShieldEnvironment(task_id="cascade_hard")
    observation = env.reset()

    for action in TASK_PLAYBOOKS["cascade_hard"][:9]:
        observation = env.step(action)

    next_action = heuristic_action(observation)
    assert next_action.action_type == "inspect"
    assert next_action.target_id == "inland_towers"


def test_websocket_transport_preserves_metadata():
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "data": {"task_id": "cascade_hard"}})
        payload = websocket.receive_json()

    observation = payload["data"]["observation"]
    assert observation["task_id"] == "cascade_hard"
    assert "metadata" in observation
    assert observation["metadata"]["priority_intel_targets"] == [
        "inland_towers",
        "inland_civic_hub",
        "creekside_field_clinic",
        "rail_junction",
    ]
    assert "available_targets" in observation["metadata"]


def test_http_routes_preserve_session_state_and_schema():
    client = TestClient(app)

    reset_response = client.post("/reset", json={"task_id": "cooling_corridor_easy"})
    assert reset_response.status_code == 200
    session_id = reset_response.headers[HTTP_SESSION_HEADER]
    reset_observation = reset_response.json()["observation"]
    assert reset_observation["metadata"]["priority_intel_targets"] == [
        "rivergate",
        "rivergate_library",
        "medical_team",
    ]

    raw_step = client.post(
        "/step",
        headers={HTTP_SESSION_HEADER: session_id},
        json={"action_type": "inspect", "target_id": "rivergate"},
    )
    assert raw_step.status_code == 200

    wrapped_step = client.post(
        "/step",
        headers={HTTP_SESSION_HEADER: session_id},
        json={"action": {"action_type": "inspect", "target_id": "rivergate_library"}},
    )
    assert wrapped_step.status_code == 200

    first_observation = raw_step.json()["observation"]
    second_observation = wrapped_step.json()["observation"]
    assert first_observation["turns_remaining"] == 7
    assert second_observation["turns_remaining"] == 6
    assert second_observation["action_history"] == [
        "inspect(target_id=rivergate)",
        "inspect(target_id=rivergate_library)",
    ]

    state_response = client.get("/state", headers={HTTP_SESSION_HEADER: session_id})
    assert state_response.status_code == 200
    assert state_response.json()["step_count"] == 2
    assert state_response.json()["task_id"] == "cooling_corridor_easy"

    schema_response = client.get("/schema")
    assert schema_response.status_code == 200
    state_schema = schema_response.json()["state"]
    assert "task_id" in state_schema["properties"]
    assert "current_score" in state_schema["properties"]
