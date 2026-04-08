"""HeatShield environment implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from heatshield_env.graders import GraderReport, grade_plan
from heatshield_env.models import (
    DistrictSnapshot,
    FacilitySnapshot,
    HeatShieldAction,
    HeatShieldObservation,
    HeatShieldState,
    ResourceSnapshot,
    ScoreBreakdown,
)
from heatshield_env.scenario_data import (
    RESOURCE_IMPACTS,
    DistrictScenario,
    FacilityScenario,
    TaskScenario,
    get_task,
    get_task_ids,
    get_task_summaries,
)


BENCHMARK_NAME = "heatshield_env"


@dataclass
class EpisodeRuntime:
    """Internal mutable runtime state."""

    scenario: TaskScenario
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    step_count: int = 0
    active_facilities: Set[str] = field(default_factory=set)
    inspected_targets: Set[str] = field(default_factory=set)
    resource_remaining: Dict[str, int] = field(default_factory=dict)
    resource_dispatches: Dict[str, Dict[str, int]] = field(default_factory=dict)
    alert_messages: Dict[str, str] = field(default_factory=dict)
    invalid_actions: int = 0
    finalized: bool = False
    final_summary: str = ""
    action_history: List[str] = field(default_factory=list)
    inspection_log: List[str] = field(default_factory=list)
    last_event: str = ""
    last_error: str | None = None
    grader_report: GraderReport | None = None

    def copy_dispatch_totals(self) -> Dict[str, int]:
        return {
            resource_type: sum(targets.values())
            for resource_type, targets in self.resource_dispatches.items()
        }


class HeatShieldEnvironment(Environment):
    """Urban heatwave response environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str | None = None):
        self._requested_task_id = task_id or os.getenv("HEATSHIELD_TASK_ID")
        self._runtime = self._new_runtime(self._requested_task_id)

    def _new_runtime(self, requested_task_id: str | None) -> EpisodeRuntime:
        task_id = requested_task_id if requested_task_id in get_task_ids() else get_task_ids()[0]
        runtime = EpisodeRuntime(
            scenario=get_task(task_id),
            resource_remaining=dict(get_task(task_id).resource_pool),
            resource_dispatches={resource_type: {} for resource_type in RESOURCE_IMPACTS},
        )
        if requested_task_id and requested_task_id not in get_task_ids():
            runtime.last_event = (
                f"Unknown task '{requested_task_id}'. Falling back to {task_id}."
            )
        else:
            runtime.last_event = f"Ready for task {task_id}."
        runtime.grader_report = self._grade(runtime)
        return runtime

    def reset(self, task_id: str | None = None) -> HeatShieldObservation:
        if task_id is not None:
            self._requested_task_id = task_id
        self._runtime = self._new_runtime(self._requested_task_id)
        self._runtime.last_event = f"Episode reset for {self._runtime.scenario.title}."
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: HeatShieldAction) -> HeatShieldObservation:  # type: ignore[override]
        runtime = self._runtime
        if runtime.finalized:
            runtime.last_error = "episode_already_finalized"
            runtime.last_event = "Episode already finalized. Call reset() to start again."
            return self._build_observation(reward=0.0, done=True)

        previous_score = self._grade(runtime).score_breakdown.total_score
        runtime.step_count += 1
        runtime.last_error = None
        action_text = self._format_action(action)

        if action.action_type == "inspect":
            self._handle_inspect(runtime, action)
        elif action.action_type == "activate_center":
            self._handle_activate(runtime, action)
        elif action.action_type == "dispatch_resource":
            self._handle_dispatch(runtime, action)
        elif action.action_type == "broadcast_alert":
            self._handle_alert(runtime, action)
        elif action.action_type == "finalize":
            runtime.finalized = True
            runtime.final_summary = action.message.strip()
            runtime.last_event = "Mission finalized and submitted for grading."
        else:
            self._mark_invalid(runtime, f"unsupported_action:{action.action_type}")

        runtime.action_history.append(action_text)
        runtime.action_history = runtime.action_history[-8:]

        if runtime.step_count >= runtime.scenario.max_steps and not runtime.finalized:
            runtime.finalized = True
            runtime.last_event = (
                f"{runtime.last_event} Step limit reached; auto-finalized."
            ).strip()

        current_report = self._grade(runtime)
        # Keep transport-facing rewards in [0.0, 1.0] for validator compatibility,
        # while still providing dense positive feedback for useful progress.
        reward = round(
            max(current_report.score_breakdown.total_score - previous_score, 0.0),
            4,
        )
        done = runtime.finalized
        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> HeatShieldState:
        runtime = self._runtime
        report = self._grade(runtime)
        return HeatShieldState(
            episode_id=runtime.episode_id,
            step_count=runtime.step_count,
            task_id=runtime.scenario.task_id,
            difficulty=runtime.scenario.difficulty,
            active_centers=sorted(runtime.active_facilities),
            inspected_targets=sorted(runtime.inspected_targets),
            dispatched_resources=runtime.copy_dispatch_totals(),
            district_alerts=sorted(runtime.alert_messages.keys()),
            current_score=report.score_breakdown.total_score,
            invalid_actions=runtime.invalid_actions,
            finalized=runtime.finalized,
        )

    def _grade(self, runtime: EpisodeRuntime) -> GraderReport:
        runtime.grader_report = grade_plan(
            runtime.scenario,
            active_facility_ids=runtime.active_facilities,
            inspected_targets=runtime.inspected_targets,
            resource_dispatches=runtime.resource_dispatches,
            alert_messages=runtime.alert_messages,
            invalid_actions=runtime.invalid_actions,
        )
        return runtime.grader_report

    def _handle_inspect(self, runtime: EpisodeRuntime, action: HeatShieldAction) -> None:
        target_id = action.target_id
        if target_id in runtime.scenario.districts:
            district = runtime.scenario.districts[target_id]
            message = f"{district.label}: {district.secret_notes}"
        elif target_id in runtime.scenario.facilities:
            facility = runtime.scenario.facilities[target_id]
            message = f"{facility.label}: {facility.secret_notes}"
        elif target_id in RESOURCE_IMPACTS:
            available = runtime.resource_remaining.get(target_id, 0)
            message = f"{target_id} inventory remaining: {available}."
        else:
            self._mark_invalid(runtime, f"unknown_inspection_target:{target_id}")
            return

        if target_id not in runtime.inspected_targets:
            runtime.inspected_targets.add(target_id)
            runtime.inspection_log.append(message)
            runtime.inspection_log = runtime.inspection_log[-8:]
            runtime.last_event = f"Inspection completed for {target_id}."
        else:
            runtime.last_event = f"No new intel for {target_id}; it was already inspected."

    def _handle_activate(self, runtime: EpisodeRuntime, action: HeatShieldAction) -> None:
        facility = runtime.scenario.facilities.get(action.target_id)
        if not facility:
            self._mark_invalid(runtime, f"unknown_facility:{action.target_id}")
            return
        if facility.facility_id in runtime.active_facilities:
            self._mark_invalid(runtime, f"facility_already_active:{facility.facility_id}")
            return
        if self._facility_blocked(runtime, facility):
            self._mark_invalid(runtime, f"facility_blocked_requires_generator:{facility.facility_id}")
            return

        runtime.active_facilities.add(facility.facility_id)
        runtime.last_event = f"Activated {facility.label}."

    def _handle_dispatch(self, runtime: EpisodeRuntime, action: HeatShieldAction) -> None:
        if action.resource_type is None:
            self._mark_invalid(runtime, "dispatch_missing_resource_type")
            return
        if action.target_id not in runtime.scenario.districts:
            self._mark_invalid(runtime, f"unknown_dispatch_target:{action.target_id}")
            return
        if action.quantity <= 0:
            self._mark_invalid(runtime, "dispatch_quantity_must_be_positive")
            return

        available = runtime.resource_remaining.get(action.resource_type, 0)
        if action.quantity > available:
            self._mark_invalid(
                runtime,
                f"insufficient_{action.resource_type}:requested={action.quantity},available={available}",
            )
            return

        runtime.resource_remaining[action.resource_type] = available - action.quantity
        district_dispatch = runtime.resource_dispatches.setdefault(action.resource_type, {})
        district_dispatch[action.target_id] = district_dispatch.get(action.target_id, 0) + action.quantity
        runtime.last_event = (
            f"Dispatched {action.quantity} {action.resource_type} to {runtime.scenario.districts[action.target_id].label}."
        )

    def _handle_alert(self, runtime: EpisodeRuntime, action: HeatShieldAction) -> None:
        target_id = action.target_id
        if target_id != "all" and target_id not in runtime.scenario.districts:
            self._mark_invalid(runtime, f"unknown_alert_target:{target_id}")
            return
        message = action.message.strip()
        if not message:
            self._mark_invalid(runtime, "alert_requires_message")
            return
        runtime.alert_messages[target_id] = message
        target_label = "all districts" if target_id == "all" else runtime.scenario.districts[target_id].label
        runtime.last_event = f"Broadcast alert sent to {target_label}."

    def _facility_blocked(self, runtime: EpisodeRuntime, facility: FacilityScenario) -> bool:
        if not facility.requires_generator:
            return False
        generators_available = runtime.resource_dispatches["generator"].get(facility.district_id, 0)
        generators_in_use = sum(
            1
            for facility_id in runtime.active_facilities
            if runtime.scenario.facilities[facility_id].district_id == facility.district_id
            and runtime.scenario.facilities[facility_id].requires_generator
        )
        return generators_available <= generators_in_use

    def _mark_invalid(self, runtime: EpisodeRuntime, reason: str) -> None:
        runtime.invalid_actions += 1
        runtime.last_error = reason
        runtime.last_event = f"Action rejected: {reason}."

    def _build_observation(self, reward: float, done: bool) -> HeatShieldObservation:
        runtime = self._runtime
        report = self._grade(runtime)
        district_snapshots = [
            DistrictSnapshot(
                district_id=district.district_id,
                label=district.label,
                population=district.population,
                heat_index_c=district.heat_index_c,
                priority=district.priority,  # type: ignore[arg-type]
                vulnerability=district.vulnerability,
                unmet_relief=round(self._district_unmet_relief(runtime, district), 2),
                power_outage=self._district_has_power_issue(runtime, district),
                must_alert=district.must_alert,
                public_notes=district.public_notes,
            )
            for district in runtime.scenario.districts.values()
        ]
        facility_snapshots = [
            FacilitySnapshot(
                facility_id=facility.facility_id,
                district_id=facility.district_id,
                label=facility.label,
                status=self._facility_status(runtime, facility),
                capacity=facility.capacity,
                requires_generator=facility.requires_generator,
                public_notes=facility.public_notes,
            )
            for facility in runtime.scenario.facilities.values()
        ]
        resource_pool = [
            ResourceSnapshot(
                resource_type=resource_type,  # type: ignore[arg-type]
                available=runtime.resource_remaining.get(resource_type, 0),
                impact_per_unit=impact,
            )
            for resource_type, impact in RESOURCE_IMPACTS.items()
        ]
        turns_remaining = max(runtime.scenario.max_steps - runtime.step_count, 0)

        return HeatShieldObservation(
            task_id=runtime.scenario.task_id,
            task_title=runtime.scenario.title,
            difficulty=runtime.scenario.difficulty,
            benchmark=BENCHMARK_NAME,
            mission_brief=runtime.scenario.mission_brief,
            public_situation_report=runtime.scenario.public_situation_report,
            step_limit=runtime.scenario.max_steps,
            turns_remaining=turns_remaining,
            district_snapshots=district_snapshots,
            facility_snapshots=facility_snapshots,
            resource_pool=resource_pool,
            inspection_log=list(runtime.inspection_log),
            action_history=list(runtime.action_history),
            last_event=runtime.last_event,
            recommended_next_actions=self._build_recommendations(runtime, turns_remaining),
            grader_summary=report.summary_lines,
            score_breakdown=report.score_breakdown,
            done=done,
            reward=reward,
            metadata={
                "available_targets": {
                    "districts": list(runtime.scenario.districts.keys()),
                    "facilities": list(runtime.scenario.facilities.keys()),
                    "resource_types": list(RESOURCE_IMPACTS.keys()),
                },
                "priority_intel_targets": [
                    target
                    for target in runtime.scenario.priority_intel_targets
                    if target not in runtime.inspected_targets
                ],
                "must_alert_targets": [
                    district_id
                    for district_id, district in runtime.scenario.districts.items()
                    if district.must_alert and district_id not in runtime.alert_messages
                ],
                "blocked_facilities": [
                    facility.facility_id
                    for facility in runtime.scenario.facilities.values()
                    if self._facility_blocked(runtime, facility)
                ],
                "activatable_facilities": [
                    facility.facility_id
                    for facility in runtime.scenario.facilities.values()
                    if facility.facility_id not in runtime.active_facilities
                    and not self._facility_blocked(runtime, facility)
                ],
                "resource_remaining": dict(runtime.resource_remaining),
                "task_catalog": [summary.model_dump(mode="json") for summary in get_task_summaries()],
                "last_action_error": runtime.last_error,
            },
        )

    def _build_recommendations(self, runtime: EpisodeRuntime, turns_remaining: int) -> List[str]:
        recommendations: List[str] = []
        uninspected = [
            target
            for target in runtime.scenario.priority_intel_targets
            if target not in runtime.inspected_targets
        ]
        if uninspected:
            recommendations.append(f"Inspect {uninspected[0]} before spending the next scarce action.")

        must_alert_remaining = [
            district.label
            for district_id, district in runtime.scenario.districts.items()
            if district.must_alert and district_id not in runtime.alert_messages
        ]
        if must_alert_remaining:
            recommendations.append(f"Send a direct alert to {must_alert_remaining[0]}.")

        blocked_critical = [
            facility.label
            for facility in runtime.scenario.facilities.values()
            if facility.critical
            and facility.facility_id not in runtime.active_facilities
            and self._facility_blocked(runtime, facility)
        ]
        if blocked_critical and runtime.resource_remaining.get("generator", 0) > 0:
            recommendations.append(f"Dispatch a generator so {blocked_critical[0]} can open.")

        inactive_facilities = [
            facility.label
            for facility in runtime.scenario.facilities.values()
            if facility.facility_id not in runtime.active_facilities
            and not self._facility_blocked(runtime, facility)
        ]
        if inactive_facilities:
            recommendations.append(f"Activate {inactive_facilities[0]} if you need quick relief points.")

        if turns_remaining <= 1:
            recommendations.append("Finalize now to lock in the current score.")

        return recommendations[:4]

    def _district_has_power_issue(self, runtime: EpisodeRuntime, district: DistrictScenario) -> bool:
        if district.district_id in {"rivergate", "clinic_belt", "inland_towers", "creekside"}:
            return True
        return any(
            facility.requires_generator and facility.district_id == district.district_id
            for facility in runtime.scenario.facilities.values()
        )

    def _facility_status(
        self, runtime: EpisodeRuntime, facility: FacilityScenario
    ) -> str:
        if facility.facility_id in runtime.active_facilities:
            return "active"
        if self._facility_blocked(runtime, facility):
            return "blocked"
        return "inactive"

    def _district_unmet_relief(
        self, runtime: EpisodeRuntime, district: DistrictScenario
    ) -> float:
        achieved = self._district_relief_points(runtime, district)
        return max(district.relief_target - achieved, 0.0)

    def _district_relief_points(
        self, runtime: EpisodeRuntime, district: DistrictScenario
    ) -> float:
        points = sum(
            runtime.scenario.facilities[facility_id].impact
            for facility_id in runtime.active_facilities
            if runtime.scenario.facilities[facility_id].district_id == district.district_id
        )
        for resource_type, per_district in runtime.resource_dispatches.items():
            points += RESOURCE_IMPACTS[resource_type] * per_district.get(district.district_id, 0)

        targeted_message = runtime.alert_messages.get(district.district_id)
        if targeted_message:
            points += 1.0 if any(k in targeted_message.lower() for k in district.alert_keywords) else 0.6
        elif "all" in runtime.alert_messages:
            points += 0.5
        return points

    def _format_action(self, action: HeatShieldAction) -> str:
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
