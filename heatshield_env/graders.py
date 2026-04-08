"""Deterministic graders for HeatShield."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Set

from .models import ScoreBreakdown
from .scenario_data import RESOURCE_IMPACTS, DistrictScenario, FacilityScenario, TaskScenario


@dataclass(frozen=True)
class GraderReport:
    """Structured grading report."""

    score_breakdown: ScoreBreakdown
    summary_lines: List[str]


def _message_matches_keywords(message: str, keywords: Iterable[str]) -> bool:
    lowered = message.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _district_relief(
    district: DistrictScenario,
    active_facilities: Iterable[FacilityScenario],
    resource_dispatches: Mapping[str, Mapping[str, int]],
    alert_messages: Mapping[str, str],
) -> float:
    relief_points = sum(facility.impact for facility in active_facilities)

    for resource_type, per_district in resource_dispatches.items():
        relief_points += RESOURCE_IMPACTS.get(resource_type, 0.0) * per_district.get(
            district.district_id, 0
        )

    targeted_message = alert_messages.get(district.district_id)
    if targeted_message:
        if _message_matches_keywords(targeted_message, district.alert_keywords):
            relief_points += 1.0
        else:
            relief_points += 0.6
    elif "all" in alert_messages:
        relief_points += 0.5

    return min(relief_points / district.relief_target, 1.0)


def grade_plan(
    scenario: TaskScenario,
    *,
    active_facility_ids: Set[str],
    inspected_targets: Set[str],
    resource_dispatches: Mapping[str, Mapping[str, int]],
    alert_messages: Mapping[str, str],
    invalid_actions: int,
) -> GraderReport:
    """Return a normalized score and human-readable summary."""

    active_facilities = {
        facility_id: scenario.facilities[facility_id]
        for facility_id in active_facility_ids
        if facility_id in scenario.facilities
    }

    total_weight = sum(district.vulnerability for district in scenario.districts.values())
    weighted_relief = 0.0
    district_ratios: Dict[str, float] = {}
    for district in scenario.districts.values():
        ratio = _district_relief(
            district,
            [
                facility
                for facility in active_facilities.values()
                if facility.district_id == district.district_id
            ],
            resource_dispatches,
            alert_messages,
        )
        district_ratios[district.district_id] = ratio
        weighted_relief += ratio * district.vulnerability
    relief_score = weighted_relief / total_weight if total_weight else 0.0

    critical_facilities = [facility for facility in scenario.facilities.values() if facility.critical]
    facility_score = (
        sum(1.0 for facility in critical_facilities if facility.facility_id in active_facility_ids)
        / len(critical_facilities)
        if critical_facilities
        else 1.0
    )

    alert_targets = [district for district in scenario.districts.values() if district.must_alert]
    alert_score = (
        sum(
            1.0
            for district in alert_targets
            if district.district_id in alert_messages
            or ("all" in alert_messages and district.district_id not in alert_messages)
        )
        / len(alert_targets)
        if alert_targets
        else 1.0
    )

    intel_score = (
        len(inspected_targets.intersection(set(scenario.priority_intel_targets)))
        / len(scenario.priority_intel_targets)
        if scenario.priority_intel_targets
        else 1.0
    )

    penalty_score = min(invalid_actions * 0.04, 0.2)
    total_score = max(
        0.0,
        min(
            1.0,
            0.55 * relief_score
            + 0.20 * facility_score
            + 0.15 * alert_score
            + 0.10 * intel_score
            - penalty_score,
        ),
    )

    score_breakdown = ScoreBreakdown(
        relief_score=round(relief_score, 4),
        facility_score=round(facility_score, 4),
        alert_score=round(alert_score, 4),
        intel_score=round(intel_score, 4),
        penalty_score=round(penalty_score, 4),
        total_score=round(total_score, 4),
    )

    summary_lines = [
        f"Relief coverage: {relief_score:.2f}",
        f"Critical facilities active: {facility_score:.2f}",
        f"Alert coverage: {alert_score:.2f}",
        f"Priority intel captured: {intel_score:.2f}",
    ]

    weakest_district_id = min(district_ratios, key=district_ratios.get)
    weakest_label = scenario.districts[weakest_district_id].label
    summary_lines.append(
        f"Biggest remaining gap: {weakest_label} ({district_ratios[weakest_district_id]:.2f} coverage)"
    )
    if invalid_actions:
        summary_lines.append(f"Penalty applied for invalid actions: -{penalty_score:.2f}")

    return GraderReport(score_breakdown=score_breakdown, summary_lines=summary_lines)
