"""Typed models for the HeatShield environment."""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


ActionType = Literal[
    "inspect",
    "activate_center",
    "dispatch_resource",
    "broadcast_alert",
    "finalize",
]
Difficulty = Literal["easy", "medium", "hard"]
Priority = Literal["moderate", "high", "critical"]
ResourceType = Literal["cooling_bus", "water_truck", "medical_team", "generator"]


class DistrictSnapshot(BaseModel):
    """Public situation summary for a district."""

    district_id: str = Field(..., description="Stable district identifier")
    label: str = Field(..., description="Human-readable district name")
    population: int = Field(..., description="Estimated people impacted")
    heat_index_c: int = Field(..., description="Approximate heat index")
    priority: Priority = Field(..., description="Operational priority band")
    vulnerability: float = Field(..., description="Relative vulnerability weight")
    unmet_relief: float = Field(..., description="Remaining relief need")
    power_outage: bool = Field(..., description="Whether the district has a power outage")
    must_alert: bool = Field(..., description="Whether the district needs direct alerting")
    public_notes: str = Field(..., description="Publicly visible operating notes")


class FacilitySnapshot(BaseModel):
    """Public facility summary."""

    facility_id: str = Field(..., description="Stable facility identifier")
    district_id: str = Field(..., description="District that owns the facility")
    label: str = Field(..., description="Human-readable facility name")
    status: Literal["inactive", "active", "blocked"] = Field(
        ..., description="Whether the center is available, active, or blocked"
    )
    capacity: int = Field(..., description="Approximate cooling capacity")
    requires_generator: bool = Field(
        ..., description="Whether the site needs a generator before activation"
    )
    public_notes: str = Field(..., description="Publicly visible facility notes")


class ResourceSnapshot(BaseModel):
    """Resource inventory summary."""

    resource_type: ResourceType = Field(..., description="Resource category")
    available: int = Field(..., description="Units still available to dispatch")
    impact_per_unit: float = Field(..., description="Relief impact of one dispatched unit")


class ScoreBreakdown(BaseModel):
    """Normalized score components."""

    relief_score: float = Field(..., ge=0.0, le=1.0)
    facility_score: float = Field(..., ge=0.0, le=1.0)
    alert_score: float = Field(..., ge=0.0, le=1.0)
    intel_score: float = Field(..., ge=0.0, le=1.0)
    penalty_score: float = Field(..., ge=0.0, le=1.0)
    total_score: float = Field(..., ge=0.0, le=1.0)


class TaskSummary(BaseModel):
    """Task registry entry."""

    task_id: str = Field(..., description="Task identifier")
    title: str = Field(..., description="Task title")
    difficulty: Difficulty = Field(..., description="Difficulty band")
    short_description: str = Field(..., description="Short task description")
    success_threshold: float = Field(..., ge=0.0, le=1.0)


class HeatShieldAction(Action):
    """One command in the HeatShield environment."""

    action_type: ActionType = Field(..., description="Action family to execute")
    target_id: str = Field(
        ...,
        description="District, facility, or task target. Use 'mission' when finalizing.",
    )
    resource_type: Optional[ResourceType] = Field(
        default=None, description="Resource used for dispatch_resource"
    )
    quantity: int = Field(
        default=1, ge=0, le=4, description="Units to dispatch for dispatch_resource"
    )
    message: str = Field(default="", description="Alert copy or final summary")


class HeatShieldObservation(Observation):
    """Full observation returned after reset and step."""

    task_id: str = Field(..., description="Current task identifier")
    task_title: str = Field(..., description="Current task title")
    difficulty: Difficulty = Field(..., description="Difficulty band")
    benchmark: str = Field(..., description="Benchmark/environment identifier")
    mission_brief: str = Field(..., description="One-paragraph mission brief")
    public_situation_report: str = Field(..., description="Operational status report")
    step_limit: int = Field(..., ge=1, description="Maximum step budget for the task")
    turns_remaining: int = Field(..., ge=0, description="Steps left in the episode")
    district_snapshots: List[DistrictSnapshot] = Field(default_factory=list)
    facility_snapshots: List[FacilitySnapshot] = Field(default_factory=list)
    resource_pool: List[ResourceSnapshot] = Field(default_factory=list)
    inspection_log: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    last_event: str = Field(default="", description="Human-readable result of the last action")
    recommended_next_actions: List[str] = Field(default_factory=list)
    grader_summary: List[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(...)


class HeatShieldState(State):
    """Typed environment state."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Current difficulty")
    active_centers: List[str] = Field(default_factory=list)
    inspected_targets: List[str] = Field(default_factory=list)
    dispatched_resources: Dict[str, int] = Field(default_factory=dict)
    district_alerts: List[str] = Field(default_factory=list)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    invalid_actions: int = Field(default=0, ge=0)
    finalized: bool = Field(default=False)
