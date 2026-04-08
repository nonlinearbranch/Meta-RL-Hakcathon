"""Scenario registry for HeatShield."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, List, Tuple

from .models import Difficulty, Priority, ResourceType, TaskSummary


RESOURCE_IMPACTS: Mapping[ResourceType, float] = MappingProxyType(
    {
        "cooling_bus": 5.0,
        "water_truck": 3.0,
        "medical_team": 4.0,
        "generator": 0.0,
    }
)


@dataclass(frozen=True)
class DistrictScenario:
    district_id: str
    label: str
    population: int
    heat_index_c: int
    priority: Priority
    vulnerability: float
    relief_target: float
    power_outage: bool
    must_alert: bool
    public_notes: str
    secret_notes: str
    alert_keywords: Tuple[str, ...]


@dataclass(frozen=True)
class FacilityScenario:
    facility_id: str
    district_id: str
    label: str
    capacity: int
    impact: float
    requires_generator: bool
    critical: bool
    public_notes: str
    secret_notes: str


@dataclass(frozen=True)
class TaskScenario:
    task_id: str
    title: str
    difficulty: Difficulty
    short_description: str
    mission_brief: str
    public_situation_report: str
    max_steps: int
    success_threshold: float
    resource_pool: Mapping[str, int]
    districts: Mapping[str, DistrictScenario]
    facilities: Mapping[str, FacilityScenario]
    priority_intel_targets: Tuple[str, ...]
    playbook: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_pool", MappingProxyType(dict(self.resource_pool)))
        object.__setattr__(self, "districts", MappingProxyType(dict(self.districts)))
        object.__setattr__(self, "facilities", MappingProxyType(dict(self.facilities)))


TASKS: Dict[str, TaskScenario] = {
    "cooling_corridor_easy": TaskScenario(
        task_id="cooling_corridor_easy",
        title="Cooling Corridor Activation",
        difficulty="easy",
        short_description="Stand up two neighborhood cooling anchors during a one-district grid wobble.",
        mission_brief=(
            "A sharp noon heat spike has overwhelmed the east-side cooling corridor. "
            "You are the duty commander for HeatShield and must open safe cooling sites, "
            "route limited field resources, and push a direct alert before ambulance demand spikes."
        ),
        public_situation_report=(
            "Rivergate is reporting repeated brownouts, long outdoor queues, and heavy senior footfall. "
            "Market Square has an available school gym, but transit shelters are overheating. "
            "One generator, one cooling bus, two water trucks, and one medical team are available."
        ),
        max_steps=8,
        success_threshold=0.76,
        resource_pool={
            "cooling_bus": 1,
            "water_truck": 2,
            "medical_team": 1,
            "generator": 1,
        },
        districts={
            "rivergate": DistrictScenario(
                district_id="rivergate",
                label="Rivergate",
                population=18000,
                heat_index_c=46,
                priority="critical",
                vulnerability=0.95,
                relief_target=15.0,
                power_outage=True,
                must_alert=True,
                public_notes=(
                    "Dense senior population near the ferry terminal. Outdoor queues are stretching into direct sun."
                ),
                secret_notes=(
                    "A generator unlocks the library HVAC and immediately relieves the worst queue pressure."
                ),
                alert_keywords=("cooling", "hydration", "heat"),
            ),
            "market_square": DistrictScenario(
                district_id="market_square",
                label="Market Square",
                population=14500,
                heat_index_c=43,
                priority="high",
                vulnerability=0.75,
                relief_target=9.0,
                power_outage=False,
                must_alert=False,
                public_notes=(
                    "Street vendors are still operating, but there is strong foot traffic and limited shade."
                ),
                secret_notes=(
                    "A staffed gym plus one mobile asset is enough to stabilize Market Square quickly."
                ),
                alert_keywords=("cooling", "shade", "water"),
            ),
        },
        facilities={
            "rivergate_library": FacilityScenario(
                facility_id="rivergate_library",
                district_id="rivergate",
                label="Rivergate Library Cooling Hall",
                capacity=180,
                impact=6.0,
                requires_generator=True,
                critical=True,
                public_notes="Best indoor site in the district, but currently offline after repeated voltage drops.",
                secret_notes="Do not attempt to activate it before a generator reaches Rivergate.",
            ),
            "market_school_gym": FacilityScenario(
                facility_id="market_school_gym",
                district_id="market_square",
                label="Market School Gym",
                capacity=140,
                impact=5.0,
                requires_generator=False,
                critical=False,
                public_notes="Ready for public opening with volunteer staff on standby.",
                secret_notes="Fastest low-risk activation in the task.",
            ),
            "bus_depot_mist_zone": FacilityScenario(
                facility_id="bus_depot_mist_zone",
                district_id="rivergate",
                label="Bus Depot Mist Zone",
                capacity=90,
                impact=3.0,
                requires_generator=False,
                critical=False,
                public_notes="Temporary shaded area next to the depot. Lower capacity but quick to open.",
                secret_notes="Useful as a secondary boost after the main indoor site is online.",
            ),
        },
        priority_intel_targets=("rivergate", "rivergate_library", "medical_team"),
        playbook=(
            "Inspect the blackout-affected district or facility before opening it.",
            "Use the generator only where it unlocks the biggest cooling gain.",
            "A direct alert to Rivergate matters in this task.",
        ),
    ),
    "blackout_triage_medium": TaskScenario(
        task_id="blackout_triage_medium",
        title="Blackout Triage Sweep",
        difficulty="medium",
        short_description="Balance three districts after a feeder outage hits a clinic corridor and waterfront block.",
        mission_brief=(
            "A feeder outage has rolled through the clinic corridor while a waterfront queue keeps growing. "
            "HeatShield has to cover patient overflow, activate cooling centers, and keep the response coherent "
            "with only one mobile cooling bus and a tight generator budget."
        ),
        public_situation_report=(
            "Clinic Belt is seeing rising heat-stress walk-ins after a cooling tent lost power. "
            "Old Port has a recreation center ready but no transport support yet. "
            "Skyline West commuters are trapped in a hot concourse during a signal delay."
        ),
        max_steps=9,
        success_threshold=0.73,
        resource_pool={
            "cooling_bus": 1,
            "water_truck": 2,
            "medical_team": 2,
            "generator": 2,
        },
        districts={
            "clinic_belt": DistrictScenario(
                district_id="clinic_belt",
                label="Clinic Belt",
                population=21000,
                heat_index_c=45,
                priority="critical",
                vulnerability=0.98,
                relief_target=14.0,
                power_outage=True,
                must_alert=True,
                public_notes="Patient overflow is spilling outdoors near two outpatient clinics.",
                secret_notes="The triage tent is the single highest-value activation, but it needs generator support.",
                alert_keywords=("clinic", "cooling", "heat"),
            ),
            "old_port": DistrictScenario(
                district_id="old_port",
                label="Old Port",
                population=17000,
                heat_index_c=44,
                priority="high",
                vulnerability=0.86,
                relief_target=12.0,
                power_outage=False,
                must_alert=True,
                public_notes="Outdoor lines at the ferry plaza are exposing families and shift workers.",
                secret_notes="Old Port performs well once its recreation center is paired with one field resource.",
                alert_keywords=("port", "cooling", "water"),
            ),
            "skyline_west": DistrictScenario(
                district_id="skyline_west",
                label="Skyline West",
                population=13000,
                heat_index_c=42,
                priority="high",
                vulnerability=0.67,
                relief_target=10.0,
                power_outage=False,
                must_alert=False,
                public_notes="Commuters are stuck in a metal-heavy transit concourse with poor airflow.",
                secret_notes="The station hall is worth activating once the two higher-risk districts are covered.",
                alert_keywords=("station", "shade", "water"),
            ),
        },
        facilities={
            "clinic_triage_tent": FacilityScenario(
                facility_id="clinic_triage_tent",
                district_id="clinic_belt",
                label="Clinic Belt Triage Tent",
                capacity=160,
                impact=6.0,
                requires_generator=True,
                critical=True,
                public_notes="Primary overflow site for heat-stress walk-ins, currently dark.",
                secret_notes="Highest-value generator target in the scenario.",
            ),
            "old_port_rec_center": FacilityScenario(
                facility_id="old_port_rec_center",
                district_id="old_port",
                label="Old Port Recreation Center",
                capacity=150,
                impact=5.0,
                requires_generator=False,
                critical=True,
                public_notes="Openable immediately if you decide to stand it up.",
                secret_notes="One of the fastest ways to stabilize Old Port.",
            ),
            "skyline_station_hall": FacilityScenario(
                facility_id="skyline_station_hall",
                district_id="skyline_west",
                label="Skyline Station Hall",
                capacity=110,
                impact=4.0,
                requires_generator=False,
                critical=False,
                public_notes="Large indoor hall with available attendants.",
                secret_notes="Helpful, but lower priority than Clinic Belt and Old Port.",
            ),
            "old_port_library_annex": FacilityScenario(
                facility_id="old_port_library_annex",
                district_id="old_port",
                label="Old Port Library Annex",
                capacity=80,
                impact=3.0,
                requires_generator=False,
                critical=False,
                public_notes="Secondary low-capacity site near the waterfront queue.",
                secret_notes="Useful when you have one spare action and still need relief points.",
            ),
        },
        priority_intel_targets=("clinic_belt", "clinic_triage_tent", "old_port"),
        playbook=(
            "Clinic Belt is your most fragile district and should be understood early.",
            "Two districts require direct alerts in this task.",
            "Do not starve Old Port while solving Clinic Belt.",
        ),
    ),
    "cascade_hard": TaskScenario(
        task_id="cascade_hard",
        title="Cascade Heat Recovery",
        difficulty="hard",
        short_description="Recover four districts after a blackout, transit stall, and clinic overflow hit at once.",
        mission_brief=(
            "A layered heat emergency is rippling across four districts: inland towers lost power, "
            "a rail junction is jammed, a creekside clinic is unstable, and the harbor north library "
            "is the only cool indoor option near a major worksite. HeatShield must spread limited assets "
            "without missing the most vulnerable populations."
        ),
        public_situation_report=(
            "Inland Towers high-rises are reporting elevator outages and limited HVAC. "
            "Rail Junction has stalled commuter trains and asphalt exposure. "
            "Creekside's clinic overflow is outdoors after a partial breaker trip. "
            "Harbor North workers are crowding around the only indoor library branch."
        ),
        max_steps=10,
        success_threshold=0.70,
        resource_pool={
            "cooling_bus": 1,
            "water_truck": 3,
            "medical_team": 2,
            "generator": 2,
        },
        districts={
            "inland_towers": DistrictScenario(
                district_id="inland_towers",
                label="Inland Towers",
                population=23000,
                heat_index_c=47,
                priority="critical",
                vulnerability=0.99,
                relief_target=16.0,
                power_outage=True,
                must_alert=True,
                public_notes="High-rise seniors are sheltering in stairwells after rolling elevator outages.",
                secret_notes="Generator-backed indoor cooling is essential; mobile assets alone are not enough.",
                alert_keywords=("tower", "cooling", "check-ins"),
            ),
            "rail_junction": DistrictScenario(
                district_id="rail_junction",
                label="Rail Junction",
                population=19000,
                heat_index_c=45,
                priority="critical",
                vulnerability=0.88,
                relief_target=13.0,
                power_outage=False,
                must_alert=True,
                public_notes="Commuters are stalled in an exposed transfer area with little shade.",
                secret_notes="A fast indoor activation plus hydration support quickly changes the rail score.",
                alert_keywords=("station", "water", "cooling"),
            ),
            "harbor_north": DistrictScenario(
                district_id="harbor_north",
                label="Harbor North",
                population=15500,
                heat_index_c=43,
                priority="high",
                vulnerability=0.74,
                relief_target=11.0,
                power_outage=False,
                must_alert=False,
                public_notes="Shipyard workers are walking to the library branch for relief.",
                secret_notes="Harbor North does not need an alert as much as it needs capacity.",
                alert_keywords=("harbor", "shade", "cooling"),
            ),
            "creekside": DistrictScenario(
                district_id="creekside",
                label="Creekside",
                population=14200,
                heat_index_c=44,
                priority="high",
                vulnerability=0.78,
                relief_target=10.0,
                power_outage=True,
                must_alert=True,
                public_notes="Families are clustering near a clinic overflow lane after a breaker trip.",
                secret_notes="A generator plus one medical or hydration asset stabilizes Creekside.",
                alert_keywords=("clinic", "cooling", "water"),
            ),
        },
        facilities={
            "inland_civic_hub": FacilityScenario(
                facility_id="inland_civic_hub",
                district_id="inland_towers",
                label="Inland Civic Hub",
                capacity=190,
                impact=6.0,
                requires_generator=True,
                critical=True,
                public_notes="Largest indoor option for tower residents, but offline after the outage.",
                secret_notes="This site should receive a generator before activation.",
            ),
            "rail_union_college": FacilityScenario(
                facility_id="rail_union_college",
                district_id="rail_junction",
                label="Rail Union College Hall",
                capacity=150,
                impact=5.0,
                requires_generator=False,
                critical=True,
                public_notes="Accessible and staffed hall adjacent to the transfer point.",
                secret_notes="High leverage and immediately activatable.",
            ),
            "harbor_library_branch": FacilityScenario(
                facility_id="harbor_library_branch",
                district_id="harbor_north",
                label="Harbor North Library Branch",
                capacity=120,
                impact=4.0,
                requires_generator=False,
                critical=False,
                public_notes="Only cooled indoor space within walking distance of the shipyard.",
                secret_notes="Important if you want to avoid starving Harbor North.",
            ),
            "creekside_field_clinic": FacilityScenario(
                facility_id="creekside_field_clinic",
                district_id="creekside",
                label="Creekside Field Clinic",
                capacity=90,
                impact=4.0,
                requires_generator=True,
                critical=True,
                public_notes="Partial breaker trip took the triage fans offline.",
                secret_notes="Needs one generator before activation.",
            ),
            "rail_mist_arcade": FacilityScenario(
                facility_id="rail_mist_arcade",
                district_id="rail_junction",
                label="Rail Mist Arcade",
                capacity=70,
                impact=3.0,
                requires_generator=False,
                critical=False,
                public_notes="Small temporary shelter that can shave the queue quickly.",
                secret_notes="A useful finishing move if you still need rail relief.",
            ),
        },
        priority_intel_targets=("inland_towers", "inland_civic_hub", "creekside_field_clinic", "rail_junction"),
        playbook=(
            "Two generator-backed facilities anchor the hard task.",
            "Three districts need direct alerts; Harbor North does not.",
            "Rail Junction is easier to stabilize than Inland Towers, but both matter.",
        ),
    ),
}


def get_task(task_id: str) -> TaskScenario:
    """Return one task scenario."""
    try:
        return TASKS[task_id]
    except KeyError as exc:
        available = ", ".join(get_task_ids())
        raise KeyError(
            f"Unknown task_id '{task_id}'. Available tasks: {available}"
        ) from exc


def get_task_ids() -> List[str]:
    """Return task ids in stable order."""

    return list(TASKS.keys())


def get_task_summaries() -> List[TaskSummary]:
    """Return task registry entries for docs and metadata."""

    return [
        TaskSummary(
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            short_description=task.short_description,
            success_threshold=task.success_threshold,
        )
        for task in TASKS.values()
    ]
