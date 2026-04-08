"""HeatShield environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import HeatShieldAction, HeatShieldObservation, HeatShieldState


class HeatShieldEnv(EnvClient[HeatShieldAction, HeatShieldObservation, HeatShieldState]):
    """Client for HeatShield."""

    def _step_payload(self, action: HeatShieldAction) -> Dict:
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[HeatShieldObservation]:
        observation = HeatShieldObservation.model_validate(
            {
                **payload.get("observation", {}),
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HeatShieldState:
        return HeatShieldState.model_validate(payload)
