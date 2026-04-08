"""FastAPI app for HeatShield."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

import openenv.core.env_server.http_server as openenv_http_server
from openenv.core.env_server import serialization as openenv_serialization
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ResetRequest, ResetResponse, SchemaResponse, StepResponse

from heatshield_env.models import HeatShieldAction, HeatShieldObservation, HeatShieldState
from server.heatshield_environment import HeatShieldEnvironment

try:
    import openenv.core.env_server.web_interface as openenv_web_interface
except ImportError:  # pragma: no cover - optional import path
    openenv_web_interface = None


HTTP_SESSION_HEADER = "X-OpenEnv-Session-Id"
HTTP_SESSION_LIMIT = 32
HTTP_SESSION_IDLE_TTL_S = 30 * 60

_ORIGINAL_SERIALIZE_OBSERVATION = openenv_serialization.serialize_observation


def _serialize_observation_with_metadata(observation: HeatShieldObservation) -> dict[str, Any]:
    """Preserve metadata in transport payloads for websocket and HTTP clients."""

    payload = _ORIGINAL_SERIALIZE_OBSERVATION(observation)
    payload["observation"]["metadata"] = dict(observation.metadata or {})
    return payload


openenv_serialization.serialize_observation = _serialize_observation_with_metadata
openenv_http_server.serialize_observation = _serialize_observation_with_metadata
if openenv_web_interface is not None:
    openenv_web_interface.serialize_observation = _serialize_observation_with_metadata


@dataclass
class HTTPSessionRecord:
    """Mutable server-side session for stateful HTTP debugging."""

    env: HeatShieldEnvironment
    last_used_at: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock)


class HTTPSessionManager:
    """Keep plain HTTP simulation routes stateful and self-consistent."""

    def __init__(self, max_sessions: int = HTTP_SESSION_LIMIT, idle_ttl_s: int = HTTP_SESSION_IDLE_TTL_S):
        self._max_sessions = max_sessions
        self._idle_ttl_s = idle_ttl_s
        self._sessions: dict[str, HTTPSessionRecord] = {}
        self._lock = RLock()

    def _close_record(self, record: HTTPSessionRecord) -> None:
        try:
            record.env.close()
        except Exception:
            pass

    def _prune_stale_sessions(self) -> None:
        now = time.monotonic()
        stale_ids = [
            session_id
            for session_id, record in self._sessions.items()
            if now - record.last_used_at > self._idle_ttl_s
        ]
        for session_id in stale_ids:
            record = self._sessions.pop(session_id)
            self._close_record(record)

    def _require_record(self, session_id: str | None) -> tuple[str, HTTPSessionRecord]:
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing {HTTP_SESSION_HEADER} header. Call /reset first.",
            )

        with self._lock:
            self._prune_stale_sessions()
            record = self._sessions.get(session_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown HTTP session '{session_id}'. Call /reset to create a session.",
            )

        return session_id, record

    def reset(self, session_id: str | None, request: ResetRequest) -> tuple[str, ResetResponse]:
        request_data = request.model_dump(exclude_unset=True)
        requested_task_id = request_data.get("task_id")

        with self._lock:
            self._prune_stale_sessions()

            if session_id:
                record = self._sessions.get(session_id)
            else:
                session_id = str(uuid4())
                record = None

            if record is None:
                if len(self._sessions) >= self._max_sessions:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="HTTP session limit reached. Reuse an existing session or wait for idle cleanup.",
                    )
                record = HTTPSessionRecord(env=HeatShieldEnvironment(task_id=requested_task_id))
                self._sessions[session_id] = record

        assert session_id is not None
        with record.lock:
            observation = record.env.reset(task_id=requested_task_id)
            record.last_used_at = time.monotonic()

        return session_id, ResetResponse(**_serialize_observation_with_metadata(observation))

    def step(self, session_id: str | None, action: HeatShieldAction) -> StepResponse:
        _, record = self._require_record(session_id)
        with record.lock:
            observation = record.env.step(action)
            record.last_used_at = time.monotonic()
        return StepResponse(**_serialize_observation_with_metadata(observation))

    def state(self, session_id: str | None) -> HeatShieldState:
        _, record = self._require_record(session_id)
        with record.lock:
            state_snapshot = record.env.state
            record.last_used_at = time.monotonic()
        return state_snapshot

    def close_all(self) -> None:
        with self._lock:
            session_items = list(self._sessions.items())
            self._sessions.clear()

        for _, record in session_items:
            self._close_record(record)


class HTTPActionEnvelope(BaseModel):
    """Compatibility wrapper for the generic OpenEnv HTTP step shape."""

    model_config = ConfigDict(extra="allow")

    action: HeatShieldAction = Field(..., description="Wrapped HeatShield action payload")


def _remove_generated_route(app: FastAPI, path: str, method: str) -> None:
    """Remove an autogenerated route so a corrected one can replace it."""

    kept_routes = []
    for route in app.router.routes:
        route_path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", None) or set()
        if route_path == path and method in route_methods:
            continue
        kept_routes.append(route)

    app.router.routes[:] = kept_routes
    app.openapi_schema = None


def _unwrap_http_action(payload: HeatShieldAction | HTTPActionEnvelope) -> HeatShieldAction:
    if isinstance(payload, HTTPActionEnvelope):
        return payload.action
    return payload


app = create_app(
    HeatShieldEnvironment,
    HeatShieldAction,
    HeatShieldObservation,
    env_name="heatshield_env",
    max_concurrent_envs=4,
)

for route_path, method in (
    ("/reset", "POST"),
    ("/step", "POST"),
    ("/state", "GET"),
    ("/schema", "GET"),
):
    _remove_generated_route(app, route_path, method)

http_sessions = HTTPSessionManager()


def _close_http_sessions() -> None:
    http_sessions.close_all()


app.router.on_shutdown.append(_close_http_sessions)


@app.post(
    "/reset",
    response_model=ResetResponse,
    tags=["Environment Control"],
    summary="Reset the environment",
    description=(
        "Reset or create a stateful HTTP session. Reuse the returned "
        f"`{HTTP_SESSION_HEADER}` header on `/step` and `/state`."
    ),
)
async def reset(
    response: Response,
    request: ResetRequest = Body(default_factory=ResetRequest),
    session_id: str | None = Header(default=None, alias=HTTP_SESSION_HEADER),
) -> ResetResponse:
    session_id, payload = await run_in_threadpool(http_sessions.reset, session_id, request)
    response.headers[HTTP_SESSION_HEADER] = session_id
    return payload


@app.post(
    "/step",
    response_model=StepResponse,
    tags=["Environment Control"],
    summary="Execute an action in the environment",
    description=(
        "Advance the active HTTP session by one step. The request body may be either "
        "a raw `HeatShieldAction` or a compatibility wrapper of the form "
        "`{\"action\": {...}}`."
    ),
)
async def step(
    payload: HeatShieldAction | HTTPActionEnvelope,
    response: Response,
    session_id: str | None = Header(default=None, alias=HTTP_SESSION_HEADER),
) -> StepResponse:
    normalized_action = _unwrap_http_action(payload)
    result = await run_in_threadpool(http_sessions.step, session_id, normalized_action)
    if session_id is not None:
        response.headers[HTTP_SESSION_HEADER] = session_id
    return result


@app.get(
    "/state",
    response_model=HeatShieldState,
    tags=["State Management"],
    summary="Get current environment state",
    description=(
        "Return the typed HeatShield state for the active HTTP session. "
        f"Requires the `{HTTP_SESSION_HEADER}` header returned by `/reset`."
    ),
)
async def state(
    response: Response,
    session_id: str | None = Header(default=None, alias=HTTP_SESSION_HEADER),
) -> HeatShieldState:
    state_payload = await run_in_threadpool(http_sessions.state, session_id)
    if session_id is not None:
        response.headers[HTTP_SESSION_HEADER] = session_id
    return state_payload


@app.get(
    "/schema",
    response_model=SchemaResponse,
    tags=["Schema"],
    summary="Get all JSON schemas",
    description="Return the action, observation, and HeatShield-specific state schemas.",
)
async def get_schemas() -> SchemaResponse:
    return SchemaResponse(
        action=HeatShieldAction.model_json_schema(),
        observation=HeatShieldObservation.model_json_schema(),
        state=HeatShieldState.model_json_schema(),
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the server locally."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
