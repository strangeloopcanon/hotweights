"""FastAPI application exposing Hotweights control-plane and planner operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover - executed when fastapi missing
    raise ImportError(
        "FastAPI integration requires the 'api' extra: install via "
        "`pip install hotweights[api]`."
    ) from exc
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.replicate import create_plan, verify_plan
from ..core.schemas import Plan
from .server import Coordinator


class RegisterRequest(BaseModel):
    worker_id: str
    caps: Dict[str, Any] = Field(default_factory=dict)


class BeginRequest(BaseModel):
    version: str
    manifest_digest: str


class CommitRequest(BaseModel):
    version: str


class AbortRequest(BaseModel):
    reason: str


class PlanRequest(BaseModel):
    prev_manifest: Dict[str, Any] = Field(alias="prev")
    next_manifest: Dict[str, Any] = Field(alias="next")
    bucket_mb: int = Field(ge=1)
    consumer_map: Optional[Dict[str, List[int]]] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("consumer_map")
    @classmethod
    def _ensure_consumer_map(
        cls, value: Optional[Dict[str, List[int]]]
    ) -> Optional[Dict[str, List[int]]]:
        if value is None:
            return None
        for key, ranks in value.items():
            if not isinstance(ranks, list) or not all(isinstance(r, int) for r in ranks):
                raise ValueError(f"consumer_map[{key!r}] must be a list of integers")
        return value


class VerifyPlanRequest(BaseModel):
    plan: Plan
    require_consumers: bool = False
    world_size: Optional[int] = None
    tp_groups: Optional[Dict[str, List[int]]] = None
    enforce_tp_superset: bool = False


class PlanResponse(BaseModel):
    plan: Plan


class VerifyPlanResponse(BaseModel):
    report: Dict[str, Any]


@dataclass
class AppConfig:
    coordinator: Coordinator


def build_app(coordinator: Optional[Coordinator] = None) -> FastAPI:
    """Construct a FastAPI application that wraps Hotweights services."""
    config = AppConfig(coordinator=coordinator or Coordinator())
    app = FastAPI(title="Hotweights API", version="0.0.1")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/coordinator/state")
    def coordinator_state() -> Dict[str, Any]:
        coord = config.coordinator
        return {
            "version": coord.version,
            "registered": {k: dict(v) for k, v in coord.registered.items()},
            "registered_count": len(coord.registered),
        }

    @app.post("/coordinator/register")
    def coordinator_register(payload: RegisterRequest) -> Dict[str, Any]:
        try:
            return config.coordinator.register(payload.worker_id, payload.caps)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/coordinator/begin")
    def coordinator_begin(payload: BeginRequest) -> Dict[str, Any]:
        try:
            return config.coordinator.begin(payload.version, payload.manifest_digest)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/coordinator/commit")
    def coordinator_commit(payload: CommitRequest) -> Dict[str, Any]:
        try:
            return config.coordinator.commit(payload.version)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/coordinator/abort")
    def coordinator_abort(payload: AbortRequest) -> Dict[str, Any]:
        try:
            return config.coordinator.abort(payload.reason)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/plan", response_model=PlanResponse)
    def create_plan_endpoint(payload: PlanRequest) -> PlanResponse:
        try:
            plan = create_plan(
                payload.prev_manifest,
                payload.next_manifest,
                bucket_mb=payload.bucket_mb,
                consumer_map=payload.consumer_map,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PlanResponse(plan=plan)

    @app.post("/plan/verify", response_model=VerifyPlanResponse)
    def verify_plan_endpoint(payload: VerifyPlanRequest) -> VerifyPlanResponse:
        try:
            report = verify_plan(
                payload.plan,
                require_consumers=payload.require_consumers,
                world_size=payload.world_size,
                tp_groups=payload.tp_groups,
                enforce_tp_superset=payload.enforce_tp_superset,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return VerifyPlanResponse(report=report)

    return app


__all__ = ["build_app"]
