from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from luckymodels_core.adapters.factory import build_adapter
from luckymodels_core.manifest import load_manifest, load_profile


class PredictRequest(BaseModel):
    text: str
    scene: str | None = None
    options: dict[str, Any] | None = None


MANIFEST_PATH = os.getenv("LUCKYMODELS_MANIFEST_PATH")
PROFILE_PATH = os.getenv("LUCKYMODELS_PROFILE_PATH")

app = FastAPI(
    title="LuckyModels Runtime",
    version="0.1.0",
    description="Generic runtime service for LuckyModels model packages",
)

adapter = None
startup_error: str | None = None

if MANIFEST_PATH and PROFILE_PATH:
    try:
        loaded_manifest = load_manifest(MANIFEST_PATH)
        loaded_profile = load_profile(PROFILE_PATH)
        adapter_cls = build_adapter(loaded_manifest.manifest.runtime_adapter)
        adapter = adapter_cls(loaded_manifest, loaded_profile)
        adapter.load()
    except Exception as exc:  # pragma: no cover - startup path
        startup_error = str(exc)
else:
    startup_error = (
        "Missing environment variables LUCKYMODELS_MANIFEST_PATH and "
        "LUCKYMODELS_PROFILE_PATH"
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    if startup_error:
        return {"status": "error", "detail": startup_error}
    return adapter.health()


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if startup_error:
        raise HTTPException(status_code=500, detail=startup_error)
    return adapter.metadata()


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if startup_error:
        raise HTTPException(status_code=500, detail=startup_error)
    try:
        return adapter.predict(request.text, request.options)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if startup_error:
        return {"runtime_status": "error"}
    health = adapter.health()
    return {
        "runtime_status": health.get("status"),
        "sessions_loaded": int(bool(health.get("sessions_loaded"))),
    }
