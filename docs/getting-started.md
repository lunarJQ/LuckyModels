# Getting Started

## 1. What Is Already Implemented

This repository now includes:

- a shared manifest schema and validator
- a model-package registry scanner
- a deployment record store
- a FastAPI-based control-plane skeleton
- a runtime service skeleton
- a first CasRel sample package

## 2. Recommended Reading Order

1. [README](../README.md)
2. [Architecture](architecture.md)
3. [Model Package Spec](model-package-spec.md)

## 3. Python Environment

The project is prepared with a `pyproject.toml`.

Recommended local workflow:

```bash
cd /home/liujingqian/LuckyModels
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 4. Run the Control Plane

```bash
cd /home/liujingqian/LuckyModels/apps/api
uvicorn main:app --reload --port 8080
```

Main endpoints:

- `GET /healthz`
- `GET /api/v1/system/summary`
- `GET /api/v1/models`
- `GET /api/v1/models/{model_id}`
- `GET /api/v1/models/{model_id}/profiles`
- `GET /api/v1/deployments`
- `POST /api/v1/deployments`

## 5. Run the Runtime Service

The runtime service is environment-driven. It expects:

- `LUCKYMODELS_MANIFEST_PATH`
- `LUCKYMODELS_PROFILE_PATH`

Example:

```bash
cd /home/liujingqian/LuckyModels
export LUCKYMODELS_MANIFEST_PATH=/home/liujingqian/LuckyModels/artifacts/models/casrel/zh/v1/manifest.yaml
export LUCKYMODELS_PROFILE_PATH=/home/liujingqian/LuckyModels/artifacts/models/casrel/zh/v1/profiles/default.yaml
uvicorn runtime.server:app --reload --port 18081
```

Main endpoints:

- `GET /healthz`
- `GET /metadata`
- `POST /predict`
- `GET /metrics`

## 6. What To Do Next

Recommended next steps:

1. Verify the CasRel package metadata and profile fields
2. Install ONNX Runtime dependencies
3. Complete the CasRel decode pipeline
4. Add a benchmark runner
5. Add container launch orchestration from the control plane

## 7. How To Read The Code

If you are worried the code may feel hard to follow, start here:

- `src/luckymodels_core/schemas.py`
  This defines the core data contracts.
- `src/luckymodels_core/manifest.py`
  This is the manifest/profile loader and validator.
- `src/luckymodels_core/registry.py`
  This scans the model packages.
- `apps/api/main.py`
  This starts the control-plane API.
- `runtime/server.py`
  This starts the runtime service.
- `src/luckymodels_core/adapters/casrel.py`
  This is the first model-family adapter scaffold.
