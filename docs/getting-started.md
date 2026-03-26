# Getting Started

## 1. What Is Already Implemented

This repository now includes:

- a shared manifest schema and validator
- a model-package registry scanner
- a deployment record store
- a benchmark record store and benchmark runner
- a FastAPI-based control plane
- a runtime service with local launch and health-check support
- a first CasRel sample package

## 2. Recommended Reading Order

1. [README](../README.md)
2. [Architecture](architecture.md)
3. [Model Package Spec](model-package-spec.md)
4. [Chinese Implementation Notes](实现说明.md)
5. [Local Run and Benchmark Guide](本地运行与基准测试.md)

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
/home/liujingqian/LuckyModels/scripts/run_api.sh
```

The script tries `127.0.0.1:38080` first.
If the port is already in use, it automatically switches to a free localhost port.
You can still override the preferred port if you want:

```bash
LUCKYMODELS_API_PORT=38180 /home/liujingqian/LuckyModels/scripts/run_api.sh
```

Main endpoints:

- `GET /healthz`
- `GET /api/v1/system/summary`
- `GET /api/v1/models`
- `GET /api/v1/models/{model_id}`
- `GET /api/v1/models/{model_id}/profiles`
- `GET /api/v1/deployments`
- `POST /api/v1/deployments`
- `GET /api/v1/deployments/{deployment_id}/health`
- `GET /api/v1/deployments/{deployment_id}/logs`
- `POST /api/v1/deployments/{deployment_id}/stop`
- `POST /api/v1/deployments/{deployment_id}/restart`
- `GET /api/v1/benchmarks`
- `POST /api/v1/benchmarks`

## 5. Run the Runtime Service Directly

```bash
/home/liujingqian/LuckyModels/scripts/run_runtime_casrel.sh
```

The script tries `127.0.0.1:38081` first.
If the port is already in use, it automatically switches to a free localhost port.
You can still override the preferred port if you want:

```bash
LUCKYMODELS_RUNTIME_PORT=38181 /home/liujingqian/LuckyModels/scripts/run_runtime_casrel.sh
```

Main endpoints:

- `GET /healthz`
- `GET /metadata`
- `POST /predict`
- `GET /metrics`

## 5.1 Run a Direct Smoke Test

```bash
cd /home/liujingqian/LuckyModels
.venv/bin/python scripts/smoke_test_casrel.py
```

This script loads the standardized CasRel package directly and prints:

- runtime health
- package metadata
- one prediction result

## 5.2 Use the Control Plane To Launch a Local Runtime

The safer engineering path is to let the control plane allocate a free localhost port automatically.
This avoids hard-coded ports and reduces conflicts with other services.

Example request:

```bash
curl -X POST http://127.0.0.1:38080/api/v1/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "casrel-zh-v1",
    "profile_name": "default",
    "launch_runtime": true
  }'
```

After the deployment is created:

```bash
curl http://127.0.0.1:38080/api/v1/deployments/<deployment_id>/health
curl http://127.0.0.1:38080/api/v1/deployments/<deployment_id>/logs
```

Run a benchmark:

```bash
curl -X POST http://127.0.0.1:38080/api/v1/benchmarks \
  -H 'Content-Type: application/json' \
  -d '{
    "deployment_id": "<deployment_id>",
    "text": "ABC设备主轴发生振动异常，需要检查主轴轴承。",
    "iterations": 5,
    "warmup_runs": 1
  }'
```

## 6. What To Do Next

Recommended next steps:

1. Verify the CasRel package metadata and profile fields
2. Improve the CasRel decode quality and thresholds
3. Add richer benchmark dimensions such as concurrency and batch size
4. Add container launch orchestration from the control plane when needed
5. Extend the same package pattern to SpERT, TPLinker, PRGC, and others

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
- `apps/api/luckymodels_api/routers/deployments.py`
  This manages local runtime launch, stop, restart, and logs.
- `apps/api/luckymodels_api/routers/benchmarks.py`
  This manages benchmark creation and result persistence.
- `runtime/server.py`
  This starts the runtime service.
- `src/luckymodels_core/adapters/casrel.py`
  This is the first model-family adapter implementation.
