from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "apps" / "api"))

    from main import app  # noqa: WPS433

    client = TestClient(app)

    print("== System Summary ==")
    print(json.dumps(client.get("/api/v1/system/summary").json(), ensure_ascii=False, indent=2))

    print("\n== Models ==")
    models = client.get("/api/v1/models").json()
    print(json.dumps(models, ensure_ascii=False, indent=2))
    if not models:
        print("No models were found under artifacts/models.")
        return 1

    model_id = models[0]["model_id"]
    print(f"\nUsing model: {model_id}")

    create_response = client.post(
        "/api/v1/deployments",
        json={
            "model_id": model_id,
            "profile_name": "default",
            "launch_runtime": True,
            "notes": "demo_local_flow",
        },
    )
    create_response.raise_for_status()
    deployment = create_response.json()
    deployment_id = deployment["deployment_id"]

    print("\n== Deployment Created ==")
    print(json.dumps(deployment, ensure_ascii=False, indent=2))

    time.sleep(2)

    health = client.get(f"/api/v1/deployments/{deployment_id}/health")
    health.raise_for_status()
    print("\n== Deployment Health ==")
    print(json.dumps(health.json(), ensure_ascii=False, indent=2))

    benchmark = client.post(
        "/api/v1/benchmarks",
        json={
            "deployment_id": deployment_id,
            "text": "ABC设备主轴发生振动异常，需要检查主轴轴承。",
            "iterations": 3,
            "warmup_runs": 1,
            "label": "demo_local_flow",
        },
    )
    benchmark.raise_for_status()
    print("\n== Benchmark Result ==")
    print(json.dumps(benchmark.json(), ensure_ascii=False, indent=2))

    logs = client.get(f"/api/v1/deployments/{deployment_id}/logs")
    logs.raise_for_status()
    print("\n== Deployment Logs ==")
    print(json.dumps(logs.json(), ensure_ascii=False, indent=2))

    stop = client.post(f"/api/v1/deployments/{deployment_id}/stop")
    stop.raise_for_status()
    print("\n== Deployment Stopped ==")
    print(json.dumps(stop.json(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
