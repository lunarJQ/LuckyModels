from __future__ import annotations

import json

from luckymodels_core.benchmark_store import BenchmarkStore
from luckymodels_core.deployment_store import DeploymentStore
from luckymodels_core.registry import ArtifactRegistry
from luckymodels_core.settings import get_settings


def main() -> int:
    settings = get_settings()
    benchmark_store = BenchmarkStore(settings.benchmarks_file)
    deployment_store = DeploymentStore(settings.deployments_file)
    registry = ArtifactRegistry(settings.artifacts_root)

    deployment_lookup = {
        item.deployment_id: item
        for item in deployment_store.list()
        if item.deployment_id
    }
    model_lookup = {
        item.model_id: item
        for item in registry.list_models()
        if item.model_id
    }

    result = benchmark_store.migrate_legacy_records(
        model_lookup=model_lookup,
        deployment_lookup=deployment_lookup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
