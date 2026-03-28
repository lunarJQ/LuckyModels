from luckymodels_core.benchmark_store import BenchmarkStore
from luckymodels_core.deployment_store import DeploymentStore
from luckymodels_core.prediction_store import PredictionStore
from luckymodels_core.registry import ArtifactRegistry
from luckymodels_core.settings import get_settings


def get_registry() -> ArtifactRegistry:
    settings = get_settings()
    return ArtifactRegistry(settings.artifacts_root)


def get_deployment_store() -> DeploymentStore:
    settings = get_settings()
    return DeploymentStore(settings.deployments_file)


def get_benchmark_store() -> BenchmarkStore:
    settings = get_settings()
    return BenchmarkStore(settings.benchmarks_file)


def get_prediction_store() -> PredictionStore:
    settings = get_settings()
    return PredictionStore(settings.prediction_archives_file)
