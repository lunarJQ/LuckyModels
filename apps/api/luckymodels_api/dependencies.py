from luckymodels_core.deployment_store import DeploymentStore
from luckymodels_core.registry import ArtifactRegistry
from luckymodels_core.settings import get_settings


def get_registry() -> ArtifactRegistry:
    settings = get_settings()
    return ArtifactRegistry(settings.artifacts_root)


def get_deployment_store() -> DeploymentStore:
    settings = get_settings()
    return DeploymentStore(settings.deployments_file)
