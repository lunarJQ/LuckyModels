from luckymodels_core.access_store import AccessStore
from luckymodels_core.benchmark_store import BenchmarkStore
from luckymodels_core.deployment_policy_store import DeploymentPolicyAutomationStore
from luckymodels_core.deployment_store import DeploymentStore
from luckymodels_core.inference_activity_tracker import InferenceActivityTracker
from luckymodels_core.prediction_store import PredictionStore
from luckymodels_core.registry import ArtifactRegistry
from luckymodels_core.settings import get_settings
from luckymodels_core.task_routing_feedback_store import TaskRoutingFeedbackStore

_inference_activity_tracker = InferenceActivityTracker()


def get_registry() -> ArtifactRegistry:
    settings = get_settings()
    return ArtifactRegistry(settings.artifacts_root)


def get_access_store() -> AccessStore:
    settings = get_settings()
    return AccessStore(settings.access_control_file)


def get_deployment_store() -> DeploymentStore:
    settings = get_settings()
    return DeploymentStore(settings.deployments_file)


def get_benchmark_store() -> BenchmarkStore:
    settings = get_settings()
    return BenchmarkStore(settings.benchmarks_file)


def get_prediction_store() -> PredictionStore:
    settings = get_settings()
    return PredictionStore(settings.prediction_archives_file)


def get_task_routing_feedback_store() -> TaskRoutingFeedbackStore:
    settings = get_settings()
    return TaskRoutingFeedbackStore(
        settings.task_routing_decisions_file,
        settings.task_routing_feedback_file,
    )


def get_inference_activity_tracker() -> InferenceActivityTracker:
    return _inference_activity_tracker


def get_deployment_policy_store() -> DeploymentPolicyAutomationStore:
    settings = get_settings()
    return DeploymentPolicyAutomationStore(settings.deployment_policy_automation_file)
