from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    artifacts_root: Path
    state_root: Path
    logs_root: Path
    reports_root: Path
    imports_root: Path
    access_control_file: Path
    deployments_file: Path
    benchmarks_file: Path
    prediction_archives_file: Path
    task_routing_decisions_file: Path
    task_routing_feedback_file: Path
    deployment_policy_automation_file: Path
    gpu_pool_config_file: Path
    casrel_demo_report_file: Path
    webnlg_eval_cases_file: Path
    dataset_eval_cases_dir: Path


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    state_root = root_dir / "data" / "state"
    logs_root = root_dir / "data" / "logs"
    reports_root = state_root / "reports"
    imports_root = state_root / "imports"
    return Settings(
        root_dir=root_dir,
        artifacts_root=root_dir / "artifacts" / "models",
        state_root=state_root,
        logs_root=logs_root,
        reports_root=reports_root,
        imports_root=imports_root,
        access_control_file=state_root / "access_control.json",
        deployments_file=state_root / "deployments.json",
        benchmarks_file=state_root / "benchmarks.json",
        prediction_archives_file=state_root / "prediction_archives.json",
        task_routing_decisions_file=state_root / "task_routing_decisions.json",
        task_routing_feedback_file=state_root / "task_routing_feedback.json",
        deployment_policy_automation_file=state_root / "deployment_policy_automation.json",
        gpu_pool_config_file=state_root / "gpu_pool_config.json",
        casrel_demo_report_file=reports_root / "casrel_demo_report.json",
        webnlg_eval_cases_file=reports_root / "webnlg_eval_cases.json",
        dataset_eval_cases_dir=reports_root / "datasets",
    )
