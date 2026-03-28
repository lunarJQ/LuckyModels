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
    deployments_file: Path
    benchmarks_file: Path
    casrel_demo_report_file: Path
    webnlg_eval_cases_file: Path
    dataset_eval_cases_dir: Path


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    state_root = root_dir / "data" / "state"
    logs_root = root_dir / "data" / "logs"
    reports_root = state_root / "reports"
    return Settings(
        root_dir=root_dir,
        artifacts_root=root_dir / "artifacts" / "models",
        state_root=state_root,
        logs_root=logs_root,
        reports_root=reports_root,
        deployments_file=state_root / "deployments.json",
        benchmarks_file=state_root / "benchmarks.json",
        casrel_demo_report_file=reports_root / "casrel_demo_report.json",
        webnlg_eval_cases_file=reports_root / "webnlg_eval_cases.json",
        dataset_eval_cases_dir=reports_root / "datasets",
    )
