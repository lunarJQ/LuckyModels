from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .schemas import DeploymentCreateRequest, DeploymentRecord


class DeploymentStore:
    def __init__(self, deployments_file: Path) -> None:
        self.deployments_file = deployments_file
        self.deployments_file.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[DeploymentRecord]:
        if not self.deployments_file.exists():
            return []

        with self.deployments_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or []
        return [DeploymentRecord.model_validate(item) for item in data]

    def create(self, request: DeploymentCreateRequest) -> DeploymentRecord:
        deployment = DeploymentRecord(
            deployment_id=f"dep-{uuid4().hex[:12]}",
            model_id=request.model_id,
            profile_name=request.profile_name,
            runtime=request.runtime,
            node_name=request.node_name,
            gpu_ids=request.gpu_ids,
            notes=request.notes,
            service_url=f"http://127.0.0.1:{self._allocate_port()}",
        )
        deployments = self.list()
        deployments.append(deployment)
        with self.deployments_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in deployments], handle, ensure_ascii=False, indent=2)
        return deployment

    def _allocate_port(self) -> int:
        base_port = 18081
        used_ports = {
            int(item.service_url.rsplit(":", maxsplit=1)[-1])
            for item in self.list()
            if item.service_url
        }
        candidate = base_port
        while candidate in used_ports:
            candidate += 1
        return candidate
