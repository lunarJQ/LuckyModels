from __future__ import annotations

import json
import socket
from pathlib import Path
from uuid import uuid4

from .schemas import DeploymentCreateRequest, DeploymentRecord, utc_now


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
        )
        deployments = self.list()
        deployments.append(deployment)
        with self.deployments_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in deployments], handle, ensure_ascii=False, indent=2)
        return deployment

    def update(self, deployment: DeploymentRecord) -> DeploymentRecord:
        deployment.updated_at = utc_now()
        deployments = self.list()
        updated: list[DeploymentRecord] = []
        found = False
        for item in deployments:
            if item.deployment_id == deployment.deployment_id:
                updated.append(deployment)
                found = True
            else:
                updated.append(item)

        if not found:
            updated.append(deployment)

        with self.deployments_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in updated], handle, ensure_ascii=False, indent=2)
        return deployment

    def delete(self, deployment_id: str) -> DeploymentRecord | None:
        deployments = self.list()
        remaining: list[DeploymentRecord] = []
        deleted: DeploymentRecord | None = None
        for item in deployments:
            if item.deployment_id == deployment_id:
                deleted = item
            else:
                remaining.append(item)

        if deleted is None:
            return None

        with self.deployments_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in remaining], handle, ensure_ascii=False, indent=2)
        return deleted

    def _allocate_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
