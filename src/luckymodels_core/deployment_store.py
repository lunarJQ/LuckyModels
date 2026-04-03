from __future__ import annotations

import json
import socket
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from .schemas import DeploymentCreateRequest, DeploymentRecord, utc_now


class DeploymentStore:
    SERVICE_HOST = "127.0.0.1"
    SERVICE_PORT_RANGE_START = 58000
    SERVICE_PORT_RANGE_END = 59000

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
            use_gpu=request.use_gpu,
            gpu_ids=request.gpu_ids,
            provider_preference=request.provider_preference,
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

    def build_service_url(self, port: int) -> str:
        return f"http://{self.SERVICE_HOST}:{int(port)}"

    def extract_service_port(self, service_url: str | None) -> int | None:
        if not service_url:
            return None
        try:
            parsed = urlsplit(service_url)
        except ValueError:
            return None
        if parsed.hostname != self.SERVICE_HOST:
            return None
        try:
            return int(parsed.port) if parsed.port is not None else None
        except ValueError:
            return None

    def is_managed_service_port(self, port: int | None) -> bool:
        if port is None:
            return False
        return self.SERVICE_PORT_RANGE_START <= int(port) <= self.SERVICE_PORT_RANGE_END

    def ensure_service_url(
        self,
        deployment: DeploymentRecord,
        *,
        keep_current_if_managed: bool = True,
    ) -> str:
        current_port = self.extract_service_port(deployment.service_url)
        if keep_current_if_managed and self.is_managed_service_port(current_port):
            has_collision = any(
                item.deployment_id != deployment.deployment_id
                and self.extract_service_port(item.service_url) == current_port
                for item in self.list()
            )
            if not has_collision:
                return self.build_service_url(current_port)
        preferred_port = current_port if self.is_managed_service_port(current_port) else None
        return self.build_service_url(
            self._allocate_port(
                exclude_deployment_id=deployment.deployment_id,
                preferred_port=preferred_port,
            )
        )

    def _allocate_port_in_range(
        self,
        *,
        exclude_deployment_id: str | None = None,
        preferred_port: int | None = None,
    ) -> int:
        reserved_ports = {
            port
            for item in self.list()
            if item.deployment_id != exclude_deployment_id
            for port in [self.extract_service_port(item.service_url)]
            if self.is_managed_service_port(port)
        }

        candidate_ports: list[int] = []
        if self.is_managed_service_port(preferred_port):
            candidate_ports.append(int(preferred_port))
        candidate_ports.extend(range(self.SERVICE_PORT_RANGE_START, self.SERVICE_PORT_RANGE_END + 1))

        tried_ports: set[int] = set()
        for port in candidate_ports:
            port = int(port)
            if port in tried_ports or port in reserved_ports:
                continue
            tried_ports.add(port)
            if self._is_port_bindable(port):
                return port

        raise RuntimeError(
            "No available deployment service ports left in managed range "
            f"{self.SERVICE_PORT_RANGE_START}-{self.SERVICE_PORT_RANGE_END}."
        )

    def _allocate_port(
        self,
        *,
        exclude_deployment_id: str | None = None,
        preferred_port: int | None = None,
    ) -> int:
        return self._allocate_port_in_range(
            exclude_deployment_id=exclude_deployment_id,
            preferred_port=preferred_port,
        )

    def _is_port_bindable(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self.SERVICE_HOST, int(port)))
            except OSError:
                return False
        return True
