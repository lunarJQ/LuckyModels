# LuckyModels

`LuckyModels` is a lightweight control plane for knowledge-extraction model packages.

中文说明：如果你想先快速看懂这套骨架，优先阅读 [实现说明](docs/实现说明.md)。

The project is built around five ideas:

1. Model packages are standardized with `manifest.yaml`.
2. Different model families are integrated through runtime adapters.
3. Deployment behavior is driven by reusable `profile.yaml` files.
4. The control plane manages assets, deployments, benchmarks, and records.
5. The runtime plane exposes uniform `/healthz`, `/metadata`, `/predict`, and `/metrics` endpoints.

This repository currently contains:

- An initial control-plane API skeleton based on FastAPI
- Shared core schemas and manifest validation logic
- A first CasRel model-package sample
- A generic runtime service skeleton with adapter loading
- Local runtime launch, health-check, stop/restart, and log-tail APIs
- A first benchmark record + runner foundation for latency/throughput experiments
- Project documentation written for implementation and onboarding

## Repository Layout

```text
LuckyModels/
  apps/
    api/                  # Control-plane API
  artifacts/
    models/               # Standardized model packages
  docs/                   # Architecture and usage docs
  infra/
    compose/              # Local docker compose examples
    prometheus/           # Monitoring examples
  onnx/                   # Raw uploaded model artifacts
  pages/                  # Thesis draft materials
  runtime/                # Runtime service entrypoint
  src/
    luckymodels_core/     # Shared domain logic
```

## Quick Start

Read these documents in order:

1. [Architecture](docs/architecture.md)
2. [Getting Started](docs/getting-started.md)
3. [Model Package Spec](docs/model-package-spec.md)

## Current Scope

This first implementation focuses on the foundation:

- Model package scanning and manifest validation
- Deployment record management
- Local runtime process management without relying on Docker
- Standard profile definitions
- CasRel package standardization
- Runtime adapter interface and CasRel adapter
- Benchmark execution and result persistence for later comparison

Triton-style performance tuning, richer dashboards, and more model families are still staged for later iterations, but the local engineering loop is now in place.
