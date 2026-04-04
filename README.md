# LuckyModels

`LuckyModels` is a lightweight control plane for knowledge-extraction model packages.

中文说明：如果你想先快速看懂这套骨架，优先阅读 [实现说明](docs/实现说明.md)。

The project is built around five ideas:

1. Model packages are standardized with `manifest.yaml`.
2. Different model families are integrated through runtime adapters.
3. Deployment behavior is driven by reusable `profile.yaml` files.
4. The control plane manages assets, deployments, benchmarks, and records.
5. The runtime plane exposes uniform `/healthz`, `/metadata`, `/predict`, and `/metrics` endpoints.

In the current repository state, the platform also includes strategy-level capabilities:

- online task routing with scene inference and model-family filtering
- online learning for routing weights
  - unsupervised calibration from benchmark and prediction archive evidence
  - explicit-feedback calibration from routing feedback and archive feedback
- deployment-policy recommendation with cross-layer coordination
  - latest effective benchmark evidence
  - routing-value signals
  - automation/restart history
  - GPU pool availability and CPU fallback suggestions

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

The currently implemented engineering loop is:

- standardized model import and package scanning
- deployment creation and runtime launch
- benchmark execution with effective-record replacement
- task-routing execution with archive persistence
- feedback write-back for routing decisions and prediction archives
- deployment-policy recommendation generation

## Online Learning and Benchmark Evidence

Task routing currently learns from three evidence channels:

1. Benchmark evidence
   The routing logic consumes the latest effective benchmark in each benchmark scope. Older superseded benchmark records are retained for traceability, but they are not repeatedly counted.
2. Unsupervised runtime evidence
   Prediction archives, empty-result rate, average latency, and scene/domain accumulation act as slow-moving calibration signals.
3. Explicit feedback evidence
   Routing feedback and archive satisfaction feedback act as faster calibration signals. For accepted feedback, the current objective's primary dimension is kept non-negative to avoid counter-intuitive results such as "accepted for accuracy, but accuracy contribution becomes negative".

## Validation Notes

The following flows have been practically verified in the current workspace:

- unsupervised task-routing execution
- explicit feedback submission and overview update
- deployment-policy recommendation generation

Triton-style performance tuning, richer dashboards, and more model families are still staged for later iterations, but the local engineering loop is now in place.
