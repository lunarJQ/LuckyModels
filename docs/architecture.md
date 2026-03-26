# LuckyModels 1.0 Architecture

## 1. Positioning

`LuckyModels` is designed as a lightweight knowledge-extraction model-library platform.

It is not a general LLM platform and it does not assume that every model can be deployed as a single raw ONNX file. Many extraction models, including CasRel, require:

- tokenizer assets
- label mappings
- multiple ONNX artifacts
- custom decode logic
- scene-aware profiles

Because of that, `LuckyModels` uses a layered design instead of binding itself tightly to a single off-the-shelf model-hosting product.

## 2. Core Technical Route

The 1.0 route is:

- Control plane: `FastAPI`
- Shared domain core: `Pydantic + YAML + filesystem registry`
- Runtime plane: adapter-driven service with ONNX Runtime as the target backend
- Profiles: YAML-based deployment templates
- Observability: Prometheus-friendly endpoints, later Grafana and DCGM Exporter
- Packaging: standard model package rooted by `manifest.yaml`

This route keeps the platform flexible enough for:

- CasRel-style multi-ONNX pipelines
- Span-based and token-pair model families
- future migration to BentoML or Triton

## 3. Layered Design

### 3.1 Presentation Layer

Responsible for:

- model list and model detail
- upload and validation results
- deployment creation
- profile selection
- benchmark result viewing
- logs and metrics browsing

The UI is intentionally allowed to evolve later. The current repository focuses first on backend contracts and model packaging.

### 3.2 Control Plane

Responsible for:

- scanning model packages
- validating manifests
- registering versions and profiles
- storing deployment records
- exposing management APIs

The control plane should not contain model-specific inference logic.

### 3.3 Runtime Plane

Responsible for:

- loading artifacts
- initializing tokenizer and sessions
- running model-family-specific preprocessing and postprocessing
- exposing unified inference endpoints

Model-specific logic lives behind adapters such as:

- `casrel_adapter`
- `spert_adapter`
- `tplinker_adapter`
- `prgc_adapter`

### 3.4 Observability Layer

Responsible for:

- request metrics
- deployment health
- benchmark result collection
- GPU and memory observations

In 1.0, the important thing is to preserve the interfaces and file structure for observability even if not every monitoring component is fully wired yet.

## 4. Core Domain Objects

`LuckyModels` manages these first-class objects:

- `Model`
- `ModelVersion`
- `Artifact`
- `Profile`
- `Deployment`
- `BenchmarkTask`
- `BenchmarkResult`

In this repository, the first implementation emphasizes artifact, profile, and deployment objects because they are the minimum needed for a working model-library control plane.

## 5. Model Package Philosophy

The platform only accepts standardized model packages.

A model package must include:

- `manifest.yaml`
- model artifacts
- tokenizer assets
- label/schema files
- profiles
- sample request files

This is the bridge between the thesis concept of "model artifactization" and a real platform implementation.

## 6. Why GPUStack Is Not the Core Here

GPUStack remains a useful reference for:

- resource-management ideas
- page structure inspiration
- deployment-module decomposition

But `LuckyModels` must own its own inference contract because knowledge-extraction models often need custom orchestration rather than standard LLM backends.

## 7. 1.0 Implementation Priorities

Priority order:

1. Standardize model packages
2. Build control-plane APIs
3. Add the first CasRel adapter
4. Add deployment records and profile selection
5. Add benchmark execution
6. Add container launch orchestration

This repository implements steps 1 through 4 as the initial foundation.
