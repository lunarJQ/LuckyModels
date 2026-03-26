# LuckyModels Model Package Specification

## 1. Goal

The model package is the smallest deployable and manageable unit in `LuckyModels`.

It allows the platform to treat a knowledge-extraction model as a stable asset instead of a loose collection of scripts.

## 2. Required Structure

```text
<package-root>/
  manifest.yaml
  README.md
  artifacts/
  tokenizer/
  labels/
  profiles/
  samples/
```

## 3. Required Files

### 3.1 `manifest.yaml`

Required fields:

- `model_id`
- `model_name`
- `model_family`
- `task_type`
- `language`
- `framework`
- `runtime_adapter`
- `version`
- `backbone`
- `artifacts`
- `tokenizer`
- `labels`
- `inference`
- `profiles`

### 3.2 `profiles/*.yaml`

At least one profile must exist.

Recommended 1.0 set:

- `default.yaml`
- `latency.yaml`
- `balanced.yaml`
- `memory.yaml`
- `regression.yaml`

### 3.3 `samples/predict.json`

Provides a minimal request example for:

- runtime smoke testing
- deployment verification
- thesis screenshots and demos

## 4. Recommended Manifest Example

```yaml
model_id: casrel-zh-v1
model_name: CasRel
model_family: casrel
task_type: relation_extraction
language: zh
framework: onnxruntime
runtime_adapter: casrel_adapter
version: v1
backbone: bert-base-chinese

artifacts:
  subject_model: artifacts/casrel_subject.onnx
  object_model: artifacts/casrel_object.onnx

tokenizer:
  type: bert_wordpiece
  vocab_path: tokenizer/vocab.txt

labels:
  relation_map: labels/rel2id.json

inference:
  max_length: 256
  batch_size: 1
  thresholds:
    subject: 0.5
    object: 0.5

profiles:
  default: profiles/default.yaml
  latency: profiles/latency.yaml
  balanced: profiles/balanced.yaml
  memory: profiles/memory.yaml
```

## 5. Adapter Responsibility

The package itself does not define how inference works.

That responsibility belongs to the adapter referenced by:

- `runtime_adapter`

The adapter must know how to:

- load artifacts
- build tokenizer inputs
- run inference
- decode outputs
- return uniform triples

## 6. Validation Rules

The platform validates:

- manifest presence
- required manifest fields
- profile existence
- referenced files exist on disk

## 7. Why This Matters

This packaging rule is what makes the thesis concepts below concrete:

- model artifactization
- engineering deployment
- version governance
- profile-driven optimization
