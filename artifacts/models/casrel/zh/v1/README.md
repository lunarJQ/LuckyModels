# CasRel Chinese DuIE v1

This package adapts the uploaded Chinese CasRel ONNX artifacts into the
standard `LuckyModels` package layout.

Included assets:

- two ONNX stages
- relation mapping
- Chinese BERT vocabulary
- reusable deployment profiles
- sample request files

Recommended profiles:

- `default`: tuned online inference baseline for current DuIE deployment
- `latency`: lower-latency single-request runtime profile
- `balanced`: balanced runtime profile for general verification
- `memory`: reduced resource footprint profile
- `regression`: compact repeated smoke-test profile

Current status:

- package validation: ready
- control-plane registration: ready
- runtime metadata loading: ready
- runtime process launch: ready
- runtime predict path: ready
- deployment benchmark path: ready

Important notes:

- the uploaded relation schema follows the Chinese DuIE relation extraction task
- the runtime keeps subject/object decoding in Python after ONNX execution
- this package is intended to be deployed and benchmarked like the English CasRel sample
