# SpERT Chinese DuIE v1

This package registers the exported Chinese SpERT ONNX artifacts for the DuIE2.0 dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- runtime metadata loading: ready
- live predict path: ready

Important notes:

- the current export contains two ONNX stages: entity encoding and relation scoring
- deployment still requires Python-side span enumeration and relation sample construction
- the tokenizer and runtime configuration are aligned with `bert-base-chinese`
- this package is ready for end-to-end LuckyModels runtime predict and benchmark flows
