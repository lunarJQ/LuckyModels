# SpERT English SciERC v1

This package registers the exported SpERT ONNX artifacts for the SciERC English dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- runtime metadata loading: ready
- live predict path: not implemented yet

Important notes:

- the current export contains two ONNX stages: entity encoding and relation scoring
- deployment still requires Python-side span enumeration and relation sample construction
- this package is ready for platform management, but not yet for end-to-end runtime predict
