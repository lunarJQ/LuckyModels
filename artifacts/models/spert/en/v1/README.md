# SpERT English SciERC v1

This package registers the exported SpERT ONNX artifacts for the SciERC English dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- runtime metadata loading: ready
- live predict path: ready

Important notes:

- the current export contains two ONNX stages: entity encoding and relation scoring
- deployment still requires Python-side span enumeration and relation sample construction
- the raw `onnx/spert_scierc_model` export was normalized with outer-level links so the package can keep a flat model artifact layout
- this package is ready for end-to-end LuckyModels runtime predict and benchmark flows
