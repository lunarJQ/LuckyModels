# TPLinker English WebNLG v1

This package registers the TPLinker ONNX export for the English WebNLG dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- live predict path: not implemented yet

Important notes:

- the ONNX export is fixed at `seq_len=100`
- deployment must preserve the shaking-tag decode logic in Python
- this package is currently ready for model-library governance, not end-to-end runtime prediction
