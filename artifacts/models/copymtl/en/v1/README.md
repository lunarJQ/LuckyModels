# CopyMTL English WebNLG v1

This package registers the CopyMTL ONNX export for the English WebNLG dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- live predict path: not implemented yet

Important notes:

- the ONNX export expects token ids padded to length 80
- deployment must still keep text encoding and final action-sequence decoding in Python
- this package is ready for platform registration and profiling experiments
