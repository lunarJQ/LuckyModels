# OneRel English WebNLG v1

This package registers the OneRel ONNX export for the English WebNLG dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- live predict path: not implemented yet

Important notes:

- the ONNX export includes the neural core only
- deployment must preserve the HBTokenizer preprocessing rule
- final triple decoding still belongs to the Python inference service
