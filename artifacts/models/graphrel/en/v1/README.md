# GraphRel English NYT v1

This package registers the GraphRel ONNX export in the LuckyModels model library with the NYT test dataset configuration.

Current status:

- package validation: ready
- control-plane registration: ready
- live predict path: not implemented yet

Important notes:

- the exported ONNX covers only the neural network core
- deployment must still run spaCy preprocessing, vector extraction, POS encoding, and dependency-graph construction
- the original export directory is `graphrel_nyt_model`, and the platform now keeps the same NYT dataset label
