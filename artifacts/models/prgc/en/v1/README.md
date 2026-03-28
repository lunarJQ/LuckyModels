# PRGC English WebNLG v1

This package registers the PRGC ONNX exports for the English WebNLG dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- live predict path: not implemented yet

Important notes:

- the export is split into stage 1 encoder filtering and stage 2 tagger decoding
- deployment still requires candidate relation filtering and final triple reconstruction in Python
- this package is suitable for registry, packaging, and profile management in LuckyModels
