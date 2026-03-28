# CopyMTL English WebNLG v1

This package registers the CopyMTL ONNX export for the English WebNLG dataset.

Current status:

- package validation: ready
- control-plane registration: ready
- live predict path: ready

Important notes:

- the ONNX export expects token ids padded to length 80
- LuckyModels runtime now performs text encoding and final action-sequence decoding in Python
- unknown vocabulary tokens can be tolerated by mapping them to padding ids, or rejected with `strict_vocab=true`
