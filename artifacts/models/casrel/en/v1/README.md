# CasRel English WebNLG v1

This package is the first standardized `LuckyModels` sample package.

It wraps the existing raw CasRel artifacts into a platform-ready layout:

- two ONNX files
- relation mapping
- tokenizer vocabulary
- reusable profiles
- sample request files

Current status:

- package validation: ready
- control-plane registration: ready
- runtime metadata loading: ready
- runtime process launch: ready
- runtime predict path: ready
- deployment benchmark path: ready

Important notes:

- the uploaded relation schema is a WebNLG-style English schema
- the uploaded package is the current live runtime compatibility sample for LuckyModels English-model integration
- the adapter now exposes warnings and debug output to help diagnose tokenization, thresholds, and score distributions
