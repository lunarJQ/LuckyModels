from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from luckymodels_core.manifest import load_manifest


class ManifestPathResolutionTests(unittest.TestCase):
    def test_load_manifest_keeps_package_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_root = root / "artifacts" / "models" / "spert" / "en" / "v1"
            raw_root = root / "onnx" / "spert_scierc_model"
            nested_root = raw_root / "spert_scierc_model"

            (package_root / "artifacts").mkdir(parents=True)
            (package_root / "tokenizer").mkdir(parents=True)
            (package_root / "labels").mkdir(parents=True)
            nested_root.mkdir(parents=True)

            (nested_root / "spert_entity_encoder.onnx").write_bytes(b"mock")
            (nested_root / "vocab.txt").write_text("hello\n", encoding="utf-8")
            (nested_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (nested_root / "types.json").write_text("{}", encoding="utf-8")
            (nested_root / "config.json").write_text("{}", encoding="utf-8")

            (raw_root / "spert_entity_encoder.onnx").symlink_to(
                nested_root / "spert_entity_encoder.onnx"
            )
            (raw_root / "vocab.txt").symlink_to(nested_root / "vocab.txt")
            (raw_root / "tokenizer_config.json").symlink_to(
                nested_root / "tokenizer_config.json"
            )
            (raw_root / "types.json").symlink_to(nested_root / "types.json")
            (raw_root / "config.json").symlink_to(nested_root / "config.json")

            (package_root / "artifacts" / "spert_entity_encoder.onnx").symlink_to(
                raw_root / "spert_entity_encoder.onnx"
            )
            (package_root / "tokenizer" / "vocab.txt").symlink_to(raw_root / "vocab.txt")
            (package_root / "tokenizer" / "tokenizer_config.json").symlink_to(
                raw_root / "tokenizer_config.json"
            )
            (package_root / "labels" / "types.json").symlink_to(raw_root / "types.json")
            (package_root / "labels" / "config.json").symlink_to(raw_root / "config.json")

            manifest_path = package_root / "manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "model_id: spert-en-scierc-v1",
                        "model_name: SpERT",
                        "model_family: spert",
                        "task_type: relation_extraction",
                        "language: en",
                        "framework: onnxruntime",
                        "runtime_adapter: spert_adapter",
                        "version: v1",
                        "dataset_name: scierc",
                        "dataset_split: raw",
                        "stage_count: 2",
                        "predict_ready: true",
                        "backbone: bert-base-cased",
                        "artifacts:",
                        "  entity_encoder: artifacts/spert_entity_encoder.onnx",
                        "tokenizer:",
                        "  type: bert_wordpiece",
                        "  vocab_path: tokenizer/vocab.txt",
                        "  tokenizer_path: tokenizer/tokenizer_config.json",
                        "labels:",
                        "  schema_path: labels/types.json",
                        "  entity_map: labels/config.json",
                        "inference:",
                        "  max_length: 256",
                        "  batch_size: 1",
                        "profiles: {}",
                    ]
                ),
                encoding="utf-8",
            )

            loaded = load_manifest(manifest_path)

            self.assertEqual(
                loaded.resolved_artifacts["entity_encoder"],
                str(package_root / "artifacts" / "spert_entity_encoder.onnx"),
            )
            self.assertEqual(
                loaded.resolved_tokenizer["vocab_path"],
                str(package_root / "tokenizer" / "vocab.txt"),
            )
            self.assertEqual(
                loaded.resolved_labels["schema_path"],
                str(package_root / "labels" / "types.json"),
            )


if __name__ == "__main__":
    unittest.main()
