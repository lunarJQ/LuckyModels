from __future__ import annotations

import fnmatch
import io
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .manifest import load_manifest, validate_loaded_manifest
from .schemas import (
    LoadedManifest,
    ModelDetail,
    ModelImportFamilyOption,
    ModelImportModeOption,
    ModelImportOptions,
    ModelImportResult,
)
from .settings import Settings

IMPORT_MODE_AUTO_PACKAGE = "auto_package"
IMPORT_MODE_DIRECT_IMPORT = "direct_import"
PACKAGE_METADATA_FILENAME = ".luckymodels_package.json"


@dataclass(frozen=True)
class _PathRule:
    target: str
    patterns: tuple[str, ...]
    entry_type: str = "file"


@dataclass(frozen=True)
class _FamilySpec:
    code: str
    label: str
    artifact_rules: dict[str, _PathRule]
    tokenizer_type: str
    tokenizer_rules: dict[str, _PathRule]
    label_rules: dict[str, _PathRule]
    default_model_name: str
    default_backbones: dict[str, str]
    default_language: str = "en"


FAMILY_SPECS: dict[str, _FamilySpec] = {
    "casrel": _FamilySpec(
        code="casrel",
        label="CasRel",
        artifact_rules={
            "subject_model": _PathRule("artifacts/casrel_subject.onnx", ("casrel_subject.onnx", "*subject*.onnx")),
            "object_model": _PathRule("artifacts/casrel_object.onnx", ("casrel_object.onnx", "*object*.onnx")),
        },
        tokenizer_type="bert_wordpiece",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/vocab.txt", ("vocab.txt", "*vocab*.txt")),
        },
        label_rules={
            "relation_map": _PathRule("labels/rel2id.json", ("rel2id.json", "*rel2id*.json")),
        },
        default_model_name="CasRel",
        default_backbones={"en": "bert-base-cased", "zh": "bert-base-chinese"},
        default_language="en",
    ),
    "tplinker": _FamilySpec(
        code="tplinker",
        label="TPLinker",
        artifact_rules={
            "model": _PathRule("artifacts/tplinker_bert.onnx", ("tplinker*.onnx", "*tplinker*.onnx", "*.onnx")),
        },
        tokenizer_type="bert_wordpiece",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/vocab.txt", ("vocab.txt", "*vocab*.txt")),
            "tokenizer_path": _PathRule("tokenizer/tokenizer.json", ("tokenizer.json", "*tokenizer*.json")),
        },
        label_rules={
            "relation_map": _PathRule("labels/rel2id.json", ("rel2id.json", "*rel2id*.json")),
            "schema_path": _PathRule("labels/config.json", ("config.json", "*config*.json")),
        },
        default_model_name="TPLinker",
        default_backbones={"en": "bert-base-cased", "zh": "bert-base-chinese"},
        default_language="en",
    ),
    "onerel": _FamilySpec(
        code="onerel",
        label="OneRel",
        artifact_rules={
            "model": _PathRule("artifacts/onerel.onnx", ("onerel*.onnx", "*.onnx")),
        },
        tokenizer_type="hb_tokenizer",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/vocab.txt", ("vocab.txt", "*vocab*.txt")),
            "tokenizer_path": _PathRule("tokenizer/hb_tokenizer.py", ("hb_tokenizer.py", "*tokenizer*.py")),
        },
        label_rules={
            "relation_map": _PathRule("labels/rel2id.json", ("rel2id.json", "*rel2id*.json")),
            "schema_path": _PathRule("labels/tag2id.json", ("tag2id.json", "*tag2id*.json")),
        },
        default_model_name="OneRel",
        default_backbones={"en": "bert-base-cased", "zh": "bert-base-chinese"},
        default_language="en",
    ),
    "prgc": _FamilySpec(
        code="prgc",
        label="PRGC",
        artifact_rules={
            "stage1_model": _PathRule(
                "artifacts/prgc_stage1_encoder.onnx",
                ("prgc_stage1_encoder.onnx", "*stage1*.onnx", "*encoder*.onnx"),
            ),
            "stage2_model": _PathRule(
                "artifacts/prgc_stage2_tagger.onnx",
                ("prgc_stage2_tagger.onnx", "*stage2*.onnx", "*tagger*.onnx"),
            ),
        },
        tokenizer_type="bert_wordpiece",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/vocab.txt", ("vocab.txt", "*vocab*.txt")),
        },
        label_rules={
            "relation_map": _PathRule("labels/rel2id.json", ("rel2id.json", "*rel2id*.json")),
            "schema_path": _PathRule("labels/config.json", ("config.json", "*config*.json")),
            "entity_map": _PathRule("labels/params.json", ("params.json", "*params*.json")),
        },
        default_model_name="PRGC",
        default_backbones={"en": "bert-base-cased", "zh": "bert-base-chinese"},
        default_language="en",
    ),
    "pure": _FamilySpec(
        code="pure",
        label="PURE",
        artifact_rules={
            "entity_model": _PathRule("artifacts/pure_entity.onnx", ("pure_entity.onnx", "*entity*.onnx")),
            "relation_model": _PathRule("artifacts/pure_relation.onnx", ("pure_relation.onnx", "*relation*.onnx")),
        },
        tokenizer_type="bert_wordpiece",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/entity_vocab.txt", ("entity_vocab.txt", "*entity*vocab*.txt", "*vocab*.txt")),
            "tokenizer_path": _PathRule(
                "tokenizer/entity_tokenizer_config.json",
                ("entity_tokenizer_config.json", "*tokenizer*config*.json", "*tokenizer*.json"),
            ),
        },
        label_rules={
            "entity_map": _PathRule("labels/ner_label_list.json", ("ner_label_list.json", "*ner*label*.json")),
            "relation_map": _PathRule(
                "labels/relation_label_list.json",
                ("relation_label_list.json", "*relation*label*.json", "label_list.json", "*label_list*.json"),
            ),
            "schema_path": _PathRule("labels/entity_config.json", ("entity_config.json", "*entity*config*.json", "*config*.json")),
            "relation_vocab": _PathRule("labels/vocab.txt", ("relation_assets/vocab.txt", "*relation*/*vocab*.txt", "vocab.txt")),
            "special_tokens": _PathRule(
                "labels/special_tokens.json",
                ("relation_assets/special_tokens.json", "*relation*/*special_tokens*.json", "special_tokens.json"),
            ),
        },
        default_model_name="PURE",
        default_backbones={"en": "bert-base-cased", "zh": "bert-base-chinese"},
        default_language="en",
    ),
    "spert": _FamilySpec(
        code="spert",
        label="SpERT",
        artifact_rules={
            "entity_encoder": _PathRule(
                "artifacts/spert_entity_encoder.onnx",
                ("spert_entity_encoder.onnx", "*entity*encoder*.onnx", "*entity*.onnx"),
            ),
            "relation_head": _PathRule(
                "artifacts/spert_relation_head.onnx",
                ("spert_relation_head.onnx", "*relation*head*.onnx", "*relation*.onnx"),
            ),
        },
        tokenizer_type="bert_wordpiece",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/vocab.txt", ("vocab.txt", "*vocab*.txt")),
            "tokenizer_path": _PathRule("tokenizer/tokenizer_config.json", ("tokenizer_config.json", "*tokenizer*config*.json")),
        },
        label_rules={
            "schema_path": _PathRule("labels/types.json", ("types.json", "*types*.json")),
            "entity_map": _PathRule("labels/config.json", ("config.json", "*config*.json")),
        },
        default_model_name="SpERT",
        default_backbones={"en": "bert-base-cased", "zh": "bert-base-chinese"},
        default_language="en",
    ),
    "graphrel": _FamilySpec(
        code="graphrel",
        label="GraphRel",
        artifact_rules={
            "model": _PathRule("artifacts/graphrel.onnx", ("graphrel.onnx", "*graphrel*.onnx", "*.onnx")),
        },
        tokenizer_type="spacy_en_core_web_lg",
        tokenizer_rules={
            "tokenizer_path": _PathRule("tokenizer/en_core_web_lg", ("en_core_web_lg", "*en_core_web_lg*"), entry_type="dir"),
        },
        label_rules={
            "relation_map": _PathRule("labels/id2rel.json", ("id2rel.json", "*id2rel*.json")),
            "entity_map": _PathRule("labels/ne_labels.json", ("ne_labels.json", "*ne_labels*.json")),
            "schema_path": _PathRule("labels/pos_tags.json", ("pos_tags.json", "*pos_tags*.json")),
        },
        default_model_name="GraphRel",
        default_backbones={"en": "graphrel-2p-spacy"},
        default_language="en",
    ),
    "copymtl": _FamilySpec(
        code="copymtl",
        label="CopyMTL",
        artifact_rules={
            "model": _PathRule("artifacts/copymtl.onnx", ("copymtl.onnx", "*copymtl*.onnx", "*.onnx")),
        },
        tokenizer_type="words2id_lookup",
        tokenizer_rules={
            "vocab_path": _PathRule("tokenizer/words2id.json", ("words2id.json", "*words2id*.json")),
        },
        label_rules={
            "relation_map": _PathRule("labels/relations2id.json", ("relations2id.json", "*relations2id*.json")),
            "schema_path": _PathRule("labels/config.json", ("config*.json", "*config*.json")),
            "entity_map": _PathRule("labels/conversion_meta.json", ("conversion_meta.json", "*conversion_meta*.json")),
        },
        default_model_name="CopyMTL",
        default_backbones={"en": "lstm-copy-decoder"},
        default_language="en",
    ),
}


def get_model_import_options() -> ModelImportOptions:
    return ModelImportOptions(
        modes=[
            ModelImportModeOption(
                code=IMPORT_MODE_AUTO_PACKAGE,
                label="模式A：原始制品自动包装",
                note="上传原始 ONNX 压缩包，平台按所选模型家族自动补齐 manifest、profiles 和标准目录结构。",
            ),
            ModelImportModeOption(
                code=IMPORT_MODE_DIRECT_IMPORT,
                label="模式B：标准包直接接入",
                note="上传已经符合 LuckyModels 标准结构的模型包，平台完成校验后直接注册接入。",
            ),
        ],
        families=[
            ModelImportFamilyOption(
                code=spec.code,
                label=spec.label,
                default_language=spec.default_language,
                note=f"支持 {spec.label} 标准接入与导出。",
            )
            for spec in FAMILY_SPECS.values()
        ],
    )


def import_model_archive(
    *,
    settings: Settings,
    archive_path: Path,
    archive_filename: str,
    import_mode: str,
    source_type: str = "builtin",
    model_family: str | None = None,
    language: str | None = None,
    dataset_name: str | None = None,
    version: str | None = None,
    model_name: str | None = None,
    backbone: str | None = None,
    alias: str | None = None,
) -> ModelImportResult:
    import_mode = str(import_mode or "").strip()
    if import_mode not in {IMPORT_MODE_AUTO_PACKAGE, IMPORT_MODE_DIRECT_IMPORT}:
        raise ValueError(f"Unsupported import_mode: {import_mode}")

    settings.imports_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=settings.imports_root) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extracted_root = temp_dir / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, extracted_root)

        if import_mode == IMPORT_MODE_DIRECT_IMPORT:
            return _direct_import_standard_package(
                settings=settings,
                extracted_root=extracted_root,
                archive_filename=archive_filename,
                source_type=source_type,
                model_family=model_family,
                language=language,
                dataset_name=dataset_name,
                version=version,
                model_name=model_name,
                backbone=backbone,
                alias=alias,
            )

        return _auto_package_raw_archive(
            settings=settings,
            extracted_root=extracted_root,
            archive_filename=archive_filename,
            source_type=source_type,
            model_family=model_family,
            language=language,
            dataset_name=dataset_name,
            version=version,
            model_name=model_name,
            backbone=backbone,
            alias=alias,
        )


def export_model_package(model: ModelDetail) -> tuple[str, bytes]:
    package_root = Path(model.summary.package_root)
    if not package_root.exists():
        raise FileNotFoundError(f"Package root not found: {package_root}")

    filename = f"{model.summary.model_id}.zip"
    manifest_data = model.manifest.model_dump(mode="python", exclude_none=True)
    export_entries: list[tuple[Path, str, str]] = []

    artifact_mapping: dict[str, str] = {}
    for key, source in model.resolved_artifacts.items():
        source_path = Path(source)
        target = f"artifacts/{source_path.name}"
        artifact_mapping[key] = target
        export_entries.append((source_path, target, "file"))

    profile_mapping: dict[str, str] = {}
    for key, source in model.resolved_profiles.items():
        source_path = Path(source)
        target = f"profiles/{source_path.name}"
        profile_mapping[key] = target
        export_entries.append((source_path, target, "file"))

    tokenizer_mapping = dict(manifest_data.get("tokenizer", {}))
    for field_name, source in model.resolved_tokenizer.items():
        source_path = Path(source)
        target = f"tokenizer/{source_path.name}"
        tokenizer_mapping[field_name] = target
        export_entries.append((source_path, target, "dir" if source_path.is_dir() else "file"))

    labels_mapping = dict(manifest_data.get("labels", {}))
    for field_name, source in model.resolved_labels.items():
        source_path = Path(source)
        target = f"labels/{source_path.name}"
        labels_mapping[field_name] = target
        export_entries.append((source_path, target, "dir" if source_path.is_dir() else "file"))

    manifest_data["artifacts"] = artifact_mapping
    manifest_data["profiles"] = profile_mapping
    manifest_data["tokenizer"] = tokenizer_mapping
    manifest_data["labels"] = labels_mapping

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        root_prefix = Path(model.summary.model_id)
        archive.writestr(
            (root_prefix / "manifest.yaml").as_posix(),
            yaml.safe_dump(manifest_data, allow_unicode=True, sort_keys=False),
        )
        for source_path, target, entry_type in export_entries:
            if entry_type == "dir":
                for child in sorted(source_path.rglob("*")):
                    if child.is_dir():
                        continue
                    child_target = root_prefix / target / child.relative_to(source_path)
                    archive.write(child, child_target.as_posix())
                continue
            archive.write(source_path, (root_prefix / target).as_posix())

        for relative_dir in ("samples",):
            source_dir = package_root / relative_dir
            if not source_dir.exists():
                continue
            for path in sorted(source_dir.rglob("*")):
                if path.is_dir():
                    continue
                archive.write(path, (root_prefix / path.relative_to(package_root)).as_posix())

        readme_path = package_root / "README.md"
        if readme_path.exists():
            archive.write(readme_path, (root_prefix / "README.md").as_posix())
        metadata_path = package_root / PACKAGE_METADATA_FILENAME
        if metadata_path.exists():
            archive.write(metadata_path, (root_prefix / PACKAGE_METADATA_FILENAME).as_posix())
    return filename, buffer.getvalue()


def _auto_package_raw_archive(
    *,
    settings: Settings,
    extracted_root: Path,
    archive_filename: str,
    source_type: str,
    model_family: str | None,
    language: str | None,
    dataset_name: str | None,
    version: str | None,
    model_name: str | None,
    backbone: str | None,
    alias: str | None,
) -> ModelImportResult:
    family_code = _normalize_family_code(model_family)
    if not family_code:
        raise ValueError("模式A必须指定模型种类。")
    spec = FAMILY_SPECS.get(family_code)
    if spec is None:
        raise ValueError(f"暂不支持该模型种类的自动包装：{model_family}")

    template = _find_template_manifest(settings.artifacts_root, family_code, language or spec.default_language)
    manifest_data = template.manifest.model_dump(mode="python", exclude_none=True)
    final_language = _normalize_language(language or template.manifest.language or spec.default_language)
    final_dataset_name = _normalize_dataset_name(dataset_name or template.manifest.dataset_name or "dataset")
    final_version = _normalize_version(version or template.manifest.version or "v1")
    final_model_name = str(model_name or manifest_data.get("model_name") or spec.default_model_name).strip() or spec.default_model_name
    final_backbone = str(
        backbone
        or manifest_data.get("backbone")
        or spec.default_backbones.get(final_language)
        or next(iter(spec.default_backbones.values()))
    ).strip()
    final_model_id = _allocate_model_id(
        artifacts_root=settings.artifacts_root,
        family_code=family_code,
        language=final_language,
        dataset_name=final_dataset_name,
        version=final_version,
        alias=alias or archive_filename,
    )
    target_root = settings.artifacts_root / family_code / final_language / final_model_id
    target_root.parent.mkdir(parents=True, exist_ok=True)

    manifest_data["model_id"] = final_model_id
    manifest_data["model_name"] = final_model_name
    manifest_data["model_family"] = family_code
    manifest_data["language"] = final_language
    manifest_data["dataset_name"] = final_dataset_name
    manifest_data["version"] = final_version
    manifest_data["backbone"] = final_backbone

    detected_files: dict[str, str] = {}
    target_root.mkdir(parents=True, exist_ok=False)

    copied_profiles = _copy_template_directory(template.package_root, "profiles", target_root)
    copied_samples = _copy_template_directory(template.package_root, "samples", target_root)
    if not copied_samples:
        _create_placeholder_samples(target_root)

    manifest_data["profiles"] = _build_profile_mapping(target_root / "profiles")
    manifest_data["artifacts"] = {}
    manifest_data["tokenizer"] = {"type": spec.tokenizer_type}
    manifest_data["labels"] = {}

    for field_name, rule in spec.artifact_rules.items():
        source = _locate_source_entry(extracted_root, rule)
        destination = target_root / rule.target
        _copy_entry(source, destination, rule.entry_type)
        manifest_data["artifacts"][field_name] = rule.target
        detected_files[field_name] = str(source.relative_to(extracted_root))

    for field_name, rule in spec.tokenizer_rules.items():
        source = _locate_source_entry(extracted_root, rule)
        destination = target_root / rule.target
        _copy_entry(source, destination, rule.entry_type)
        manifest_data["tokenizer"][field_name] = rule.target
        detected_files[f"tokenizer.{field_name}"] = str(source.relative_to(extracted_root))

    for field_name, rule in spec.label_rules.items():
        source = _locate_source_entry(extracted_root, rule)
        destination = target_root / rule.target
        _copy_entry(source, destination, rule.entry_type)
        manifest_data["labels"][field_name] = rule.target
        detected_files[f"labels.{field_name}"] = str(source.relative_to(extracted_root))

    runtime_notes = [str(item).strip() for item in manifest_data.get("runtime_notes", []) if str(item).strip()]
    runtime_notes.append(f"Imported via LuckyModels 模式A，源压缩包：{archive_filename}。")
    manifest_data["runtime_notes"] = _unique_text_items(runtime_notes)

    if not (target_root / "README.md").exists():
        _write_readme(
            target_root=target_root,
            model_id=final_model_id,
            model_name=final_model_name,
            import_mode=IMPORT_MODE_AUTO_PACKAGE,
            archive_filename=archive_filename,
        )

    _write_yaml(target_root / "manifest.yaml", manifest_data)
    _write_package_metadata(
        target_root=target_root,
        origin=source_type,
        import_mode=IMPORT_MODE_AUTO_PACKAGE,
        archive_filename=archive_filename,
    )
    loaded = load_manifest(target_root / "manifest.yaml")
    validate_loaded_manifest(loaded)
    return ModelImportResult(
        model_id=final_model_id,
        model_name=final_model_name,
        model_family=family_code,
        language=final_language,
        dataset_name=final_dataset_name,
        version=final_version,
        import_mode=IMPORT_MODE_AUTO_PACKAGE,
        package_root=str(target_root),
        archive_filename=archive_filename,
        detected_files=detected_files,
        warnings=[] if copied_profiles else ["未复制到模板配置，平台已自动生成 profiles 映射。"],
    )


def _direct_import_standard_package(
    *,
    settings: Settings,
    extracted_root: Path,
    archive_filename: str,
    source_type: str,
    model_family: str | None,
    language: str | None,
    dataset_name: str | None,
    version: str | None,
    model_name: str | None,
    backbone: str | None,
    alias: str | None,
) -> ModelImportResult:
    manifest_path = _locate_manifest_file(extracted_root)
    loaded = load_manifest(manifest_path)
    validate_loaded_manifest(loaded)

    family_code = _normalize_family_code(model_family or loaded.manifest.model_family)
    if family_code != _normalize_family_code(loaded.manifest.model_family):
        raise ValueError("上传时所选模型种类与标准包 manifest 中声明的模型种类不一致。")

    final_language = _normalize_language(language or loaded.manifest.language)
    final_dataset_name = _normalize_dataset_name(dataset_name or loaded.manifest.dataset_name or "dataset")
    final_version = _normalize_version(version or loaded.manifest.version or "v1")
    final_model_name = str(model_name or loaded.manifest.model_name).strip() or loaded.manifest.model_name
    final_backbone = str(backbone or loaded.manifest.backbone).strip() or loaded.manifest.backbone
    final_model_id = _allocate_model_id(
        artifacts_root=settings.artifacts_root,
        family_code=family_code,
        language=final_language,
        dataset_name=final_dataset_name,
        version=final_version,
        alias=alias or archive_filename or loaded.manifest.model_id,
    )

    package_root = Path(loaded.package_root)
    target_root = settings.artifacts_root / family_code / final_language / final_model_id
    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_root, target_root)

    manifest_data = loaded.manifest.model_dump(mode="python", exclude_none=True)
    manifest_data["model_id"] = final_model_id
    manifest_data["model_name"] = final_model_name
    manifest_data["model_family"] = family_code
    manifest_data["language"] = final_language
    manifest_data["dataset_name"] = final_dataset_name
    manifest_data["version"] = final_version
    manifest_data["backbone"] = final_backbone
    runtime_notes = [str(item).strip() for item in manifest_data.get("runtime_notes", []) if str(item).strip()]
    runtime_notes.append(f"Imported via LuckyModels 模式B，源压缩包：{archive_filename}。")
    manifest_data["runtime_notes"] = _unique_text_items(runtime_notes)
    _write_yaml(target_root / "manifest.yaml", manifest_data)
    _write_package_metadata(
        target_root=target_root,
        origin=source_type,
        import_mode=IMPORT_MODE_DIRECT_IMPORT,
        archive_filename=archive_filename,
    )

    direct_loaded = load_manifest(target_root / "manifest.yaml")
    validate_loaded_manifest(direct_loaded)
    return ModelImportResult(
        model_id=final_model_id,
        model_name=final_model_name,
        model_family=family_code,
        language=final_language,
        dataset_name=final_dataset_name,
        version=final_version,
        import_mode=IMPORT_MODE_DIRECT_IMPORT,
        package_root=str(target_root),
        archive_filename=archive_filename,
        detected_files={"manifest": str(manifest_path.relative_to(extracted_root))},
        warnings=[],
    )


def _find_template_manifest(artifacts_root: Path, family_code: str, language: str) -> LoadedManifest:
    manifests = sorted(artifacts_root.rglob("manifest.yaml"))
    candidates: list[LoadedManifest] = []
    for manifest_path in manifests:
        loaded = load_manifest(manifest_path)
        if _normalize_family_code(loaded.manifest.model_family) != family_code:
            continue
        candidates.append(loaded)
    if not candidates:
        raise FileNotFoundError(f"未找到可复用的模板模型包：{family_code}")

    normalized_language = _normalize_language(language)
    exact_language = [item for item in candidates if _normalize_language(item.manifest.language) == normalized_language]
    pool = exact_language or candidates
    pool.sort(key=lambda item: (len(Path(item.package_root).parts), item.manifest.model_id))
    return pool[0]


def _locate_manifest_file(extracted_root: Path) -> Path:
    manifest_paths = [
        path for path in extracted_root.rglob("manifest.yaml")
        if "__MACOSX" not in path.parts
    ]
    if not manifest_paths:
        raise FileNotFoundError("未在上传压缩包中找到 manifest.yaml，无法按模式B导入。")
    if len(manifest_paths) > 1:
        manifest_paths.sort(key=lambda path: (len(path.parts), str(path)))
    return manifest_paths[0]


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    suffixes = [suffix.lower() for suffix in archive_path.suffixes]
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _validate_archive_member(member.filename)
            archive.extractall(target_dir)
        return
    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-2:] == [".tar", ".bz2"] or suffixes[-2:] == [".tar", ".xz"] or archive_path.suffix.lower() in {".tgz", ".tar"}:
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            for member in members:
                _validate_archive_member(member.name)
            archive.extractall(target_dir)
        return
    raise ValueError("当前仅支持 zip、tar、tar.gz、tgz 等常见压缩包格式。")


def _validate_archive_member(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"压缩包中包含非法路径：{name}")


def _locate_source_entry(extracted_root: Path, rule: _PathRule) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in extracted_root.rglob("*"):
        if "__MACOSX" in path.parts:
            continue
        if rule.entry_type == "file" and not path.is_file():
            continue
        if rule.entry_type == "dir" and not path.is_dir():
            continue
        relative = path.relative_to(extracted_root).as_posix().lower()
        basename = path.name.lower()
        for pattern in rule.patterns:
            lower_pattern = pattern.lower()
            if fnmatch.fnmatch(relative, lower_pattern) or fnmatch.fnmatch(basename, lower_pattern):
                exact_bonus = 0 if basename == lower_pattern else 1
                candidates.append((exact_bonus + relative.count("/"), path))
                break
    if not candidates:
        raise FileNotFoundError(
            f"未在上传压缩包中找到 {rule.target} 所需的文件/目录，匹配规则：{', '.join(rule.patterns)}"
        )
    candidates.sort(key=lambda item: (item[0], len(item[1].as_posix()), item[1].as_posix()))
    return candidates[0][1]


def _copy_entry(source: Path, destination: Path, entry_type: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if entry_type == "dir":
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return
    shutil.copy2(source, destination)


def _copy_template_directory(template_root: str, relative_dir: str, target_root: Path) -> bool:
    source_dir = Path(template_root) / relative_dir
    if not source_dir.exists():
        return False
    destination_dir = target_root / relative_dir
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    return True


def _create_placeholder_samples(target_root: Path) -> None:
    samples_dir = target_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "predict.json").write_text(
        json.dumps({"text": "Please replace this sample with a dataset-specific sentence."}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (samples_dir / "expected_output.json").write_text(
        json.dumps({"triples": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_readme(
    *,
    target_root: Path,
    model_id: str,
    model_name: str,
    import_mode: str,
    archive_filename: str,
) -> None:
    readme = "\n".join(
        [
            f"# {model_name}",
            "",
            f"- model_id: `{model_id}`",
            f"- import_mode: `{import_mode}`",
            f"- source_archive: `{archive_filename}`",
            "",
            "This package was imported by LuckyModels and can be deployed through the control plane.",
            "",
        ]
    )
    (target_root / "README.md").write_text(readme, encoding="utf-8")


def read_package_metadata(package_root: str | Path) -> dict[str, Any]:
    package_root = Path(package_root)
    metadata_path = package_root / PACKAGE_METADATA_FILENAME
    if not metadata_path.exists():
        return {
            "origin": "builtin",
            "origin_label": _origin_label("builtin"),
            "deletable": False,
        }
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except Exception:
        return {
            "origin": "builtin",
            "origin_label": _origin_label("builtin"),
            "deletable": False,
        }

    origin = str(payload.get("origin") or "builtin").strip().lower() or "builtin"
    normalized = {
        **payload,
        "origin": origin,
        "origin_label": _origin_label(origin),
        "deletable": bool(payload.get("deletable")) if origin == "user_upload" else False,
    }
    return normalized


def _write_package_metadata(
    *,
    target_root: Path,
    origin: str,
    import_mode: str,
    archive_filename: str,
) -> None:
    normalized_origin = str(origin or "builtin").strip().lower() or "builtin"
    payload = {
        "origin": normalized_origin,
        "origin_label": _origin_label(normalized_origin),
        "deletable": normalized_origin == "user_upload",
        "import_mode": import_mode,
        "archive_filename": archive_filename,
    }
    metadata_path = target_root / PACKAGE_METADATA_FILENAME
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _origin_label(origin: str) -> str:
    return "用户上传" if str(origin or "").strip().lower() == "user_upload" else "本机自带"


def _build_profile_mapping(profiles_dir: Path) -> dict[str, str]:
    if not profiles_dir.exists():
        return {}
    mapping: dict[str, str] = {}
    for path in sorted(profiles_dir.glob("*.yaml")):
        mapping[path.stem] = f"profiles/{path.name}"
    return mapping


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _normalize_family_code(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower().strip())


def _normalize_language(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"zh", "cn", "zh-cn", "chinese"}:
        return "zh"
    return "en" if text in {"", "en", "eng", "english"} else text


def _normalize_dataset_name(value: str) -> str:
    return _slugify(value or "dataset", default="dataset")


def _normalize_version(value: str) -> str:
    text = str(value or "v1").strip().lower()
    if not text:
        return "v1"
    return text if text.startswith("v") else f"v{text}"


def _slugify(value: str | None, *, default: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or default


def _allocate_model_id(
    *,
    artifacts_root: Path,
    family_code: str,
    language: str,
    dataset_name: str,
    version: str,
    alias: str | None,
) -> str:
    cleaned_alias = _clean_alias(alias, family_code=family_code, dataset_name=dataset_name, language=language, version=version)
    base = f"{family_code}-{language}-{dataset_name}"
    if cleaned_alias:
        base = f"{base}-{cleaned_alias}"
    candidate = f"{base}-{version}"
    taken = set()
    family_root = artifacts_root / family_code / language
    if family_root.exists():
        for manifest_path in family_root.rglob("manifest.yaml"):
            try:
                taken.add(load_manifest(manifest_path).manifest.model_id)
            except Exception:
                continue
    if candidate not in taken:
        return candidate
    index = 2
    while f"{candidate}-r{index}" in taken:
        index += 1
    return f"{candidate}-r{index}"


def _clean_alias(alias: str | None, *, family_code: str, dataset_name: str, language: str, version: str) -> str:
    text = _slugify(alias, default="")
    if not text:
        return ""
    removable = {
        family_code,
        language,
        dataset_name,
        version,
        "model",
        "onnx",
        "zip",
        "package",
        "artifact",
        "artifacts",
        "luckymodels",
    }
    parts = [part for part in text.split("-") if part and part not in removable]
    if not parts:
        return ""
    return "-".join(parts[:4])


def _unique_text_items(items: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results
