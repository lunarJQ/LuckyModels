from __future__ import annotations

ADAPTER_IMPORTS = {
    "casrel_adapter": ("luckymodels_core.adapters.casrel", "CasRelAdapter"),
    "metadata_only_adapter": ("luckymodels_core.adapters.metadata_only", "MetadataOnlyAdapter"),
}


def build_adapter(adapter_name: str):
    if adapter_name not in ADAPTER_IMPORTS:
        raise KeyError(f"Unknown adapter: {adapter_name}")
    module_name, class_name = ADAPTER_IMPORTS[adapter_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def has_adapter(adapter_name: str) -> bool:
    return adapter_name in ADAPTER_IMPORTS
