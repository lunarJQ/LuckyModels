from __future__ import annotations

ADAPTER_IMPORTS = {
    "casrel_adapter": ("luckymodels_core.adapters.casrel", "CasRelAdapter"),
    "copymtl_adapter": ("luckymodels_core.adapters.copymtl", "CopyMTLAdapter"),
    "graphrel_adapter": ("luckymodels_core.adapters.graphrel", "GraphRelAdapter"),
    "onerel_adapter": ("luckymodels_core.adapters.onerel", "OneRelAdapter"),
    "pure_adapter": ("luckymodels_core.adapters.pure", "PUREAdapter"),
    "prgc_adapter": ("luckymodels_core.adapters.prgc", "PRGCAdapter"),
    "spert_adapter": ("luckymodels_core.adapters.spert", "SpERTAdapter"),
    "metadata_only_adapter": ("luckymodels_core.adapters.metadata_only", "MetadataOnlyAdapter"),
    "tplinker_adapter": ("luckymodels_core.adapters.tplinker", "TPLinkerAdapter"),
}


def build_adapter(adapter_name: str):
    if adapter_name not in ADAPTER_IMPORTS:
        raise KeyError(f"Unknown adapter: {adapter_name}")
    module_name, class_name = ADAPTER_IMPORTS[adapter_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def has_adapter(adapter_name: str) -> bool:
    return adapter_name in ADAPTER_IMPORTS
