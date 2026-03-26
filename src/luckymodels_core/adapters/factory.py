from __future__ import annotations

from .casrel import CasRelAdapter


ADAPTERS = {
    "casrel_adapter": CasRelAdapter,
}


def build_adapter(adapter_name: str):
    if adapter_name not in ADAPTERS:
        raise KeyError(f"Unknown adapter: {adapter_name}")
    return ADAPTERS[adapter_name]
