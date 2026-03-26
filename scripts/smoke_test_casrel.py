from __future__ import annotations

import json

from luckymodels_core.adapters.casrel import CasRelAdapter
from luckymodels_core.manifest import load_manifest, load_profile


def main() -> None:
    manifest = load_manifest(
        "/home/liujingqian/LuckyModels/artifacts/models/casrel/zh/v1/manifest.yaml"
    )
    profile = load_profile(
        "/home/liujingqian/LuckyModels/artifacts/models/casrel/zh/v1/profiles/default.yaml"
    )
    adapter = CasRelAdapter(manifest, profile)
    adapter.load()

    print("Health:")
    print(json.dumps(adapter.health(), ensure_ascii=False, indent=2))
    print()

    print("Metadata:")
    print(json.dumps(adapter.metadata(), ensure_ascii=False, indent=2))
    print()

    text = "ABC设备主轴发生振动异常，需要检查主轴轴承。"
    result = adapter.predict(text)
    print("Predict Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
