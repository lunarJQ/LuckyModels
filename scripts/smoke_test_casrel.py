from __future__ import annotations

import json

from luckymodels_core.adapters.casrel import CasRelAdapter
from luckymodels_core.manifest import load_manifest, load_profile


def main() -> None:
    manifest = load_manifest(
        "/home/liujingqian/LuckyModels/artifacts/models/casrel/en/v1/manifest.yaml"
    )
    profile = load_profile(
        "/home/liujingqian/LuckyModels/artifacts/models/casrel/en/v1/profiles/default.yaml"
    )
    adapter = CasRelAdapter(manifest, profile)
    adapter.load()

    print("Health:")
    print(json.dumps(adapter.health(), ensure_ascii=False, indent=2))
    print()

    print("Metadata:")
    print(json.dumps(adapter.metadata(), ensure_ascii=False, indent=2))
    print()

    samples = [
        "Tim Brooke-Taylor was the star of Bananaman , an STV series first aired on 10/03/1983 and created by Steve Bright .",
        "Baymax is a character in Big Hero 6 that stars Damon Wayans Jr . The character was created by the American , Duncan Rouleau .",
    ]
    for text in samples:
        result = adapter.predict(
            text,
            {
                "subject_threshold": 0.001,
                "object_threshold": 0.001,
                "debug": True,
            },
        )
        print(f"Predict Result: {text}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
