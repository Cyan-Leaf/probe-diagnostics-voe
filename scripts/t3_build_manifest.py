"""Build the deterministic clip manifest for T3/T4 from LikePhys-Benchmark.

Label protocol (fixed here *before* any result is computed; see
results/T3_null_floor/protocol_as_executed.md):

  group=physics   label 0 = `valid_*`            (physically plausible)
                  label 1 = every violation kind (physically implausible)
  group=temporal  `temporal_disorder_*`  -> excluded from the main probe, kept as the
                  frame-order control (cf. Punzo et al. 2606.09646 temporal controls)
  group=appearance `color_change_*`      -> excluded from the main probe, kept as the
                  appearance-only control candidate (SPEC E7)

Pairing key for paired statistics: (scenario, subgroup).

Run: uv run --no-project python scripts/t3_build_manifest.py
"""

import glob
import json
import os
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "T3_null_floor"

TEMPORAL_KINDS = {"temporal_disorder"}
APPEARANCE_KINDS = {"color_change"}


def main() -> None:
    hub = Path(os.environ["HF_HOME"]) / "hub"
    roots = sorted(glob.glob(str(hub / "datasets--JianhaoDYDY--LikePhys-Benchmark" / "snapshots" / "*" / "data")))
    assert roots, "LikePhys not downloaded"
    root = Path(roots[0])

    clips = []
    for p in sorted(root.glob("*/*/*.mp4")):
        scenario, subgroup, fname = p.parts[-3], p.parts[-2], p.parts[-1]
        kind = re.sub(r"_\d+\.mp4$", "", fname)
        rep = re.search(r"_(\d+)\.mp4$", fname).group(1)
        if kind in TEMPORAL_KINDS:
            group, label = "temporal", None
        elif kind in APPEARANCE_KINDS:
            group, label = "appearance", None
        elif kind == "valid":
            group, label = "physics", 0
        else:
            group, label = "physics", 1
        clips.append(
            {
                "clip_id": f"{scenario}/{subgroup}/{kind}_{rep}",
                "path": str(p),
                "scenario": scenario.replace("_videos", ""),
                "subgroup": subgroup,
                "kind": kind,
                "rep": rep,
                "group": group,
                "label": label,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False))

    from collections import Counter

    summary = {
        "n_clips": len(clips),
        "by_group": dict(Counter(c["group"] for c in clips)),
        "by_label_physics": dict(Counter(c["label"] for c in clips if c["group"] == "physics")),
        "n_scenarios": len({c["scenario"] for c in clips}),
        "per_scenario": {
            s: dict(Counter(c["group"] for c in clips if c["scenario"] == s))
            for s in sorted({c["scenario"] for c in clips})
        },
        "dataset_root": str(root),
    }
    (OUT / "manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
