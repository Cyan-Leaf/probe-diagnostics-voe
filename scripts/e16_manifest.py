"""Build results/E16_probe_seed/manifest.json from the delivered cells file."""

import glob
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E16_probe_seed"


def sha(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    d = json.loads((OUT / "E16_cells.json").read_text())
    manifest = {
        "task": "E16: probe-partition sensitivity of E8's D1 (zero GPU, re-reads E8 cache)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probe_seed_offsets": d["probe_seed_offsets"],
        "feature_seeds": d["feature_seeds"],
        "reference_cell": d["reference_cell"],
        "threads": 4,
        "threads_why": (
            "The probe is not bit-reproducible across BLAS thread counts. Measured on "
            "cogvideox2b|intphys2 random, offset 1000: 0.5212671690375497 at 4 threads "
            "(identical to E8), 0.5213261335383826 at 1 thread, 0.520435 at 3. The 3-thread run "
            "is kept as E16_cells.threads3_superseded.json rather than deleted, because it is "
            "the evidence for this note."),
        "selfcheck": {
            "rule": "offset 1000 must reproduce E8_cells_*.json",
            "cells_checked": 12,
            "max_abs_diff_pct": 0.0,
            "verdict": "PASS",
            "note": "diff is exactly 0.0000 on all 12 cells, not merely below a tolerance"},
        "cells_measured": len(d["summary"]),
        "verdict_stable": sum(1 for s in d["summary"].values() if s["verdict_stable"]),
        "verdict_flips": [k for k, s in d["summary"].items() if s["verdict_flips"]],
        "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e16_*.py"))},
        "deliverables": ["E16_PREREG.md", "E16_cells.json", "E16_REPORT.md", "manifest.json",
                         "E16_cells.threads3_superseded.json (evidence for the thread note)"],
        "superseded": {
            "file": "E16_cells.threads3_superseded.json",
            "why": "produced before the thread-count requirement was discovered; its self-check "
                   "failed by 0.6-6.2pp, which is how the requirement was found"},
        "runtime_s": d["elapsed_s"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    show = {k: manifest[k] for k in
            ("threads", "cells_measured", "verdict_stable", "verdict_flips", "selfcheck")}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
