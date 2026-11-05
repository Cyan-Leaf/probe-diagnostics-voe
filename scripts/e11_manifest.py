"""Build results/E11_readout/manifest.json from the delivered cells + verdict files."""

import glob
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E11_readout"


def sha(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    d = json.loads((OUT / "E11_cells.json").read_text())
    v = json.loads((OUT / "E11_verdict.json").read_text())["verdict"]
    ps_path = OUT / "E11_vjepa_probe_seeds.json"
    ps = json.loads(ps_path.read_text()) if ps_path.exists() else {}

    manifest = {
        "task": "E11: does a STRONGER non-linear readout overturn D1? (zero GPU)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threads": 4,
        "threads_why": (
            "Same requirement as E16: the probe is not bit-reproducible across BLAS thread "
            "counts. The linear arm imports e8_analyse.probe verbatim, so 4 threads is the only "
            "setting in which its output can be compared to E8's delivered numbers."),
        "readouts": {
            "linear": "e8_analyse.probe, imported verbatim (frozen reference)",
            "nystrom": "RBF Nystrom map (512 comps, gamma from the training fold) + L2 logistic",
            "mlp": "2-layer, 256 hidden, FULL input (no PCA), max_iter 5000, convergence checked"},
        "noninferiority_margin": d["noninferiority_margin"],
        "gate_why": (
            "A readout more than 0.01 AUC below the linear baseline is barred from the verdict "
            "(E11_PREREG.md Q2). Without this gate the MLP would appear to 'prove' capacity "
            "matters (D1 77-87%) when it is actually over-fitting on n~750 (all folds converged, so this "
            "is generalisation failure, not an optimisation failure): AUC_random collapses to "
            "0.50-0.60 while AUC_real barely moves, which shrinks D1's denominator and raises D1 "
            "mechanically."),
        "selfcheck": {
            "rule": "the linear arm must reproduce E8's delivered D1 on every cell",
            "all_cells_reproduce": v.get("linear_reproduces_e8_all"),
            "detail": {k: {"linear_pct": c["d1_linear"],
                           "e8_pct": c["e8_delivered_d1_pct"],
                           "d1_source": c.get("d1_source"),
                           "reproduces": c.get("linear_reproduces_e8")}
                       for k, c in v["cells"].items()},
            "verdict": "PASS" if v.get("linear_reproduces_e8_all") else "FAIL"},
        "verdict": {
            "q1": v["q1"],
            "cells": len(v["cells"]),
            "per_cell": {k: {"linear": c["d1_linear"], "strongest": c["d1_strongest"],
                             "arm": c["strongest_eligible"], "verdict": c["verdict"]}
                         for k, c in v["cells"].items()}},
        "vjepa_special_case": {
            "why": ("V-JEPA's LikePhys D1 is not a single-seed quantity: E15/E16 measured "
                    "43.8-54.0% across probe partitions. Its cells are read from the "
                    "six-partition means, so the linear baseline reproduces the delivered 49.4%."),
            "six_partition_means": v["cells"].get("vjepa|likephys", {}).get("multi_seed") or {},
            "single_seed_would_have_been": (
                "linear 51.11% at seed 1000 (E11_cells.json), which would NOT have reproduced "
                "E8's delivered 49.4%; the six-partition mean is what makes the comparison valid")},
        "scope_limits": [
            "only 4 cells, chosen because capacity can only flip a verdict near the 50% bar",
            "only ONE non-linear family (Nystrom) survives the gate; the strongest readout "
            "class (temporal attention probe, Punzo et al. 2606.09646) is NOT tested",
            "no cross-model comparison is drawn from these numbers"],
        "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e11_*.py"))},
        "deliverables": ["E11_PREREG.md", "E11_cells.json", "E11_verdict.json",
                         "E11_vjepa_probe_seeds.json", "E11_REPORT.md", "manifest.json"],
        "runtime_s": d["elapsed_s"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["selfcheck"], indent=1))
    print(json.dumps(manifest["verdict"], indent=1))


if __name__ == "__main__":
    main()
