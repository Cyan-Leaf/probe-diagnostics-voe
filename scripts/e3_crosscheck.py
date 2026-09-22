"""Cross-machine consistency check for the E3 grid completion.

The grid run recomputes denoising errors at (t, seed) pairs that the *original*
`cache/feats_real/` already contains (real backbone, t in {200,400,800,950},
seeds 1 and 2).  Those numbers are a free, zero-cost cross-machine control:

  old box (GPU with defective SM retired, results from healthy cards)
  vs new box (4x RTX 4090, all cards verified bit-exact on 2026-09-17)

If the two agree bit-for-bit, the new box reproduces the delivered E0/E1 pipeline
exactly and the new grid cells can be analysed together with the old ones.  If they
drift, the grid must NOT be merged with the existing results and this has to be
reported before any analysis (MACHINE_HANDOFF §3.5 point 3: comparability).

Run: python scripts/e3_crosscheck.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "cache" / "feats_real"
NEW = ROOT / "cache" / "feats_real_grid"
OUT = ROOT / "results" / "E3_readout_grid"

T_CHECK = ["200", "400", "800", "950"]
SEED_CHECK = ["1", "2"]


def main() -> None:
    new_files = sorted(NEW.glob("*.pt"))
    if not new_files:
        print("no files in cache/feats_real_grid yet")
        return

    rows = []
    n_cmp = 0
    n_exact = 0
    worst = {"abs": 0.0, "rel": 0.0, "where": None}

    for f in new_files:
        old_f = OLD / f.name
        if not old_f.exists():
            continue
        new = torch.load(f, map_location="cpu", weights_only=False)
        old = torch.load(old_f, map_location="cpu", weights_only=False)
        for t in T_CHECK:
            for s in SEED_CHECK:
                try:
                    a = old["errors"][t][s]["mse"]
                    b = new["errors"][t][s]["mse"]
                except KeyError:
                    continue
                n_cmp += 1
                d = abs(a - b)
                rel = d / abs(a) if a else 0.0
                if d == 0.0:
                    n_exact += 1
                if d > worst["abs"]:
                    worst = {"abs": d, "rel": rel,
                             "where": f"{f.name} t={t} s={s} old={a!r} new={b!r}"}
                rows.append({"clip": f.name, "t": int(t), "seed": int(s),
                             "old_mse": a, "new_mse": b, "abs_diff": d})

    if not n_cmp:
        print("no overlapping (t, seed) cells found yet")
        return

    diffs = np.array([r["abs_diff"] for r in rows])
    summary = {
        "n_clips_compared": len({r["clip"] for r in rows}),
        "n_cells_compared": n_cmp,
        "n_bit_exact": n_exact,
        "frac_bit_exact": n_exact / n_cmp,
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "worst_cell": worst["where"],
        "verdict": ("IDENTICAL -- new box reproduces the old pipeline bit-for-bit; "
                    "the grid may be analysed together with the delivered results"
                    if n_exact == n_cmp else
                    "DRIFT -- do NOT merge the new grid with existing results; report first"),
    }
    print(json.dumps(summary, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crosscheck_old_vs_new_box.json").write_text(
        json.dumps({"summary": summary, "cells": rows}, indent=2))
    print(f"\nwrote {OUT/'crosscheck_old_vs_new_box.json'}")

    if n_exact != n_cmp:
        sys.exit(1)


if __name__ == "__main__":
    main()
