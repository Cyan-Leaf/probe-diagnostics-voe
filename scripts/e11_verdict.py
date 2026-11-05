"""E11 verdict from the already-computed cells (E11_PREREG.md §3).

Separate from e11_readout.py so the 65-minute matrix is never recomputed just to re-derive the
verdict. The rule is the one fixed before the run: a readout more than 1 AUC point below linear
is rejected as under-fitting and cannot speak to capacity; among the rest, the verdict flips
only if the STRONGEST arm lands on the opposite side of the 50% bar from the linear baseline.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "E11_readout"
MARGIN = 0.01
PROBE_SD_PP = 2.51
THRESHOLD = 50.0


def main():
    d = json.loads((OUT / "E11_cells.json").read_text())

    # V-JEPA's LikePhys D1 is NOT a single-seed quantity: E15/E16 showed it moves 43.8-54.0%
    # across probe partitions. Its cells are therefore read from the six-partition means, not
    # from the seed-1000 column, otherwise the "linear baseline" here (51.1%) would not even
    # reproduce the delivered 49.4% and the comparison would be meaningless.
    ps_path = OUT / "E11_vjepa_probe_seeds.json"
    probe_sweep = json.loads(ps_path.read_text()) if ps_path.exists() else {}
    per_cell = {}
    for key, cell in d["cells"].items():
        lin_auc = cell["readouts"]["linear"]["auc_real"]
        eligible = {r: v for r, v in cell["readouts"].items()
                    if v["auc_real"] - lin_auc >= -MARGIN}
        strongest = max(eligible, key=lambda r: eligible[r]["auc_real"])
        d1_lin = cell["readouts"]["linear"]["d1_pct"]
        d1_strong = eligible[strongest]["d1_pct"]
        multi = None
        if key.startswith("vjepa") and probe_sweep:
            multi = {r: {"d1_mean_pct": probe_sweep[r]["d1_mean_pct"],
                         "d1_sd_pct": probe_sweep[r]["d1_sd_pct"],
                         "n_below_50": probe_sweep[r]["n_below_50"],
                         "n_seeds": probe_sweep[r]["n_seeds"]}
                     for r in eligible if r in probe_sweep}
            if "linear" in multi and strongest in multi:
                d1_lin = multi["linear"]["d1_mean_pct"]
                d1_strong = multi[strongest]["d1_mean_pct"]
        same_side = (d1_lin >= THRESHOLD) == (d1_strong >= THRESHOLD)
        per_cell[key] = {
            "eligible": sorted(eligible),
            "rejected": sorted(set(cell["readouts"]) - set(eligible)),
            "d1_by_eligible": {r: v["d1_pct"] for r, v in eligible.items()},
            "rejected_d1": {r: cell["readouts"][r]["d1_pct"]
                            for r in set(cell["readouts"]) - set(eligible)},
            "strongest_eligible": strongest,
            "d1_linear": d1_lin, "d1_strongest": d1_strong,
            "delta_d1_pp": d1_strong - d1_lin,
            "delta_within_probe_noise": abs(d1_strong - d1_lin) < PROBE_SD_PP,
            "verdict": "robust_to_readout" if same_side else "readout_dependent",
            "multi_seed": multi,
            "d1_source": "six-probe-partition means" if multi else "three feature seeds, seed 1000",
            "linear_reproduces_e8": abs(d1_lin - cell["e8_delivered_d1_pct"]) < 0.05,
            "e8_delivered_d1_pct": cell["e8_delivered_d1_pct"]}
    all_robust = all(v["verdict"] == "robust_to_readout" for v in per_cell.values())
    verdict = {
        "cells": per_cell,
        "q1": "robust_to_readout" if all_robust else "readout_dependent",
        "q1_note": ("D1's side of the 50% bar is unchanged by readout capacity in all four cells"
                    if all_robust else "at least one cell flips under a stronger readout"),
        "q2_rejected": {k: v["rejected"] for k, v in per_cell.items() if v["rejected"]},
        "q2_note": ("the MLP is rejected wherever it appears: it scores 3.2-12.7 AUC points BELOW "
                    "the linear baseline, so its dramatic-looking D1 (77-87%) measures a "
                    "degenerate random-init arm, not capacity"),
        "q3_note": ("all deltas are below the 2.51pp probe-partition noise except the LikePhys "
                    "cells' 5-8pp shifts, which still do not cross the bar in the low cells"),
        "linear_reproduces_e8_all": all(v["linear_reproduces_e8"] for v in per_cell.values()),
        # Derived deltas are stored, not just narrated in the report: audit_reports.py traces
        # every number in E11_REPORT.md to a JSON, and the Nystrom-vs-linear / MLP-vs-linear AUC
        # gaps were computed inside the report prose and existed nowhere on disk.
        # The report's "a single feature seed gives 99.5% against E8's 91.2%" note is a computed
        # quantity on the delivered per-seed AUCs, so it is stored rather than left in prose.
        "linear_random_per_feature_seed_d1_pct": {
            k: [float((a - b) / (a - 0.5) * 100) if a > 0.5 else float("nan")
                for a, b in zip(cell["readouts"]["linear"]["per_feature_seed_auc_real"],
                                cell["readouts"]["linear"]["per_feature_seed_auc_random"])]
            for k, cell in d["cells"].items() if "linear" in cell["readouts"]},
        # D1's denominator (AUC_real - 0.5) per arm: the report's point is that the MLP's
        # denominator collapses while its numerator does not, so the denominator is stored.
        "d1_denominator": {
            k: {r: v["auc_real"] - 0.5 for r, v in cell["readouts"].items()}
            for k, cell in d["cells"].items()},
        "auc_delta_vs_linear": {
            k: {r: v["auc_real"] - cell["readouts"]["linear"]["auc_real"]
                for r, v in cell["readouts"].items() if r != "linear"}
            for k, cell in d["cells"].items()},
    }
    (OUT / "E11_verdict.json").write_text(json.dumps({"verdict": verdict}, indent=1))
    print(f"linear reproduces E8 in all cells: {verdict['linear_reproduces_e8_all']}")
    for k, v in per_cell.items():
        print(f"  {k:22s} {v['verdict']:18s} linear {v['d1_linear']:6.2f}% -> "
              f"{v['strongest_eligible']:8s} {v['d1_strongest']:6.2f}%  Δ{v['delta_d1_pp']:+6.2f}pp"
              f"  (rejected: {v['rejected']})")
    print(f"Q1: {verdict['q1']}")


if __name__ == "__main__":
    main()
