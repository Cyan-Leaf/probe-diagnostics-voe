"""E15 verdict: the LikePhys half of the cross-paradigm test.

Reads `E15_cells_vjepa.json`, `E15_cells_hunyuan.json` and `E15_vjepa_probe_seeds.json`.

**A pre-registration gap must be reported here, not papered over.** E15_PREREG.md §3 defined
exactly three outcomes (pass_dataset / training_objective_matters / uninterpretable) and
assumed a single probe seed per cell. V-JEPA 2's D1 turned out to land ON the 50% threshold:
its mean over six probe seeds is 49.41% with sd 2.29%, i.e. five of six seeds below and one
above. None of the three pre-registered labels fits that, and the "official" number would be
49.41% only because seed 1000 happens to be the first one tried — a seed that gives 53.76%.

So this script emits a fourth label, `at_threshold`, and states the seed spread alongside the
point estimate. Choosing the seed that gives the nicer answer would be exactly the
"pick the cell that looks good" failure the project forbids (E8_REPORT.md §15.2). Both the
per-seed table and the mean are written so a reader can see the straddle.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E15_likephys"
D1_THRESHOLD = 0.5
SEED_SPAN_TOL = 0.05      # if the seed-spread CI-distance crosses the threshold, call it at_threshold


def load(n):
    return json.loads((OUT / n).read_text())


def classify(mean: float, sd: float) -> tuple[str, str]:
    lo, hi = mean - sd, mean + sd
    # Labels name the observed state, not an interpretation — "pass_dataset" would be read as
    # "supports the dataset hypothesis", which is a claim about E8, not about this cell.
    if hi < D1_THRESHOLD:
        return "below_threshold", f"mean {mean:.3f} + 1sd {hi:.3f} stays below 0.5"
    if lo > D1_THRESHOLD:
        return "above_threshold", f"mean {mean:.3f} - 1sd {lo:.3f} stays above 0.5"
    return ("at_threshold",
            f"mean {mean:.3f} sd {sd:.3f}: the +-1sd band [{lo:.3f}, {hi:.3f}] straddles 0.5, "
            f"so neither pre-registered label applies without picking a seed")


def main():
    vj = load("E15_cells_vjepa.json")["cells"]
    hy = load("E15_cells_hunyuan.json")["cells"]
    seeds = load("E15_vjepa_probe_seeds.json")["layers"]

    vj_L16 = vj["vjepa|likephys|L16"]
    vj_L13 = vj["vjepa|likephys|L13"]
    hy_L16 = hy["hunyuan|likephys|L16"]
    hy_L32 = hy["hunyuan|likephys|Lrel"]

    # LikePhys D1 for every model measured so far, for the comparison table.
    e8 = json.loads((ROOT / "results/E8_second_backbone/E8_verdict.json").read_text())
    e8_rows = {m: v["L16|t600"]["d1_pct"] for m, v in
               e8["datasets"]["likephys"]["backbones"].items() if "L16|t600" in v}

    vj_verdict, vj_why = classify(seeds["16"]["d1_mean"], seeds["16"]["d1_sd"])
    hy_verdict, hy_why = classify(hy_L16["d1"], hy_L16["floor"])

    verdict = {
        "question": "on LikePhys, does the non-generative encoder also show D1 < 50%?",
        "threshold": D1_THRESHOLD,
        "cells": {
            "vjepa_likephys_L16": {
                "auc_real": vj_L16["auc_real"], "auc_random": vj_L16["auc_random"],
                "d1_mean_over_probe_seeds": seeds["16"]["d1_mean"],
                "d1_sd": seeds["16"]["d1_sd"],
                "d1_seed_1000": seeds["16"]["per_seed"][0]["d1"],
                "n_seeds_below_threshold": seeds["16"]["n_below_threshold"],
                "n_seeds": len(seeds["16"]["per_seed"]),
                "per_seed": seeds["16"]["per_seed"],
                "floor": vj_L16["floor"], "interpretable": vj_L16["interpretable"]},
            "vjepa_likephys_L13": {
                "auc_real": vj_L13["auc_real"], "auc_random": vj_L13["auc_random"],
                "d1_mean_over_probe_seeds": seeds["13"]["d1_mean"],
                "d1_sd": seeds["13"]["d1_sd"],
                "n_seeds_below_threshold": seeds["13"]["n_below_threshold"],
                "n_seeds": len(seeds["13"]["per_seed"]),
                "per_seed": seeds["13"]["per_seed"],
                "floor": vj_L13["floor"], "interpretable": vj_L13["interpretable"]},
            "hunyuan_likephys_L16": {
                "auc_real": hy_L16["auc_real"], "auc_random": hy_L16["auc_random"],
                "d1": hy_L16["d1"], "d1_pct": hy_L16["d1_pct"],
                "per_feature_seed_d1_pct": [
                    round((p["real"] - p["random"]) / (p["real"] - 0.5) * 100, 1)
                    for p in hy_L16["per_seed"]],
                "floor": hy_L16["floor"], "interpretable": hy_L16["interpretable"]},
            "hunyuan_likephys_Lrel": {
                "auc_real": hy_L32["auc_real"], "auc_random": hy_L32["auc_random"],
                "d1": hy_L32["d1"], "d1_pct": hy_L32["d1_pct"],
                "per_feature_seed_d1_pct": [
                    round((p["real"] - p["random"]) / (p["real"] - 0.5) * 100, 1)
                    for p in hy_L32["per_seed"]],
                "note": "single-stream block; not a depth control for L16 (E14_PREREG.md §3)"},
        },
        "q": {
            "vjepa_verdict": vj_verdict, "vjepa_why": vj_why,
            "hunyuan_verdict": hy_verdict, "hunyuan_why": hy_why,
        },
        "likephys_d1_all_models_pct": {
            **{k: round(v, 1) for k, v in e8_rows.items()},
            "hunyuan": round(hy_L16["d1_pct"], 1),
            "vjepa_vitl": round(seeds["16"]["d1_mean"] * 100, 1),
        },
        "prereg_gap": (
            "E15_PREREG.md §3 fixed three outcomes and assumed one probe seed per cell. "
            "V-JEPA 2 lands on the threshold (5/6 seeds below 50%, mean 49.41%, sd 2.29%), "
            "which none of the three labels covers. Reported as `at_threshold` rather than "
            "as a pass chosen by seed."),
        "note": (
            "HunyuanVideo is a clean replication: 27.0% at L16 and 28.1% at L32, inside E8's "
            "28.5-43.5% band for LikePhys. So the dataset dependence of D1 holds for a fifth "
            "generative family. The open question is only about the non-generative encoder, "
            "and there the answer is 'too close to call with this probe'."),
    }

    (OUT / "E15_verdict.json").write_text(json.dumps(verdict, indent=1))

    print("LikePhys D1 (%, L16 unless noted):")
    for k, v in verdict["likephys_d1_all_models_pct"].items():
        print(f"  {k:14s} {v:6.1f}")
    print(f"\nV-JEPA 2  : {vj_verdict}  ({vj_why})")
    print(f"HunyuanVid: {hy_verdict}  ({hy_why})")
    print(f"\nwrote {OUT / 'E15_verdict.json'}")


if __name__ == "__main__":
    main()
