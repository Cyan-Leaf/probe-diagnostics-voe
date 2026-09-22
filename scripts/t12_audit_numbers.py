"""T12.1(a): trace every number in EXPERIMENTS.md back to a field path in the result JSONs.

Method matters here.  The audit must read the *source* and compare against the manuscript,
not read the manuscript and go looking for confirmation -- the second direction cannot
detect a number that was mistyped in a way that still looks plausible.

So: each claim below names a file and a field path, this script resolves that path in the
actual JSON, and then checks the manuscript contains the resolved value.  A claim whose
path does not resolve, or whose resolved value is absent from the text, is reported.

Run: python scripts/t12_audit_numbers.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "results" / "T11_writing" / "EXPERIMENTS.md"
OUT = ROOT / "results" / "T12_audit"


def get(obj, path):
    """Resolve 'a.b[0].c' against nested json. Returns (ok, value)."""
    cur = obj
    for tok in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        try:
            if tok.startswith("["):
                cur = cur[int(tok[1:-1])]
            else:
                cur = cur[tok]
        except (KeyError, IndexError, TypeError):
            return False, None
    return True, cur


def fmt(v, dp=None, pct=False, sign=False):
    if v is None:
        return None
    if pct:
        v = v * 100
    if dp is None:
        return str(v)
    s = f"{v:+.{dp}f}" if sign else f"{v:.{dp}f}"
    return s


# section, claim, file, field path, formatting, what the manuscript should contain
CLAIMS = [
    # ---- 5.3 D1 -------------------------------------------------------------------
    ("5.3", "best real AUC", "T4_caliper/E1_summary.json", "best_real.auc_mean", dict(dp=4)),
    ("5.3", "best real sd", "T4_caliper/E1_summary.json", "best_real.auc_std", dict(dp=4)),
    ("5.3", "random lower bound", "T4_caliper/E1_summary.json",
     "lower_bounds.random_dit(best layer).auc_mean", dict(dp=4)),
    ("5.3", "random lb sd", "T4_caliper/E1_summary.json",
     "lower_bounds.random_dit(best layer).auc_std", dict(dp=4)),
    ("5.3", "vae latent", "T4_caliper/E1_summary.json",
     "lower_bounds.vae_latent.auc_mean", dict(dp=4)),
    ("5.3", "raw pixel", "T4_caliper/E1_summary.json",
     "lower_bounds.raw_pixel.auc_mean", dict(dp=4)),
    ("5.3", "MDL trained compression", "T10_validity/T10_5_mdl.json",
     "comparison.compression_real", dict(dp=3)),
    ("5.3", "MDL random compression", "T10_validity/T10_5_mdl.json",
     "comparison.compression_random", dict(dp=3)),
    ("5.3", "MDL training share", "T10_validity/T10_5_mdl.json",
     "comparison.mdl_based_training_share", dict(dp=1, pct=True)),
    ("5.3", "AUC training share", "T10_validity/T10_5_mdl.json",
     "comparison.auc_based_D1_share", dict(dp=1, pct=True)),

    # ---- 5.4 D2 -------------------------------------------------------------------
    ("5.4", "LikePhys appearance t600", "E3_readout_grid/E3_summary.json",
     "valid_vs_color_change.600.real_dit.auc_mean", dict(dp=4)),
    ("5.4", "LikePhys physics t600", "E3_readout_grid/E3_summary.json",
     "valid_vs_violation.600.real_dit.auc_mean", dict(dp=4)),
    ("5.4", "LikePhys physics t200", "E3_readout_grid/E3_summary.json",
     "valid_vs_violation.200.real_dit.auc_mean", dict(dp=4)),
    ("5.4", "LikePhys physics t950", "E3_readout_grid/E3_summary.json",
     "valid_vs_violation.950.real_dit.auc_mean", dict(dp=4)),
    ("5.4", "IntPhys2 physics real", "E7_intphys2/E7_verdict.json",
     "D2.physics_auc_real", dict(dp=4)),
    ("5.4", "IntPhys2 appearance real", "E7_intphys2/E7_verdict.json",
     "D2.appearance_auc_real", dict(dp=4)),
    ("5.4", "IntPhys2 physics random", "E7_intphys2/E7_verdict.json",
     "D1.rows[0].random_auc", dict(dp=4)),
    ("5.4", "IntPhys2 D2 gap", "E7_intphys2/E7_verdict.json",
     "D2.gap_auc", dict(dp=4, sign=True)),
    ("5.4", "weak jitter AUC", "E7_intphys2/E7_d2_sensitivity.json",
     "verdict.weak_appearance_auc", dict(dp=4)),
    ("5.4", "weak jitter RGB delta", "E7_intphys2/E7_d2_sensitivity.json",
     "levels.cjweak.rgb_mean_abs_change", dict(dp=4)),

    # ---- 5.5 lower bounds ----------------------------------------------------------
    ("5.5", "IntPhys2 raw pixel", "E7_intphys2/E7_lower_bounds.json",
     "sources.raw_pixel.auc_mean", dict(dp=4)),
    ("5.5", "IntPhys2 vae latent", "E7_intphys2/E7_lower_bounds.json",
     "sources.vae_latent.auc_mean", dict(dp=4)),
    ("5.5", "IntPhys2 random dit", "E7_intphys2/E7_lower_bounds.json",
     "comparison.random_dit_auc", dict(dp=4)),
    ("5.5", "IntPhys2 trained dit", "E7_intphys2/E7_lower_bounds.json",
     "comparison.real_dit_auc", dict(dp=4)),

    # ---- 5.6 transfer --------------------------------------------------------------
    ("5.6", "IntPhys2 D1 pooled", "E7_intphys2/E7_verdict.json",
     "D1.pooled_pct_auc", dict(dp=1)),
    ("5.6", "solidity effect", "E7_intphys2/E7_verdict.json",
     "D4.rows[1].effect", dict(dp=4)),
    ("5.6", "camera fixed auc", "E7_intphys2/E7_verdict.json",
     "camera.fixed.real_auc", dict(dp=4)),
    ("5.6", "camera moving auc", "E7_intphys2/E7_verdict.json",
     "camera.moving.real_auc", dict(dp=4)),
    ("5.6", "n shared games", "E7_intphys2/E7_verdict.json",
     "camera.n_games_shared", dict()),

    # ---- 5.7 E4 --------------------------------------------------------------------
    ("5.7", "global mean d_err", "E4_complexity/E4_decomposition.json",
     "verdict.global_mean_d_err", dict(dp=5, sign=True)),
    ("5.7", "pooled intercept", "E4_complexity/E4_decomposition.json",
     "verdict.pooled_intercept", dict(dp=5, sign=True)),
    ("5.7", "within-scenario intercept", "E4_complexity/E4_decomposition.json",
     "verdict.within_scenario_weighted_intercept", dict(dp=5, sign=True)),
    ("5.7", "top2 share", "E4_complexity/E4_decomposition.json",
     "verdict.share_of_negative_effect_from_top2", dict(dp=0, pct=True)),
    ("5.7", "explained share", "E4_complexity/E4_decomposition.json",
     "verdict.share_of_negative_effect_explained_by_complexity", dict(dp=0, pct=True)),

    # ---- 5.8 E10 -------------------------------------------------------------------
    ("5.8", "S2 real mean", "E10_shadow_camera/E10_S1_S3.json",
     "S2.real.auc_mean", dict(dp=4)),
    ("5.8", "S2 CI low", "E10_shadow_camera/E10_S1_S3.json",
     "S2.real.auc_ci95[0]", dict(dp=4)),
    ("5.8", "S2 CI high", "E10_shadow_camera/E10_S1_S3.json",
     "S2.real.auc_ci95[1]", dict(dp=4)),
    # §5.8 quotes the observed null RANGE (min/max), not the CI -- checking the CI here
    # would flag a correct sentence.
    ("5.8", "null floor observed min", "E10_shadow_camera/E10_S1_S3.json",
     "S1.null_auc_min", dict(dp=2)),
    ("5.8", "null floor observed max", "E10_shadow_camera/E10_S1_S3.json",
     "S1.null_auc_max", dict(dp=2)),
    ("5.8", "null floor CI low", "E10_shadow_camera/E10_S1_S3.json",
     "S1.null_auc_ci95[0]", dict(dp=2)),
    ("5.8", "null frac above 0.60", "E10_shadow_camera/E10_S1_S3.json",
     "S1.null_frac_above_060", dict(dp=0, pct=True)),
    ("5.8", "inversion permutation p", "E10_shadow_camera/E10_S1_S3.json",
     "S2b_inversion.permutation_p_two_sided", dict(dp=2)),
    ("5.8", "inversion wilcoxon p", "E10_shadow_camera/E10_S1_S3.json",
     "S2b_inversion.wilcoxon_p", dict(dp=5)),
    ("5.8", "S4 zero gate max drop", "E10_S4/S4_verdict.json",
     "verdict.zero_level_gate.max_drop", dict(dp=4)),
    ("5.8", "S4 threshold", "E10_S4/S4_verdict.json",
     "verdict.threshold", dict(dp=4)),
    ("5.8", "S4 max real drop", "E10_S4/S4_verdict.json",
     "verdict.max_real_drop_L0_to_L3", dict(dp=4)),
    ("5.8", "S4 dprime retained", "E10_S4/S4_exploratory_dprime.json",
     "verdict.real_dprime_L3_over_L0_mean", dict(dp=3)),

    # ---- 5.9 human -----------------------------------------------------------------
    ("5.9", "human overall accuracy", "T10_validity/human_eval/human_baseline_results.json",
     "overall.human_accuracy", dict(dp=3)),
    ("5.9", "shadow_camera human acc",
     "T10_validity/human_eval/human_baseline_results.json",
     "per_scenario.shadow_camera.human_accuracy", dict(dp=3)),
    ("5.9", "shadow_camera confidence",
     "T10_validity/human_eval/human_baseline_results.json",
     "per_scenario.shadow_camera.mean_confidence", dict(dp=2)),
    ("5.9", "n pairs", "T10_validity/human_eval/human_baseline_results.json",
     "n_pairs_answered", dict()),

    # ---- 5.10 validity -------------------------------------------------------------
    ("5.10", "ablation D1", "T10_validity/T10_1_ablation.json", "ablation.D1.n_missed", dict()),
    ("5.10", "ablation D2", "T10_validity/T10_1_ablation.json", "ablation.D2.n_missed", dict()),
    ("5.10", "ablation D3", "T10_validity/T10_1_ablation.json", "ablation.D3.n_missed", dict()),
    ("5.10", "ablation D4", "T10_validity/T10_1_ablation.json", "ablation.D4.n_missed", dict()),
    ("5.10", "ablation D5", "T10_validity/T10_1_ablation.json", "ablation.D5.n_missed", dict()),
    ("5.10", "ablation D6", "T10_validity/T10_1_ablation.json", "ablation.D6.n_missed", dict()),
    ("5.10", "n units", "T10_validity/T10_1_ablation.json", "summary.n_units", dict()),
    ("5.10", "construct n correct", "T10_validity/T10_2_construct.json",
     "summary.n_correct", dict()),
    ("5.10", "C2 real auc", "T10_validity/T10_2_construct.json",
     "cases.C2.real.auc_mean", dict(dp=4)),
    ("5.10", "C2 random auc", "T10_validity/T10_2_construct.json",
     "cases.C2.random.auc_mean", dict(dp=4)),
    ("5.10", "C2 appearance auc", "T10_validity/T10_2_construct.json",
     "cases.C2.appearance.auc_mean", dict(dp=4)),
    ("5.10", "A2 degenerate D1", "T10_validity/T10_2_construct.json",
     "cases.A2.checks.D1.value", dict(dp=3)),
    ("5.10", "A2 real auc", "T10_validity/T10_2_construct.json",
     "cases.A2.real.auc_mean", dict(dp=4)),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The manuscript uses the typographic minus U+2212, not ASCII hyphen; normalise both
    # sides or every negative number reads as missing.
    text = MS.read_text().replace("\u2212", "-").replace("\u2013", "-")
    cache = {}
    rows = []

    for sec, name, fn, path, f in CLAIMS:
        p = ROOT / "results" / fn
        if fn not in cache:
            cache[fn] = json.loads(p.read_text()) if p.exists() else None
        data = cache[fn]
        if data is None:
            rows.append({"section": sec, "claim": name, "source": f"{fn}:{path}",
                         "resolved": None, "in_text": False,
                         "status": "SOURCE MISSING"})
            continue
        ok, val = get(data, path)
        if not ok:
            rows.append({"section": sec, "claim": name, "source": f"{fn}:{path}",
                         "resolved": None, "in_text": False,
                         "status": "PATH NOT FOUND"})
            continue
        s = fmt(val, **f)
        # accept the value with or without a leading +, and tolerate trailing-zero styles
        cands = {s, s.lstrip("+")}
        if f.get("dp") is not None and not f.get("pct"):
            cands.add(str(round(val, f["dp"])))
        found = any(c in text for c in cands if c)
        rows.append({"section": sec, "claim": name, "source": f"{fn}:{path}",
                     "resolved": s, "raw": val, "in_text": bool(found),
                     "status": "ok" if found else "NOT IN TEXT"})

    bad = [r for r in rows if r["status"] != "ok"]
    (OUT / "T12_number_audit.json").write_text(
        json.dumps({"n_claims": len(rows), "n_ok": len(rows) - len(bad),
                    "rows": rows}, indent=2, ensure_ascii=False))

    print(f"{len(rows)} claims checked, {len(rows)-len(bad)} ok, {len(bad)} need attention\n")
    print(f"{'sec':6s} {'claim':32s} {'resolved':>12s}  status")
    for r in rows:
        mark = "  " if r["status"] == "ok" else "<<"
        print(f"{r['section']:6s} {r['claim']:32s} {str(r['resolved']):>12s}  "
              f"{r['status']} {mark}")
    if bad:
        print("\n--- needing attention ---")
        for r in bad:
            print(f"  §{r['section']} {r['claim']}")
            print(f"      {r['source']}")
            print(f"      resolved={r['resolved']}  status={r['status']}")
    print(f"\nwrote {OUT/'T12_number_audit.json'}")


if __name__ == "__main__":
    main()
