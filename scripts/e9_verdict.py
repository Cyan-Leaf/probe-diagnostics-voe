"""E9 verdict + report inputs, from the already-computed cells.

Split out of e9_saturation.py the same way e11_verdict.py is: the 156-second run must never be
repeated just to re-derive a judgement, or someone will eventually "fix" a number by re-running
until it looks right.

Two things this file deliberately does NOT do:
  * it does not touch E8's delivered LikePhys D1 (32.9%) — E9_PREREG.md §5 forbids replacement;
  * it does not turn the L16/t=600 strict count (2) into a correction of the 5/12 in
    EXPERIMENTS.md §5.6, because that was measured on L14/t=400 whose features are not cached.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "E9_saturation"
THRESHOLD = 50.0
PROBE_SD_PP = 2.51


def main():
    d = json.loads((OUT / "E9_cells.json").read_text())
    agg, per = d["aggregates"], d["per_scenario"]
    all_c, drop = agg["all_clips"], agg["drop_saturated"]
    psm = agg["per_scenario_mean"]
    recon = agg["count_reconciliation"]

    delta = drop["d1_pct"] - all_c["d1_pct"]
    below_chance = {s: v["auc_real"] for s, v in per.items() if v.get("auc_real", 1) < 0.5}

    verdict = {
        "q1": "still_below_threshold" if drop["d1_pct"] < THRESHOLD else "crosses_threshold",
        "q1_detail": {
            "d1_all_pct": all_c["d1_pct"], "d1_drop_pct": drop["d1_pct"],
            "threshold_pct": THRESHOLD,
            "note": ("removing the ceiling scenarios moves D1 from 32.90% to 35.83%; the LikePhys "
                     "D1 verdict (< 50%) does not depend on the saturated scenarios")},
        "q2": {"delta_pp": delta, "probe_sd_pp": PROBE_SD_PP,
               "within_noise": abs(delta) < PROBE_SD_PP,
               "note": (f"+{delta:.2f} pp is {abs(delta)/PROBE_SD_PP:.2f}x the E16 probe-partition "
                        "sd, so it is a real but small movement in the expected direction — the bias "
                        "is measurable, just far too small to matter. It also cannot be attributed "
                        "cleanly: dropping clips changes the group set and therefore the folds.")},
        "q3": {"n_saturated": recon["n_strict"], "saturated": recon["strict_both_arms_1.000"],
               "n_clips_dropped": drop["n_dropped"], "n_clips_total": all_c["n"],
               "fraction_dropped": drop["n_dropped"] / all_c["n"],
               "note": ("16.0% of clips dropped; under the prereg's 30% concern line, so the kept "
                        "estimate is not sample-starved")},
        "count_discrepancy": {
            "claim": recon["claim_under_test"],
            "measured_at_L16_t600": recon["n_zero_gap"],
            "status": "not_reproduced_and_not_refuted",
            "why": recon["note"]},
        "below_chance_scenarios": below_chance,
        # The AUC gaps of the "real pinned at 1.000 but not saturated" scenarios. The report
        # quotes their range, so the range is stored here rather than computed in prose.
        # real - random for the scenarios whose real arm is pinned at 1.000 but whose random arm
        # is not, i.e. NOT saturated under the preregistered rule yet carrying almost no
        # information about training. A subset of these (gap < 0.02) is what the report calls
        # "effectively uninformative"; the rest (flag, pendulum, river) really do differ.
        "near_saturated_gaps": {
            s: v["auc_real"] - v["auc_random"] for s, v in per.items()
            if v.get("auc_real", 0) >= 1.0 - 1e-9 and not v.get("saturated")},
        "effectively_uninformative": sorted(
            s for s, v in per.items()
            if v.get("auc_real", 0) >= 1.0 - 1e-9 and not v.get("saturated")
            and v["auc_real"] - v["auc_random"] < 0.02),
        "per_scenario_mean": psm,
        "caveats": [
            "the two means (pooled vs per-scenario) are different quantities and are reported "
            "separately per E9_PREREG.md §3; neither is promoted to 'the' main result",
            "shadow_camera has AUC_real 0.3662 and is excluded from both per-scenario means: D1's "
            "denominator is negative there, so it is undefined rather than small",
            "the per-scenario spread is enormous (0.00% to 63.82% on 50-80 clips each), which is "
            "the real finding: LikePhys is not one task",
        ],
    }
    (OUT / "E9_verdict.json").write_text(json.dumps(verdict, indent=1))

    print(f"Q1 {verdict['q1']}: {all_c['d1_pct']:.2f}% -> {drop['d1_pct']:.2f}% "
          f"(threshold {THRESHOLD})")
    print(f"Q2 delta {delta:+.2f} pp vs noise {PROBE_SD_PP} -> within_noise="
          f"{verdict['q2']['within_noise']}")
    print(f"Q3 sat {recon['n_strict']} -> dropped {drop['n_dropped']}/{all_c['n']} clips "
          f"({drop['n_dropped']/all_c['n']*100:.1f}%)")
    print(f"per-scenario mean {psm['all']:.2f}% -> {psm['drop_saturated']:.2f}% "
          f"(n {psm['n_all']} -> {psm['n_kept']}, undefined {psm['undefined_scenarios']})")
    print(f"below chance: {below_chance}")
    print("near-saturated gaps: "
          + ", ".join(f"{k} {g:.4f}" for k, g in sorted(
              verdict["near_saturated_gaps"].items(), key=lambda kv: kv[1])))
    print(f"wrote {OUT / 'E9_verdict.json'}")


if __name__ == "__main__":
    main()
