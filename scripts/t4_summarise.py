"""T4 summary: turn E1_probe_raw.json into the CALIPER verdict + heatmap inputs.

Verdict rule (protocol_as_executed.md §5):
  real_dit(best) - max(lower bounds) must exceed 2 x probe-level null floor,
  where the null floor is the AUC spread of the *same* config across noise seeds.

Run: uv run --no-project python scripts/t4_summarise.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T4 = ROOT / "results" / "T4_caliper"
T3 = ROOT / "results" / "T3_null_floor"


def main() -> None:
    raw = json.loads((T4 / "E1_probe_raw.json").read_text())
    t_ref = 600

    real = [r for r in raw if r["source"] == "real_dit"]
    rand = [r for r in raw if r["source"] == "random_dit"]
    vae = next((r for r in raw if r["source"] == "vae_latent"), None)
    pix = next((r for r in raw if r["source"] == "raw_pixel"), None)

    # probe-level null floor: spread across noise seeds of the same (source, layer, t)
    spreads = []
    for src_rows in (real, rand):
        by = {}
        for r in src_rows:
            if r["t"] != t_ref:
                continue
            by.setdefault((r["source"], r["layer"]), {})[r["seed"]] = r["auc_mean"]
        for key, d in by.items():
            if len(d) >= 2:
                spreads.append(max(d.values()) - min(d.values()))
    floor = float(np.median(spreads)) if spreads else float("nan")
    floor_p90 = float(np.percentile(spreads, 90)) if spreads else float("nan")

    real_seed0 = [r for r in real if r["seed"] == 0]
    best_real = max(real_seed0, key=lambda r: r["auc_mean"])
    best_real_ref = max([r for r in real_seed0 if r["t"] == t_ref], key=lambda r: r["auc_mean"])
    best_rand = max([r for r in rand if r["seed"] == 0], key=lambda r: r["auc_mean"], default=None)

    lower = {}
    if best_rand:
        lower["random_dit(best layer)"] = {
            "auc_mean": best_rand["auc_mean"], "auc_std": best_rand["auc_std"],
            "layer": best_rand["layer"], "t": best_rand["t"],
            "acc_mean": best_rand["acc_mean"],
        }
    if vae:
        lower["vae_latent"] = {"auc_mean": vae["auc_mean"], "auc_std": vae["auc_std"],
                               "acc_mean": vae["acc_mean"], "dim": vae["dim"]}
    if pix:
        lower["raw_pixel"] = {"auc_mean": pix["auc_mean"], "auc_std": pix["auc_std"],
                              "acc_mean": pix["acc_mean"], "dim": pix["dim"]}

    strongest = max(lower.items(), key=lambda kv: kv[1]["auc_mean"]) if lower else (None, {"auc_mean": 0.5})
    margin = best_real["auc_mean"] - strongest[1]["auc_mean"]

    summary = {
        "reference_timestep": t_ref,
        "n_configs": len(raw),
        "best_real": {k: best_real[k] for k in
                      ("layer", "t", "seed", "auc_mean", "auc_std", "acc_mean", "acc_std",
                       "chosen_C", "n", "n_pos", "dim")},
        "best_real_at_reference_t": {k: best_real_ref[k] for k in
                                     ("layer", "t", "auc_mean", "auc_std", "acc_mean")},
        "lower_bounds": lower,
        "strongest_lower_bound": {"name": strongest[0], **strongest[1]},
        "probe_null_floor": {
            "auc_seed_spread": floor,
            "auc_seed_spread_p90": floor_p90,
            "n_configs_with_multiseed": len(spreads),
        },
        "verdict": {
            "margin_real_minus_strongest_lower": margin,
            "threshold_2x_null_floor": 2 * floor,
            "discriminative": bool(margin > 2 * floor),
        },
        "layer_profile_at_reference_t": sorted(
            [{"layer": r["layer"], "auc_mean": r["auc_mean"], "auc_std": r["auc_std"],
              "acc_mean": r["acc_mean"]} for r in real_seed0 if r["t"] == t_ref],
            key=lambda d: d["layer"],
        ),
        "timestep_profile": {
            str(t): max([r["auc_mean"] for r in real_seed0 if r["t"] == t], default=None)
            for t in sorted({r["t"] for r in real_seed0})
        },
    }

    if (T3 / "E0_null_floor.json").exists():
        e0 = json.loads((T3 / "E0_null_floor.json").read_text())
        summary["e0_cross_reference"] = {
            "clip_seed_cv_pct_at_ref_t": e0["clip_level_seed_noise"][str(t_ref)]["median_cv_pct"],
            "effect_over_seed_spread_at_ref_t":
                e0["seed_spread_of_effect"][str(t_ref)]["effect_over_seed_spread"],
        }

    (T4 / "E1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("layer_profile_at_reference_t",)}, indent=2))
    print(f"\nwrote {T4/'E1_summary.json'}")


if __name__ == "__main__":
    main()
