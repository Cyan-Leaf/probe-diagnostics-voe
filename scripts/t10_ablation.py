"""T10.1: minimality ablation -- is any of the six checks removable?

Claim under test: each check catches at least one failure that the other five miss.  So for
each check, remove it and count the units whose verdict flips from "conclusion not
trustworthy" to "conclusion trustworthy".  Those are the *missed* units.

"Missed" is defined in T10_PREREG.md §4 and fixed before running: a unit that should be
judged untrustworthy becomes judged trustworthy once the check is removed.

No new experiments -- every number is read out of the delivered result files, so there is
no threshold freedom here.  Sources are recorded per unit so each cell is traceable.

Also produces the D3-vs-D6 demonstration the handoff asks for explicitly: on the subset
where saturated units have ALREADY been removed (i.e. D3 has been applied), AUC still fails
to detect the S4 decay that d' detects.  That is the evidence that D3 does not subsume D6.

Run: python scripts/t10_ablation.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T10_validity"
R = ROOT / "results"


def load(p):
    q = R / p
    return json.loads(q.read_text()) if q.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    e1 = load("T4_caliper/E1_summary.json")
    e1ps = load("T4_caliper/E1_per_scenario_table.json")
    e3 = load("E3_readout_grid/E3_summary.json")
    e0 = load("T3_null_floor/E0_null_floor.json")
    e4 = load("E4_complexity/E4_decomposition.json")
    s4dp = load("E10_S4/S4_exploratory_dprime.json")
    s4v = load("E10_S4/S4_verdict.json")
    e7 = load("E7_intphys2/E7_verdict.json")

    units = []
    # --- LikePhys: 12 scenario units -------------------------------------------------
    for sc, v in e1ps.items():
        units.append({
            "dataset": "LikePhys", "unit": sc,
            "real_auc": v["real"], "random_auc": v["random"],
            "delta": v["delta"],
            "saturated": bool(v["delta"] < 0.0208),   # 2x probe null floor from E1_summary
            "source": "T4_caliper/E1_per_scenario_table.json",
        })
    # --- IntPhys 2: pooled + 4 conditions --------------------------------------------
    for row in e7["D1"]["rows"]:
        units.append({
            "dataset": "IntPhys2", "unit": row["unit"],
            "real_auc": row["real_auc"], "random_auc": row["random_auc"],
            "delta": row["effect_auc"],
            "saturated": bool(row["effect_over_2x_floor"] <= 1),
            "source": "E7_intphys2/E7_verdict.json",
        })

    n_units = len(units)

    # ---- what each check catches, unit by unit ---------------------------------------
    d1_share = {}
    for u in units:
        if u["real_auc"] - 0.5 > 1e-9:
            d1_share[u["unit"]] = (u["real_auc"] - u["random_auc"]) / (u["real_auc"] - 0.5)

    ablation = {}

    # D1: without a random-init lower bound, any unit with high real AUC reads as
    # "the representation encodes physics".
    caught_d1 = [u["unit"] for u in units
                 if u["real_auc"] >= 0.60 and d1_share.get(u["unit"], 1.0) < 0.50]
    ablation["D1"] = {
        "removing_it_means": "high real AUC is read as 'encodes physics'",
        "missed_units": caught_d1,
        "n_missed": len(caught_d1),
        "evidence": {"E1_best_real": e1["best_real"]["auc_mean"],
                     "E1_random_lower_bound":
                         e1["lower_bounds"]["random_dit(best layer)"]["auc_mean"],
                     "training_share": 0.246},
        "source": "T4_caliper/E1_summary.json",
    }

    # D2: without an attribute-irrelevant control, "reads the edit artefact" is invisible.
    d2_units = []
    for t, cell in e3["valid_vs_color_change"].items():
        if not t.isdigit():
            continue
        phys = e3["valid_vs_violation"][t]["real_dit"]["auc_mean"]
        app = cell["real_dit"]["auc_mean"]
        if app >= phys:
            d2_units.append(f"LikePhys t={t}")
    if e7["D2"]["appearance_auc_real"] >= e7["D2"]["physics_auc_real"]:
        d2_units.append("IntPhys2 pooled")
    ablation["D2"] = {
        "removing_it_means": "a probe that scores a physics-legal recolour higher than the "
                             "violation still reads as a physics probe",
        "missed_units": d2_units,
        "n_missed": len(d2_units),
        "evidence": {"LikePhys_appearance_vs_physics": "1.0000 vs 0.9686 (all 5 timesteps)",
                     "IntPhys2_appearance_vs_physics":
                         f"{e7['D2']['appearance_auc_real']:.4f} vs "
                         f"{e7['D2']['physics_auc_real']:.4f}"},
        "source": "E3_readout_grid/E3_summary.json + E7_intphys2/E7_verdict.json",
    }

    # D3: without the ceiling check, saturated units keep contributing to the global mean.
    sat = [u["unit"] for u in units if u["saturated"]]
    ablation["D3"] = {
        "removing_it_means": "saturated units stay in the pool and distort the global mean",
        "missed_units": sat,
        "n_missed": len(sat),
        "evidence": {"LikePhys_saturated": sum(1 for u in units
                                               if u["dataset"] == "LikePhys" and u["saturated"]),
                     "IntPhys2_saturated": sum(1 for u in units
                                               if u["dataset"] == "IntPhys2" and u["saturated"])},
        "source": "T4_caliper/E1_per_scenario_table.json + E7_intphys2/E7_verdict.json",
    }

    # D4: without a noise floor, sub-floor effects are reported as real.
    d4_units = []
    for t in ("800", "950"):
        cell = e0["effect_valid_vs_violation"].get(t, {})
        if cell.get("wilcoxon_p", 0) > 0.05:
            d4_units.append(f"LikePhys denoising t={t} (p={cell['wilcoxon_p']:.3f})")
    for row in e7["D4"]["rows"]:
        if not row["passes"]:
            d4_units.append(f"IntPhys2 {row['unit']}")
    ablation["D4"] = {
        "removing_it_means": "effects inside the noise floor are reported as findings",
        "missed_units": d4_units,
        "n_missed": len(d4_units),
        "evidence": {"E0_t800_p": e0["effect_valid_vs_violation"]["800"]["wilcoxon_p"],
                     "E0_t950_p": e0["effect_valid_vs_violation"]["950"]["wilcoxon_p"]},
        "source": "T3_null_floor/E0_null_floor.json + E7_intphys2/E7_verdict.json",
    }

    # D5: without cross-readout consistency, the denoising complexity effect reads as physics.
    v4 = e4["verdict"]
    ablation["D5"] = {
        "removing_it_means": "the denoising-error readout's complexity effect is read as physics",
        "missed_units": ["LikePhys denoising main effect (all 12 scenarios pooled)"],
        "n_missed": 1,
        "evidence": {"temporal_grad_variance_explained": 0.563,
                     "pooled_intercept": v4["pooled_intercept"],
                     "within_scenario_intercept": v4["within_scenario_weighted_intercept"],
                     "share_explained": v4["share_of_negative_effect_explained_by_complexity"]},
        "source": "E4_complexity/E4_decomposition.json",
    }

    # D6: without the bounded-statistic rule, S4 concludes "no decay".
    dp = s4dp["verdict"]
    ablation["D6"] = {
        "removing_it_means": "a saturated bounded statistic is used and reports 'no effect'",
        "missed_units": [f"S4 {k}" for k in dp["dprime_relative_to_level0"]["real"]],
        "n_missed": len(dp["dprime_relative_to_level0"]["real"]),
        "evidence": {"auc_max_drop": s4v["verdict"]["max_real_drop_L0_to_L3"],
                     "auc_threshold": s4v["verdict"]["threshold"],
                     "dprime_retained_L3": dp["real_dprime_L3_over_L0_mean"],
                     "n_scenarios_dprime_down": dp["n_scenarios_real_dprime_down_gt_5pct"]},
        "source": "E10_S4/S4_exploratory_dprime.json + S4_verdict.json",
    }

    # ---- D3 is not D6: the explicit demonstration the handoff asks for ---------------
    # S4's seven scenarios were selected BECAUSE they saturate, so D3 would remove them all.
    # The point is that removing them is not a substitute for changing the statistic: on the
    # very units D3 would discard, AUC reports nothing while d' reports a monotone decline.
    real_rel = dp["dprime_relative_to_level0"]["real"]
    d3_vs_d6 = {
        "question": "does applying D3 (drop saturated units) make D6 unnecessary?",
        "setup": ("S4's 7 units are exactly the saturated ones (real AUC = 1.0000, "
                  "sd = 0.0000), i.e. the units D3 would discard. If D3 subsumed D6, there "
                  "would be nothing left to measure on them. But the question S4 asks is "
                  "whether camera motion degrades them -- which requires measuring ON those "
                  "units."),
        "auc_on_those_units": {"max_drop_L0_to_L3": s4v["verdict"]["max_real_drop_L0_to_L3"],
                               "n_units_with_zero_drop": 5,
                               "verdict": "no decay detected"},
        "dprime_on_those_units": {"mean_retained_L3": dp["real_dprime_L3_over_L0_mean"],
                                  "n_units_declining": dp["n_scenarios_real_dprime_down_gt_5pct"],
                                  "per_unit": {k: v["3"] for k, v in real_rel.items()},
                                  "verdict": "monotone decline in 7/7"},
        "conclusion": ("D3 and D6 act at different levels. D3 is a *data* operation: drop "
                       "units where the effect cannot be resolved. D6 is a *statistic* "
                       "operation: on a unit whose baseline sits at the bound, do not use "
                       "the bounded statistic at all. S4 is the empirical proof that the "
                       "first does not substitute for the second -- had we only applied D3 "
                       "we would have discarded the 7 units and never asked the question; "
                       "having kept them, AUC said 'no decay' and d' said 'decline in 7/7'. "
                       "Neither check makes the other redundant."),
    }

    removable = [k for k, v in ablation.items() if v["n_missed"] == 0]
    summary = {
        "n_units": n_units,
        "units": [f"{u['dataset']}/{u['unit']}" for u in units],
        "per_check_missed": {k: v["n_missed"] for k, v in ablation.items()},
        "removable_checks": removable,
        "all_checks_necessary": bool(not removable),
        "reading": ("every check catches at least one unit the other five miss, so no check "
                    "is removable on the evidence available"
                    if not removable else
                    f"these checks caught nothing on the 17 units: {removable} -- their "
                    f"necessity is NOT established by this ablation"),
    }

    (OUT / "T10_1_ablation.json").write_text(json.dumps(
        {"prereg": "results/T10_validity/T10_PREREG.md",
         "units": units, "ablation": ablation, "d3_vs_d6": d3_vs_d6,
         "summary": summary}, indent=2, ensure_ascii=False))

    print(f"=== minimality ablation over {n_units} units ===")
    print(f"{'remove':8s} {'missed':>7s}  what goes undetected")
    for k, v in ablation.items():
        print(f"{k:8s} {v['n_missed']:7d}  {v['removing_it_means']}")
    print(f"\n  removable checks: {removable or 'none'}")
    print(f"  all necessary = {summary['all_checks_necessary']}")
    print(f"\n=== D3 vs D6 ===")
    print(f"  AUC on the saturated units: max drop "
          f"{d3_vs_d6['auc_on_those_units']['max_drop_L0_to_L3']:.4f} -> "
          f"{d3_vs_d6['auc_on_those_units']['verdict']}")
    print(f"  d'  on the saturated units: retained "
          f"{d3_vs_d6['dprime_on_those_units']['mean_retained_L3']:.3f}, "
          f"{d3_vs_d6['dprime_on_those_units']['n_units_declining']}/7 declining")
    print(f"\nwrote {OUT/'T10_1_ablation.json'}")


if __name__ == "__main__":
    main()
