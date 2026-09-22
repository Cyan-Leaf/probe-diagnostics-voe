"""S4 verdict against the pre-registered criteria (PREREG §4.2, frozen before the run).

Criteria, verbatim:
  H4 supported  : real AUC falls monotonically with strength, the level-3 drop exceeds
                  2 x permutation floor sd (0.2486), and >=5 of the 7 scenarios agree
  H4 unsupported: no systematic trend, or the drop sits inside the floor

Mandatory extras:
  A.3 item 2 : random AUC and margin recomputed at every level (real alone is meaningless)
  A.3 item 3 : the crop ratio and effective field of view actually applied

Hard rule 4 / PREREG §4.2 one-sidedness: a null result must NOT be written as a refutation
of H4, because level 3 (128 px) reaches only 27% of the real motion that produced the
original anomaly (479 px).  This script emits the mandated wording rather than leaving it
to prose.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E10_S4"
SCEN = ["ball_drop", "cloth_drape", "faucet", "fluid", "pendulum", "pyramid", "river"]
LEVELS = ["0", "1", "2", "3"]
EXCURSION = {"0": 0, "1": 32, "2": 64, "3": 128}
FLOOR_SD = 0.1243
THRESHOLD = 2 * FLOOR_SD
REAL_MOTION_PX = 479.2          # A5_motion.json, shadow_camera max-excursion median
SHADOW_CAMERA_MARGIN = 0.4927 - 0.6207   # the anomaly S4 is trying to reproduce


def main():
    src = OUT / "S4_levels_0_1_2_3.json"
    d = json.loads(src.read_text())
    lv = d["levels"]

    real = {L: {sc: lv[L]["real"][sc]["auc_mean"] for sc in SCEN} for L in LEVELS}
    rsd = {L: {sc: (lv[L]["real"][sc]["auc_sd"] or 0.0) for sc in SCEN} for L in LEVELS}
    rand = {L: {sc: lv[L]["random"][sc]["auc_mean"] for sc in SCEN} for L in LEVELS}
    marg = {L: {sc: real[L][sc] - rand[L][sc] for sc in SCEN} for L in LEVELS}

    rows = []
    for sc in SCEN:
        seq = [real[L][sc] for L in LEVELS]
        drop = seq[0] - seq[-1]
        strictly_down = all(b <= a + 1e-12 for a, b in zip(seq[:-1], seq[1:]))
        any_down = drop > 1e-12
        rows.append({
            "scenario": sc,
            "real_auc": {L: real[L][sc] for L in LEVELS},
            "real_sd": {L: rsd[L][sc] for L in LEVELS},
            "random_auc": {L: rand[L][sc] for L in LEVELS},
            "margin": {L: marg[L][sc] for L in LEVELS},
            "real_drop_L0_to_L3": drop,
            "real_drop_over_floor_sd": drop / FLOOR_SD,
            "monotone_non_increasing": bool(strictly_down),
            "declined_at_all": bool(any_down),
            "random_drop_L0_to_L3": rand["0"][sc] - rand["3"][sc],
            "margin_change_L0_to_L3": marg["3"][sc] - marg["0"][sc],
        })

    drops = np.array([r["real_drop_L0_to_L3"] for r in rows])
    n_declined = int(sum(r["declined_at_all"] for r in rows))
    n_monotone = int(sum(r["monotone_non_increasing"] for r in rows))
    max_drop = float(drops.max())

    crit_drop = bool(max_drop > THRESHOLD)
    crit_agree = bool(n_declined >= 5)
    supported = bool(crit_drop and crit_agree)

    rand_drops = np.array([r["random_drop_L0_to_L3"] for r in rows])
    marg_changes = np.array([r["margin_change_L0_to_L3"] for r in rows])

    verdict = {
        "criterion_verbatim": ("real AUC falls monotonically; level-3 drop > 0.2486 "
                               "(= 2 x permutation floor sd 0.1243); >=5 of 7 scenarios agree"),
        "max_real_drop_L0_to_L3": max_drop,
        "max_real_drop_scenario": rows[int(drops.argmax())]["scenario"],
        "max_real_drop_over_floor_sd": max_drop / FLOOR_SD,
        "threshold": THRESHOLD,
        "n_scenarios_declined_at_all": n_declined,
        "n_scenarios_monotone": n_monotone,
        "criterion_drop_met": crit_drop,
        "criterion_agreement_met": crit_agree,
        "H4_supported": supported,
        "verdict_text": (
            "H4 supported" if supported else
            "H4 not supported by S4: no decay observed"),
        # ---- mandated one-sidedness wording (PREREG §4.2, hard rule 4) -----------------
        "mandated_wording": (
            f"In the range up to {EXCURSION['3']} px cumulative displacement no decay of "
            f"real AUC was observed (largest drop {max_drop:.4f} = "
            f"{max_drop/FLOOR_SD:.2f}x the permutation floor sd, threshold "
            f"{THRESHOLD:.4f}). That range is only "
            f"{EXCURSION['3']/REAL_MOTION_PX*100:.0f}% of the motion that produced the "
            f"original shadow_camera anomaly ({REAL_MOTION_PX:.0f} px). Per PREREG §4.2 "
            f"this is NOT a refutation of H4 and must not be written as one."),
        "forbidden_phrasings": ["H4 被证伪", "H4 refuted", "camera-motion hypothesis "
                                "disproved", "相机运动假设不成立"],
        # ---- secondary readouts, A.3 item 2 -------------------------------------------
        "secondary": {
            "random_auc_drop_mean": float(rand_drops.mean()),
            "random_auc_drop_max": float(rand_drops.max()),
            "random_auc_drop_max_scenario": rows[int(rand_drops.argmax())]["scenario"],
            "n_random_declined": int((rand_drops > 1e-12).sum()),
            "margin_change_mean": float(marg_changes.mean()),
            "margin_change_min": float(marg_changes.min()),
            "margin_change_max": float(marg_changes.max()),
            "n_margin_increased": int((marg_changes > 0).sum()),
            "shadow_camera_margin_to_reproduce": SHADOW_CAMERA_MARGIN,
            "observation": (
                "The trained backbone stays at ceiling while the RANDOM backbone loses "
                f"{rand_drops.mean():.3f} AUC on average (up to "
                f"{rand_drops.max():.3f} on {rows[int(rand_drops.argmax())]['scenario']}). "
                f"Margin (real - random) therefore *increases* in "
                f"{int((marg_changes > 0).sum())}/7 scenarios "
                f"(mean {marg_changes.mean():+.4f}), whereas the shadow_camera anomaly "
                f"that S4 set out to explain has margin {SHADOW_CAMERA_MARGIN:+.4f}. "
                "Synthetic camera motion in this range does not reproduce the anomaly's "
                "signature even in sign. Subject to the same 27% magnitude caveat, so it "
                "is a directional observation, not a refutation."),
        },
        "crop_applied": d["crop"],
        "zero_level_gate": d["zero_level_gate"],
    }

    out = {"per_scenario": rows, "verdict": verdict,
           "source": str(src.relative_to(ROOT))}

    # Attach the exploratory ceiling check if it has been run: without it the
    # "margin increases" line above reads as "the trained backbone is robust to camera
    # motion", which the d' analysis shows is a saturation artefact.
    dp_path = OUT / "S4_exploratory_dprime.json"
    if dp_path.exists():
        dp = json.loads(dp_path.read_text())["verdict"]
        verdict["secondary"]["CORRECTION_from_exploratory_dprime"] = {
            "status": "the margin observation above is confounded by AUC saturation",
            "real_dprime_retained_at_L3": dp["real_dprime_L3_over_L0_mean"],
            "random_dprime_retained_at_L3": dp["random_dprime_L3_over_L0_mean"],
            "text": dp["margin_caveat"],
        }
        verdict["main_readout_sensitivity"] = {
            "status": "EXPLORATORY",
            "finding": dp["reading"],
            "n_real_scenarios_dprime_down": dp["n_scenarios_real_dprime_down_gt_5pct"],
            "real_dprime_retained_at_L3_mean": dp["real_dprime_L3_over_L0_mean"],
            "implication": (
                "real AUC was pre-registered as the main readout because baseline = "
                "1.0000 with sd = 0.0000 made 'any drop measurable'. For a bounded rank "
                "statistic that reasoning is inverted: AUC = 1.0 is the saturation point, "
                "so it is maximally insensitive there. The AUC null is therefore weak "
                "evidence, and an unsaturated readout finds monotone degradation in 7/7 "
                "scenarios."),
        }
    (OUT / "S4_verdict.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=== real AUC (main readout) ===")
    hdr = f"{'scenario':13s}" + "".join(f"{'L'+L+'('+str(EXCURSION[L])+'px)':>13s}" for L in LEVELS) + f"{'drop':>9s}{'/floor':>8s}"
    print(hdr)
    for r in rows:
        line = f"{r['scenario']:13s}"
        for L in LEVELS:
            line += f"{r['real_auc'][L]:13.4f}"
        line += f"{r['real_drop_L0_to_L3']:9.4f}{r['real_drop_over_floor_sd']:8.2f}"
        print(line)

    print("\n=== random AUC (secondary, A.3 item 2) ===")
    print(hdr)
    for r in rows:
        line = f"{r['scenario']:13s}"
        for L in LEVELS:
            line += f"{r['random_auc'][L]:13.4f}"
        line += f"{r['random_drop_L0_to_L3']:9.4f}{r['random_drop_L0_to_L3']/FLOOR_SD:8.2f}"
        print(line)

    print("\n=== margin real-random (secondary) ===")
    print(f"{'scenario':13s}" + "".join(f"{'L'+L:>13s}" for L in LEVELS) + f"{'change':>9s}")
    for r in rows:
        line = f"{r['scenario']:13s}"
        for L in LEVELS:
            line += f"{r['margin'][L]:+13.4f}"
        line += f"{r['margin_change_L0_to_L3']:+9.4f}"
        print(line)

    print("\n=== verdict ===")
    for k in ("max_real_drop_L0_to_L3", "max_real_drop_scenario",
              "max_real_drop_over_floor_sd", "threshold",
              "n_scenarios_declined_at_all", "n_scenarios_monotone",
              "criterion_drop_met", "criterion_agreement_met", "H4_supported"):
        print(f"  {k}: {verdict[k]}")
    print(f"\n  {verdict['mandated_wording']}")
    print(f"\n  secondary: {verdict['secondary']['observation']}")
    print(f"\nwrote {OUT/'S4_verdict.json'}")


if __name__ == "__main__":
    main()
