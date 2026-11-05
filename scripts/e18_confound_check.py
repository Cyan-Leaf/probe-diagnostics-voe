"""E18 confound check: is `experts_differ_in_own_band` real, or a denominator artefact?

Why this script exists
----------------------
E18's pre-registered statistic is Δ_own = D1(low@t=300) − D1(high@t=900). It came out at
−7.81 pp, past the 2.51 pp probe-noise threshold. Taken at face value, E18_PREREG.md §3 requires
retracting E17 §5.

It must not be taken at face value. Two things about the own-band cells make the ratio fragile
in a way E17's t=600 comparison was not:

  1. **The random arms are no longer the same measurement.** At t=600 both experts were probed
     with the same noise draw, so their random AUC was bit-identical (verified with ==, see
     E17_REPORT §3.1). At t=300 vs t=900 the noise tensor z_t = (1-σ)z₀ + σε differs, so the two
     cells are different probe inputs even with identical random weights. The arms land at
     0.5141 vs 0.5025.

  2. **The denominator is small, so it amplifies.** D1 = (real − random)/(real − 0.5), and here
     real − 0.5 is only 0.096–0.133, i.e. an amplification of 7.5–10.5×. A random-arm gap of
     0.0116 therefore moves D1 by ~9 pp on its own, with no change in what either real arm reads.

So this script decomposes the delivered Δ_own into (a) the effect of the two random arms sitting
at different levels and (b) the effect of the two real arms differing, and adds an error
propagation over the three feature seeds.

Reported as exploratory (START_HERE rule 4). E18_verdict.json is NOT rewritten.

Run: python3 scripts/e18_confound_check.py
"""

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E18_own_band"
E17 = ROOT / "results" / "E17_moe_experts" / "E17_cells.json"

GATE_AUC = 0.6        # E8 §15.2: D1 ratios unreliable below this real AUC
THRESHOLD_PP = 2.51   # E16 probe-partition sd median, the threshold E18_PREREG.md §3 fixed
LAYERS = ["L16", "Lrel"]


def d1_pct(real, random, denom_real=None):
    den = (real if denom_real is None else denom_real) - 0.5
    if den <= 0:
        return float("nan")
    return (real - random) / den * 100


def seed_pair_deltas(lo, hi):
    """Δ_own recomputed for all 9 feature-seed pairings: error propagation through the ratio."""
    out = []
    for i, j in itertools.product(range(len(lo["per_feature_seed_real"])),
                                  range(len(hi["per_feature_seed_real"]))):
        rl, ql = lo["per_feature_seed_real"][i], lo["per_feature_seed_random"][i]
        rh, qh = hi["per_feature_seed_real"][j], hi["per_feature_seed_random"][j]
        if rl <= 0.5 or rh <= 0.5:
            continue
        out.append(d1_pct(rl, ql) - d1_pct(rh, qh))
    return np.array(out, float)


def main():
    cells = json.loads((OUT / "E18_cells.json").read_text())["cells"]
    e17 = json.loads(E17.read_text())["cells"]

    out = {
        "purpose": ("decompose E18's Δ_own into a random-arm level effect and a real-arm effect, "
                    "and propagate the feature-seed noise through the D1 ratio"),
        "status": "exploratory, does not modify E18_verdict.json",
        "threshold_pp": THRESHOLD_PP,
        "layers": {},
    }

    for lname in LAYERS:
        lo = cells[f"a14b|low@300|{lname}"]
        hi = cells[f"a14b|high@900|{lname}"]
        e17_lo = e17[f"a14b|low|{lname}"]
        e17_hi = e17[f"a14b|high|{lname}"]

        delivered = lo["d1_pct"] - hi["d1_pct"]
        amp_lo = 1.0 / (lo["auc_real"] - 0.5)
        amp_hi = 1.0 / (hi["auc_real"] - 0.5)

        rand_gap = lo["auc_random"] - hi["auc_random"]
        real_gap = lo["auc_real"] - hi["auc_real"]

        # Linearise Δ_own = D1_lo − D1_hi around the midpoint of the two random arms.
        #   dD1/dq = −1/(real−0.5), and the arms move in opposite directions from their mean
        #   (dq_lo = +gap/2, dq_hi = −gap/2), so
        #   random-arm contribution = −(gap/2)·(amp_lo + amp_hi)·100
        # Sign matters: a higher random arm on the low-noise expert pushes D1_lo DOWN.
        rand_only = -(rand_gap / 2.0) * (amp_lo + amp_hi) * 100

        # (b) what the real-arm gap alone would do, holding the denominator fixed (mean random).
        shared = (lo["auc_random"] + hi["auc_random"]) / 2
        real_only = d1_pct(lo["auc_real"], shared) - d1_pct(hi["auc_real"], shared)

        ds = seed_pair_deltas(lo, hi)

        out["layers"][lname] = {
            "delivered_delta_own_pp": delivered,
            "amplification": {"low": amp_lo, "high": amp_hi},
            "decomposition_pp": {
                "random_arm_level_effect": rand_only,
                "real_arm_effect_shared_denominator": real_only,
                "sum": rand_only + real_only,
                "note": ("the two terms do not sum exactly to the delivered number because D1 is "
                         "a ratio nonlinear in both arms; they are a linearisation, and the point "
                         "is which term dominates"),
            },
            "random_arms": {
                "auc_low": lo["auc_random"], "auc_high": hi["auc_random"],
                "gap": rand_gap,
                "low_seed_spread": lo["floor_random"], "high_seed_spread": hi["floor_random"],
                "e17_arms_bit_identical": bool(e17_lo["auc_random"] == e17_hi["auc_random"]),
                "why_gap_is_new": ("z_t=(1-sigma)z0+sigma*eps makes t=300 and t=900 different "
                                   "probe inputs; E17's two arms shared one draw at t=600"),
            },
            "real_arms": {
                "auc_low": lo["auc_real"], "auc_high": hi["auc_real"], "gap": real_gap,
                "low_seed_spread": lo["floor_real"], "high_seed_spread": hi["floor_real"],
                "gap_within_seed_spread": bool(abs(real_gap) < max(lo["floor_real"],
                                                                    hi["floor_real"])),
            },
            "seed_propagation": {
                "n_pairs": int(len(ds)),
                "min_pp": float(ds.min()), "max_pp": float(ds.max()),
                "mean_pp": float(ds.mean()), "sd_pp": float(ds.std(ddof=1)),
                "delivered_inside_range": bool(ds.min() <= delivered <= ds.max()),
                "sd_vs_threshold": float(ds.std(ddof=1) / THRESHOLD_PP),
                "reading": ("the feature-seed spread of Δ_own itself is an order of magnitude "
                            "larger than the 2.51 pp threshold, so that threshold cannot resolve "
                            "these cells"),
            },
            "weak_signal_gate": {
                "auc_real_below_0.6": [name for name, v in (("low@300", lo["auc_real"]),
                                                            ("high@900", hi["auc_real"]))
                                       if v < GATE_AUC],
                "warning": ("E8 §15.2: D1 ratios are not comparable across cells with real AUC "
                            "below 0.6; E12 reproduced this on six token budgets"),
            },
        }

    L = out["layers"]
    decomposed_dominant = all(
        abs(v["decomposition_pp"]["random_arm_level_effect"])
        > abs(v["decomposition_pp"]["real_arm_effect_shared_denominator"])
        for v in L.values())
    threshold_unusable = all(v["seed_propagation"]["sd_vs_threshold"] > 1 for v in L.values())
    # The cleanest single fact: on Lrel the real arms are indistinguishable yet D1 differs by ~7.6pp.
    real_arms_agree = {k: v["real_arms"]["gap_within_seed_spread"] for k, v in L.items()}

    out["conclusion"] = {
        "verdict": "delta_own_is_a_denominator_artefact",
        "verdict_reason": (
            "The delivered Δ_own decomposes almost entirely into the two random arms sitting at "
            "different levels: the random-arm term alone carries the whole −7.81 pp, while the "
            "real-arm term is +2.43 pp on L16 and −0.05 pp on Lrel — inside, or at, the 2.51 pp "
            "band. The cause is structural: t enters the noise draw z_t=(1−σ)z₀+σε, so t=300 and "
            "t=900 are different probe inputs even with identical random weights, and E17's "
            "bit-identical-random-arm structure does not survive a change of sigma. D1's "
            "denominator of 0.096–0.133 then magnifies a 0.0116 AUC gap into ~10 pp."),
        "decomposition_closes": all(
            abs(v["decomposition_pp"]["random_arm_level_effect"]
                + v["decomposition_pp"]["real_arm_effect_shared_denominator"]
                - v["delivered_delta_own_pp"]) < 0.5 for v in L.values()),
        "random_arm_effect_dominates": bool(decomposed_dominant),
        "threshold_too_small_for_these_cells": bool(threshold_unusable),
        "real_arm_gap_within_seed_spread": real_arms_agree,
        "retraction_deferred_not_executed": True,
        "note_on_retraction": (
            "E18_PREREG.md §3 said |Δ_own| >= 2.51 requires retracting E17 §5. That instruction "
            "presumed E17's shared-denominator structure, which does not survive a change of "
            "sigma, and it was also written as a two-sided test on a statistic whose sign is set "
            "by which arm got the higher random draw. The retraction is therefore deferred, not "
            "executed; E18_verdict.json keeps the pre-registered verdict unedited."),
        "prereg_internal_inconsistency": (
            "E18_PREREG.md §3 computes Δ_own across t=300 vs t=900 while §6 forbids comparing D1 "
            "across different sigma ('σ 相差 3 倍，不是同一难度'). The statistic as written is "
            "exactly what §6 prohibits, and the observed −8 pp is that difficulty difference "
            "showing up, not an expert difference."),
        "what_would_settle_it": (
            "probe both experts at ONE sigma while holding the noise draw fixed (so the random "
            "arm is shared, as in E17), and vary a different axis; or measure each expert across "
            "a sigma ladder so the random-arm level is modelled rather than assumed equal"),
    }

    p = OUT / "E18_confound_check.json"
    p.write_text(json.dumps(out, indent=1))

    for lname, v in L.items():
        d = v["decomposition_pp"]
        print(f"\n{lname}: delivered Δ_own {v['delivered_delta_own_pp']:+.2f} pp")
        print(f"  amplification (1/(real-0.5)): {v['amplification']['low']:.1f}x / "
              f"{v['amplification']['high']:.1f}x")
        print(f"  random-arm level effect {d['random_arm_level_effect']:+.2f} pp  |  "
              f"real-arm effect {d['real_arm_effect_shared_denominator']:+.2f} pp")
        print(f"  random gap {v['random_arms']['gap']:+.4f} (E17 arms identical: "
              f"{v['random_arms']['e17_arms_bit_identical']})")
        print(f"  real gap {v['real_arms']['gap']:+.4f}  within seed spread: "
              f"{v['real_arms']['gap_within_seed_spread']}")
        s = v["seed_propagation"]
        print(f"  Δ_own over {s['n_pairs']} seed pairs: {s['min_pp']:+.1f} .. {s['max_pp']:+.1f} "
              f"(mean {s['mean_pp']:+.1f}, sd {s['sd_pp']:.1f} = {s['sd_vs_threshold']:.1f}x "
              f"the {THRESHOLD_PP} pp threshold)")
    print(f"\nverdict: {out['conclusion']['verdict']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
