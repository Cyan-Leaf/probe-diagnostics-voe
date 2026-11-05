"""E9: how much do the saturated scenarios depress D1?

The claim being tested (NEXT_STEPS_AND_DIVISION.md §1): "5 of the 12 LikePhys scenarios have
real-random = 0, which drags the global mean down". That is a *predictable bias*, not noise, so
it is worth quantifying rather than asserting.

Two aggregates, reported separately because they are not the same number and the exclusion moves
them for different reasons (E9_PREREG.md §3):

  pooled D1     recompute AUC on all clips, then (A-B)/(A-0.5)  — the aggregate E8 quotes
  per-scenario  per scenario AUC -> per scenario D1 -> mean over 12

Saturation is defined strictly: AUC >= 1 - 1e-9 on BOTH trained and random. A looser delta
would conflate "the rank statistic hit its ceiling" (protocol validity D3/D6) with "the signal
is strong", and only the former justifies exclusion.

Uses e8_analyse.probe verbatim, so the numbers are comparable to the delivered cells.

Run: python scripts/e9_saturation.py --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_analyse as A8

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E9_saturation"

BACKBONE = "wan21_1b3"
DATASET = "likephys"
LAYER = 16
T = 600
# EXPERIMENTS.md §5.3 quotes its headline at L14/t=400, but every cross-task comparison in
# E8/E12-E17 uses L16/t=600. Both are computed: the L14/t400 cell is what the "5 saturated
# scenarios" claim was derived from, so reconciling it needs that cell, not the reference one.
FEATURE_SEEDS = [0, 1, 2]
D1_THRESHOLD = 0.5
PROBE_SD_PP = 2.51          # E16's median probe-partition sd
SAT_EPS = 1e-9              # strictly 1.000, per E9_PREREG.md §2


def load(weights: str, fs: int):
    X, meta = A8.load_X(BACKBONE, DATASET, "plain", fs, LAYER, T, "", weights)
    assert X is not None
    y = np.array([-1 if m["label"] is None else int(m["label"]) for m in meta], dtype=np.int64)
    keep = np.isin(y, [0, 1])
    return np.asarray(X, dtype=np.float32)[keep], y[keep], [m for m, k in zip(meta, keep) if k]


def scenario_auc(Xr, Xq, y, meta, scen, s, seed):
    """AUC for one scenario, trained vs random-init, using the frozen probe."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from t4_caliper import _decision, _fit_path, _rowspace
    from e10_s1s3 import _shuffled_group_folds

    # `scen` is the full per-clip scenario list, so the mask must be built against the single
    # scenario name `s`. Comparing the array to the list itself broadcast to an empty mask and
    # silently skipped all 12 scenarios (the first run reported "drop 100% saturated").
    idx = np.flatnonzero(np.array(scen, dtype=object) == s)
    if len(idx) < 8 or len(np.unique(y[idx])) < 2:
        return None
    groups = np.array([f"{meta[i]['scenario']}/{meta[i]['subgroup']}" for i in idx])
    rng = np.random.default_rng(seed)
    aucs = {0: [], 1: []}       # 0 = trained, 1 = random-init
    for tr, te in _shuffled_group_folds(groups, 3, rng):
        if len(np.unique(y[idx][te])) < 2:
            continue
        for tag, X in ((0, Xr), (1, Xq)):
            sub = X[idx]
            sc = StandardScaler().fit(sub[tr])
            A, B = _rowspace(sc.transform(sub[tr]), sc.transform(sub[te]))
            _, coef, b = _fit_path(A, y[idx][tr], [1.0])[-1]
            aucs[tag].append(roc_auc_score(y[idx][te], _decision(B, coef, b)))
    if not aucs[0] or not aucs[1]:
        return None
    return float(np.mean(aucs[0])), float(np.mean(aucs[1])), int(len(idx))


def main():
    ap = argparse.ArgumentParser()
    global LAYER, T
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "E9_cells.json"))
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--t", type=int, default=T)
    args = ap.parse_args()
    from threadpoolctl import threadpool_limits

    LAYER, T = args.layer, args.t
    t0 = time.time()
    report = {"backbone": BACKBONE, "dataset": DATASET, "layer": LAYER, "t": T,
              "sat_eps": SAT_EPS, "per_scenario": {}, "aggregates": {}}

    with threadpool_limits(args.threads):
        # ---- the pooled aggregate, over all three feature seeds (E8's convention) ----------
        pooled = {"real": [], "random": []}
        y_ref = meta_ref = None
        for fs in FEATURE_SEEDS:
            Xr, y, meta = load("real", fs)
            Xq, yq, _ = load("random", fs)
            assert np.array_equal(y, yq)
            g = np.array([A8.group_of(m, "subkey") for m in meta])
            a_r, _ = A8.probe(Xr, y, g, seed=1000 + fs)
            a_q, _ = A8.probe(Xq, y, g, seed=1000 + fs)
            pooled["real"].append(a_r)
            pooled["random"].append(a_q)
            y_ref, meta_ref = y, meta
        A = float(np.mean(pooled["real"]))
        B = float(np.mean(pooled["random"]))
        report["aggregates"]["all_clips"] = {
            "auc_real": A, "auc_random": B, "d1": (A - B) / (A - 0.5),
            "d1_pct": (A - B) / (A - 0.5) * 100, "n": int(len(y_ref)),
            "per_feature_seed_real": pooled["real"],
            "per_feature_seed_random": pooled["random"]}

        # ---- per scenario ----------------------------------------------------------------
        Xr, y, meta = load("real", FEATURE_SEEDS[0])
        Xq, _, _ = load("random", FEATURE_SEEDS[0])
        scen = [m["scenario"] for m in meta]
        for s in sorted(set(scen)):
            got = scenario_auc(Xr, Xq, y, meta, scen, s, seed=1000)
            if got is None:
                report["per_scenario"][s] = {"skipped": "too few rows or single class"}
                continue
            a_r, a_q, n = got
            sat = (a_r >= 1.0 - SAT_EPS) and (a_q >= 1.0 - SAT_EPS)
            report["per_scenario"][s] = {
                "auc_real": a_r, "auc_random": a_q, "n": n,
                "d1": (a_r - a_q) / (a_r - 0.5) if a_r > 0.5 else float("nan"),
                "d1_pct": ((a_r - a_q) / (a_r - 0.5) * 100) if a_r > 0.5 else float("nan"),
                "saturated": bool(sat)}

    # ---- Q2: drop the saturated scenarios and recompute the pooled D1 -------------------
    sat = [s for s, v in report["per_scenario"].items() if v.get("saturated")]
    keep = [s for s in report["per_scenario"] if s not in sat and "skipped" not in
            report["per_scenario"][s]]
    mask = np.array([m["scenario"] in keep for m in meta_ref])
    g_all = np.array([A8.group_of(m, "subkey") for m in meta_ref])

    dropped, kept = {"real": [], "random": []}, {"real": [], "random": []}
    for fs in FEATURE_SEEDS:
        Xr, y, meta = load("real", fs)
        Xq, _, _ = load("random", fs)
        # NOTE: filtering clips changes the group set, so the folds change too. That is the
        # honest cost of dropping data and is why Q2's delta is compared against probe noise.
        a_r, _ = A8.probe(Xr[mask], y[mask], g_all[mask], seed=1000 + fs)
        a_q, _ = A8.probe(Xq[mask], y[mask], g_all[mask], seed=1000 + fs)
        kept["real"].append(a_r)
        kept["random"].append(a_q)
    Ak, Bk = float(np.mean(kept["real"])), float(np.mean(kept["random"]))
    d1_all = report["aggregates"]["all_clips"]["d1_pct"]
    d1_kept = (Ak - Bk) / (Ak - 0.5) * 100
    report["aggregates"]["drop_saturated"] = {
        "dropped_scenarios": sat, "kept_scenarios": keep,
        "auc_real": Ak, "auc_random": Bk, "d1_pct": d1_kept,
        "n": int(mask.sum()), "n_dropped": int((~mask).sum()),
        "per_feature_seed_real": kept["real"],
        "per_feature_seed_random": kept["random"]}

    # The per-scenario mean must average over the SAME set of defined scenarios on both sides.
    # shadow_camera has AUC_real = 0.3662 < 0.5, so (A-B)/(A-0.5) is not a "training share" at
    # all — the denominator is negative. Its d1 is undefined (nan) and the first version of this
    # block fed that nan straight into np.mean, making the whole drop_saturated aggregate nan.
    defined = [s for s, v in report["per_scenario"].items() if np.isfinite(v.get("d1_pct", np.nan))]
    undefined = [s for s, v in report["per_scenario"].items() if s not in defined]
    per_scen_vals = [report["per_scenario"][s]["d1_pct"] for s in defined]
    per_scen_kept = [report["per_scenario"][s]["d1_pct"] for s in keep if s in defined]
    report["aggregates"]["per_scenario_mean"] = {
        "all": float(np.mean(per_scen_vals)),
        "n_all": len(per_scen_vals),
        "drop_saturated": float(np.mean(per_scen_kept)),
        "n_kept": len(per_scen_kept),
        "undefined_scenarios": undefined,
        "undefined_reason": ("AUC_real < 0.5 — the training-share denominator (A-0.5) is negative, "
                             "so D1 is undefined rather than small. Excluded from BOTH means so "
                             "the two sides average over the same scenario set.")}

    # ---- reconciliation with the "5 of 12 saturated" claim in NEXT_STEPS_AND_DIVISION.md -----
    # NOT the preregistered definition (E9_PREREG.md §2 fixes delta = 1e-9 on BOTH arms) and it
    # never enters the verdict. It exists only to explain where "5" could have come from, because
    # the strict count here is 2 and a silent discrepancy between a delivered claim and a fresh
    # measurement is worse than either number on its own.
    real_sat = [s for s, v in report["per_scenario"].items()
                if v.get("auc_real", 0) >= 1.0 - SAT_EPS]
    zero_gap = [s for s, v in report["per_scenario"].items()
                if np.isfinite(v.get("d1_pct", np.nan)) and abs(v["d1_pct"]) < 1e-12]
    report["aggregates"]["count_reconciliation"] = {
        "strict_both_arms_1.000": sorted(sat), "n_strict": len(sat),
        "real_arm_1.000_only": sorted(real_sat), "n_real_only": len(real_sat),
        "exact_zero_gap": sorted(zero_gap), "n_zero_gap": len(zero_gap),
        "claim_under_test": "NEXT_STEPS_AND_DIVISION.md §1: '12 场景中 5 个 real-random=0'",
        "note": ("at L16/t=600 only 2 scenarios reach 1.000 on BOTH arms and exactly 2 have a "
                 "zero gap, so '5' does not reproduce here. The 5/12 figure in EXPERIMENTS.md "
                 "§5.6 (D3) was measured on the L14/t=400 best cell, whose features are NOT in "
                 "cache, so it cannot be re-derived on this machine — it is neither confirmed "
                 "nor refuted by E9, and must not be silently replaced by the L16/t=600 count.")}

    verdict = {
        "q1": "still_below_threshold" if d1_kept < D1_THRESHOLD * 100
              else "crosses_threshold",
        "q1_note": ("dropping saturated scenarios does not move D1 past the 50% bar"
                    if d1_kept < D1_THRESHOLD * 100 else
                    "D1 crosses 50% once saturated scenarios are removed — this would mean E8's "
                    "LikePhys verdict was depressed by ceiling effects"),
        "q2": {"delta_pp": d1_kept - d1_all,
               "probe_sd_pp": PROBE_SD_PP,
               "within_noise": abs(d1_kept - d1_all) < PROBE_SD_PP},
        "q3": {"n_saturated": len(sat), "saturated": sat,
               "n_clips_dropped": int((~mask).sum()), "n_clips_total": int(len(mask)),
               "fraction_dropped": float((~mask).sum() / len(mask)),
               "caveat": ("filtering clips also changes the group set and therefore the CV folds; "
                          "the dropped fraction is reported because a large one would make the "
                          "kept estimate noisy rather than cleaner")},
        "elapsed_s": round(time.time() - t0, 1)}

    report["verdict"] = verdict
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"all clips          : real {A:.4f} random {B:.4f}  D1 {d1_all:.2f}%  n={len(mask)}")
    print(f"drop {len(sat)} saturated: real {Ak:.4f} random {Bk:.4f}  D1 {d1_kept:.2f}%  "
          f"n={int(mask.sum())}")
    print(f"saturated scenarios: {sat}")
    print(f"Q1 {verdict['q1']}   Q2 delta {d1_kept - d1_all:+.2f} pp "
          f"(noise {PROBE_SD_PP})   Q3 dropped {verdict['q3']['fraction_dropped']*100:.1f}%")
    print(f"wrote {args.out} ({verdict['elapsed_s']}s)")


if __name__ == "__main__":
    main()
