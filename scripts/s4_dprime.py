"""S4 exploratory addendum: is the null on real AUC a ceiling artefact?

NOT pre-registered -- labelled exploratory per hard rule 5, kept out of the main verdict.

Motivation.  PREREG §4.2 picked real AUC as the main readout precisely because the seven
scenarios sit at 1.0000 with sd 0.0000, so "any drop is measurable".  That is true, but AUC
is a *rank* statistic: it stays at exactly 1.0 until the class score distributions actually
overlap.  A probe whose class separation is shrinking under camera motion can therefore
show no AUC movement at all right up to the point of collapse.  Since 5/7 scenarios
returned exactly 0.0000 drop, the null needs a sensitivity check with a readout that can
see sub-rank degradation.

Readout used here: the cross-validated separation of the probe's decision values,

    d' = (mean(s | y=1) - mean(s | y=0)) / pooled_sd(s)

computed on the held-out folds with the identical protocol (same downsampling, same
partitions, same C selection).  d' keeps decreasing after AUC has pinned at 1.0.

Interpretation rule fixed before looking:
  - AUC flat AND d' flat        -> the null is real; camera motion does nothing here
  - AUC flat BUT d' declining   -> AUC saturation was masking degradation; the main
                                   readout was insensitive and the null is weaker than it
                                   appears
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_s1s3 import _match_design, _shuffled_group_folds
from s4_analyse import (LAYER, N_POS_PER_SUB, N_RESAMPLE, S3_SEED, SCENARIOS, load_level)
from t4_caliper import CS, _decision, _fit_path, _rowspace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E10_S4"
LEVELS = ["0", "1", "2", "3"]
EXC = {"0": 0, "1": 32, "2": 64, "3": 128}


def probe_dprime(X, y, groups, seed):
    """Same protocol as _probe_auc, but returns held-out d' as well as AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    aucs, dps = [], []
    for tr, te in _shuffled_group_folds(groups, 5, rng):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, gtr = y[tr], groups[tr]

        scores = {c: [] for c in CS}
        for itr, ite in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
            if len(np.unique(ytr[ite])) < 2:
                continue
            sc2 = StandardScaler().fit(Xtr[itr])
            A, B = sc2.transform(Xtr[itr]), sc2.transform(Xtr[ite])
            A, B = _rowspace(A, B)
            for c, coef, b in _fit_path(A, ytr[itr], CS):
                scores[c].append(roc_auc_score(ytr[ite], _decision(B, coef, b)))
        best_c = max(CS, key=lambda c: np.mean(scores[c]) if scores[c] else -1)

        Ztr, Zte = _rowspace(Xtr, Xte)
        _, coef, b = _fit_path(Ztr, ytr, [best_c])[-1]
        s = _decision(Zte, coef, b)
        aucs.append(roc_auc_score(y[te], s))
        s1, s0 = s[y[te] == 1], s[y[te] == 0]
        if len(s1) > 1 and len(s0) > 1:
            pooled = np.sqrt((s1.var(ddof=1) + s0.var(ddof=1)) / 2.0)
            if pooled > 1e-12:
                dps.append(float((s1.mean() - s0.mean()) / pooled))
    return (float(np.mean(aucs)) if aucs else np.nan,
            float(np.mean(dps)) if dps else np.nan)


def main():
    from threadpoolctl import threadpool_limits

    table = {"real": {}, "random": {}}
    with threadpool_limits(limits=8):
        for bb in ("real", "random"):
            for sc in SCENARIOS:
                table[bb][sc] = {}
                for L in LEVELS:
                    got = load_level(bb, int(L), sc)
                    if got is None:
                        continue
                    X, y, g, _ = got
                    aucs, dps = [], []
                    for r in range(N_RESAMPLE):
                        rng = np.random.default_rng(S3_SEED + 7919 * r)
                        keep = _match_design(y, g, N_POS_PER_SUB, rng)
                        if len(np.unique(y[keep])) < 2:
                            continue
                        a, dp = probe_dprime(X[keep], y[keep], g[keep],
                                             seed=S3_SEED + 104729 * r)
                        if not np.isnan(a):
                            aucs.append(a)
                        if not np.isnan(dp):
                            dps.append(dp)
                    table[bb][sc][L] = {
                        "auc_mean": float(np.mean(aucs)) if aucs else None,
                        "dprime_mean": float(np.mean(dps)) if dps else None,
                        "dprime_sd": float(np.std(dps, ddof=1)) if len(dps) > 1 else None,
                    }
                    print(f"  {bb:6s} {sc:12s} L{L}({EXC[L]:3d}px)  "
                          f"AUC {table[bb][sc][L]['auc_mean']:.4f}  "
                          f"d' {table[bb][sc][L]['dprime_mean']:7.3f} "
                          f"+-{(table[bb][sc][L]['dprime_sd'] or 0):.3f}", flush=True)

    dp_rel = {bb: {} for bb in table}
    for bb in table:
        for sc in SCENARIOS:
            base = table[bb][sc]["0"]["dprime_mean"]
            dp_rel[bb][sc] = {L: (table[bb][sc][L]["dprime_mean"] / base if base else None)
                              for L in LEVELS}

    l3_real = np.array([dp_rel["real"][sc]["3"] for sc in SCENARIOS], dtype=float)
    l3_rand = np.array([dp_rel["random"][sc]["3"] for sc in SCENARIOS], dtype=float)
    auc_flat = all(abs(table["real"][sc]["0"]["auc_mean"]
                       - table["real"][sc]["3"]["auc_mean"]) < 0.03 for sc in SCENARIOS)
    n_dp_down = int((l3_real < 0.95).sum())

    verdict = {
        "status": "EXPLORATORY -- not pre-registered, excluded from the main verdict",
        "readout": "held-out d' of the probe decision values, same folds/protocol as S4",
        "dprime_relative_to_level0": dp_rel,
        "n_scenarios_real_dprime_down_gt_5pct": n_dp_down,
        "real_dprime_L3_over_L0_mean": float(np.nanmean(l3_real)),
        "real_dprime_L3_over_L0_min": float(np.nanmin(l3_real)),
        "random_dprime_L3_over_L0_mean": float(np.nanmean(l3_rand)),
        "random_dprime_L3_over_L0_min": float(np.nanmin(l3_rand)),
        "auc_essentially_flat_for_real": bool(auc_flat),
        "reading": (
            "AUC flat and d' also flat -> the S4 null is genuine, not a ceiling artefact"
            if n_dp_down <= 2 else
            "AUC flat while d' declines in most scenarios -> AUC saturation was masking "
            "degradation; the pre-registered readout is insensitive in this regime and the "
            "null should be reported as weaker than the AUC table alone suggests"),
        "margin_caveat": (
            f"The AUC-margin increase reported in S4_verdict.json is at least partly a "
            f"ceiling asymmetry: real AUC is pinned at 1.0000 and cannot fall, while "
            f"random AUC has headroom. On the unsaturated d' readout real retains "
            f"{np.nanmean(l3_real):.3f} of its level-0 separation and random retains "
            f"{np.nanmean(l3_rand):.3f}, i.e. "
            + ("both degrade comparably, so 'the trained backbone is robust to camera "
               "motion' is NOT supported once the ceiling is removed"
               if abs(np.nanmean(l3_real) - np.nanmean(l3_rand)) < 0.10 else
               "the two backbones degrade by materially different amounts even off the "
               "ceiling, so the asymmetry is not purely an artefact")),
    }

    (OUT / "S4_exploratory_dprime.json").write_text(
        json.dumps({"per_scenario": table, "verdict": verdict}, indent=2))

    print("\n=== d' relative to level 0 ===")
    for bb in ("real", "random"):
        print(f"  -- {bb} --")
        print(f"  {'scenario':13s}" + "".join(f"{'L'+L:>10s}" for L in LEVELS))
        for sc in SCENARIOS:
            print(f"  {sc:13s}" + "".join(
                f"{dp_rel[bb][sc][L]:10.3f}" if dp_rel[bb][sc][L] is not None
                else f"{'-':>10s}" for L in LEVELS))
    print(f"\n  real   d' retained at L3: mean {np.nanmean(l3_real):.3f} "
          f"(min {np.nanmin(l3_real):.3f})")
    print(f"  random d' retained at L3: mean {np.nanmean(l3_rand):.3f} "
          f"(min {np.nanmin(l3_rand):.3f})")
    print(f"  scenarios with real d' down >5% at L3: {n_dp_down}/7")
    print(f"\n  {verdict['reading']}")
    print(f"\n  {verdict['margin_caveat']}")
    print(f"\nwrote {OUT/'S4_exploratory_dprime.json'}")


if __name__ == "__main__":
    main()
