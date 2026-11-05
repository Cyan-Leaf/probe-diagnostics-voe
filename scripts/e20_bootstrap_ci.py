"""E20: cluster-bootstrap confidence intervals for the D1 and D2 verdicts.

Every verdict in this project is a point estimate. E16 measured the probe-partition sd at
2.51 pp, E18 found a feature-seed sd of 20-26 pp in the weak-signal region, and D1 is a
*ratio* whose denominator was observed as low as 0.096 — which can turn an AUC gap of 0.0116
into ~10 pp of training share. A point estimate cannot say whether a verdict survives its own
resampling error.

**Method, and why it is not a re-fit bootstrap.** The design is frozen in
`results/E20_bootstrap_ci/E20_PREREG.md`. PREREG §2/§3 as written implied refitting the probe
inside every draw. Measured, that is infeasible here: one probe fit over 1012 x 24960 features
is 13-30 s depending on BLAS threading, so 36 cells x 300 draws would be tens of hours of CPU
and, at 64 workers, thrash memory bandwidth until each fit slows to several minutes. Instead
this uses the standard **out-of-fold score bootstrap**: fit the frozen probe ONCE per cell,
keep its held-out decision scores, then resample CLUSTERS and recompute AUC on the resampled
score set for each draw.

What that changes, stated plainly: the interval conditions on the fitted model and measures
the sampling variability of the *evaluation set*, not the additional variability of refitting
the classifier. It is therefore slightly narrower than a full refit bootstrap would be. That
is a real limitation and it is written in the report; the measured probe-partition sd (E16,
2.51 pp) is the scale of the component being omitted. The alternative was to run no interval
at all.

Other frozen decisions, unchanged:

  1. Resample CLUSTERS (scenes / subgroups), never rows — both probes use group-disjoint CV,
     so row resampling would put a scene on both sides of a fold.
  2. PAIR the arms: within a draw, every arm of a cell uses the same drawn clusters, and the
     interval is over the composed quantity (a ratio for D1, a difference for D2), never over
     two marginal intervals.
  3. Duplicated clusters keep their cluster id.
  4. Percentile 95% interval, B=300.

`probe` internals are imported, never re-implemented (E11: a rewrite shifts AUC by 2.3 pp).

Run (all CPU, ~15 min):
  python scripts/e20_bootstrap_ci.py --workers 24 --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_analyse as A8
from e10_s1s3 import _shuffled_group_folds
from e8_verdict import d1_ratio
from t4_caliper import CS, _decision, _fit_path, _rowspace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E20_bootstrap_ci"

B = 300
DRAW_SEED = 20261105
FEATURE_SEEDS = [0, 1, 2]   # E8 averages over these three; matching it is mandatory
CI_LO, CI_HI = 2.5, 97.5
D1_THRESHOLD = 0.50

BACKBONES = ["cogvideox2b", "cogvideox5b", "ltx", "mochi", "wan21_14b", "wan21_1b3"]
DATASETS = ["intphys2", "likephys"]
LAYER = 16
T = 600

# E8 assigns a different probe CV seed per arm; reproducing the delivered numbers needs the same
# ones. `_appearance_cj` uses 2000 + seed, `cell()` uses 1000 + seed (PREREG §6.4).
ARM_SEED = {"real": 1000, "random": 1000, "cj": 2000, "appearance_native": 1000}


# ---------------------------------------------------------------------------------------
# out-of-fold scores from the frozen probe
# ---------------------------------------------------------------------------------------
def oof_scores(X, y, groups, seed: int) -> np.ndarray:
    """Held-out decision scores from `e8_analyse.probe`'s exact fitting path.

    Mirrors probe() statement for statement — same fold dealer, same inner CV over C, same
    `_rowspace` reduction — with one difference: it returns the per-row scores instead of
    collapsing them to a fold-averaged AUC. `verify_against_probe` checks that the pooled AUC
    it induces agrees with probe() to well within the noise.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
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
            A, Bx = sc2.transform(Xtr[itr]), sc2.transform(Xtr[ite])
            A, Bx = _rowspace(A, Bx)
            for c, coef, b in _fit_path(A, ytr[itr], CS):
                scores[c].append(roc_auc_score(ytr[ite], _decision(Bx, coef, b)))
        best_c = max(CS, key=lambda c: np.mean(scores[c]) if scores[c] else -1)

        Ztr, Zte = _rowspace(Xtr, Xte)
        _, coef, b = _fit_path(Ztr, ytr, [best_c])[-1]
        oof[te] = _decision(Zte, coef, b)
    return oof


def verify_against_probe(name: str, X, y, groups, seed: int, oof: np.ndarray) -> dict:
    """Self-check (PREREG §6.2): the pooled OOF AUC must reproduce probe()'s AUC.

    It will not be bit-identical: probe() averages per-fold AUCs, while pooling the scores
    averages over the whole set. The gap is reported per cell rather than hidden, and the
    tolerance is the same 1e-3 AUC used for the identity check plus the known fold-averaging
    difference, which is measured here instead of assumed.
    """
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(oof)
    pooled = float(roc_auc_score(y[ok], oof[ok])) if ok.any() else float("nan")
    ref, _ = A8.probe(X, y, groups, seed)
    return {"cell": name, "pooled_oof_auc": pooled, "probe_auc": float(ref),
            "delta": pooled - float(ref)}


# ---------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------
def arm_data(bb, dataset, arm, seed: int):
    """-> (X, y, groups) for one arm, built exactly the way E8 builds it."""
    _, gkey = A8.load_dataset(dataset)

    if arm in ("real", "random"):
        X, meta = A8.load_X(bb, dataset, "plain", seed, LAYER, T, weights=arm)
        if X is None:
            return None
        sel = [i for i, m in enumerate(meta) if m["label"] in (0, 1)]
        return (np.asarray(X[sel], dtype=np.float32),
                np.array([meta[i]["label"] for i in sel]),
                np.array([A8.group_of(meta[i], gkey) for i in sel]))

    if arm == "cj":
        Xp, mp = A8.load_X(bb, dataset, "plain", seed, LAYER, T, weights="real")
        Xc, mc = A8.load_X(bb, dataset, "cj", seed, LAYER, T, weights="real")
        if Xp is None or Xc is None:
            return None
        keep = [i for i, m in enumerate(mp) if m["label"] == 0]
        Xa = np.asarray(Xp[keep], dtype=np.float32)
        Xb = np.asarray(Xc, dtype=np.float32)
        return (np.concatenate([Xa, Xb], axis=0),
                np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)]),
                np.array([A8.group_of(mp[i], gkey) for i in keep]
                         + [A8.group_of(m, gkey) for m in mc]))

    if arm == "appearance_native":
        X, meta = A8.load_X(bb, dataset, "plain", seed, LAYER, T, weights="real")
        if X is None:
            return None
        sel = [i for i, m in enumerate(meta)
               if (m["group"] == "physics" and m["label"] == 0) or m["kind"] == "color_change"]
        return (np.asarray(X[sel], dtype=np.float32),
                np.array([0 if (meta[i]["group"] == "physics" and meta[i]["label"] == 0) else 1
                          for i in sel]),
                np.array([A8.group_of(meta[i], gkey) for i in sel]))

    raise ValueError(arm)


def _arm_task(spec: dict) -> dict:
    """One (backbone, dataset, arm) fit. Top-level and primitive-only so it pickles."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=spec["threads"]):
        d = arm_data(spec["bb"], spec["dataset"], spec["arm"], spec["seed"])
        if d is None:
            return {"key": spec["key"], "ok": False}
        X, y, g = d
        seed = ARM_SEED[spec["arm"]]
        oof = oof_scores(X, y, g, seed)
        chk = verify_against_probe(spec["key"], X, y, g, seed, oof)
        return {"key": spec["key"], "ok": True, "y": y, "groups": g, "oof": oof,
                "check": chk}


# ---------------------------------------------------------------------------------------
# cluster bootstrap over the frozen scores
# ---------------------------------------------------------------------------------------
def cluster_draws(groups: np.ndarray, b: int, seed: int) -> list:
    """`b` draws of CLUSTER ORDINALS (not row indices), from the sorted unique cluster list.

    Storing ordinals is what keeps the arms paired: every arm of a cell maps the same ordinal
    sequence onto its own rows, so draw *i* always means "the same clusters" in every arm even
    though the arms hold different row sets (PREREG §3). Duplicated clusters keep their id, so
    their rows carry double weight — which is what a cluster bootstrap means.
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    return [rng.integers(0, len(uniq), size=len(uniq)) for _ in range(b)]


def expand(ordinals: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cluster ordinals -> row indices for THIS arm, whose clusters are in sorted order."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return np.concatenate([rows_of[k] for k in ordinals])


def boot_auc(y: np.ndarray, oof: np.ndarray, groups: np.ndarray, ords: list) -> np.ndarray:
    """AUC of the frozen scores on each resampled cluster set."""
    from sklearn.metrics import roc_auc_score

    vals = np.empty(len(ords))
    for i, o in enumerate(ords):
        idx = expand(o, groups)
        yy, ss = y[idx], oof[idx]
        vals[i] = roc_auc_score(yy, ss) if len(np.unique(yy)) > 1 else np.nan
    return vals


def pct(v: np.ndarray) -> list:
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, CI_LO)), float(np.percentile(v, CI_HI))]


def mc(v: np.ndarray) -> dict:
    """Two disjoint halves: how much of the width is Monte-Carlo noise (PREREG §6.3)."""
    v = v[np.isfinite(v)]
    if len(v) < 40:
        return {"fraction": float("nan")}
    a, b = v[0::2], v[1::2]
    la, ha = np.percentile(a, [CI_LO, CI_HI])
    lb, hb = np.percentile(b, [CI_LO, CI_HI])
    w = float(ha - la)
    return {"lo_disagree": float(abs(la - lb)), "hi_disagree": float(abs(ha - hb)),
            "width": w,
            "fraction": float(max(abs(la - lb), abs(ha - hb)) / w) if w > 0 else float("nan")}


def compose(bb, dataset, arms_res, draws) -> dict:
    """D1 and D2 intervals from the per-arm frozen scores, paired over the same draws.

    `arms_res[arm]` maps field -> LIST over feature seeds (y, oof, groups). The point estimate
    is the 3-seed mean, matching E8's `auc_mean`; the interval is taken over draws with the
    per-draw statistic computed as the mean over the same three seeds, so the interval width
    includes feature-seed variability the way the delivered point estimate does.
    """
    from sklearn.metrics import roc_auc_score

    out = {}
    real, rand = arms_res["real"], arms_res["random"]
    app_key = "cj" if dataset == "intphys2" else "appearance_native"
    app = arms_res[app_key]

    def draw_curves(arm_res):
        """[n_draws, n_seeds] AUC of the frozen scores on each resampled cluster set."""
        cols = [boot_auc(y, oof, g, draws)
                for y, oof, g in zip(arm_res["y"], arm_res["oof"], arm_res["groups"])]
        return np.vstack(cols).T

    c_real, c_rand, c_app = draw_curves(real), draw_curves(rand), draw_curves(app)

    # Point estimates: 3-seed mean of the pooled OOF AUC, then E8's ratio on those means.
    p_real = float(np.mean([roc_auc_score(y, oof)
                            for y, oof in zip(real["y"], real["oof"])]))
    p_rand = float(np.mean([roc_auc_score(y, oof)
                            for y, oof in zip(rand["y"], rand["oof"])]))
    p_app = float(np.mean([roc_auc_score(y, oof)
                           for y, oof in zip(app["y"], app["oof"])]))

    # D1 = (AUC_real - AUC_random) / (AUC_real - 0.5). The denominator comes from
    # e8_verdict.d1_ratio rather than being rewritten: dividing by AUC_real instead turns 91%
    # into 22% and silently inverts the project's headline verdict (this happened on E20's
    # first run, which is why the function is imported).
    d1_draws = np.array([d1_ratio(r, q) for r, q in
                         zip(c_real.mean(axis=1), c_rand.mean(axis=1))])
    p_share = d1_ratio(p_real, p_rand)
    lo, hi = pct(d1_draws)

    # Per-seed sanity: how much does seed 0 alone differ from the 3-seed mean? E8 has this
    # spread on the record (LTX: 48.47% vs 18.08%), so it is reported, not hidden.
    per_seed = [d1_ratio(roc_auc_score(real["y"][i], real["oof"][i]),
                         roc_auc_score(rand["y"][i], rand["oof"][i]))
                for i in range(len(real["y"]))]

    out[f"{bb}|{dataset}|D1"] = {
        "cell": f"{bb}|{dataset}|D1", "backbone": bb, "dataset": dataset,
        "layer": LAYER, "timestep": T, "n_draws_used": int(np.isfinite(d1_draws).sum()),
        "auc_real_point": p_real, "auc_random_point": p_rand,
        "d1_pct_point": p_share * 100,
        "d1_pct_per_seed": [round(100.0 * v, 2) for v in per_seed],
        "d1_pct_ci95": [lo * 100, hi * 100], "d1_pct_ci_width": (hi - lo) * 100,
        "threshold_pct": D1_THRESHOLD * 100,
        "verdict_point": "pass" if p_share >= D1_THRESHOLD else "fail",
        "verdict_fragile": bool(lo < D1_THRESHOLD <= hi),
        "monte_carlo": mc(d1_draws),
        "method": "out-of-fold score bootstrap, 3-seed mean (see module docstring)",
    }

    gap_draws = c_app.mean(axis=1) - c_real.mean(axis=1)
    p_gap = p_app - p_real
    lo2, hi2 = pct(gap_draws)
    out[f"{bb}|{dataset}|D2"] = {
        "cell": f"{bb}|{dataset}|D2", "backbone": bb, "dataset": dataset,
        "layer": LAYER, "timestep": T, "n_draws_used": int(np.isfinite(gap_draws).sum()),
        "appearance_arm": app_key,
        "auc_physics_point": p_real, "auc_appearance_point": p_app,
        "gap_point": p_gap, "gap_ci95": [lo2, hi2], "gap_ci_width": hi2 - lo2,
        "verdict_point": "fail" if p_gap > 0 else "pass",
        "verdict_fragile": bool(lo2 <= 0 < hi2),
        "monte_carlo": mc(gap_draws),
        "method": "out-of-fold score bootstrap, 3-seed mean (see module docstring)",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "E20: cluster-bootstrap 95% CI for the D1/D2 verdicts",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prereg": "results/E20_bootstrap_ci/E20_PREREG.md",
        "method_note": ("Out-of-fold score bootstrap: the frozen probe is fit once per cell and "
                        "the resampling acts on the held-out scores. PREREG §2/§3 implied a "
                        "refit-per-draw bootstrap, which measured infeasible (13-30 s per fit). "
                        "Consequence: intervals omit the refit component of the variance "
                        "(E16 puts that at 2.51 pp). See the module docstring and E20_REPORT."),
        "protocol": {"B": B, "draw_seed": DRAW_SEED, "feature_seeds": FEATURE_SEEDS,
                     "arm_probe_seed": ARM_SEED, "threads": args.threads,
                     "ci": [CI_LO, CI_HI], "layer": LAYER, "timestep": T,
                     "resample_unit": "cluster (scene_index / scenario-subgroup)",
                     "estimator": "percentile, no BCa"},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__},
        "cells": {}, "self_check": {},
    }

    arms_by_ds = {"intphys2": ["real", "random", "cj"],
                  "likephys": ["real", "random", "appearance_native"]}
    t_all = time.time()

    for dataset in args.datasets:
        arms = arms_by_ds[dataset]
        specs = [{"bb": bb, "dataset": dataset, "arm": a, "threads": args.threads,
                  "key": f"{bb}|{dataset}|{a}|s{seed}", "seed": seed}
                 for bb in args.backbones for a in arms for seed in FEATURE_SEEDS]
        print(f"[{dataset}] {len(specs)} probe fits ({len(args.backbones)} backbones x "
              f"{len(arms)} arms x {len(FEATURE_SEEDS)} feature seeds)", flush=True)

        t0 = time.time()
        res = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_arm_task, specs):
                if not r["ok"]:
                    print(f"  [skip] {r['key']}: missing cache", flush=True)
                    continue
                res[r["key"]] = r
                c = r["check"]
                print(f"  {r['key']:34s} pooled {c['pooled_oof_auc']:.4f} vs probe "
                      f"{c['probe_auc']:.4f} (d={c['delta']:+.2e})", flush=True)
                with (OUT / "E20_arms.journal.jsonl").open("a") as fh:
                    fh.write(json.dumps({k: v for k, v in r.items()
                                         if k not in ("oof",)},
                                        default=lambda o: o.tolist() if hasattr(o, "tolist")
                                        else None) + "\n")
        print(f"  {len(res)} arms fitted in {time.time()-t0:.0f}s", flush=True)

        for bb in args.backbones:
            try:
                # Every arm carries all three feature seeds; the point estimate averages the
                # per-seed AUCs exactly as E8 does (d1_pct/auc_mean are 3-seed means, and
                # seed-0 alone differs by up to 30 pp of D1 on LTX).
                arms_res = {a: {k: [res[f"{bb}|{dataset}|{a}|s{sd}"][k]
                                    for sd in FEATURE_SEEDS]
                                for k in ("y", "oof", "groups")} for a in arms}
            except KeyError:
                continue
            # Cluster counts must agree across arms, otherwise one ordinal sequence cannot
            # mean the same thing in every arm and the pairing is void. Assert, never paper over.
            n_cl = {a: len(np.unique(v["groups"][0])) for a, v in arms_res.items()}
            assert len(set(n_cl.values())) == 1, f"{bb}|{dataset}: cluster counts differ {n_cl}"
            draws = cluster_draws(arms_res["real"]["groups"][0], B, DRAW_SEED)
            payload["self_check"][f"{bb}|{dataset}"] = {
                a: [res[f"{bb}|{dataset}|{a}|s{sd}"]["check"] for sd in FEATURE_SEEDS]
                for a in arms}
            payload["cells"].update(compose(bb, dataset, arms_res, draws))

    payload["elapsed_min"] = round((time.time() - t_all) / 60, 1)
    (OUT / "E20_cells.json").write_text(json.dumps(payload, indent=1, default=lambda o:
                                                   o.tolist() if hasattr(o, "tolist") else None))
    print(f"\nwrote {OUT/'E20_cells.json'} in {payload['elapsed_min']} min", flush=True)
    tally(payload)


def tally(payload):
    cells = payload["cells"]
    d1 = [c for c in cells.values() if c["cell"].endswith("D1")]
    d2 = [c for c in cells.values() if c["cell"].endswith("D2")]
    t = {"task": payload["task"], "created": payload["created"],
         "B": payload["protocol"]["B"],
         "method": payload["method_note"],
         "d1": {"cells": len(d1), "robust": sum(not c["verdict_fragile"] for c in d1),
                "fragile": sum(c["verdict_fragile"] for c in d1),
                "fragile_cells": [{"cell": c["cell"], "d1_pct_point": c["d1_pct_point"],
                                   "ci95": c["d1_pct_ci95"]}
                                  for c in d1 if c["verdict_fragile"]]},
         "d2": {"cells": len(d2), "robust": sum(not c["verdict_fragile"] for c in d2),
                "fragile": sum(c["verdict_fragile"] for c in d2),
                "fragile_cells": [{"cell": c["cell"], "gap_point": c["gap_point"],
                                   "ci95": c["gap_ci95"]}
                                  for c in d2 if c["verdict_fragile"]]},
         "why": ("fragile = the 95% interval straddles the decision boundary. The point-estimate "
                 "verdicts in D1_TALLY/D2_TALLY are NOT retracted by this file.")}
    (OUT / "E20_verdict.json").write_text(json.dumps(t, indent=1))
    print(f"D1 robust {t['d1']['robust']}/{t['d1']['cells']}  "
          f"D2 robust {t['d2']['robust']}/{t['d2']['cells']}", flush=True)


if __name__ == "__main__":
    main()
