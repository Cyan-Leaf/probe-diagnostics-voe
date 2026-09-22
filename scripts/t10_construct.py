"""T10.2 / T10.3: construct validity (A1-A4) and the sufficiency attacks (C1, C2).

Criteria are frozen in results/T10_validity/T10_PREREG.md, written before this ran.

The idea is the one Hewitt & Liang used to establish control tasks: rather than arguing in
prose that the six checks are the right ones, build representations whose ground truth we
*define by construction*, then see whether the protocol's verdict matches.  The protocol is
the object under test here, not the tool.

Every case is a synthetic 256-dim feature matrix over the real LikePhys design (750 rows,
group = (scenario, subgroup)), so the probe protocol, the folds and the floor definition are
identical to the ones used on real data -- only the features are constructed.

Run: python scripts/t10_construct.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_s1s3 import _shuffled_group_folds
from t4_caliper import CS, _decision, _fit_path, _rowspace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T10_validity"
DIM = 256
N_SEEDS = 3
SAT_AUC = 0.99
FLOOR_AUC = 0.51


# ----------------------------------------------------------------------------------------
def design():
    """The real LikePhys physics design: labels, groups, plus an appearance label.

    Using the real grouping matters: a synthetic case that passed only because of an
    unrealistically clean fold structure would prove nothing about the protocol as applied.
    """
    idx = json.loads((ROOT / "cache/consolidated/real_index.json").read_text())
    keep = [i for i, r in enumerate(idx) if r["group"] == "physics"]
    y = np.array([idx[i]["label"] for i in keep], dtype=int)
    g = np.array([f"{idx[i]['scenario']}/{idx[i]['subgroup']}" for i in keep])
    scen = np.array([idx[i]["scenario"] for i in keep])
    # appearance label: an attribute-irrelevant split that is *independent* of y.
    # Built from scenario identity parity so it is stable and has nothing to do with physics.
    uniq = {s: k for k, s in enumerate(sorted(set(scen)))}
    a = np.array([uniq[s] % 2 for s in scen], dtype=int)
    return y, g, scen, a


def probe(X, y, groups, seed):
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


def over_seeds(build, y, g, seed0):
    """Run one constructed case over N_SEEDS construction seeds -> mean + seed spread."""
    A, D = [], []
    for k in range(N_SEEDS):
        X = build(seed0 + k)
        a, d = probe(X, y, g, seed=90_000 + k)
        A.append(a)
        D.append(d)
    A, D = np.array(A, float), np.array(D, float)
    return {"auc_mean": float(np.nanmean(A)), "auc_sd": float(np.nanstd(A, ddof=1)),
            "dprime_mean": float(np.nanmean(D)), "dprime_sd": float(np.nanstd(D, ddof=1)),
            "per_seed_auc": [float(x) for x in A]}


# ----------------------------------------------------------------------------------------
def make_signal(y, strength, dim=DIM, n_sig=8, seed=0, extra=None):
    """Linear-encoded label in the first n_sig dims, isotropic noise elsewhere.

    `extra` lets a case inject additional structured dimensions (appearance, artefact,
    batch id ...) without changing how the physics part is built.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(y), dim)).astype(np.float32)
    if strength > 0:
        X[:, :n_sig] += strength * (2.0 * y - 1.0)[:, None]
    if extra is not None:
        for start, vals, sc in extra:
            X[:, start:start + vals.shape[1]] += sc * vals
    return X


def onehotish(v, width, rng):
    """A structured block correlated with integer vector v (used for appearance/artefact)."""
    lv = np.asarray(v).astype(float)
    base = rng.standard_normal((width,))
    return np.outer(2.0 * lv - 1.0, base)


# ----------------------------------------------------------------------------------------
def evaluate_protocol(name, real_build, random_build, y, g, a, truth, expect, seed0=1000):
    """Apply D1-D6 to one constructed case and record each verdict with its number.

    D2 must interrogate *the same representation* on an attribute-irrelevant label -- that
    is what it does on real data (same cached features, valid-vs-colour_change instead of
    valid-vs-violation).  An earlier version of this harness built a fresh feature matrix
    for the appearance task, which meant D2 was testing a different representation and
    could never detect an appearance-only case.  So the appearance probe below reuses
    `real_build` and swaps the *label*.
    """
    real = over_seeds(real_build, y, g, seed0)
    rand = over_seeds(random_build, y, g, seed0 + 100)
    app = over_seeds(real_build, a, g, seed0 + 200)   # same features, appearance label

    floor = max(real["auc_sd"], 1e-6)
    effect = real["auc_mean"] - rand["auc_mean"]
    d1 = ((real["auc_mean"] - rand["auc_mean"]) / (real["auc_mean"] - 0.5)
          if real["auc_mean"] - 0.5 > 1e-9 else float("nan"))
    d1_dp = ((real["dprime_mean"] - rand["dprime_mean"]) / real["dprime_mean"]
             if abs(real["dprime_mean"]) > 1e-9 else float("nan"))

    checks = {}
    # D1 as originally written is a ratio whose denominator vanishes as real AUC -> chance,
    # so it can report a large "training share" of an effect that does not exist.  The
    # construct-validity run below exposes this (case A2).  The ratio is therefore reported
    # together with an interpretability flag rather than silently trusted.
    d1_interpretable = bool(effect > 2 * floor and real["auc_mean"] - 0.5 > 2 * floor)
    checks["D1"] = {"metric": "training-increment share",
                    "value": d1, "value_dprime": d1_dp, "threshold": 0.50,
                    "interpretable": d1_interpretable,
                    "passed": bool(np.isfinite(d1) and d1 >= 0.50 and d1_interpretable)}
    checks["D2"] = {"metric": "AUC(appearance) < AUC(physics), same representation",
                    "appearance_auc": app["auc_mean"], "physics_auc": real["auc_mean"],
                    "passed": bool(app["auc_mean"] < real["auc_mean"])}
    checks["D3"] = {"metric": "not saturated (effect > 2x floor AND auc < 0.99)",
                    "effect": effect, "floor": floor, "effect_over_2x_floor": effect / (2 * floor),
                    "real_auc": real["auc_mean"],
                    "passed": bool(effect > 2 * floor and real["auc_mean"] < SAT_AUC)}
    checks["D4"] = {"metric": "effect > 2x seed-spread floor",
                    "effect": effect, "floor": floor,
                    "passed": bool(effect > 2 * floor)}
    same_dir = (np.sign(real["auc_mean"] - 0.5) == np.sign(real["dprime_mean"]))
    checks["D5"] = {"metric": "AUC and d' agree in direction",
                    "auc_above_chance": real["auc_mean"] - 0.5,
                    "dprime": real["dprime_mean"],
                    "passed": bool(same_dir)}
    at_bound = real["auc_mean"] >= SAT_AUC or real["auc_mean"] <= FLOOR_AUC
    checks["D6"] = {"metric": "bounded statistic usable (baseline off the boundary)",
                    "real_auc": real["auc_mean"], "at_boundary": bool(at_bound),
                    "passed": bool(not at_bound)}

    n_pass = sum(1 for v in checks.values() if v["passed"] is True)
    n_fail = sum(1 for v in checks.values() if v["passed"] is False)

    verdict = {
        "case": name, "truth": truth, "expectation": expect,
        "real": real, "random": rand, "appearance": app,
        "checks": checks, "n_pass": n_pass, "n_fail": n_fail,
        "failed_checks": [k for k, v in checks.items() if v["passed"] is False],
    }
    return verdict


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y, g, scen, a = design()
    rng_app = np.random.default_rng(7)
    app_block = onehotish(a, 8, rng_app)
    print(f"design: n={len(y)} n_pos={int(y.sum())} groups={len(set(g))} "
          f"appearance split={np.bincount(a)}", flush=True)

    results = {}

    # ---- A1 positive: physics genuinely encoded ------------------------------------
    S = 0.42
    results["A1"] = evaluate_protocol(
        "A1_positive",
        lambda s: make_signal(y, S, seed=s),
        lambda s: make_signal(y, 0.25 * S, seed=s),
        y, g, a,
        truth="representation genuinely encodes physics",
        expect="D1-D6 all pass")

    # ---- A2 negative: appearance only ----------------------------------------------
    results["A2"] = evaluate_protocol(
        "A2_negative_appearance_only",
        lambda s: make_signal(np.zeros_like(y), 0.0, seed=s,
                              extra=[(0, app_block, 1.4)]),
        lambda s: make_signal(np.zeros_like(y), 0.0, seed=s,
                              extra=[(0, app_block, 0.35)]),
        y, g, a,
        truth="encodes no physics at all (appearance only)",
        expect="D1 or D2 must fail")

    # ---- A3 mixed: physics + edit artefact ------------------------------------------
    # The artefact has to be visible to BOTH labels: it tracks the edit (so it helps the
    # physics task) and it is the kind of low-level trace an appearance control detects.
    art_app = onehotish(y, 8, np.random.default_rng(11)) + onehotish(a, 8, np.random.default_rng(12))
    results["A3"] = evaluate_protocol(
        "A3_mixed_physics_plus_artefact",
        lambda s: make_signal(y, S, seed=s, extra=[(64, art_app, 1.1)]),
        lambda s: make_signal(y, 0.25 * S, seed=s, extra=[(64, art_app, 0.3)]),
        y, g, a,
        truth="physics partly encoded, plus an exploitable edit artefact",
        expect="D2 fails, D1 passes")

    # ---- A4 saturated ----------------------------------------------------------------
    results["A4"] = evaluate_protocol(
        "A4_saturated",
        lambda s: make_signal(y, 3.0, seed=s),
        lambda s: make_signal(y, 0.25 * 3.0, seed=s),
        y, g, a,
        truth="AUC is inflated; the statistic has no resolution left",
        expect="D3 or D6 must trigger")

    # ---- C1 attack: training-dependent, appearance-independent pseudo-signal --------
    # A leak that correlates with the label but is not physics: a "render batch id".
    rb = np.random.default_rng(23)
    batch = (np.asarray([hash(x) % 2 for x in g]) ^ y) & 1   # correlated with y, not appearance
    batch_block = onehotish(batch, 8, rb)
    results["C1"] = evaluate_protocol(
        "C1_correlated_pseudo_signal",
        lambda s: make_signal(np.zeros_like(y), 0.0, seed=s,
                              extra=[(0, batch_block, 1.6)]),
        lambda s: make_signal(np.zeros_like(y), 0.0, seed=s,
                              extra=[(0, batch_block, 0.40)]),
        y, g, a,
        truth="NOT physics: a label-correlated batch identifier",
        expect="attack succeeds if all six pass")

    # ---- C2 attack: pipeline-level difference, invisible to an appearance control ----
    pipe = onehotish(y, 8, np.random.default_rng(29))
    results["C2"] = evaluate_protocol(
        "C2_pipeline_leak",
        lambda s: make_signal(np.zeros_like(y), 0.0, seed=s, extra=[(128, pipe, 1.5)]),
        lambda s: make_signal(np.zeros_like(y), 0.0, seed=s, extra=[(128, pipe, 0.38)]),
        y, g, a,
        truth="NOT physics: the two classes went through different render pipelines",
        expect="attack succeeds if all six pass")

    # ---- adjudicate ------------------------------------------------------------------
    def judge(case, rule):
        v = results[case]
        f = set(v["failed_checks"])
        if rule == "all_pass":
            ok = v["n_fail"] == 0
        elif rule == "d1_or_d2_fail":
            ok = bool({"D1", "D2"} & f)
        elif rule == "d2_fail_d1_pass":
            ok = ("D2" in f) and ("D1" not in f)
        elif rule == "d3_or_d6_fail":
            ok = bool({"D3", "D6"} & f)
        else:
            ok = None
        v["protocol_correct"] = bool(ok)
        return ok

    judge("A1", "all_pass")
    judge("A2", "d1_or_d2_fail")
    judge("A3", "d2_fail_d1_pass")
    judge("A4", "d3_or_d6_fail")
    for c in ("C1", "C2"):
        results[c]["attack_succeeded"] = bool(results[c]["n_fail"] == 0)

    summary = {
        "construct_validity": {
            c: {"truth": results[c]["truth"], "expectation": results[c]["expectation"],
                "failed_checks": results[c]["failed_checks"],
                "protocol_correct": results[c]["protocol_correct"]}
            for c in ("A1", "A2", "A3", "A4")},
        "n_correct": sum(results[c]["protocol_correct"] for c in ("A1", "A2", "A3", "A4")),
        "attacks": {
            c: {"truth": results[c]["truth"],
                "failed_checks": results[c]["failed_checks"],
                "attack_succeeded": results[c]["attack_succeeded"]}
            for c in ("C1", "C2")},
        "any_attack_succeeded": any(results[c]["attack_succeeded"] for c in ("C1", "C2")),
    }
    summary["sufficiency_wording"] = (
        "minimal necessary (boundary identified)" if summary["any_attack_succeeded"]
        else "no successful attack in these two constructions; this does NOT license "
             "calling the protocol sufficient")

    (OUT / "T10_2_construct.json").write_text(
        json.dumps({"prereg": "results/T10_validity/T10_PREREG.md",
                    "cases": results, "summary": summary}, indent=2, ensure_ascii=False))

    print("\n=== construct validity (A) ===")
    for c in ("A1", "A2", "A3", "A4"):
        v = results[c]
        print(f"  {c} {v['case']:34s} real AUC {v['real']['auc_mean']:.4f} "
              f"rand {v['random']['auc_mean']:.4f} "
              f"app {v['appearance']['auc_mean']:.4f}")
        print(f"      D1 {v['checks']['D1']['value']:.3f}"
              f"{'' if v['checks']['D1']['interpretable'] else ' (NOT interpretable)'}"
              f" | failed={v['failed_checks'] or 'none'}"
              f" | expected: {v['expectation']} -> "
              f"{'CORRECT' if v['protocol_correct'] else 'PROTOCOL WRONG'}")
    print(f"\n  cases judged correctly: {summary['n_correct']}/4")

    print("\n=== sufficiency attacks (C) ===")
    for c in ("C1", "C2"):
        v = results[c]
        print(f"  {c} real AUC {v['real']['auc_mean']:.4f} rand {v['random']['auc_mean']:.4f} "
              f"app {v['appearance']['auc_mean']:.4f}")
        print(f"      failed={v['failed_checks'] or 'none'} -> "
              f"attack {'SUCCEEDED' if v['attack_succeeded'] else 'failed'}")
    print(f"\n  wording: {summary['sufficiency_wording']}")
    print(f"\nwrote {OUT/'T10_2_construct.json'}")


if __name__ == "__main__":
    main()
