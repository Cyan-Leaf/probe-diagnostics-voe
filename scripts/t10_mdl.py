"""T10.5: redo D1 with MDL codelength, the statistic Voita & Titov actually recommend.

Why this exists.  T10.4 found that our D1 genealogy claim has a hole: Voita & Titov (2020)
do say probe accuracy fails to separate trained from randomly initialised representations,
but their paper's contribution is the *remedy* -- MDL codelength -- and they report that
codelength DOES separate them:

    "With MDL probes, we will see that codelength shows large difference between trained and
     randomly initialized representations."   (§4)
    "gain from using context for the randomly initialized model is at least twice smaller
     than for the trained model."             (§4.2)

Our D1 is a ratio of AUC gaps, i.e. exactly the accuracy-family statistic they criticise.
So the obvious objection is: your 24.6% is an artefact of using the wrong measure.

This settles it by computing the online (prequential) code from their §2.2.2 on our cached
features.  Two outcomes, both informative:

  codelength also shows trained ~ random  -> our conclusion survives their own statistic
  codelength shows a large gap            -> our 24.6% understates training's contribution
                                             and D1's number must be revised

Online code (their eq. 4): split the training data into blocks t_1 < t_2 < ... < t_S = n;
for each block, train on everything before it and pay -log2 p(y | x) on the block:

    L_online = t_1 * log2(K) + sum_i [ -sum_{j in block i} log2 p_hat(y_j | x_j) ]

with the first block charged at the uniform code.  Compression = uniform / L_online.

Run: python scripts/t10_mdl.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T10_validity"
CONS = ROOT / "cache" / "consolidated"
LAYER = 16
T = 600
# their §2.2.2 uses fractions of the training set; these are the same style of schedule
FRACTIONS = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
EPS = 1e-12


def load(prefix: str, seed: int = 0):
    idx = json.loads((CONS / f"{prefix}_index.json").read_text())
    keep = [i for i, r in enumerate(idx) if r["group"] == "physics"]
    mm = np.load(CONS / f"{prefix}_t{T}_s{seed}.npy", mmap_mode="r")
    X = np.asarray(mm[:, LAYER][keep], dtype=np.float32).reshape(len(keep), -1)
    y = np.array([idx[i]["label"] for i in keep], dtype=int)
    g = np.array([f"{idx[i]['scenario']}/{idx[i]['subgroup']}" for i in keep])
    return X, y, g


def online_codelength(X, y, groups, seed=0, C=1.0):
    """Prequential codelength in bits, with group-aware ordering.

    Blocks are formed over *groups* rather than rows so that a clip's subgroup-mates are
    never split across the train/charge boundary; otherwise the code would be paid on rows
    whose near-duplicates were already seen, which understates the codelength for both
    models equally but by a data-dependent amount.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    order = np.concatenate([np.flatnonzero(groups == u) for u in uniq])
    Xo, yo = X[order], y[order]
    n = len(yo)

    bounds = sorted({max(2, int(round(f * n))) for f in FRACTIONS})
    if bounds[-1] != n:
        bounds.append(n)

    # first block: uniform code over K=2 classes
    L = bounds[0] * np.log2(2.0)
    per_block = [{"upto": bounds[0], "bits": float(L), "mode": "uniform"}]

    for a, b in zip(bounds[:-1], bounds[1:]):
        Xtr, ytr = Xo[:a], yo[:a]
        Xte, yte = Xo[a:b], yo[a:b]
        if len(np.unique(ytr)) < 2:
            bits = (b - a) * np.log2(2.0)
        else:
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(C=C, max_iter=2000, class_weight=None)
            m.fit(sc.transform(Xtr), ytr)
            p = m.predict_proba(sc.transform(Xte))
            cls = list(m.classes_)
            pt = np.array([p[i, cls.index(yte[i])] for i in range(len(yte))])
            bits = float(-np.sum(np.log2(np.clip(pt, EPS, 1.0))))
        L += bits
        per_block.append({"upto": int(b), "bits": float(bits), "mode": "model"})

    uniform_total = n * np.log2(2.0)
    return {"codelength_bits": float(L),
            "codelength_kbits": float(L / 1000.0),
            "uniform_bits": float(uniform_total),
            "compression": float(uniform_total / L),
            "n": int(n), "per_block": per_block}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"method": ("online / prequential code, Voita & Titov (2020) eq. 4; blocks formed "
                      "over (scenario, subgroup) groups"),
           "layer": LAYER, "t": T, "fractions": FRACTIONS, "runs": {}}

    # C sensitivity: the AUC pipeline selects C by inner CV, so a codelength computed at a
    # single arbitrary C would not be a fair comparison -- a badly scaled C could inflate the
    # random model's codelength and manufacture the very gap we are testing for.
    C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    res["C_grid"] = C_GRID
    res["C_sensitivity"] = {}
    for prefix in ("real", "random"):
        X, y, g = load(prefix)
        res["C_sensitivity"][prefix] = {
            str(c): online_codelength(X, y, g, seed=0, C=c)["compression"] for c in C_GRID}
    print("  C sensitivity (compression, seed 0):", flush=True)
    for prefix in ("real", "random"):
        row = "  ".join(f"C={c:<6g} {res['C_sensitivity'][prefix][str(c)]:.3f}"
                        for c in C_GRID)
        print(f"    {prefix:7s} {row}", flush=True)
    best_C = max(C_GRID, key=lambda c: min(res["C_sensitivity"]["real"][str(c)],
                                           res["C_sensitivity"]["random"][str(c)]))
    res["C_used"] = best_C
    print(f"  -> using C={best_C} (maximises the WEAKER of the two, so the comparison "
          f"cannot be rigged in favour of the trained model)\n", flush=True)

    for prefix, name in (("real", "trained DiT"), ("random", "randomly initialised DiT")):
        X, y, g = load(prefix)
        per_seed = [online_codelength(X, y, g, seed=s, C=best_C) for s in (0, 1, 2)]
        cl = np.array([r["codelength_kbits"] for r in per_seed])
        cp = np.array([r["compression"] for r in per_seed])
        res["runs"][prefix] = {
            "name": name, "n": per_seed[0]["n"],
            "codelength_kbits_mean": float(cl.mean()),
            "codelength_kbits_sd": float(cl.std(ddof=1)),
            "compression_mean": float(cp.mean()),
            "compression_sd": float(cp.std(ddof=1)),
            "per_seed": per_seed,
        }
        print(f"  {name:28s} codelength {cl.mean():7.3f} +- {cl.std(ddof=1):.3f} kbits   "
              f"compression {cp.mean():.3f}", flush=True)

    r, q = res["runs"]["real"], res["runs"]["random"]
    # AUC-side D1 for comparison (delivered numbers, layer 14 @ t=400 best cell)
    auc_real, auc_rand = 0.9740, 0.8576
    auc_share = (auc_real - auc_rand) / (auc_real - 0.5)

    gap_kbits = q["codelength_kbits_mean"] - r["codelength_kbits_mean"]
    uniform_kbits = r["per_seed"][0]["uniform_bits"] / 1000.0
    gain_real = uniform_kbits - r["codelength_kbits_mean"]
    gain_rand = uniform_kbits - q["codelength_kbits_mean"]
    # The "share" formula that mirrors the AUC one is only meaningful while BOTH models
    # compress at all.  When the random model's code is longer than uniform (compression
    # < 1) its gain is negative and the ratio exceeds 100%, which is a sign the quantity has
    # broken down, not a finding.  So report compression directly and flag the degeneracy.
    share_well_defined = bool(gain_real > 0 and gain_rand > 0)
    mdl_share = ((gain_real - gain_rand) / gain_real
                 if share_well_defined else float("nan"))

    res["comparison"] = {
        "auc_based_D1_share": auc_share,
        "auc_real": auc_real, "auc_random": auc_rand,
        "codelength_real_kbits": r["codelength_kbits_mean"],
        "codelength_random_kbits": q["codelength_kbits_mean"],
        "codelength_gap_kbits": gap_kbits,
        "compression_real": r["compression_mean"],
        "compression_random": q["compression_mean"],
        "mdl_based_training_share": mdl_share,
        "share_well_defined": share_well_defined,
        "share_degenerate_reason": (None if share_well_defined else
                                    "the randomly initialised features do not compress at "
                                    "all (compression < 1, codelength longer than the "
                                    "uniform code), so a 'share of the gain' has no "
                                    "denominator to divide"),
        "C_used": best_C,
        "voita_titov_claim": ("codelength shows a large difference between trained and "
                              "randomly initialised representations, where accuracy does not"),
        "reading": None,
    }
    big_gap = (not share_well_defined) or (mdl_share > auc_share + 0.10)
    res["comparison"]["reading"] = (
        f"AUC-based D1 attributes {auc_share*100:.1f}% of the above-chance signal to "
        f"training (0.9740 vs 0.8576). Under the online code at C={best_C}: trained "
        f"compression {r['compression_mean']:.3f}, randomly initialised "
        f"{q['compression_mean']:.3f} "
        f"(codelength {r['codelength_kbits_mean']:.2f} vs {q['codelength_kbits_mean']:.2f} "
        f"kbits). "
        + ("The two statistics agree that training contributes a minority of the signal, so "
           "our D1 conclusion is NOT an artefact of using an accuracy-family measure -- it "
           "survives the statistic Voita & Titov actually recommend."
           if not big_gap else
           "Codelength separates the two far more sharply than AUC does -- exactly what "
           "Voita & Titov report for NLP. The randomly initialised representation, which "
           "looks like 88% of the trained one on AUC, does not compress the labels at all. "
           "Our headline 'training contributes only 24.6%' is therefore specific to the "
           "accuracy-family statistic and must be reported together with the codelength "
           "result, which points the other way."))
    res["comparison"]["mdl_and_auc_agree"] = bool(not big_gap)

    (OUT / "T10_5_mdl.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n  AUC-based D1 share:  {auc_share*100:.1f}%")
    print(f"  MDL-based D1 share:  {mdl_share*100:.1f}%")
    print(f"\n  {res['comparison']['reading']}")
    print(f"\nwrote {OUT/'T10_5_mdl.json'}")


if __name__ == "__main__":
    main()
