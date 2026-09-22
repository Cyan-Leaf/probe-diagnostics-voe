"""T12.3: the two defensibility gaps that existing cached data can close.

T12.1(d) found five likely reviewer questions. Two of them have no answer anywhere in the
manuscript:

  Q4  "how stable is the 0.8576 random-init baseline?"
  Q1  "you only used a linear probe -- would a non-linear probe change the conclusion?"

Q4 is answerable from cache at zero extra cost: E3 already extracted three independent
random initialisations (feature seeds 0/1/2) at every timestep, so the spread across
initialisations can just be read out.

Q1 needs a new fit but not new extraction: the cached features support an MLP probe under
the identical folds. Whatever it returns is reported as-is -- if a non-linear probe closes
the trained-vs-random gap, that weakens D1 and must be said.

Run: python scripts/t12_defensibility_gaps.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_s1s3 import _shuffled_group_folds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T12_audit"
CONS = ROOT / "cache" / "consolidated"
LAYER, T = 16, 600


# ---------------------------------------------------------------- Q4: baseline stability
def q4_random_init_stability():
    raw = json.loads((ROOT / "results/E3_readout_grid/E3_probe_raw.json").read_text())["results"]
    rows = [r for r in raw
            if r["source"] == "random_dit" and r["task"] == "valid_vs_violation"]
    by_t = {}
    for r in rows:
        by_t.setdefault(r["t"], {})[r["seed"]] = r["auc_mean"]

    per_t = {}
    for t, d in sorted(by_t.items()):
        v = np.array([d[s] for s in sorted(d)], float)
        per_t[str(t)] = {
            "seeds": sorted(d),
            "aucs": [float(x) for x in v],
            "mean": float(v.mean()),
            "sd_across_inits": float(v.std(ddof=1)),
            "range": float(v.max() - v.min()),
        }

    e1 = json.loads((ROOT / "results/T4_caliper/E1_summary.json").read_text())
    lb = e1["lower_bounds"]["random_dit(best layer)"]
    margin = e1["best_real"]["auc_mean"] - lb["auc_mean"]
    sds = [v["sd_across_inits"] for v in per_t.values()]

    return {
        "question": "how stable is the random-init lower bound?",
        "per_timestep": per_t,
        "sd_across_inits_range": [float(min(sds)), float(max(sds))],
        "sd_across_inits_max": float(max(sds)),
        "reported_cv_sd_at_best_layer": lb["auc_std"],
        "margin_real_minus_random": float(margin),
        "margin_over_max_init_sd": float(margin / max(sds)),
        "reading": (
            f"Three independent random initialisations give a spread of "
            f"{min(sds):.4f}-{max(sds):.4f} AUC across timesteps. The trained-minus-random "
            f"margin is {margin:.4f}, i.e. {margin/max(sds):.0f}x the largest "
            f"initialisation spread, so the D1 conclusion does not hinge on which random "
            f"initialisation was drawn. Note this is the spread across *initialisations*; "
            f"the {lb['auc_std']:.4f} quoted in E1 is the across-fold sd, a different "
            f"quantity."),
    }


# ---------------------------------------------------------------- Q1: non-linear probe
def load_physics(prefix, seed=0):
    idx = json.loads((CONS / f"{prefix}_index.json").read_text())
    keep = [i for i, r in enumerate(idx) if r["group"] == "physics"]
    mm = np.load(CONS / f"{prefix}_t{T}_s{seed}.npy", mmap_mode="r")
    X = np.asarray(mm[:, LAYER][keep], dtype=np.float32).reshape(len(keep), -1)
    y = np.array([idx[i]["label"] for i in keep], dtype=int)
    g = np.array([f"{idx[i]['scenario']}/{idx[i]['subgroup']}" for i in keep])
    return X, y, g


def mlp_auc(X, y, groups, seed=0, hidden=(256,)):
    """Same folds and same scaling as the linear probe; only the head changes.

    PCA to 256 dims first: an MLP on 39,936 raw dims with n=750 would be dominated by the
    optimiser's behaviour in a p >> n regime rather than by what the representation holds,
    which would not answer the question being asked.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    aucs = []
    for tr, te in _shuffled_group_folds(groups, 5, rng):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pca = PCA(n_components=min(256, len(tr) - 1), random_state=0).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        m = MLPClassifier(hidden_layer_sizes=hidden, alpha=1.0, max_iter=800,
                          random_state=seed, early_stopping=True, n_iter_no_change=20)
        m.fit(Ztr, y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Zte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def q1_nonlinear_probe():
    from threadpoolctl import threadpool_limits

    out = {}
    with threadpool_limits(limits=8):
        for prefix, name in (("real", "trained"), ("random", "random-init")):
            X, y, g = load_physics(prefix)
            per = [mlp_auc(X, y, g, seed=s) for s in (0, 1, 2)]
            a = np.array([p[0] for p in per])
            out[name] = {"auc_mean": float(a.mean()), "auc_sd": float(a.std(ddof=1)),
                         "per_seed": [float(x) for x in a]}
            print(f"  MLP {name:12s} AUC {a.mean():.4f} +- {a.std(ddof=1):.4f}", flush=True)

    e3 = json.loads((ROOT / "results/E3_readout_grid/E3_summary.json").read_text())
    lin_real = e3["valid_vs_violation"]["600"]["real_dit"]["auc_mean"]
    lin_rand = e3["valid_vs_violation"]["600"]["random_dit"]["auc_mean"]
    lin_share = (lin_real - lin_rand) / (lin_real - 0.5)

    mr, mq = out["trained"]["auc_mean"], out["random-init"]["auc_mean"]
    mlp_share = (mr - mq) / (mr - 0.5) if mr > 0.5 else float("nan")

    return {
        "question": "would a non-linear probe change the conclusion?",
        "setup": ("same folds, same scaling, PCA to 256 dims, one hidden layer of 256 with "
                  "alpha=1.0 and early stopping; layer 16 / t=600 to match the fixed cell"),
        "mlp": out,
        "linear_reference": {"real": lin_real, "random": lin_rand,
                             "training_share": lin_share},
        "mlp_training_share": float(mlp_share),
        "share_change": float(mlp_share - lin_share),
        "reading": (
            f"A one-hidden-layer MLP reaches {mr:.4f} on the trained backbone and {mq:.4f} "
            f"on the random-initialised one, a training share of {mlp_share*100:.1f}% "
            f"against {lin_share*100:.1f}% for the linear probe "
            f"({(mlp_share-lin_share)*100:+.1f} points). "
            + ("The D1 verdict is unchanged: a stronger readout raises both sides, so the "
               "share attributable to training stays below the 50% threshold."
               if mlp_share < 0.5 else
               "The stronger readout pushes the training share ABOVE the 50% threshold, so "
               "D1's verdict on this cell is readout-dependent and must be reported as "
               "such.")),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Q4: random-init baseline stability (from cache) ===")
    q4 = q4_random_init_stability()
    for t, v in q4["per_timestep"].items():
        print(f"  t={t:>3s}  aucs={[f'{x:.4f}' for x in v['aucs']]}  "
              f"sd={v['sd_across_inits']:.4f}")
    print(f"  -> {q4['reading']}\n")

    print("=== Q1: non-linear probe (new fit, cached features) ===")
    q1 = q1_nonlinear_probe()
    print(f"  -> {q1['reading']}")

    (OUT / "T12_defensibility_gaps.json").write_text(
        json.dumps({"Q4_baseline_stability": q4, "Q1_nonlinear_probe": q1},
                   indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT/'T12_defensibility_gaps.json'}")


if __name__ == "__main__":
    main()
