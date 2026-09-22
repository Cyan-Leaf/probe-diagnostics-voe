"""T4 / E1 — CALIPER discriminability check + layer x timestep linear-probe scan.

Per results/T3_null_floor/protocol_as_executed.md:

  sources : real_dit (layer L, timestep t) | random_dit (same protocol) | vae_latent | raw_pixel
  probe   : StandardScaler -> LogisticRegression(L2), class_weight="balanced"
            C in {1e-3,1e-2,1e-1,1,10} chosen by 3-fold inner CV inside the training fold only
  split   : GroupKFold(5), group = (scenario, subgroup)  -> no scene leaks across folds
  metrics : AUC-ROC and accuracy, mean +- std over the 5 outer folds
  verdict : real_dit must beat the strongest lower bound by more than 2x the probe-level
            null floor (E0), otherwise the scenario set is not discriminative -> stop.

Compute note: with fresh (non-warm-started) lbfgs fits one fit on 750 x 39,936 features
costs ~1-2 s, so the full inner-CV protocol is ~2 min per config and is applied uniformly
to every source / layer / timestep / seed. (Warm-starting along the C path was tried first
and is 10x *slower* in this p >> n regime.)

Run:
  uv run --no-project python scripts/t4_caliper.py --stage consolidate --backbone real
  uv run --no-project python scripts/t4_caliper.py --stage probe --workers 12 --threads 8
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T4_caliper"
CONS = ROOT / "cache" / "consolidated"
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
FIXED_C = 1e-2
REF_T = 600


# --------------------------------------------------------------------------------------
def consolidate(backbone: str) -> None:
    """Turn per-clip .pt files into [n_clips, n_layers, F, D] fp16 .npy per (t, seed)."""
    cache = ROOT / "cache" / f"feats_{backbone}"
    files = sorted(cache.glob("*.pt"))
    assert files, f"no cache in {cache}"
    CONS.mkdir(parents=True, exist_ok=True)

    index = []
    buffers: dict[tuple[str, str], np.memmap] = {}
    shapes: dict[tuple[str, str], tuple] = {}
    extras = {"vae_latent": [], "raw_pixel": []}

    for i, f in enumerate(files):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        c = rec["clip"]
        index.append({k: c[k] for k in ("clip_id", "scenario", "subgroup", "kind", "group", "label")})
        extras["vae_latent"].append(rec["vae_latent_pooled"].numpy())
        extras["raw_pixel"].append(rec["raw_pixel_32"].numpy())
        for t, per_seed in rec["feats"].items():
            for seed, arr in per_seed.items():
                key = (t, seed)
                if key not in buffers:
                    shapes[key] = (len(files), *arr.shape)
                    path = CONS / f"{backbone}_t{t}_s{seed}.npy"
                    buffers[key] = np.lib.format.open_memmap(
                        path, mode="w+", dtype=np.float16, shape=shapes[key]
                    )
                buffers[key][i] = arr.numpy()
        if (i + 1) % 100 == 0:
            print(f"  consolidated {i+1}/{len(files)}", flush=True)

    for key, mm in buffers.items():
        mm.flush()
        print(f"  {backbone}_t{key[0]}_s{key[1]}.npy {shapes[key]}")
    for name, lst in extras.items():
        np.save(CONS / f"{backbone}_{name}.npy", np.stack(lst).astype(np.float16))
    (CONS / f"{backbone}_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print(f"consolidated {len(files)} clips for backbone={backbone}")


# --------------------------------------------------------------------------------------
def _fit_path(Xtr, ytr, cs):
    """Fit LogisticRegression once per C (fresh start).

    Warm-starting along the C path was tried and is *much* slower here: with p >> n a
    previous solution at a smaller C is a poor initialisation, and lbfgs needs an order of
    magnitude more iterations (measured 1038 s vs ~100 s for the same config).
    """
    from sklearn.linear_model import LogisticRegression

    models = []
    for c in cs:
        m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        models.append((c, m.coef_.copy(), m.intercept_.copy()))
    return models


def _decision(X, coef, intercept):
    return X @ coef.ravel() + intercept[0]


def _rowspace(A: np.ndarray, B: np.ndarray, tol: float = 1e-10):
    """Exact p >> n reduction: express both matrices in the row space of the training matrix.

    For an L2-regularised linear model the optimal w lies in span(rows of A), so with
    A = U S V^T, fitting on Z_tr = U S (n x r) at the same C gives the *identical* objective
    (||w||^2 = ||V^T w||^2, A w = Z_tr (V^T w)) and identical decision values; test points map
    with the same V, Z_te = B V. V is obtained from the n x n Gram matrix, which is far
    cheaper than an SVD of the n x p matrix.

    This is an exact reformulation, not a dimensionality-reduction approximation — the
    equivalence is checked numerically in scripts/t4_verify_rowspace.py.
    """
    K = A @ A.T
    w, Q = np.linalg.eigh(K)
    keep = w > max(tol, float(w.max()) * 1e-12)
    w, Q = w[keep], Q[:, keep]
    s = np.sqrt(w)
    Z_tr = Q * s
    Z_te = (B @ A.T) @ (Q / s)
    return np.ascontiguousarray(Z_tr), np.ascontiguousarray(Z_te)


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, mode: str = "innerCV") -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    aucs, accs, chosen, ranks = [], [], [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, gtr = y[tr], groups[tr]

        if mode == "innerCV":
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
        else:
            best_c = FIXED_C

        Ztr, Zte = _rowspace(Xtr, Xte)
        ranks.append(int(Ztr.shape[1]))
        _, coef, b = _fit_path(Ztr, ytr, [best_c])[-1]
        s = _decision(Zte, coef, b)
        aucs.append(roc_auc_score(y[te], s))
        accs.append(accuracy_score(y[te], (s > 0).astype(int)))
        chosen.append(best_c)

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "auc_folds": [float(a) for a in aucs],
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)),
        "chosen_C": chosen,
        "rowspace_rank": ranks,
        "mode": mode,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "dim": int(X.shape[1]),
    }


def _load_xy(spec: dict):
    idx = json.loads((CONS / f"{spec['backbone']}_index.json").read_text())
    keep = [i for i, r in enumerate(idx) if r["group"] == "physics"]
    y = np.array([idx[i]["label"] for i in keep], dtype=int)
    groups = np.array([f"{idx[i]['scenario']}/{idx[i]['subgroup']}" for i in keep])

    src = spec["source"]
    if src in ("real_dit", "random_dit"):
        mm = np.load(CONS / f"{spec['backbone']}_t{spec['t']}_s{spec['seed']}.npy", mmap_mode="r")
        X = np.asarray(mm[:, spec["layer"]][keep], dtype=np.float32)  # slice layer first (cheap)
    elif src == "vae_latent":
        X = np.asarray(np.load(CONS / "real_vae_latent.npy")[keep], dtype=np.float32)
    elif src == "raw_pixel":
        X = np.asarray(np.load(CONS / "real_raw_pixel.npy")[keep], dtype=np.float32)
    else:
        raise ValueError(src)
    return X.reshape(len(keep), -1), y, groups


def _job(spec: dict) -> dict:
    import time

    from threadpoolctl import threadpool_limits

    t0 = time.time()
    X, y, groups = _load_xy(spec)
    with threadpool_limits(limits=spec.get("threads", 8)):
        res = fit_probe(X, y, groups, mode=spec.get("mode", "innerCV"))
    return {**spec, **res, "secs": round(time.time() - t0, 1)}


def build_specs(layers: list[int], timesteps: list[int], seeds: list[int]) -> list[dict]:
    coarse = [2, 6, 10, 14, 18, 22, 26, 29]
    multiseed_layers = [4, 9, 14, 19, 24, 29]
    specs = []
    # --- verdict configs: full protocol ---
    for L in layers:
        specs.append({"source": "real_dit", "backbone": "real", "layer": L, "t": REF_T,
                      "seed": 0, "mode": "innerCV"})
    if (CONS / "random_index.json").exists():
        for L in layers:
            specs.append({"source": "random_dit", "backbone": "random", "layer": L, "t": REF_T,
                          "seed": 0, "mode": "innerCV"})
    specs.append({"source": "vae_latent", "backbone": "real", "layer": -1, "t": -1, "seed": -1,
                  "mode": "innerCV"})
    specs.append({"source": "raw_pixel", "backbone": "real", "layer": -1, "t": -1, "seed": -1,
                  "mode": "innerCV"})
    # --- probe-level null floor: repeat with other noise seeds ---
    for s in [x for x in seeds if x != 0]:
        for L in multiseed_layers:
            specs.append({"source": "real_dit", "backbone": "real", "layer": L, "t": REF_T,
                          "seed": s, "mode": "innerCV"})
            if (CONS / "random_index.json").exists():
                specs.append({"source": "random_dit", "backbone": "random", "layer": L,
                              "t": REF_T, "seed": s, "mode": "innerCV"})
    # --- layer x timestep scan: fixed C ---
    for t in timesteps:
        if t == REF_T:
            continue  # already covered by the verdict configs above
        for L in coarse:
            specs.append({"source": "real_dit", "backbone": "real", "layer": L, "t": t,
                          "seed": 0, "mode": "innerCV"})
    return specs


def probe_all(workers: int, threads: int, layers: list[int], timesteps: list[int],
              seeds: list[int]) -> None:
    from joblib import Parallel, delayed

    specs = build_specs(layers, timesteps, seeds)
    for s in specs:
        s["threads"] = threads
    print(f"running {len(specs)} probe configs on {workers} workers x {threads} threads",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=workers, verbose=10)(delayed(_job)(s) for s in specs)
    (OUT / "E1_probe_raw.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'E1_probe_raw.json'}")


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["consolidate", "probe"], required=True)
    ap.add_argument("--backbone", default="real")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--timesteps", type=int, nargs="+", default=[200, 400, 600, 800, 950])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    if args.stage == "consolidate":
        consolidate(args.backbone)
    else:
        probe_all(args.workers, args.threads, args.layers, args.timesteps, args.seeds)


if __name__ == "__main__":
    main()
