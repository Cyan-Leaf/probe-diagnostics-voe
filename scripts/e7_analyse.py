"""E7 analysis: run D1-D4 on IntPhys 2 and check whether the diagnostic protocol transfers.

Everything here follows results/E7_intphys2/E7_PREREG.md, frozen before extraction:

  grouping ....... GroupKFold(5) on SceneIndex (the official pairing key)
  readouts ....... d' (unbounded) AND AUC, BOTH pre-declared primary, because IntPhys 2's
                   own README states most models sit at chance -- i.e. the bounded statistic
                   is expected at its lower boundary, which is exactly what D6 forbids
                   relying on.  Declared before the run, not after seeing a null.
  D4 floor ....... definition A, seed-spread over feature seeds 0/1/2 (pooled n=1012 and
                   per-condition n>=240, so sample size is not the bottleneck).  Units with
                   n < 100 additionally get definition B, flagged in the output.
  layer / t ...... 16 / 600, matching the LikePhys reference cell for comparability

Run: python scripts/e7_analyse.py --workers 12
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_s1s3 import _permute_within_groups, _shuffled_group_folds
from t4_caliper import CS, _decision, _fit_path, _rowspace
from wan_probe_lib import env_stamp, preflight

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E7_intphys2"
LAYER = 16
T = "600"
SEEDS = [0, 1, 2]
CONDITIONS = ["solidity", "permanence", "immutability", "continuity"]
SMALL_UNIT = 100          # below this, also report the definition-B floor


def cache_dir(backbone: str, variant: str) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    return ROOT / "cache" / f"feats_{backbone}_intphys2{suffix}"


def load(backbone: str, variant: str, seed: int):
    """-> X [n, 30*13*3072 sliced to LAYER], meta list. Cached per (backbone,variant,seed)."""
    npy = OUT / f"X_{backbone}_{variant}_s{seed}_L{LAYER}.npy"
    mjs = OUT / f"meta_{backbone}_{variant}.json"
    if npy.exists() and mjs.exists():
        return np.load(npy, mmap_mode="r"), json.loads(mjs.read_text())

    d = cache_dir(backbone, variant)
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d}"
    rows, meta = [], []
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        # Guard against a directory/variant mismatch: an earlier cache_dir mapped every
        # non-"cj" variant onto the plain cache, which silently turned the weak-jitter
        # control into Possible-vs-everything instead of Possible-vs-recoloured.  The
        # record carries the variant that produced it, so check rather than trust.
        assert rec.get("variant") == variant, (
            f"{f.name} was produced with variant={rec.get('variant')!r} but loaded as "
            f"{variant!r} -- cache_dir is pointing at the wrong directory")
        rows.append(rec["feats"][T][str(seed)][LAYER].reshape(-1).numpy().astype(np.float32))
        c = rec["clip"]
        meta.append({k: c[k] for k in ("clip_id", "scene_index", "type", "label",
                                       "condition", "camera", "difficulty")})
    X = np.stack(rows)
    np.save(npy, X)
    mjs.write_text(json.dumps(meta, indent=1))
    return X, meta


def probe(X, y, groups, seed: int):
    """Held-out AUC and d' under the frozen probe protocol."""
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


def cell(backbone, variant, rows_filter, label_fn, tag, args):
    """Average AUC/d' over feature seeds + the seed-spread floor (D4 definition A)."""
    per_seed = []
    n = n_pos = 0
    for seed in SEEDS:
        X, meta = load(backbone, variant, seed)
        sel = [i for i, m in enumerate(meta) if rows_filter(m)]
        if not sel:
            return None
        y = np.array([label_fn(meta[i]) for i in sel])
        g = np.array([meta[i]["scene_index"] for i in sel])
        if len(np.unique(y)) < 2:
            return None
        n, n_pos = len(sel), int(y.sum())
        a, dp = probe(np.asarray(X[sel], dtype=np.float32), y, g, seed=1000 + seed)
        per_seed.append((a, dp))
    A = np.array([p[0] for p in per_seed], float)
    D = np.array([p[1] for p in per_seed], float)
    return {
        "tag": tag, "backbone": backbone, "variant": variant,
        "n": n, "n_pos": n_pos, "n_scenes": None,
        "auc_mean": float(np.nanmean(A)), "auc_seed_spread": float(np.nanstd(A, ddof=1)),
        "dprime_mean": float(np.nanmean(D)), "dprime_seed_spread": float(np.nanstd(D, ddof=1)),
        "per_seed_auc": [float(x) for x in A], "per_seed_dprime": [float(x) for x in D],
        "small_unit": bool(n < SMALL_UNIT),
    }


def perm_floor(backbone, variant, rows_filter, label_fn, n_draw, args):
    """Definition-B floor: labels permuted within SceneIndex. Only for small units."""
    X, meta = load(backbone, variant, 0)
    sel = [i for i, m in enumerate(meta) if rows_filter(m)]
    y = np.array([label_fn(meta[i]) for i in sel])
    g = np.array([meta[i]["scene_index"] for i in sel])
    Xs = np.asarray(X[sel], dtype=np.float32)
    aucs, dps = [], []
    for r in range(n_draw):
        yp = _permute_within_groups(y, g, np.random.default_rng(50_000 + r))
        a, dp = probe(Xs, yp, g, seed=60_000 + r)
        if not np.isnan(a):
            aucs.append(a)
        if not np.isnan(dp):
            dps.append(dp)
    return {"n_draw": len(aucs),
            "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
            "dprime_sd": float(np.std(dps, ddof=1)) if len(dps) > 1 else None,
            "auc_mean": float(np.mean(aucs)) if aucs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--perm-draws", type=int, default=20)
    args = ap.parse_args()

    pf = preflight("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits

    payload = {"args": vars(args), "env": env_stamp(), "preflight": pf,
               "prereg": "results/E7_intphys2/E7_PREREG.md",
               "protocol": {"layer": LAYER, "t": int(T), "feat_seeds": SEEDS,
                            "grouping": "SceneIndex", "folds": 5,
                            "readouts": ["dprime (unbounded, primary)",
                                         "auc (bounded, primary, reported alongside)"],
                            "d4_floor": "definition A (seed-spread); definition B added "
                                        "for units with n < 100"},
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": {}}

    units = [("pooled", lambda m: True)]
    units += [(f"cond:{c}", (lambda c: (lambda m: m["condition"] == c))(c))
              for c in CONDITIONS]
    units += [(f"camera:{cam}", (lambda cam: (lambda m: m["camera"] == cam))(cam))
              for cam in ("Fixed", "Moving")]

    with threadpool_limits(limits=args.threads):
        # ---- main task: possible vs impossible, real & random (D1, D3, D4) -------------
        print("=== main task: Possible vs Impossible ===", flush=True)
        for tag, filt in units:
            for bb in ("real", "random"):
                r = cell(bb, "plain", filt, lambda m: m["label"], tag, args)
                if r is None:
                    continue
                payload["cells"][f"{tag}|{bb}|physics"] = r
                print(f"  {tag:18s} {bb:7s} n={r['n']:4d} "
                      f"AUC {r['auc_mean']:.4f}±{r['auc_seed_spread']:.4f}  "
                      f"d' {r['dprime_mean']:6.3f}±{r['dprime_seed_spread']:.3f}", flush=True)

        # ---- D2: possible vs colour-jittered possible ----------------------------------
        print("\n=== D2 control: Possible vs colour-jittered Possible ===", flush=True)
        for bb in ("real", "random"):
            Xp, mp = load(bb, "plain", 0)
            Xc, mc = load(bb, "cj", 0)
            per_seed = []
            for seed in SEEDS:
                Xp, mp = load(bb, "plain", seed)
                Xc, mc = load(bb, "cj", seed)
                keep = [i for i, m in enumerate(mp) if m["label"] == 0]
                Xa = np.asarray(Xp[keep], dtype=np.float32)
                Xb = np.asarray(Xc, dtype=np.float32)
                X = np.concatenate([Xa, Xb], axis=0)
                y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
                g = np.array([mp[i]["scene_index"] for i in keep]
                             + [m["scene_index"] for m in mc])
                a, dp = probe(X, y, g, seed=2000 + seed)
                per_seed.append((a, dp))
            A = np.array([p[0] for p in per_seed], float)
            D = np.array([p[1] for p in per_seed], float)
            payload["cells"][f"pooled|{bb}|appearance"] = {
                "tag": "pooled", "backbone": bb, "variant": "plain_vs_cj",
                "n": int(len(y)), "n_pos": int(y.sum()),
                "auc_mean": float(np.nanmean(A)),
                "auc_seed_spread": float(np.nanstd(A, ddof=1)),
                "dprime_mean": float(np.nanmean(D)),
                "dprime_seed_spread": float(np.nanstd(D, ddof=1)),
                "per_seed_auc": [float(x) for x in A],
                "per_seed_dprime": [float(x) for x in D],
                "small_unit": False,
            }
            c = payload["cells"][f"pooled|{bb}|appearance"]
            print(f"  {bb:7s} n={c['n']:4d} AUC {c['auc_mean']:.4f}±"
                  f"{c['auc_seed_spread']:.4f}  d' {c['dprime_mean']:6.3f}", flush=True)

        # ---- definition-B floor for any small unit ------------------------------------
        small = [k for k, v in payload["cells"].items() if v.get("small_unit")]
        payload["perm_floors"] = {}
        for k in small:
            v = payload["cells"][k]
            tag = v["tag"]
            filt = dict(units)[tag]
            payload["perm_floors"][k] = perm_floor(v["backbone"], "plain", filt,
                                                   lambda m: m["label"],
                                                   args.perm_draws, args)
            print(f"  perm floor for {k}: {payload['perm_floors'][k]}", flush=True)

    payload["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "E7_cells.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT/'E7_cells.json'}")


if __name__ == "__main__":
    main()
