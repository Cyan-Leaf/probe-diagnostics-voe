"""E14 analysis: HunyuanVideo under the E8 probe.

Reads the e14_extract.py records and produces the same cells that e8_analyse.py would, but
without touching that script — E8's `BACKBONES` registry has no entry for a model whose block
list is two concatenated stacks, and adding one there would change a delivered artifact.

The one structural difference that matters for the probe: this model has 60 blocks in two
stacks (0-19 double-stream, 20-59 single-stream). L16 sits in the double stack, the 53.3%
relative-depth cell L32 sits in the single stack, so the two are NOT two depths of the same
quantity (E14_PREREG.md §3). Both are reported; the report says this every time it shows them.

Probe, folds, C grid, floors and the D1/D2 arithmetic are imported from the frozen code paths
(e12_analyse.probe_oof -> e4_caliper/e10_s1s3), so a difference between this model and E8 can
only come from the model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_analyse import auc_of, probe_oof
from e8_verdict import interpretable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E14_hunyuan"
CACHE = ROOT / "cache"

SEEDS = [0, 1, 2]
T = 600
N_BLOCKS = 60
N_DOUBLE = 20
REL_DEPTH = 16 / 30.0
LAYERS = {"L16": 16, "Lrel": int(round(REL_DEPTH * N_BLOCKS))}


def load(weights: str, variant: str) -> list[dict]:
    suffix = "" if variant == "plain" else f"_{variant}"
    d = CACHE / f"hunyuan_feats_{weights}_intphys2{suffix}"
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d}"
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def X_of(recs, layer, seed):
    return np.stack([r["feats"][str(T)][str(seed)][layer].reshape(-1).numpy().astype(np.float32)
                     for r in recs])


def cell(recs, layer, seed_base=1000):
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X_of(recs, layer, s), y, g, seed=seed_base + s)) for s in SEEDS]
    A = np.array(per, float)
    return {"auc_mean": float(np.nanmean(A)), "seed_spread": float(np.nanstd(A, ddof=1)),
            "per_seed_auc": per, "n": int(len(y)), "n_pos": int(y.sum())}


def appearance(recs_p, recs_c, layer, seed_base=2000):
    keep = [i for i, r in enumerate(recs_p) if r["clip"]["label"] == 0]
    assert len(recs_c) == len(keep), f"cj {len(recs_c)} vs Possible {len(keep)}"
    per = []
    for s in SEEDS:
        Xa = X_of(recs_p, layer, s)[keep]
        Xb = X_of(recs_c, layer, s)
        X = np.concatenate([Xa, Xb], axis=0)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(recs_p[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in recs_c])
        per.append(auc_of(probe_oof(X, y, g, seed=seed_base + s)))
    A = np.array(per, float)
    return {"auc_mean": float(np.nanmean(A)), "seed_spread": float(np.nanstd(A, ddof=1)),
            "per_seed_auc": per, "n": int(len(y))}


def lower_bound(recs, field: str, seed_base=7000):
    X = np.stack([r[field].reshape(-1).numpy().astype(np.float32) for r in recs])
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X, y, g, seed=seed_base + s)) for s in SEEDS]
    return {"auc": float(np.nanmean(per)),
            "seed_spread": float(np.nanstd(per, ddof=1)), "per_seed_auc": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(OUT / "E14_cells.json"))
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits

    t0 = time.time()
    payload = {"model": "hunyuanvideo-community/HunyuanVideo@e8c2aaa66fe3",
               "reference_cell": {"layer": 16, "t": T},
               "relative_depth_cell": {"layer": LAYERS["Lrel"], "depth": REL_DEPTH,
                                       "stack": "single-stream",
                                       "note": "not a depth control for L16: different stack"},
               "n_blocks": N_BLOCKS, "n_double": N_DOUBLE,
               "n_single": N_BLOCKS - N_DOUBLE, "cells": {}}

    with threadpool_limits(args.threads):
        real = load("real", "plain")
        rand = load("random", "plain")
        cj = load("real", "cj")
        print(f"n: real {len(real)} random {len(rand)} cj {len(cj)}", flush=True)

        for lname, layer in LAYERS.items():
            print(f"\n== {lname} (block {layer}, "
                  f"{'double' if layer < N_DOUBLE else 'single'}-stream) ==", flush=True)
            cr = cell(real, layer)
            cq = cell(rand, layer)
            ca = appearance(real, cj, layer)
            a_r, a_q = cr["auc_mean"], cq["auc_mean"]
            floor = max(cr["seed_spread"], cq["seed_spread"])
            d = (a_r - a_q) / (a_r - 0.5) if a_r > 0.5 else float("nan")
            ok, why = interpretable(a_r, a_q, floor)
            payload["cells"][f"hunyuan|intphys2|{lname}|flat|physics"] = {
                "layer": layer, "auc_real": a_r, "auc_random": a_q,
                "auc_appearance": ca["auc_mean"], "d1": d,
                "d1_pct": d * 100 if np.isfinite(d) else float("nan"),
                "floor": floor, "floor_real": cr["seed_spread"],
                "floor_random": cq["seed_spread"],
                "interpretable": ok, "why": why, "n": cr["n"], "n_pos": cr["n_pos"],
                "per_seed_real": cr["per_seed_auc"], "per_seed_random": cq["per_seed_auc"],
                "gap_appearance_minus_physics": ca["auc_mean"] - a_r}
            print(f"  real {a_r:.4f} rand {a_q:.4f} appear {ca['auc_mean']:.4f} "
                  f"D1 {d*100:6.1f}% floor {floor:.4f} "
                  f"{'OK' if ok else 'NOT interpretable'}", flush=True)

        for field, nm in (("raw_pixel_32", "raw_pixel"), ("vae_latent_pooled", "vae_latent")):
            lb = lower_bound(real, field)
            payload["cells"][f"hunyuan|intphys2|{nm}"] = lb
            print(f"  {nm:12s} {lb['auc']:.4f}", flush=True)

    payload["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
