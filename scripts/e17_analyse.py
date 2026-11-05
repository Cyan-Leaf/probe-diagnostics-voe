"""E17 analysis: the two Wan2.2-T2V-A14B experts under the E8 probe.

This model is the only one in the study whose weight package contains **two independently
trained experts** with identical architecture, data and recipe, differing only in the noise
band they were trained on. Everything else is held fixed, so the comparison in Q2 is the one
contrast that no cross-family comparison can provide.

Kept separate from e8_analyse.py for the usual reason: that script's BACKBONES registry has no
notion of an "expert" axis, and adding one there would change a delivered artifact. The probe,
folds, C grid, floor definition and D1/D2 arithmetic are all imported, not re-implemented.

Two layers per expert, as in E8 for 40-layer models:
  L16   absolute reference cell (40% depth for a 40-layer model)
  L21   53.3% relative depth, matching Wan2.2-TI2V-5B's layer 16

Floors use seed-spread (PREREG.md definition A), because these are diffusion models with a
noising step and therefore have 3 feature seeds — unlike V-JEPA 2 in E13.

Run: python scripts/e17_analyse.py --threads 8
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
OUT = ROOT / "results" / "E17_moe_experts"
CACHE = ROOT / "cache"

SEEDS = [0, 1, 2]
T = 600
N_LAYERS = 40
REL_DEPTH = 16 / 30.0
LAYERS = {"L16": 16, "Lrel": int(round(REL_DEPTH * N_LAYERS))}
EXPERTS = ("low", "high")


def load(expert: str, weights: str, variant: str) -> list[dict]:
    suffix = "" if variant == "plain" else f"_{variant}"
    d = CACHE / f"wan22a14b_{expert}_feats_{weights}_intphys2{suffix}"
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d} — run e17_extract.py for (expert={expert}, {weights}, {variant})"
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def X_of(recs, layer, seed):
    return np.stack([r["feats"][str(T)][str(seed)][layer].reshape(-1).numpy().astype(np.float32)
                     for r in recs])


def cell(recs, layer, seed_base):
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X_of(recs, layer, s), y, g, seed=seed_base + s)) for s in SEEDS]
    A = np.array(per, float)
    return {"auc_mean": float(np.nanmean(A)), "seed_spread": float(np.nanstd(A, ddof=1)),
            "per_feature_seed_auc": per, "n": int(len(y)), "n_pos": int(y.sum())}


def appearance(recs_p, recs_c, layer, seed_base):
    """D2 on IntPhys2: Possible vs the same Possible clips recoloured (E8's own control)."""
    keep = [i for i, r in enumerate(recs_p) if r["clip"]["label"] == 0]
    assert len(recs_c) == len(keep), (
        f"cj cache has {len(recs_c)} clips but plain has {len(keep)} Possible — incomplete")
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
            "per_feature_seed_auc": per, "n": int(len(y))}


def lower_bound(recs, field, seed_base):
    X = np.stack([r[field].reshape(-1).numpy().astype(np.float32) for r in recs])
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X, y, g, seed=seed_base + s)) for s in SEEDS]
    return {"auc": float(np.nanmean(per)), "seed_spread": float(np.nanstd(per, ddof=1)),
            "per_feature_seed_auc": per}


def main():
    ap = argparse.ArgumentParser()
    # Must be 4. The probe is not bit-reproducible across BLAS thread counts (E16); a 6-thread
    # run of this very script shifted every D1 by ~0.13pp, which is the same order as the
    # expert-to-expert delta Q2 is trying to measure. 6-thread output is kept as
    # E17_cells.threads6_superseded.json.
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "E17_cells.json"))
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits

    t0 = time.time()
    payload = {"model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers@5be7df9619b5",
               "reference_cell": {"layer": 16, "t": T},
               "relative_depth_cell": {"layer": LAYERS["Lrel"], "depth": REL_DEPTH},
               "boundary_ratio": 0.875,
               "experts": {"low": "transformer_2 (in range at t=600, sigma 0.600 < 0.875)",
                           "high": "transformer (OUT of range at t=600; E17_PREREG.md §4.1)"},
               "cells": {}}

    with threadpool_limits(args.threads):
        for expert in EXPERTS:
            real = load(expert, "real", "plain")
            rand = load(expert, "random", "plain")
            cj = load(expert, "real", "cj")
            print(f"\n== expert={expert}  n: real {len(real)} random {len(rand)} cj {len(cj)} ==",
                  flush=True)
            for lname, layer in LAYERS.items():
                cr = cell(real, layer, 1000)
                cq = cell(rand, layer, 1000)
                ca = appearance(real, cj, layer, 2000)
                a_r, a_q = cr["auc_mean"], cq["auc_mean"]
                floor = max(cr["seed_spread"], cq["seed_spread"])
                d = (a_r - a_q) / (a_r - 0.5) if a_r > 0.5 else float("nan")
                ok, why = interpretable(a_r, a_q, floor)
                payload["cells"][f"a14b|{expert}|{lname}"] = {
                    "expert": expert, "layer": layer, "auc_real": a_r, "auc_random": a_q,
                    "auc_appearance": ca["auc_mean"], "d1": d,
                    "d1_pct": d * 100 if np.isfinite(d) else float("nan"),
                    "floor": floor, "floor_real": cr["seed_spread"],
                    "floor_random": cq["seed_spread"],
                    "interpretable": ok, "why": why, "n": cr["n"], "n_pos": cr["n_pos"],
                    "per_feature_seed_real": cr["per_feature_seed_auc"],
                    "per_feature_seed_random": cq["per_feature_seed_auc"],
                    "per_feature_seed_appearance": ca["per_feature_seed_auc"],
                    "gap_appearance_minus_physics": ca["auc_mean"] - a_r}
                print(f"  {lname:4s} L{layer:<3d} real {a_r:.4f} rand {a_q:.4f} "
                      f"appear {ca['auc_mean']:.4f}  D1 {d*100:6.1f}% floor {floor:.4f} "
                      f"{'OK' if ok else 'NOT interpretable'}", flush=True)

            for field, nm in (("raw_pixel_32", "raw_pixel"),
                              ("vae_latent_pooled", "vae_latent")):
                lb = lower_bound(real, field, 7000)
                payload["cells"][f"a14b|{expert}|{nm}"] = lb
                print(f"  {nm:12s} {lb['auc']:.4f}", flush=True)

    payload["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
