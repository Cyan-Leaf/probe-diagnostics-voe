"""E18 analysis: the two Wan2.2 experts, each measured inside its own noise band.

E17 measured both experts at t=600 and got identical D1 (99.1% / 99.1%). That comparison is
unfair: at sigma 0.600 the low-noise expert is in range (boundary 0.875) and the high-noise
expert is not, so the null can be read two ways. E18_PREREG.md §2 fixes this by taking the
diagonal -- low@t=300, high@t=900 -- where each expert is the one the schedule would actually
route to.

Everything numeric is imported, not re-implemented: `probe_oof`/`auc_of` from e12_analyse (same
arithmetic as e8_analyse.probe) and `interpretable` from e8_verdict (the T10 gate). E11 already
paid for that lesson -- a locally rewritten linear probe drifted 2.3pp.

Only difference from e17_analyse.py: the timestep is a parameter, and the cache dir carries it
for t != 600. Keeping this as a separate script rather than a flag on e17_analyse.py protects
E17's delivered E17_cells.json from a rerun with a different t.

Run: python scripts/e18_analyse.py --threads 4
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
OUT = ROOT / "results" / "E18_own_band"
CACHE = ROOT / "cache"

SEEDS = [0, 1, 2]
N_LAYERS = 40
REL_DEPTH = 16 / 30.0
LAYERS = {"L16": 16, "Lrel": int(round(REL_DEPTH * N_LAYERS))}
BOUNDARY_RATIO = 0.875

# The two diagonal cells of E18_PREREG.md §2.
OWN_BAND = {"low": 300, "high": 900}


def out_dir(expert: str, weights: str, variant: str, t: int) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    tpart = "" if t == 600 else f"_t{t}"
    return CACHE / f"wan22a14b_{expert}_feats_{weights}_intphys2{suffix}{tpart}"


def load(expert: str, weights: str, variant: str, t: int) -> list[dict]:
    d = out_dir(expert, weights, variant, t)
    files = sorted(d.glob("*.pt"))
    assert files, (f"empty cache {d} -- run: scripts/e17_extract.py --expert {expert} "
                   f"--weights {weights} --variant {variant} --t {t}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def X_of(recs, layer, seed, t):
    return np.stack([r["feats"][str(t)][str(seed)][layer].reshape(-1).numpy().astype(np.float32)
                     for r in recs])


def cell(recs, layer, t, seed_base):
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X_of(recs, layer, s, t), y, g, seed=seed_base + s)) for s in SEEDS]
    A = np.array(per, float)
    return {"auc_mean": float(np.nanmean(A)), "seed_spread": float(np.nanstd(A, ddof=1)),
            "per_feature_seed_auc": per, "n": int(len(y)), "n_pos": int(y.sum())}


def appearance(recs_p, recs_c, layer, t, seed_base):
    keep = [i for i, r in enumerate(recs_p) if r["clip"]["label"] == 0]
    assert len(recs_c) == len(keep), (
        f"cj cache has {len(recs_c)} clips but plain has {len(keep)} Possible -- incomplete")
    per = []
    for s in SEEDS:
        Xa = X_of(recs_p, layer, s, t)[keep]
        Xb = X_of(recs_c, layer, s, t)
        X = np.concatenate([Xa, Xb], axis=0)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(recs_p[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in recs_c])
        per.append(auc_of(probe_oof(X, y, g, seed=seed_base + s)))
    A = np.array(per, float)
    return {"auc_mean": float(np.nanmean(A)), "seed_spread": float(np.nanstd(A, ddof=1)),
            "per_feature_seed_auc": per, "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    # Must be 4: the probe is not bit-reproducible across BLAS thread counts (E16), and E8's
    # delivered numbers are 4-thread.
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "E18_cells.json"))
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits

    t0 = time.time()
    payload = {
        "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers@5be7df9619b5",
        "design": "diagonal cells only: each expert measured inside its own noise band",
        "boundary_ratio": BOUNDARY_RATIO,
        "own_band": {e: {"t": t, "sigma": t / 1000.0,
                         "in_range": (t / 1000.0 >= BOUNDARY_RATIO) if e == "high"
                                     else (t / 1000.0 < BOUNDARY_RATIO)}
                     for e, t in OWN_BAND.items()},
        "cross_band_reference": {"t": 600, "source": "results/E17_moe_experts/E17_cells.json",
                                 "note": "E17's cells, NOT recomputed here; read by e18_verdict.py"},
        "cells": {},
    }

    with threadpool_limits(args.threads):
        for expert, t in OWN_BAND.items():
            real = load(expert, "real", "plain", t)
            rand = load(expert, "random", "plain", t)
            cj = load(expert, "real", "cj", t)
            sigma = t / 1000.0
            print(f"\n== expert={expert} t={t} sigma={sigma:.3f} "
                  f"n: real {len(real)} random {len(rand)} cj {len(cj)} ==", flush=True)
            for lname, layer in LAYERS.items():
                cr = cell(real, layer, t, 1000)
                cq = cell(rand, layer, t, 1000)
                ca = appearance(real, cj, layer, t, 2000)
                a_r, a_q = cr["auc_mean"], cq["auc_mean"]
                floor = max(cr["seed_spread"], cq["seed_spread"])
                d = (a_r - a_q) / (a_r - 0.5) if a_r > 0.5 else float("nan")
                ok, why = interpretable(a_r, a_q, floor)
                payload["cells"][f"a14b|{expert}@{t}|{lname}"] = {
                    "expert": expert, "t": t, "sigma": sigma, "layer": layer,
                    "auc_real": a_r, "auc_random": a_q, "auc_appearance": ca["auc_mean"],
                    "d1": d, "d1_pct": d * 100 if np.isfinite(d) else float("nan"),
                    "floor": floor, "floor_real": cr["seed_spread"],
                    "floor_random": cq["seed_spread"],
                    "interpretable": ok, "why": why,
                    "n": cr["n"], "n_pos": cr["n_pos"],
                    "per_feature_seed_real": cr["per_feature_seed_auc"],
                    "per_feature_seed_random": cq["per_feature_seed_auc"],
                    "per_feature_seed_appearance": ca["per_feature_seed_auc"],
                    "gap_appearance_minus_physics": ca["auc_mean"] - a_r,
                }
                print(f"  {lname:4s} L{layer:<3d} real {a_r:.4f} rand {a_q:.4f} "
                      f"appear {ca['auc_mean']:.4f}  D1 {d*100:6.1f}% floor {floor:.4f} "
                      f"{'OK' if ok else 'NOT interpretable: ' + why}", flush=True)

    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["threads"] = args.threads
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out} ({payload['elapsed_s']}s, threads={args.threads})")


if __name__ == "__main__":
    main()
