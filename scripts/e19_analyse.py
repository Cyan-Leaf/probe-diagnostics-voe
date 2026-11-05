"""E19 analysis: MiniMax-H3 (33B) -- does the D1/D2 direction survive the 14B -> 33B jump?

Same arithmetic as every other backbone, imported rather than rewritten: `probe_oof`/`auc_of`
from e12_analyse (which is E8's `probe` verified bit-identical) and `interpretable` from
e8_verdict (the T10 gate). E11 paid for that lesson already -- a locally rewritten linear probe
drifted 2.3pp, and E18's cross-check proved this import path reproduces delivered numbers
bit-for-bit.

Two things are specific to H3 and are handled here rather than in the extractor:

  1. **The pooled feature is per-latent-frame, like every other backbone**, but the frame count
     is 12 (39 pixel frames -> `5*n+2`), not 13. Flattening 12x5376 gives the probe a 64512-dim
     vector -- an order of magnitude wider than the Wan cells (13x5120 = 66560, so actually
     comparable). Kept flat for consistency with E8's `X_of`.
  2. **The raw-pixel arm is recomputed**, not read from the extractor's `raw_pixel_32`. That
     field bins the frames as `1 + 4*i` (the Wan f4t4 layout) and would give 10 bins for H3's 39
     frames, while the latent arm has 12 bins. Comparing arms with different bin counts is not a
     comparison. `pixel_features()` uses H3's own chunking (17/17/5 -> 12 groups) so the two
     arms are on the same temporal grid.

Run: .venv_h3/bin/python scripts/e19_analyse.py --threads 4
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
OUT = ROOT / "results" / "E19_h3_scale"
CACHE = ROOT / "cache"

SEEDS = [0, 1, 2]
# 50 layers in the released checkpoint. L16 is PREREG.md's fixed reference depth; the second
# cell continues E8's relative-depth convention (16/30 of the stack -> 27 of 50).
LAYERS = {"L16": 16, "Lrel": 27}
T = 600
D1_THRESHOLD = 0.5

# H3's temporal geometry: 17 pixel frames per encoder chunk, 3 trailing latents dropped, so
# 39 = 17+17+5 maps to 12 latent frames in groups of (5,5,2).
FRAME_GROUPS = [17, 17, 5]
LATENT_PER_GROUP = [5, 5, 2]


def out_dir(weights: str, variant: str) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    return CACHE / f"minimax_h3_feats_{weights}_intphys2{suffix}"


def load(weights: str, variant: str) -> list[dict]:
    d = out_dir(weights, variant)
    files = sorted(d.glob("*.pt"))
    assert files, (f"empty cache {d} -- run scripts/e19_queue.sh first")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def layer_slot(layer: int) -> int:
    """Map a layer number to its position in the extractor's stacked feature tensor.

    The extractor hooks `layers = [16, 27]` and stores `torch.stack([hs[k] for k in layers])`,
    so dim 0 is the position in that list. Both this module and `e19_extract.py` derive the
    list the same way (L16 plus 16/30 of the stack), so the mapping is just the index.
    """
    order = sorted({16, int(round(16 / 30.0 * 50))})
    if layer not in order:
        raise KeyError(f"layer {layer} was not hooked; extractor stored {order}")
    return order.index(layer)


def X_of(recs, layer_slot, seed, t=T):
    """Features for one hooked layer.

    `layer_slot` is the POSITION in the extractor's `layers` list, not the layer number: the
    extractor stacks `[hs[k] for k in layers]`, so dim 0 of the stored tensor is an index into
    that list (size 2 here), not into the 50-block stack. Indexing it with 16 raised
    `IndexError: index 16 is out of bounds for dimension 0 with size 2` -- the same class of
    "layer number vs slot" confusion that E14 had to document for its two block stacks.
    """
    return np.stack([r["feats"][str(t)][str(seed)][layer_slot].reshape(-1).numpy()
                     .astype(np.float32) for r in recs])


def latent_frame_indices() -> list[int]:
    """Which pixel frame each of the 12 latent frames is anchored to.

    H3's encoder is causal with a 4x temporal stride, and the chunking is `17 + 17 + 5` frames
    -> `5 + 5 + 2` latents. Within a chunk, latent `j` is anchored at `chunk_start + 4*j`:
    17 frames give latents at 0, 4, 8, 12, 16 (five of them, the last touching frame 16), and the
    5-frame tail gives two at 0 and 4 within itself. So the global anchors are

        0, 4, 8, 12, 16 | 17, 21, 25, 29, 33 | 34, 38

    Derived from the geometry rather than guessed: `17 -> 5` is `(17-1)//4 + 1`, and `5 -> 2`
    is `(5-1)//4 + 1`, which is exactly the `clip_length`/`token_drop` contract in the VAE doc.
    A flat `linspace` over 39 frames would NOT land on these, so the two arms would not be
    temporally aligned.
    """
    starts, n = [], 0
    for g, k in zip(FRAME_GROUPS, LATENT_PER_GROUP):
        starts += [n + 4 * j for j in range(k)]
        n += g
    return starts


def pixel_features(recs, size: int = 32) -> np.ndarray:
    """Per-latent-frame 32x32 RGB, anchored on H3's own latent grid.

    The extractor already stored `raw_pixel_32`, but that uses the Wan f4t4 grouping
    (`1 + 4*i` bins) and yields 10 bins for 39 frames. Recomputing keeps the pixel arm on the
    same 12-bin temporal grid as the latent arm -- comparing arms with different bin counts is
    not a comparison.
    """
    import decord

    anchors = latent_frame_indices()
    rows = []
    for r in recs:
        vr = decord.VideoReader(str(r["clip"]["path"]))
        total = len(vr)
        # Same spanning rule the extractor used (`read_spanning`), so anchor indices mean the
        # same thing in both arms.
        span = np.linspace(0, total - 1, sum(FRAME_GROUPS)).round().astype(int)
        frames = vr.get_batch(span[anchors].tolist()).asnumpy()   # uint8 [12,H,W,3]
        x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                            align_corners=False)
        rows.append(x.flatten().numpy().astype(np.float32))
    return np.stack(rows)


def cell(recs, layer, seed_base, t=T):
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X_of(recs, layer, s, t), y, g, seed=seed_base + s)) for s in SEEDS]
    A = np.array(per, float)
    return {"auc_mean": float(np.nanmean(A)), "seed_spread": float(np.nanstd(A, ddof=1)),
            "per_feature_seed_auc": per, "n": int(len(y)), "n_pos": int(y.sum())}


def appearance(recs_p, recs_c, layer, seed_base, t=T):
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


def latent_pooled_of(rec) -> np.ndarray:
    """Read the VAE-latent arm, tolerating both field names.

    The extractor wrote `latent_pooled` for its first passes and `vae_latent_pooled` afterwards
    (renamed to match E8's reader). Restarting a 1012-clip arm to rename a key would throw away
    ~40 minutes of GPU time, so the reader accepts either -- but an absent key is an error, not
    a silent zero.
    """
    for k in ("vae_latent_pooled", "latent_pooled"):
        if k in rec:
            return rec[k].reshape(-1).numpy().astype(np.float32)
    raise KeyError(f"neither vae_latent_pooled nor latent_pooled in {rec.get('clip', {})}")


def lower_bound(recs, layer, seed_base, kind):
    """One of the four E8 lower bounds. `kind` selects which feature matrix is probed."""
    if kind == "raw_pixel":
        X = pixel_features(recs)
    elif kind == "vae_latent":
        X = np.stack([latent_pooled_of(r) for r in recs])
    else:
        raise ValueError(kind)
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    per = [auc_of(probe_oof(X, y, g, seed=seed_base + s)) for s in range(len(SEEDS))]
    return {"auc_mean": float(np.nanmean(per)), "per_feature_seed_auc": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "E19_cells.json"))
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits

    t0 = time.time()
    payload = {
        "model": "MiniMaxAI/MiniMax-H3 (33.1B dense single-stream Omni Transformer)",
        "reference_cell": {"layer": 16, "t": T},
        "design": "E8 protocol: four lower bounds + D1 + D2, IntPhys 2 only",
        "frame_geometry": {"pixel_frames": sum(FRAME_GROUPS), "groups": FRAME_GROUPS,
                           "latent_frames": sum(LATENT_PER_GROUP),
                           "note": "39 = 17+17+5 pixel frames -> 12 latent frames"},
        "cells": {},
    }

    with threadpool_limits(args.threads):
        real = load("real", "plain")
        rand = load("random", "plain")
        cj = load("real", "cj")
        print(f"loaded: real {len(real)} random {len(rand)} cj {len(cj)}", flush=True)

        for lname, layer in LAYERS.items():
            slot = layer_slot(layer)
            cr = cell(real, slot, 1000)
            cq = cell(rand, slot, 1000)
            ca = appearance(real, cj, slot, 2000)
            a_r, a_q = cr["auc_mean"], cq["auc_mean"]
            floor = max(cr["seed_spread"], cq["seed_spread"])
            d = (a_r - a_q) / (a_r - 0.5) if a_r > 0.5 else float("nan")
            ok, why = interpretable(a_r, a_q, floor)
            payload["cells"][f"h3|{lname}"] = {
                "layer": layer, "layer_slot": slot, "t": T,
                "auc_real": a_r, "auc_random": a_q,
                "auc_appearance": ca["auc_mean"], "d1": d,
                "d1_pct": d * 100 if np.isfinite(d) else float("nan"),
                "floor": floor, "floor_real": cr["seed_spread"],
                "floor_random": cq["seed_spread"],
                "interpretable": ok, "why": why, "n": cr["n"], "n_pos": cr["n_pos"],
                "per_feature_seed_real": cr["per_feature_seed_auc"],
                "per_feature_seed_random": cq["per_feature_seed_auc"],
                "per_feature_seed_appearance": ca["per_feature_seed_auc"],
                "gap_appearance_minus_physics": ca["auc_mean"] - a_r,
            }
            print(f"  {lname:4s} L{layer:<3d} real {a_r:.4f} rand {a_q:.4f} "
                  f"appear {ca['auc_mean']:.4f}  D1 {d*100:6.1f}% floor {floor:.4f} "
                  f"{'OK' if ok else 'NOT interpretable: ' + why}", flush=True)

        # Lower bounds at the reference cell. `trained` and `random` are the DiT arms already
        # computed above; `raw_pixel` and `vae_latent` are the two non-DiT arms.
        lb = {"trained_dit": payload["cells"]["h3|L16"]["auc_real"],
              "random_dit": payload["cells"]["h3|L16"]["auc_random"]}
        for kind in ("vae_latent", "raw_pixel"):
            r = lower_bound(real, layer_slot(LAYERS["L16"]), 3000, kind)
            lb[kind] = r["auc_mean"]
            payload.setdefault("lower_bound_details", {})[kind] = r
            print(f"  lower bound {kind}: {r['auc_mean']:.4f}", flush=True)
        payload["lower_bounds"] = lb
        payload["lower_bound_order"] = sorted(lb, key=lambda k: -lb[k])
        payload["order_reproduces"] = payload["lower_bound_order"] == \
            ["trained_dit", "random_dit", "raw_pixel"] or \
            payload["lower_bound_order"][0] == "trained_dit"

    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["threads"] = args.threads
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out} ({payload['elapsed_s']}s, threads={args.threads})")


if __name__ == "__main__":
    main()
