"""E13 analysis: the same probe on a non-generative encoder (V-JEPA 2).

Differences from e12_analyse.py, all forced by the model rather than chosen:

  * There is no timestep.  Records key their features under `clean`, not `"600"`, and there
    is exactly one feature draw per clip (a ViT forward is deterministic; only the diffusion
    backbones got 3 draws for the noise seed).  So the seed-spread floor is unavailable and
    the T10 gate is applied with the floor taken from the probe's own feature-seed spread,
    which does not exist here.
    -> Floor definition used instead (stated in the report, not silently swapped): the
       **permutation floor** — refit the probe with labels shuffled within scenes and take
       the spread of that AUC distribution.  This is definition B in PREREG.md D4, which was
       pre-registered for exactly the question "does this generalise beyond these clips".

  * No VAE.  `vae_latent_pooled` is None by construction; the VAE-latent lower bound is
    reported as absent rather than substituted (E13_PREREG.md §3.3).

  * The CogVideoX-2b t=0 cells (`--t-grid 0 --tag T0`) are read alongside, because the
    V-JEPA D1 is only comparable to E8 through that clean-input cell (§3.1).

Run:
  python scripts/e13_analyse.py
  python scripts/e13_analyse.py --perm-draws 40
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_s1s3 import _shuffled_group_folds
from t4_caliper import CS, _decision, _fit_path, _rowspace
from e8_verdict import interpretable
from e12_analyse import auc_of, probe_oof

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E13_non_generative"
CACHE = ROOT / "cache"

VJEPA_N_LAYERS = 24
COG_N_LAYERS = 30
REL_DEPTH = 16 / 30.0          # same relative-depth rule as E8/E12


def rel(n: int) -> int:
    return int(round(REL_DEPTH * n))


def load(dirname: str) -> list[dict]:
    d = CACHE / dirname
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d}"
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def stack(recs, layers, caliber="flat", tkey="clean", seed="0"):
    """-> {layer: X}.  V-JEPA records hold one deterministic draw under key `clean`."""
    key = "feats" if caliber == "flat" else "feats_timeavg"
    out = {}
    for L in layers:
        out[L] = np.stack([r[key][tkey][seed][L].reshape(-1).numpy().astype(np.float32)
                           for r in recs])
    return out


def perm_floor(X, y, g, seed, draws):
    """Definition B (PREREG.md D4): shuffle labels within scenes, refit, spread of AUC.

    This is the floor that answers "does this generalise beyond these clips" — the question
    E13 actually asks.  It is not interchangeable with the seed-spread floor used for the
    diffusion backbones, which relies on 3 noise draws that a ViT does not have.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(draws):
        yp = y.copy()
        for gv in np.unique(g):
            idx = np.flatnonzero(g == gv)
            yp[idx] = rng.permutation(yp[idx])
        if len(np.unique(yp)) < 2:
            continue
        f = probe_oof(X, yp, g, seed=seed)
        a = auc_of(f)
        if np.isfinite(a):
            aucs.append(a)
    return float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")


def cell_from(recs, layers, caliber, label_fn=None, perm_draws=20, seed=0):
    y = np.array([r["clip"]["label"] for r in recs])
    g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    res = {}
    for L, X in stack(recs, layers, caliber).items():
        f = probe_oof(X, y, g, seed=seed)
        auc = auc_of(f)
        floor = perm_floor(X, y, g, seed=9000 + L, draws=perm_draws)
        res[L] = {"layer": L, "caliber": caliber, "auc": auc, "floor": floor,
                  "n": int(len(y)), "n_pos": int(y.sum())}
    return res


def appearance_cell(recs_p, recs_c, layers, caliber, perm_draws, seed=0):
    """D2: Possible vs the same Possible clips recoloured."""
    keep = [i for i, r in enumerate(recs_p) if r["clip"]["label"] == 0]
    Xp = stack([recs_p[i] for i in keep], layers, caliber)
    Xc = stack(recs_c, layers, caliber)
    assert len(recs_c) == len(keep), (
        f"cj cache has {len(recs_c)} clips, plain has {len(keep)} Possible")
    y = np.concatenate([np.zeros(len(keep), int), np.ones(len(keep), int)])
    g = np.array([str(recs_p[i]["clip"]["scene_index"]) for i in keep]
                 + [str(r["clip"]["scene_index"]) for r in recs_c])
    res = {}
    for L in layers:
        X = np.concatenate([Xp[L], Xc[L]], axis=0)
        f = probe_oof(X, y, g, seed=seed)
        auc = auc_of(f)
        floor = perm_floor(X, y, g, seed=9100 + L, draws=perm_draws)
        res[L] = {"layer": L, "caliber": caliber, "auc": auc, "floor": floor,
                  "n": int(len(y))}
    return res


def _cell_task(spec: dict) -> dict:
    """One (cache set, layer, caliber) probe plus its permutation floor.

    Top-level and primitive-only so it pickles into the process pool. Serial execution cost
    32 s per probe and ~300 probes; 224 cores were sitting idle, so the cells run in parallel
    at a fixed per-worker BLAS thread count (the same reason E8's workers pinned threads).
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(spec["threads"]):
        recs = load(spec["dir"])
        L = spec["layer"]
        X = np.stack([r["feats" if spec["caliber"] == "flat" else "feats_timeavg"]
                       ["clean"]["0"][L].reshape(-1).numpy().astype(np.float32)
                      for r in recs])
        if spec["mode"] == "physics":
            y = np.array([r["clip"]["label"] for r in recs])
            g = np.array([str(r["clip"]["scene_index"]) for r in recs])
        else:  # raw pixel
            X = np.stack([r["raw_pixel_32"].reshape(-1).numpy().astype(np.float32)
                          for r in recs])
            y = np.array([r["clip"]["label"] for r in recs])
            g = np.array([str(r["clip"]["scene_index"]) for r in recs])
        auc = auc_of(probe_oof(X, y, g, seed=spec["seed"]))
        floor = perm_floor(X, y, g, seed=spec["floor_seed"], draws=spec["draws"])
    return {"key": spec["key"], "auc": auc, "floor": floor,
            "n": int(len(y)), "n_pos": int(y.sum())}


def _appear_task(spec: dict) -> dict:
    from threadpoolctl import threadpool_limits

    pre = spec.get("prefix", "vjepa")
    with threadpool_limits(spec["threads"]):
        rp = load(f"{pre}_feats_real_intphys2")
        rc = load(f"{pre}_feats_real_intphys2_cj")
        L = spec["layer"]
        key = "feats" if spec["caliber"] == "flat" else "feats_timeavg"
        keep = [i for i, r in enumerate(rp) if r["clip"]["label"] == 0]
        Xa = np.stack([rp[i][key]["clean"]["0"][L].reshape(-1).numpy().astype(np.float32)
                       for i in keep])
        Xb = np.stack([r[key]["clean"]["0"][L].reshape(-1).numpy().astype(np.float32)
                       for r in rc])
        X = np.concatenate([Xa, Xb], axis=0)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(rp[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in rc])
        auc = auc_of(probe_oof(X, y, g, seed=spec["seed"]))
        floor = perm_floor(X, y, g, seed=spec["floor_seed"], draws=spec["draws"])
    return {"key": spec["key"], "auc": auc, "floor": floor, "n": int(len(y))}


def _cog_t0_task(spec: dict) -> dict:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(spec["threads"]):
        L = spec["layer"]
        out = {}
        for wname in ("real", "random"):
            recs = load(f"cogvideox_feats_{wname}_intphys2_T0")
            X = np.stack([r["feats"]["0"]["0"][L].reshape(-1).numpy().astype(np.float32)
                          for r in recs])
            y = np.array([r["clip"]["label"] for r in recs])
            g = np.array([str(r["clip"]["scene_index"]) for r in recs])
            out[wname] = auc_of(probe_oof(X, y, g, seed=1000))
            if wname == "real":
                Xr, yr, gr = X, y, g
        floor = perm_floor(Xr, yr, gr, seed=spec["floor_seed"], draws=spec["draws"])
    return {"key": spec["key"], "real": out["real"], "random": out["random"],
            "floor": floor, "n": int(len(yr))}


def d1(real, rand):
    return (real - rand) / (real - 0.5) if real > 0.5 else float("nan")


def main():
    # Without this the BLAS inside the probe spreads a 300x300 Gram matrix over all 224 cores
    # and thrashes: measured 118 s per probe at n=400 versus 9.6 s pinned to 4 threads.
    from threadpoolctl import threadpool_limits

    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-draws", type=int, default=20,
                    help="label permutations per cell for the D4-B floor")
    ap.add_argument("--out", default=str(OUT / "E13_cells.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with threadpool_limits(args.threads):
        _main(args)


def _main(args):
    t0 = time.time()
    Lv = [16, rel(VJEPA_N_LAYERS)]
    Lc = [16, rel(COG_N_LAYERS)]

    payload = {"model": "facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34",
               "reference_cell": {"layer": 16, "t": None,
                                  "why_null": "non-generative: no noising step"},
               "geometry": {"height": 256, "width": 256, "frames": 64, "tokens": 8192},
               "rel_depth_layer": rel(VJEPA_N_LAYERS),
               "floor_definition": "permutation within scenes (PREREG.md D4 definition B)",
               "cells": {}}

    print("== V-JEPA 2 ==", flush=True)
    for d_, nm in (("vjepa_feats_real_intphys2", "real"),
                   ("vjepa_feats_random_intphys2", "random"),
                   ("vjepa_feats_real_intphys2_cj", "cj")):
        print(f"  n: {nm} {len(load(d_))}", flush=True)

    # Every (cache set, layer, caliber) probe is independent, and each one carries its own
    # permutation floor. Serial that is ~19 tasks x (1 probe + 20 permutations) x 30 s ~ 3.5 h;
    # 224 cores were idle, so they go to a process pool with pinned BLAS threads.
    from concurrent.futures import ProcessPoolExecutor

    common = {"threads": args.threads, "draws": args.perm_draws, "seed": 1000}
    tasks = []
    for caliber in ("flat", "timeavg"):
        for L in Lv:
            for wdir, wname in (("vjepa_feats_real_intphys2", "real"),
                                ("vjepa_feats_random_intphys2", "random")):
                tasks.append((_cell_task, {**common, "dir": wdir, "layer": L,
                                           "caliber": caliber, "mode": "physics",
                                           "floor_seed": 9000 + L,
                                           "key": f"vjepa|{wname}|L{L}|{caliber}"}))
            tasks.append((_appear_task, {**common, "layer": L, "caliber": caliber,
                                         "floor_seed": 9100 + L,
                                         "key": f"vjepa|appear|L{L}|{caliber}"}))
    tasks.append((_cell_task, {**common, "dir": "vjepa_feats_real_intphys2", "layer": -1,
                               "caliber": "flat", "mode": "raw_pixel",
                               "floor_seed": 9200, "key": "vjepa|raw_pixel"}))
    for L in Lc:
        tasks.append((_cog_t0_task, {**common, "layer": L, "floor_seed": 9300 + L,
                                     "key": f"cogvideox2b|L{L}|flat|t0"}))

    print(f"  {len(tasks)} probe tasks, {args.workers} workers", flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn, spec): spec["key"] for fn, spec in tasks}
        for fut in futs:
            r = fut.result()
            res[r["key"]] = r
            print(f"    done {r['key']:32s} auc {r['auc'] if 'auc' in r else r['real']:.4f} "
                  f"floor {r['floor']:.4f}", flush=True)

    for caliber in ("flat", "timeavg"):
        for L in Lv:
            cr, cq, ca = (res[f"vjepa|real|L{L}|{caliber}"],
                          res[f"vjepa|random|L{L}|{caliber}"],
                          res[f"vjepa|appear|L{L}|{caliber}"])
            real, rand = cr["auc"], cq["auc"]
            floor = max(cr["floor"], cq["floor"])
            d = d1(real, rand)
            ok, why = interpretable(real, rand, floor)
            payload["cells"][f"vjepa|intphys2|L{L}|{caliber}|physics"] = {
                "auc_real": real, "auc_random": rand, "auc_appearance": ca["auc"],
                "d1": d, "d1_pct": d * 100 if np.isfinite(d) else float("nan"),
                "floor": floor, "floor_real": cr["floor"], "floor_random": cq["floor"],
                "interpretable": ok, "why": why, "n": cr["n"], "n_pos": cr["n_pos"],
                "gap_appearance_minus_physics": ca["auc"] - real}
            print(f"  {caliber:8s} L{L:<3d} real {real:.4f} rand {rand:.4f} "
                  f"appear {ca['auc']:.4f}  D1 {d*100:6.1f}% floor {floor:.4f} "
                  f"{'OK' if ok else 'NOT interpretable'}", flush=True)

    payload["cells"]["vjepa|intphys2|raw_pixel"] = dict(res["vjepa|raw_pixel"])
    payload["cells"]["vjepa|intphys2|raw_pixel"].pop("key", None)
    print(f"  raw_pixel {res['vjepa|raw_pixel']['auc']:.4f}", flush=True)
    payload["cells"]["vjepa|intphys2|vae_latent"] = {
        "auc": None, "why": "model has no VAE (E13_PREREG.md §3.3)"}

    print("\n== CogVideoX-2b @ t=0 (clean latents) ==", flush=True)
    for L in Lc:
        r = res[f"cogvideox2b|L{L}|flat|t0"]
        real, rand, floor = r["real"], r["random"], r["floor"]
        d = d1(real, rand)
        ok, why = interpretable(real, rand, floor)
        payload["cells"][f"cogvideox2b|intphys2|L{L}|flat|t0"] = {
            "auc_real": real, "auc_random": rand, "d1": d,
            "d1_pct": d * 100 if np.isfinite(d) else float("nan"),
            "floor": floor, "interpretable": ok, "why": why, "n": r["n"]}
        print(f"  L{L:<3d} real {real:.4f} rand {rand:.4f} D1 {d*100:6.1f}% "
              f"floor {floor:.4f} {'OK' if ok else 'NOT interpretable'}", flush=True)

    payload["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}  ({payload['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
