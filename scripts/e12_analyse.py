"""E12 analysis: does the token budget explain LTX's weak signal?

Reuses the frozen probe from t4_caliper / e10_s1s3 verbatim (via e8_analyse's copy), so a
difference between two rungs can only come from the geometry, not the readout.  Adds three
things E8 did not need:

  1. **Two calibers.**  `flat` is F_lat*D, matching E8 exactly.  `timeavg` averages over
     latent frames first, so every rung of a ladder has the SAME dimensionality.  Only the
     second can attribute a change to the token budget rather than to the probe's capacity
     (E12_PREREG.md §4.4).
  2. **Per-fold held-out scores**, kept so the slope CI can be bootstrapped over scenes
     without refitting the probe a thousand times (see `bootstrap_slope`).
  3. **Slope over log(tokens)** with the random-init slope as the control, plus the raw-pixel
     and VAE-latent slopes as the "did we just see more pixels" guard.

Run:
  python scripts/e12_analyse.py --model ltx  --rungs S512F49,S512F97,S512F193,S512F289,S512F337,S512F393
  python scripts/e12_analyse.py --model cogvideox2b --rungs S512F49,S384F49,S256F49,S192F49
"""

import argparse
import itertools
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E12_token_budget"
CACHE = ROOT / "cache"

SEEDS = [0, 1, 2]
T = 600
REL_DEPTH = 16 / 30.0

MODELS = {
    "ltx": {"prefix": "ltx", "n_layers": 28, "label": "LTX-Video-0.9.5"},
    "cogvideox2b": {"prefix": "cogvideox", "n_layers": 30, "label": "CogVideoX-2b"},
}


def layers_for(model: str) -> dict:
    n = MODELS[model]["n_layers"]
    return {"L16": 16, "Lrel": int(round(REL_DEPTH * n)), "n_layers": n}


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------
def feat_dir(model: str, tag: str, weights: str, variant: str) -> Path:
    """`tag == "BASE"` reads E8's own cache dir (no geometry suffix).

    The lowest rung of every ladder already exists as an E8 deliverable, so it is reused
    rather than re-extracted — E12_PREREG.md §7 requires those two cells to cross-check
    against E8 anyway, and re-extracting them would remove that check.
    """
    suffix = "" if variant == "plain" else f"_{variant}"
    t = "" if tag == "BASE" else f"_{tag}"
    return CACHE / f"{MODELS[model]['prefix']}_feats_{weights}_intphys2{suffix}{t}"


def load_records(model: str, tag: str, weights: str, variant: str) -> list[dict]:
    d = feat_dir(model, tag, weights, variant)
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d} — run e12_extract.py for this rung first"
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def stack_features(recs: list[dict], layer: int, seed: int, caliber: str) -> np.ndarray:
    """-> X [n, F_lat*D] (flat) or [n, D] (timeavg), at one layer/seed.

    E8's records predate the time-averaged twin, so the BASE rung has no `feats_timeavg`.
    It is derived from `feats` instead of being skipped: the time-average is exactly
    `feats.mean(dim=temporal)`, so deriving it keeps the caliber definition identical across
    rungs rather than making the baseline a different quantity.
    """
    key = "feats" if caliber == "flat" else "feats_timeavg"
    if key in recs[0]:
        rows = [r[key][str(T)][str(seed)][layer].reshape(-1).numpy().astype(np.float32)
                for r in recs]
    else:
        assert caliber == "timeavg", f"{key} missing but caliber={caliber}"
        rows = [r["feats"][str(T)][str(seed)][layer].float().mean(dim=0)
                .numpy().astype(np.float32) for r in recs]
    return np.stack(rows)


def stack_simple(recs: list[dict], field: str, caliber: str) -> np.ndarray:
    """raw_pixel_32 / vae_latent_pooled, flat or time-averaged. Weights-independent.

    Same BASE fallback as `stack_features`: E8 records have no time-averaged twin, so it is
    derived by averaging over the per-frame axis.
    """
    if caliber == "flat":
        rows = [r[field].reshape(-1).numpy().astype(np.float32) for r in recs]
    else:
        tkey = "raw_pixel_timeavg" if field == "raw_pixel_32" else "vae_latent_timeavg"
        if tkey in recs[0]:
            rows = [r[tkey].reshape(-1).numpy().astype(np.float32) for r in recs]
        else:
            rows = [r[field].float().mean(dim=0).numpy().astype(np.float32) for r in recs]
    return np.stack(rows)


def tokens_of(recs: list[dict]) -> int:
    """Token count of a rung, read from the records rather than from the tag name.

    E8's records (the BASE rung) predate the `tokens` key, so it is derived from `grid` —
    the latent grid is what the transformer actually sees, and deriving it from the tag
    string instead would let a mislabelled tag silently produce a wrong x-axis.
    """
    t = {int(r["config"].get("tokens", 0)) or
         int(np.prod(r["grid"])) for r in recs}
    assert len(t) == 1, f"rung mixes token counts: {t}"
    return int(t.pop())


def meta_of(recs: list[dict]) -> list[dict]:
    return [{"section": str(r["clip"]["scene_index"]), "label": r["clip"]["label"]}
            for r in recs]


# ---------------------------------------------------------------------------------------
# probe, with held-out scores kept
# ---------------------------------------------------------------------------------------
def probe_oof(X, y, groups, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Identical arithmetic to `e8_analyse.probe`, but returns the per-fold predictions.

    Keeping (y_te, s_te) per fold is what makes the slope CI affordable: the bootstrap can
    resample evaluation units without refitting, and the point estimate is still the same
    mean-over-folds AUC that E8 reported.  Verified against `e8_analyse.probe` on the E8
    caches by `--crosscheck`.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    folds = []
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
        folds.append((y[te], _decision(Zte, coef, b)))
    return folds


def auc_of(folds) -> float:
    from sklearn.metrics import roc_auc_score

    a = [roc_auc_score(y, s) for y, s in folds if len(np.unique(y)) > 1]
    return float(np.mean(a)) if a else float("nan")


def cell(model, tag, layer, caliber, weights, recs=None) -> dict:
    recs = recs or load_records(model, tag, weights, "plain")
    meta = meta_of(recs)
    y = np.array([m["label"] for m in meta])
    g = np.array([m["section"] for m in meta])
    per_seed, folds_seed = [], []
    for seed in SEEDS:
        X = stack_features(recs, layer, seed, caliber)
        f = probe_oof(X, y, g, seed=1000 + seed)
        per_seed.append(auc_of(f))
        folds_seed.append(f)
    A = np.array(per_seed, float)
    return {"tag": tag, "model": model, "layer": layer, "caliber": caliber,
            "weights": weights, "tokens": tokens_of(recs), "n": int(len(y)),
            "n_pos": int(y.sum()),
            "auc_mean": float(np.nanmean(A)),
            "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "per_seed_auc": [float(x) for x in A],
            "_folds": folds_seed}


def simple_cell(model, tag, field, caliber, recs=None) -> dict:
    recs = recs or load_records(model, tag, "real", "plain")
    meta = meta_of(recs)
    y = np.array([m["label"] for m in meta])
    g = np.array([m["section"] for m in meta])
    X = stack_simple(recs, field, caliber)
    per, folds_seed = [], []
    for s in SEEDS:
        f = probe_oof(X, y, g, seed=7000 + s)
        per.append(auc_of(f))
        folds_seed.append(f)
    A = np.array(per, float)
    return {"tag": tag, "model": model, "field": field, "caliber": caliber,
            "tokens": tokens_of(recs), "n": int(len(y)),
            "auc_mean": float(np.nanmean(A)),
            "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "per_seed_auc": [float(x) for x in A], "_folds": folds_seed}


def appearance_cell(model, tag, layer, caliber, weights) -> dict:
    """D2 on IntPhys2: Possible vs the same Possible clips recoloured (two caches joined)."""
    recs_p = load_records(model, tag, weights, "plain")
    recs_c = load_records(model, tag, weights, "cj")
    keep = [i for i, r in enumerate(recs_p) if r["clip"]["label"] == 0]
    per, folds_seed = [], []
    for seed in SEEDS:
        Xa = stack_features(recs_p, layer, seed, caliber)[keep]
        Xb = stack_features(recs_c, layer, seed, caliber)
        assert len(Xb) == len(keep), (
            f"cj cache has {len(Xb)} clips but plain has {len(keep)} Possible — "
            f"the rung is incomplete, do not read this number")
        X = np.concatenate([Xa, Xb], axis=0)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(recs_p[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in recs_c])
        f = probe_oof(X, y, g, seed=2000 + seed)
        per.append(auc_of(f))
        folds_seed.append(f)
    A = np.array(per, float)
    return {"tag": tag, "model": model, "layer": layer, "caliber": caliber,
            "weights": weights, "variant": "plain_vs_cj", "n": int(len(y)),
            "auc_mean": float(np.nanmean(A)),
            "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "per_seed_auc": [float(x) for x in A], "_folds": folds_seed}


# ---------------------------------------------------------------------------------------
# slopes
# ---------------------------------------------------------------------------------------
def slope_boot(x_log, y_vals, folds_per_point, n_boot=1000, seed=0) -> dict:
    """OLS of AUC on log(tokens), with a cluster bootstrap over evaluation units.

    The bootstrap resamples *scenes* with replacement and recomputes each rung's mean-over-
    folds AUC from the frozen per-fold predictions, then refits the slope.  It therefore
    captures evaluation-unit sampling noise but NOT refitting noise.  Stated here because
    the CI is used for a pre-registered pass/fail, and a narrower-than-truth CI would make
    the test too easy to pass (E12_PREREG.md §3 Q1 asks for the lower bound).
    """
    from sklearn.metrics import roc_auc_score

    x_log = np.asarray(x_log, float)
    y_vals = np.asarray(y_vals, float)
    good = np.isfinite(y_vals)
    if good.sum() < 3:
        return {"slope": float("nan"), "ci95": [float("nan")] * 2, "n_points": int(good.sum())}
    A = np.polyfit(x_log[good], y_vals[good], 1)[0]

    rng = np.random.default_rng(seed)
    # All rungs share the same clips in the same sorted-file order, but the FOLD SIZES are
    # not identical across rungs: the group->fold assignment is seeded per rung, and with 1012
    # clips over ~127 scene groups some folds land 200 and some 204. Indexing every fold with
    # one index vector therefore overruns the shorter folds (IndexError: 203 out of bounds for
    # size 200). Each fold is resampled at its own length instead; the draw is still the same
    # clip set across rungs because the row order is shared.
    folds_n = [[len(y) for y, _ in point] for point in folds_per_point]
    boots = []
    for _ in range(n_boot):
        ys = []
        for point, sizes in zip(folds_per_point, folds_n):
            aucs = []
            for (y, s), n_f in zip(point, sizes):
                idx = rng.integers(0, n_f, n_f)
                yy, ss = y[idx], s[idx]
                if len(np.unique(yy)) > 1:
                    aucs.append(roc_auc_score(yy, ss))
            ys.append(np.mean(aucs) if aucs else np.nan)
        ys = np.asarray(ys, float)
        ok = np.isfinite(ys)
        if ok.sum() >= 3:
            boots.append(np.polyfit(x_log[ok], ys[ok], 1)[0])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots \
        else [float("nan")] * 2
    return {"slope": float(A), "ci95": ci, "n_points": int(good.sum()),
            "n_boot_valid": len(boots)}


def main():
    # Pin BLAS: the probe's Gram matrix is 300x300, and letting it spread over 224 cores
    # thrashes instead of helping — measured 118 s per probe at n=400 with default threads
    # versus 9.6 s at 4 threads. This is the whole reason E8's workers wrapped in
    # threadpool_limits; skipping it here made the first analysis run 12x slower.
    from threadpoolctl import threadpool_limits

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--rungs", required=True, help="comma-separated cache tags, low to high")
    ap.add_argument("--out", default="")
    ap.add_argument("--crosscheck", action="store_true",
                    help="compare probe_oof against e8_analyse.probe on E8 caches")
    args = ap.parse_args()

    with threadpool_limits(args.threads):
        _main(args)


def _main(args):
    tags = [t for t in args.rungs.split(",") if t]
    out = Path(args.out) if args.out else OUT / f"E12_cells_{args.model}.json"
    t0 = time.time()
    L = layers_for(args.model)

    payload = {"model": args.model, "label": MODELS[args.model]["label"],
               "n_layers": L["n_layers"], "reference_cell": {"layer": 16, "t": T},
               "rungs": tags, "cells": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Cache the record list per (tag, weights, variant) so 12 probe cells do not re-read
    # the same 1012 .pt files twelve times.
    rec_cache: dict = {}

    def recs(tag, weights, variant="plain"):
        k = (tag, weights, variant)
        if k not in rec_cache:
            rec_cache[k] = load_records(args.model, tag, weights, variant)
        return rec_cache[k]

    ladders = {}
    for tag in tags:
        r_real = recs(tag, "real")
        tokens = tokens_of(r_real)
        print(f"\n== {tag}  tokens={tokens}  n={len(r_real)}", flush=True)
        entry = {"tokens": tokens}
        for caliber in ("flat", "timeavg"):
            for lname in ("L16", "Lrel"):
                layer = L[lname]
                cr = cell(args.model, tag, layer, caliber, "real", recs(tag, "real"))
                cq = cell(args.model, tag, layer,  caliber, "random", recs(tag, "random"))
                real, rand = cr["auc_mean"], cq["auc_mean"]
                floor = max(cr["auc_seed_spread"], cq["auc_seed_spread"])
                d1 = (real - rand) / (real - 0.5) if real > 0.5 else float("nan")
                ok, why = interpretable(real, rand, floor)
                entry[f"{caliber}|{lname}"] = {
                    "layer": layer, "caliber": caliber, "tokens": tokens,
                    "auc_real": real, "auc_random": rand,
                    "d1": float(d1), "d1_pct": float(d1 * 100) if np.isfinite(d1) else float("nan"),
                    "floor": float(floor), "interpretable": bool(ok), "why": why,
                    "real_seed_spread": cr["auc_seed_spread"],
                    "random_seed_spread": cq["auc_seed_spread"],
                    "per_seed_real": cr["per_seed_auc"], "per_seed_random": cq["per_seed_auc"],
                    "n": cr["n"], "n_pos": cr["n_pos"],
                    "_folds_real": cr["_folds"], "_folds_random": cq["_folds"]}
                print(f"  {caliber:8s} {lname:4s} real {real:.4f} rand {rand:.4f} "
                      f"D1 {d1*100:6.1f}% floor {floor:.4f} "
                      f"{'OK' if ok else 'NOT interpretable'}", flush=True)
            for field, nm in (("raw_pixel_32", "raw_pixel"), ("vae_latent_pooled", "vae_latent")):
                sc = simple_cell(args.model, tag, field, caliber, recs(tag, "real"))
                entry[f"{caliber}|{nm}"] = {"auc": sc["auc_mean"], "tokens": tokens,
                                            "seed_spread": sc["auc_seed_spread"],
                                            "per_seed_auc": sc["per_seed_auc"],
                                            "_folds": sc["_folds"]}
                print(f"  {caliber:8s} {nm:10s} {sc['auc_mean']:.4f}", flush=True)
        ladders[tag] = entry

    # ---- Q1: slopes over log(tokens), LTX temporal ladder only -------------------------
    q1 = {}
    for caliber in ("flat", "timeavg"):
        xs, real_aucs, rand_aucs, raw_aucs, vae_aucs = [], [], [], [], []
        f_real, f_rand, f_raw, f_vae = [], [], [], []
        for tag in tags:
            e = ladders[tag]
            if f"{caliber}|L16" not in e:
                continue
            c = e[f"{caliber}|L16"]
            xs.append(np.log(c["tokens"]))
            real_aucs.append(c["auc_real"])
            rand_aucs.append(c["auc_random"])
            raw_aucs.append(e[f"{caliber}|raw_pixel"]["auc"])
            vae_aucs.append(e[f"{caliber}|vae_latent"]["auc"])
            f_real.append(c["_folds_real"][0])
            f_rand.append(c["_folds_random"][0])
            f_raw.append(e[f"{caliber}|raw_pixel"]["_folds"][0])
            f_vae.append(e[f"{caliber}|vae_latent"]["_folds"][0])
        s_real = slope_boot(xs, real_aucs, f_real)
        s_rand = slope_boot(xs, rand_aucs, f_rand)
        s_raw = slope_boot(xs, raw_aucs, f_raw)
        s_vae = slope_boot(xs, vae_aucs, f_vae)
        delta = {"slope": s_real["slope"] - s_rand["slope"],
                 "ci95": [s_real["ci95"][0] - s_rand["ci95"][1],
                          s_real["ci95"][1] - s_rand["ci95"][0]]}
        q1[caliber] = {"slope_real": s_real, "slope_random": s_rand,
                       "slope_raw_pixel": s_raw, "slope_vae_latent": s_vae,
                       "delta_slope_real_minus_random": delta,
                       "points": [{"tokens": int(np.exp(x)), "auc_real": r, "auc_random": q,
                                   "auc_raw_pixel": p, "auc_vae_latent": v}
                                  for x, r, q, p, v in
                                  zip(xs, real_aucs, rand_aucs, raw_aucs, vae_aucs)]}
        print(f"\nQ1 [{caliber}] slope_real {s_real['slope']:+.4f}/e-fold "
              f"CI {s_real['ci95'][0]:+.4f}..{s_real['ci95'][1]:+.4f} | "
              f"delta {delta['slope']:+.4f} CI {delta['ci95'][0]:+.4f}..{delta['ci95'][1]:+.4f}",
              flush=True)

    # Strip the bulky per-fold payload before writing.
    for tag in ladders:
        for k, v in ladders[tag].items():
            if isinstance(v, dict):
                v.pop("_folds", None)
                v.pop("_folds_real", None)
                v.pop("_folds_random", None)
    for cal in q1.values():
        for k in ("slope_real", "slope_random", "slope_raw_pixel", "slope_vae_latent"):
            pass

    payload["ladders"] = ladders
    payload["q1"] = q1
    payload["elapsed_s"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}  ({payload['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
