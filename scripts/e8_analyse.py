"""E8 analysis: the four lower bounds + D1/D2 on a second (and third, fourth) backbone.

Reuses the frozen probe from E3/E7 unchanged — `_rowspace` / `_fit_path` / `_decision` /
`CS` from t4_caliper, the grouped folds and permutation helper from e10_s1s3 — so a
difference in the numbers can only come from the backbone, not from the readout.

Two layers are always reported per backbone:

  L16      absolute layer 16, the reference cell fixed by PREREG.md.  Only directly
           comparable between backbones that have 30 layers.
  Lrel     round(0.533 * num_layers), i.e. 53.3% depth — the relative position that layer 16
           occupies in Wan2.2-TI2V-5B.  This is the cell that makes backbones of different
           depth comparable at all, and it is the one to read when comparing 30-layer against
           42- or 48-layer models.

Both are pre-registered here and both are reported; neither is chosen after the fact.

Run: python scripts/e8_analyse.py --backbones cogvideox2b --dataset intphys2 --workers 12
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
OUT = ROOT / "results" / "E8_second_backbone"

SEEDS = [0, 1, 2]

# Extra feature directories to merge in for a given (backbone, dataset, variant, weights).
# Populated by main() from --merge-tags; empty by default so the protocol run is untouched.
_EXTRA_DIRS: dict = {}
SMALL_UNIT = 100
REL_DEPTH = 16 / 30.0          # Wan2.2-TI2V-5B: layer 16 of 30

# Timesteps, per E8_PREREG.md §2.  600 is the literal protocol cell; 298 is SNR-matched to
# Wan's t=600.  Backbones on a flow-matching schedule keep 600 for both roles.
TIMESTEPS = [600, 298]


# ---------------------------------------------------------------------------------------
# backbone registry
# ---------------------------------------------------------------------------------------
# prefix is the cache/DIR name, fixed here so two sizes of the same family never collide
# (cogvideox2b writes cache/cogvideox_feats_*, cogvideox5b writes cache/cogvideox5b_feats_*).
# n_layers drives the Lrel cell; sched tells whether t=600 is already noise-matched.
BACKBONES = {
    "cogvideox2b": {"n_layers": 30, "prefix": "cogvideox", "label": "CogVideoX-2b",
                    "params_b": 2.0, "sched": "vp"},
    "cogvideox5b": {"n_layers": 42, "prefix": "cogvideox5b", "label": "CogVideoX-5b",
                    "params_b": 5.0, "sched": "vp"},
    "wan21_1b3": {"n_layers": 30, "prefix": "wan21_1b3", "label": "Wan2.1-T2V-1.3B",
                  "params_b": 1.3, "sched": "flow"},
    "wan21_14b": {"n_layers": 40, "prefix": "wan21_14b", "label": "Wan2.1-T2V-14B",
                  "params_b": 14.0, "sched": "flow"},
    "ltx": {"n_layers": 28, "prefix": "ltx", "label": "LTX-Video-0.9.5",
            "params_b": 2.0, "sched": "flow"},
    "mochi": {"n_layers": 48, "prefix": "mochi", "label": "Mochi-1-preview",
              "params_b": 10.0, "sched": "flow"},
    # Not extracted through E8: the E7 reference numbers already exist and are reused
    # read-only (see WAN_REF in e8_verdict.py). Listed for documentation only.
    "wan22_5b": {"n_layers": 30, "prefix": "wan22_5b", "label": "Wan2.2-TI2V-5B (E7)",
                 "params_b": 5.0, "sched": "flow"},
}


def _prefix(backbone: str) -> str:
    return BACKBONES[backbone]["prefix"]


def feat_dir(backbone: str, dataset: str, variant: str, tag: str = "",
             weights: str = "real") -> Path:
    """Feature cache layout: cache/<prefix>_feats_<real|random>_<dataset>[_<variant>][_<tag>].

    Three components, all of them load-bearing:
      prefix   separates model sizes within a family (cogvideox vs cogvideox5b)
      weights  real vs random-init DiT — the D1 contrast is exactly this pair
      tag      separates runs on different t-grids; without it a sweep silently reads the
               pre-registered two-timestep records instead
    """
    suffix = "" if variant == "plain" else f"_{variant}"
    tag_suffix = f"_{tag}" if tag else ""
    return ROOT / "cache" / f"{_prefix(backbone)}_feats_{weights}_{dataset}{suffix}{tag_suffix}"


def layers_for(backbone: str) -> dict:
    n = BACKBONES[backbone]["n_layers"]
    rel = int(round(REL_DEPTH * n))
    return {"L16": 16, "Lrel": rel, "n_layers": n, "rel_depth": REL_DEPTH}


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------
def load_dataset(dataset: str) -> tuple[list, str]:
    """-> (manifest rows, grouping key).  Grouping follows E7 (IntPhys2) / E3 (LikePhys)."""
    if dataset == "intphys2":
        man = json.loads((ROOT / "results" / "E7_intphys2" / "manifest.json").read_text())
        return man, "scene_index"
    if dataset == "likephys":
        man = json.loads((ROOT / "results" / "T3_null_floor" / "manifest.json").read_text())
        return man, "subgroup_pair"
    raise ValueError(dataset)


def group_of(row: dict, key: str) -> str:
    if key == "scene_index":
        return str(row["scene_index"])
    return f"{row['scenario']}/{row['subgroup']}"


def load_X(backbone: str, dataset: str, variant: str, seed: int, layer: int,
           timestep: int, tag: str = "", weights: str = "real") -> tuple[np.ndarray, list]:
    """-> X [n, F*D] at one (layer, timestep), meta list.  Cached as .npy like E7 does.

    `timestep` must index the record's `feats` dict — records hold every pre-registered t,
    so reading a fixed "600" here would silently return the same features for every t and
    make the sweep look flat.
    """
    tag_extra = f"_{tag}" if tag else ""
    ftag = (f"{_prefix(backbone)}_{weights}_{dataset}{tag_extra}"
            f"{'' if variant == 'plain' else '_' + variant}_t{timestep}_s{seed}_L{layer}")
    npy = OUT / f"X_{ftag}.npy"
    mjs = OUT / f"meta_{_prefix(backbone)}_{weights}_{dataset}{tag_extra}_{variant}.json"
    if npy.exists() and mjs.exists():
        return np.load(npy, mmap_mode="r"), json.loads(mjs.read_text())

    d = feat_dir(backbone, dataset, variant, tag, weights)
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d} — run e8_extract.py first"
    # A run may split its timesteps across directories (t=600/298 from the protocol run,
    # t=100/200/400/800 from a sweep run). The same clip therefore appears in more than one
    # directory and only SOME copies carry the requested timestep. Deduplicating by file
    # name would keep the protocol copy and then fail on `key in rec["feats"]`; so the
    # extra directories are consulted per clip and the first copy that actually has the
    # timestep wins.
    extras = _EXTRA_DIRS.get((backbone, dataset, variant, weights), [])
    by_name = {f.name: f for f in files}
    if extras:
        for extra in extras:
            for f in sorted(extra.glob("*.pt")):
                by_name.setdefault(f.name, f)
    files = [by_name[k] for k in sorted(by_name)]
    rows, meta = [], []
    key = str(timestep)
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        if key not in rec["feats"]:
            # Look for the clip in the other directories before giving up.
            alt = None
            for extra in extras:
                cand = extra / f.name
                if cand.exists():
                    r2 = torch.load(cand, map_location="cpu", weights_only=False)
                    if key in r2["feats"]:
                        alt = r2
                        break
            if alt is None:
                # This backbone was never extracted at this timestep. A flow-matching model
                # has no separate SNR-matched cell (its t=600 IS the match), so asking for
                # 298 is a caller mistake, not a corrupt cache: report "unavailable" and let
                # the caller skip the cell rather than aborting the whole run.
                return None, None
            rec = alt
        assert rec["variant"] == variant, (
            f"{f.name} produced with variant={rec['variant']!r}, loaded as {variant!r}")
        rows.append(rec["feats"][key][str(seed)][layer].reshape(-1).numpy().astype(np.float32))
        c = rec["clip"]
        meta.append({"clip_id": c.get("clip_id"), "label": c.get("label"),
                     "condition": c.get("condition"), "scene_index": c.get("scene_index"),
                     "scenario": c.get("scenario"), "subgroup": c.get("subgroup"),
                     "kind": c.get("kind"), "group": c.get("group"),
                     "camera": c.get("camera")})
    X = np.stack(rows)
    np.save(npy, X)
    mjs.write_text(json.dumps(meta, indent=1))
    return X, meta


def load_simple(backbone: str, dataset: str, field: str,
                run_tag: str = "") -> tuple[np.ndarray, list]:
    """vae_latent_pooled or raw_pixel_32, flattened — the two non-DiT lower bounds.

    These come from the `real` cache: both are functions of the pixels alone, so the DiT
    weights cannot influence them. Read from the real directory only to avoid loading the
    same numbers twice.
    """
    d = feat_dir(backbone, dataset, "plain", run_tag, "real")
    rows, meta = [], []
    for f in sorted(d.glob("*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        rows.append(rec[field].reshape(-1).numpy().astype(np.float32))
        c = rec["clip"]
        meta.append({"clip_id": c.get("clip_id"), "label": c.get("label"),
                     "scene_index": c.get("scene_index"), "scenario": c.get("scenario"),
                     "subgroup": c.get("subgroup"), "kind": c.get("kind"),
                     "group": c.get("group"), "condition": c.get("condition")})
    return np.stack(rows), meta


# ---------------------------------------------------------------------------------------
# the probe (identical to E7's)
# ---------------------------------------------------------------------------------------
def probe(X, y, groups, seed: int):
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


def cell(backbone, dataset, variant, layer, rows_filter, label_fn, gkey, tag,
         timestep=600, run_tag="", weights="real"):
    """One probe cell: mean AUC/d' over feature seeds + the seed-spread floor.

    `weights` is the real/random-init axis. It must be threaded through explicitly —
    defaulting it to "real" for the random cells would make AUC_real == AUC_random and
    report a D1 of exactly 0% for every backbone, which reads as a strong finding
    ("training contributes nothing") when it is really a path bug.
    """
    per_seed, n, n_pos = [], 0, 0
    for seed in SEEDS:
        X, meta = load_X(backbone, dataset, variant, seed, layer, timestep, run_tag, weights)
        if X is None:
            return None      # timestep not extracted for this backbone
        sel = [i for i, m in enumerate(meta) if rows_filter(m)]
        if not sel:
            return None
        y = np.array([label_fn(meta[i]) for i in sel])
        g = np.array([group_of(meta[i], gkey) for i in sel])
        if len(np.unique(y)) < 2:
            return None
        n, n_pos = len(sel), int(y.sum())
        a, dp = probe(np.asarray(X[sel], dtype=np.float32), y, g, seed=1000 + seed)
        per_seed.append((a, dp))
    A = np.array([p[0] for p in per_seed], float)
    D = np.array([p[1] for p in per_seed], float)
    return {"tag": tag, "backbone": backbone, "dataset": dataset, "variant": variant,
            "layer": layer, "timestep": timestep, "weights": weights,
            "n": n, "n_pos": n_pos,
            "auc_mean": float(np.nanmean(A)), "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "dprime_mean": float(np.nanmean(D)),
            "dprime_seed_spread": float(np.nanstd(D, ddof=1)),
            "per_seed_auc": [float(x) for x in A],
            "per_seed_dprime": [float(x) for x in D],
            "small_unit": bool(n < SMALL_UNIT)}


def _run_spec(spec: dict) -> dict:
    """Worker: one (backbone, layer, timestep, weights, task) probe. Must be top-level and
    take only primitives, so it pickles under the process pool."""
    from threadpoolctl import threadpool_limits

    _, gkey = load_dataset(spec["dataset"])
    if spec["task"] == "physics":
        filt = lambda m: m["label"] in (0, 1)
        lab = lambda m: m["label"]
        variant = "plain"
    elif spec["task"] == "appearance_native":
        filt = lambda m: ((m["group"] == "physics" and m["label"] == 0)
                          or m["kind"] == "color_change")
        lab = lambda m: 0 if (m["group"] == "physics" and m["label"] == 0) else 1
        variant = "plain"
    else:
        raise ValueError(spec["task"])

    with threadpool_limits(limits=spec["threads"]):
        return cell(spec["backbone"], spec["dataset"], variant, spec["layer"], filt, lab,
                    gkey, spec["task"], timestep=spec["timestep"],
                    run_tag=spec["run_tag"], weights=spec["weights"])


def _appearance_cj(bb, dataset, layer, gkey, run_tag, src):
    """D2 on IntPhys2: Possible vs the same Possible clips recoloured (two caches joined).

    Cannot go through `cell()` because the two classes come from different cache
    directories rather than from rows of one.
    """
    per_seed = []
    for seed in SEEDS:
        Xp, mp = load_X(bb, dataset, "plain", seed, layer, 600, run_tag, src)
        Xc, mc = load_X(bb, dataset, "cj", seed, layer, 600, run_tag, src)
        keep = [i for i, m in enumerate(mp) if m["label"] == 0]
        Xa = np.asarray(Xp[keep], dtype=np.float32)
        Xb = np.asarray(Xc, dtype=np.float32)
        X = np.concatenate([Xa, Xb], axis=0)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([group_of(mp[i], gkey) for i in keep]
                     + [group_of(m, gkey) for m in mc])
        per_seed.append(probe(X, y, g, seed=2000 + seed))
    A = np.array([p[0] for p in per_seed], float)
    D = np.array([p[1] for p in per_seed], float)
    return {"tag": "pooled", "backbone": bb, "dataset": dataset, "variant": "plain_vs_cj",
            "layer": layer, "timestep": 600, "weights": src,
            "n": int(len(y)), "n_pos": int(y.sum()),
            "auc_mean": float(np.nanmean(A)),
            "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "dprime_mean": float(np.nanmean(D)),
            "dprime_seed_spread": float(np.nanstd(D, ddof=1)),
            "per_seed_auc": [float(x) for x in A],
            "per_seed_dprime": [float(x) for x in D], "small_unit": False}


def _lower_bound(bb, dataset, field, name, gkey, run_tag):
    """VAE-latent and raw-pixel bounds: functions of the pixels, so no weights involved."""
    per = []
    X, meta = load_simple(bb, dataset, field, run_tag)
    sel = [i for i, m in enumerate(meta) if m["label"] in (0, 1)]
    y = np.array([meta[i]["label"] for i in sel])
    g = np.array([group_of(meta[i], gkey) for i in sel])
    Xs = np.asarray(X[sel], dtype=np.float32)
    for s in SEEDS:
        per.append(probe(Xs, y, g, seed=7000 + s))
    A = np.array([p[0] for p in per], float)
    D = np.array([p[1] for p in per], float)
    return {"tag": name, "backbone": bb, "dataset": dataset, "variant": "plain",
            "layer": -1, "timestep": -1, "weights": "none",
            "n": int(len(y)), "n_pos": int(y.sum()),
            "auc_mean": float(np.nanmean(A)), "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "dprime_mean": float(np.nanmean(D)),
            "dprime_seed_spread": float(np.nanstd(D, ddof=1)),
            "per_seed_auc": [float(x) for x in A], "small_unit": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["cogvideox2b"])
    ap.add_argument("--dataset", default="intphys2", choices=["intphys2", "likephys"])
    ap.add_argument("--timesteps", type=int, nargs="+", default=None,
                    help="override TIMESTEPS (for the noise-schedule sweep)")
    ap.add_argument("--run-tag", default="",
                    help="suffix for the OUTPUT json only; does not affect which cache is read")
    ap.add_argument("--base-tag", default="",
                    help="feature-cache tag of the PRIMARY input directory (default: the "
                         "protocol run, i.e. no tag)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel processes; every cell is independent")
    ap.add_argument("--merge-tags", nargs="*", default=[],
                    help="additional cache tags whose records are merged in (for a t-grid "
                         "split across runs)")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="override the layer list (default: L16 and the 53.3%% depth layer)")
    ap.add_argument("--perm-draws", type=int, default=20)
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    pf = preflight("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    man, gkey = load_dataset(args.dataset)
    timesteps = args.timesteps or TIMESTEPS

    payload = {"args": vars(args), "dataset": args.dataset, "grouping": gkey,
               "env": env_stamp(), "preflight": pf,
               "prereg": "results/E8_second_backbone/E8_PREREG.md",
               "protocol": {"seeds": SEEDS, "readouts": ["auc", "dprime"],
                            "folds": 5, "timesteps": timesteps,
                            "rel_depth": REL_DEPTH},
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": {}}

    # Build the work list. Every cell is independent, so they go to a process pool; the
    # np.npy caches written by load_X are per (layer, t, seed, weights), so concurrent
    # workers writing different cells never collide.
    specs = []
    for bb in args.backbones:
        L = layers_for(bb)
        # --layers overrides the two named reference cells with an explicit sweep grid.
        layer_cells = ([("L%d" % l, l) for l in args.layers] if args.layers
                       else [("L16", L["L16"]), ("Lrel", L["Lrel"])])
        for lname, lidx in layer_cells:
            for t in timesteps:
                for src in ("real", "random"):
                    specs.append({"backbone": bb, "dataset": args.dataset, "layer": lidx,
                                  "timestep": t, "weights": src, "task": "physics",
                                  "run_tag": args.base_tag, "threads": args.threads,
                                  "key": f"{bb}|{args.dataset}|t{t}|{lname}|{src}|physics"})
            if args.dataset == "likephys":
                for src in ("real", "random"):
                    specs.append({"backbone": bb, "dataset": args.dataset, "layer": lidx,
                                  "timestep": 600, "weights": src,
                                  "task": "appearance_native", "run_tag": args.base_tag,
                                  "threads": args.threads,
                                  "key": f"{bb}|{args.dataset}|{lname}|{src}|appearance"})

    for bb in args.backbones:
        for ds in (args.dataset,):
            for v in ("plain", "cj"):
                for wsrc in ("real", "random"):
                    dirs = [feat_dir(bb, ds, v, t, wsrc) for t in args.merge_tags]
                    if dirs:
                        _EXTRA_DIRS[(bb, ds, v, wsrc)] = dirs
    print(f"{len(specs)} probe cells, {args.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_spec, {k: v for k, v in s.items() if k != "key"}): s
                for s in specs}
        for fut, spec in futs.items():
            r = fut.result()
            if not r:
                continue
            payload["cells"][spec["key"]] = r
            print(f"  {spec['key']:46s} n={r['n']:4d} AUC {r['auc_mean']:.4f}"
                  f"±{r['auc_seed_spread']:.4f} d' {r['dprime_mean']:6.3f}", flush=True)

    # IntPhys2's D2 needs the jittered cache, so it is built here rather than in the pool.
    # The layer list must be the same one the physics cells used, including --layers
    # overrides; hardcoding L16/Lrel here silently ran the same layer twice in a sweep.
    if args.dataset == "intphys2":
        for bb in args.backbones:
            # D2 concatenates the *plain* cache with the *jittered* one for BOTH the real
            # and the random backbone, so all four directories have to exist. Checking only
            # "real" let a run start and then die halfway, after burning the appearance cost.
            need = [feat_dir(bb, args.dataset, "cj", args.base_tag, w) for w in ("real", "random")]
            missing = [d.name for d in need if not any(d.glob("*.pt"))]
            if missing:
                print(f"  [skip D2] {bb}: missing jittered cache(s) {missing}", flush=True)
                continue
            L = layers_for(bb)
            layer_cells = ([("L%d" % l, l) for l in args.layers] if args.layers
                           else [("L16", L["L16"]), ("Lrel", L["Lrel"])])
            for lname, lidx in layer_cells:
                for src in ("real", "random"):
                    key = f"{bb}|{args.dataset}|{lname}|{src}|appearance"
                    payload["cells"][key] = _appearance_cj(bb, args.dataset, lidx,
                                                           gkey, args.base_tag, src)
                    c = payload["cells"][key]
                    print(f"  {key:46s} n={c['n']:4d} AUC {c['auc_mean']:.4f}", flush=True)

    for bb in args.backbones:
        for field, name in (("vae_latent_pooled", "vae_latent"), ("raw_pixel_32", "raw_pixel")):
            key = f"{bb}|{args.dataset}|{name}"
            payload["cells"][key] = _lower_bound(bb, args.dataset, field, name, gkey,
                                                 args.base_tag)
            print(f"  {key:46s} AUC {payload['cells'][key]['auc_mean']:.4f}", flush=True)

    print(f"cells done in {time.time()-t0:.0f}s", flush=True)
    payload["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    outp = OUT / f"E8_cells_{args.dataset}{suffix}.json"
    # Merge into an existing result file instead of replacing it. Running one backbone at a
    # time is the normal workflow (each takes minutes), and overwriting silently discarded
    # the other backbones' cells — which reads afterwards as "that backbone was never run".
    if outp.exists():
        try:
            prev = json.loads(outp.read_text())
            merged = dict(prev.get("cells", {}))
            merged.update(payload["cells"])
            payload["cells"] = merged
            payload["merged_from_previous"] = True
            payload["previous_started"] = prev.get("started")
        except Exception as exc:      # noqa: BLE001
            print(f"WARNING: could not merge into existing {outp.name}: {exc}", flush=True)
    outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
