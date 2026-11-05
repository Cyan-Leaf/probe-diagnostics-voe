"""E20-ext: extend the cluster-bootstrap intervals to the 22 cells E20 did not cover.

E20 measured E8's six backbones on both datasets. Every other verdict in
`D1_TALLY.json` / `D2_TALLY.json` -- the V-JEPA 2 pair, HunyuanVideo, the Wan2.2 MoE experts,
the own-band cells, and MiniMax-H3 -- still carries only a point estimate. That is an
inconsistency in the project's own evidence: two cells with identical wording ("fail") differ
in whether anyone checked they survive resampling.

What this does NOT do: it does not re-derive any point estimate. Each cell's AUC is read from
the experiment's own delivered JSON via its own loader, and the interval is computed on
out-of-fold scores refit from the same cache with the SAME arithmetic E20 used
(`oof_scores` / `cluster_draws` / `boot_auc` / `compose` are imported, not copied). The
self-check in `compose`'s caller compares each refit point estimate against the delivered value
and reports the delta.

Method note, same as E20: out-of-fold score bootstrap, so the interval conditions on the
fitted probe and omits the refit component (E16: 2.51 pp). See `E20_REPORT.md` §2.

Run (CPU):
  python3 scripts/e20_extend.py --workers 12 --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20_bootstrap_ci as E20

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E20_bootstrap_ci"
CACHE = ROOT / "cache"

# E19 (H3) needs its own interpreter for the extraction, but the cached records load fine under
# the system python3 -- verified before writing this (feats/600/<seed> -> [2, 12, 5376]).
SEEDS = [0, 1, 2]

# V-JEPA 2 is deterministic: there is no noise draw, so its records carry exactly one feature
# seed (`feats/clean/0`). Asking for seeds 1 and 2 raised KeyError: '1'. That is also why E13's
# delivered numbers have no seed spread -- the seed axis simply does not exist there, and the
# interval for those cells therefore reflects cluster resampling only.
SEEDS_BY_FAMILY = {"vjepa": [0], "cogvideox_clean": [0]}

# OOF scores are the expensive product (a probe fit takes 1-10 min at these widths), and the
# bootstrap stage can fail for a reason unrelated to them. Cache per (cell, seed, arm) so a
# rerun after a downstream fix costs seconds instead of an hour. Keyed on a hash of the cell id
# so filenames stay legal.
OOF_DIR = OUT / "oof_cache"


def _oof_path(cid: str, seed: int, arm: str) -> Path:
    import hashlib
    h = hashlib.sha1(f"{cid}|{seed}|{arm}".encode()).hexdigest()[:16]
    return OOF_DIR / f"{h}.npz"


def _load_oof(cid, seed, arm):
    p = _oof_path(cid, seed, arm)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    if str(z["cid"]) != cid:
        return None
    return {"ok": True, "y": z["y"], "oof": z["oof"], "groups": z["groups"]}


def _save_oof(cid, seed, arm, y, oof, groups):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_oof_path(cid, seed, arm), cid=cid, seed=seed, arm=arm,
                        y=y, oof=oof, groups=groups)

# Wan2.2-A14B: 40 DiT layers, relative-depth cell = round(16/30 * 40) = 21. Same convention E17
# and E18 used. H3 has 50 blocks, its second cell is the delivered 27 (not round(16/30*50)=27).
A14B_N, A14B_REL = 40, int(round(16 / 30.0 * 40))
H3_LAYERS = {"L16": 16, "Lrel": 27}
HUNYUAN_N, HUNYUAN_REL = 60, int(round(16 / 30.0 * 60))
VJEPA_LAYERS = {"L16": 16, "L13": 13}
VJEPA_H_LAYERS = {"L16": 16, "L17": 17}


# ---------------------------------------------------------------------------------------
# per-experiment arm builders: each returns (X, y, groups) for one (arm, seed)
# ---------------------------------------------------------------------------------------
def _recs(dirname: str):
    d = CACHE / dirname
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d}"
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def _lab_groups(recs, dataset="intphys2"):
    """Labels and CLUSTERS. IntPhys 2 groups by `scene_index`; LikePhys has no such field and
    groups by (scenario, subgroup) — the same key E3/E8 use, and the one the delivered verdicts
    were computed with. Getting this wrong would silently change the cluster count, which the
    per-arm assertion in main() catches rather than letting a wrong interval through.
    """
    y = np.array([r["clip"]["label"] for r in recs])
    if dataset == "likephys":
        g = np.array([f"{r['clip']['scenario']}/{r['clip']['subgroup']}" for r in recs])
    else:
        g = np.array([str(r["clip"]["scene_index"]) for r in recs])
    return y, g


def _labelled(recs, get, dataset):
    """(X, y, g) for the physics task, filtering to rows that actually carry a 0/1 label.

    LikePhys' manifest also carries `temporal_disorder` and `color_change` rows whose label is
    `None`. Passing those to `np.unique` raises `TypeError: '<' not supported between 'int' and
    'NoneType'`; E8's own `cell()` filters with `m["label"] in (0, 1)`, and this reproduces that
    selection exactly (750 physics rows / 120 subgroups on LikePhys, 1012 / 253 on IntPhys 2).
    """
    sel = [i for i, r in enumerate(recs) if r["clip"]["label"] in (0, 1)]
    y, g = _lab_groups([recs[i] for i in sel], dataset)
    return get(recs)[sel], y, g


def _flat(rec, layer, seed, t=600, slot=None):
    """One record's feature vector at (layer, timestep, seed). `slot` for H3's stacked tensor."""
    idx = slot if slot is not None else layer
    return rec["feats"][str(t)][str(seed)][idx].reshape(-1).numpy().astype(np.float32)


def arms_vjepa(dataset, size, seed):
    """V-JEPA 2 (E13 for IntPhys 2, E15 for LikePhys). No timestep; `flat` caliber.

    The `cj` cache exists for IntPhys 2 only. LikePhys ships its own `color_change` clips in
    the plain cache (E8_PREREG §5), so `appearance` on LikePhys is built from those rows and
    must NOT touch a nonexistent jitter directory.
    """
    prefix = "vjepa_h" if size == "vith" else "vjepa"
    plain = _recs(f"{prefix}_feats_real_{dataset}")
    rand = _recs(f"{prefix}_feats_random_{dataset}")
    cj = _recs(f"{prefix}_feats_real_{dataset}_cj") if dataset == "intphys2" else None
    layers = VJEPA_H_LAYERS if size == "vith" else VJEPA_LAYERS
    return plain, rand, cj, layers


def arms_hunyuan(dataset, seed):
    plain = _recs(f"hunyuan_feats_real_{dataset}")
    rand = _recs(f"hunyuan_feats_random_{dataset}")
    cj = _recs(f"hunyuan_feats_real_{dataset}_cj") if dataset == "intphys2" else None
    return plain, rand, cj, {"L16": 16, "Lrel": HUNYUAN_REL}


def arms_a14b(dataset, expert, tag, seed, t):
    """Wan2.2-A14B expert. `tag` separates the own-band runs (t300/t900) from E17's."""
    plain = _recs(f"wan22a14b_{expert}_feats_real_{dataset}{tag}")
    rand = _recs(f"wan22a14b_{expert}_feats_random_{dataset}{tag}")
    cj = _recs(f"wan22a14b_{expert}_feats_real_{dataset}_cj{tag}")
    return plain, rand, cj, {"L16": 16, "Lrel": A14B_REL}, t


def arms_h3(seed):
    plain = _recs("minimax_h3_feats_real_intphys2")
    rand = _recs("minimax_h3_feats_random_intphys2")
    cj = _recs("minimax_h3_feats_real_intphys2_cj")
    return plain, rand, cj, H3_LAYERS


# ---------------------------------------------------------------------------------------
# cell registry: one entry per delivered cell that E20 did not cover
# ---------------------------------------------------------------------------------------
def registry():
    R = []
    # --- E13: V-JEPA 2 ViT-L (4 cells: two calibers x two layers) + ViT-H (2 cells)
    for layer in (16, 13):
        for caliber in ("flat", "timeavg"):
            R.append({"id": f"E13|intphys2|vjepa|intphys2|L{layer}|{caliber}",
                      "family": "vjepa", "size": "vitl", "dataset": "intphys2",
                      "layer": layer, "caliber": caliber, "group": "scene_index"})
    for layer in (16, 17):
        R.append({"id": f"E13|intphys2|vith|intphys2|L{layer}",
                  "family": "vjepa", "size": "vith", "dataset": "intphys2",
                  "layer": layer, "caliber": "flat", "group": "scene_index"})
    # --- E13's cjweak-style single cell: cogvideox2b clean (`t0`) reference
    R.append({"id": "E13|intphys2|cogvideox2b|intphys2|L16|flat|t0",
              "family": "cogvideox_clean", "dataset": "intphys2", "layer": 16,
              "group": "scene_index"})
    # --- E13 ViT-H on LikePhys (2 cells)
    for layer in (16, 17):
        R.append({"id": f"E13vithD2|likephys|vith|likephys|L{layer}",
                  "family": "vjepa", "size": "vith", "dataset": "likephys",
                  "layer": layer, "caliber": "flat", "group": "subgroup"})
    # --- E14: HunyuanVideo, IntPhys 2, 2 layers
    for lname, layer in (("L16", 16), ("Lrel", HUNYUAN_REL)):
        R.append({"id": f"E14|intphys2|{lname}", "family": "hunyuan",
                  "dataset": "intphys2", "layer": layer, "group": "scene_index"})
    # --- E17: both experts, 2 layers each
    for lname, layer in (("L16", 16), ("Lrel", A14B_REL)):
        for expert in ("low", "high"):
            R.append({"id": f"E17|intphys2|a14b|{expert}|{lname}", "family": "a14b",
                      "expert": expert, "tag": "", "t": 600, "dataset": "intphys2",
                      "layer": layer, "group": "scene_index"})
    # --- E18: own-band cells
    for lname, layer in (("L16", 16), ("Lrel", A14B_REL)):
        R.append({"id": f"E18-ownband|intphys2|a14b|low@300|{lname}", "family": "a14b",
                  "expert": "low", "tag": "_t300", "t": 300, "dataset": "intphys2",
                  "layer": layer, "group": "scene_index"})
        R.append({"id": f"E18-ownband|intphys2|a14b|high@900|{lname}", "family": "a14b",
                  "expert": "high", "tag": "_t900", "t": 900, "dataset": "intphys2",
                  "layer": layer, "group": "scene_index"})
    # --- E19: H3, 2 layers
    for lname, layer in H3_LAYERS.items():
        R.append({"id": f"E19-h3|intphys2|h3|{lname}", "family": "h3",
                  "dataset": "intphys2", "layer": layer, "group": "scene_index"})
    # --- E15: V-JEPA 2 ViT-L and HunyuanVideo on LikePhys (D2 addendum + D1)
    R.append({"id": "E15-vjepa|likephys|L16", "family": "vjepa", "size": "vitl",
              "dataset": "likephys", "layer": 16, "caliber": "flat", "group": "subgroup"})
    R.append({"id": "E15-vjepa|likephys|L13", "family": "vjepa", "size": "vitl",
              "dataset": "likephys", "layer": 13, "caliber": "flat", "group": "subgroup"})
    R.append({"id": "E15-hunyuan|likephys|L16", "family": "hunyuan",
              "dataset": "likephys", "layer": 16, "group": "subgroup"})
    R.append({"id": "E15-hunyuan|likephys|Lrel", "family": "hunyuan",
              "dataset": "likephys", "layer": HUNYUAN_REL, "group": "subgroup"})
    return R


def _native_appearance(plain, get):
    """LikePhys' built-in appearance contrast: physics-valid vs `color_change`, same subgroups.

    Identical row selection to E8's `appearance_native` arm, so the interval describes the same
    quantity the delivered point estimate does.
    """
    sel = [i for i, r in enumerate(plain)
           if (r["clip"]["group"] == "physics" and r["clip"]["label"] == 0)
           or r["clip"]["kind"] == "color_change"]
    X = get(plain)[sel]
    y = np.array([0 if (plain[i]["clip"]["group"] == "physics"
                        and plain[i]["clip"]["label"] == 0) else 1 for i in sel])
    g = np.array([f"{plain[i]['clip']['scenario']}/{plain[i]['clip']['subgroup']}"
                  for i in sel])
    return X, y, g


def build(spec: dict, arm: str, seed: int):
    """-> (X, y, groups) or None. `arm` is 'real' | 'random' | 'appearance'."""
    fam = spec["family"]
    ds = spec["dataset"]
    layer = spec["layer"]

    if fam == "vjepa":
        plain, rand, cj, _ = arms_vjepa(ds, spec.get("size", "vitl"), seed)
        caliber = spec.get("caliber", "flat")
        key = "feats" if caliber == "flat" else "feats_timeavg"
        get = lambda recs: np.stack([r[key]["clean"][str(seed)][layer].reshape(-1)
                                     .numpy().astype(np.float32) for r in recs])
        if arm == "real":
            return _labelled(plain, get, ds)
        if arm == "random":
            return _labelled(rand, get, ds)
        if ds == "likephys":
            return _native_appearance(plain, get)
        # appearance on IntPhys 2: Possible vs the same Possible recoloured
        keep = [i for i, r in enumerate(plain) if r["clip"]["label"] == 0]
        Xa, Xb = get(plain)[keep], get(cj)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(plain[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in cj])
        return np.concatenate([Xa, Xb]), y, g

    if fam == "cogvideox_clean":
        # E13's `t0` cell: CogVideoX-2b on CLEAN video (no noise), its non-generative-style
        # reference. The cache is E8's cogvideox dir; t0 records carry t="0".
        plain = _recs("cogvideox_feats_real_intphys2_T0")
        rand = _recs("cogvideox_feats_random_intphys2_T0")
        t = "0"
        get = lambda recs: np.stack([r["feats"][t][str(seed)][layer].reshape(-1)
                                     .numpy().astype(np.float32) for r in recs])
        if arm == "real":
            return _labelled(plain, get, ds)
        if arm == "random":
            return _labelled(rand, get, ds)
        cj = _recs("cogvideox_feats_real_intphys2_cj_T0")
        keep = [i for i, r in enumerate(plain) if r["clip"]["label"] == 0]
        Xa, Xb = get(plain)[keep], get(cj)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(plain[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in cj])
        return np.concatenate([Xa, Xb]), y, g

    if fam == "hunyuan":
        plain, rand, cj, _ = arms_hunyuan(ds, seed)
        get = lambda recs: np.stack([_flat(r, layer, seed) for r in recs])
        if arm == "real":
            return _labelled(plain, get, ds)
        if arm == "random":
            return _labelled(rand, get, ds)
        if ds == "intphys2":
            keep = [i for i, r in enumerate(plain) if r["clip"]["label"] == 0]
            Xa, Xb = get(plain)[keep], get(cj)
            y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
            g = np.array([str(plain[i]["clip"]["scene_index"]) for i in keep]
                         + [str(r["clip"]["scene_index"]) for r in cj])
            return np.concatenate([Xa, Xb]), y, g
        return _native_appearance(plain, get)

    if fam == "a14b":
        plain, rand, cj, _, t = arms_a14b(ds, spec["expert"], spec["tag"], seed, spec["t"])
        get = lambda recs: np.stack([_flat(r, layer, seed, t=t) for r in recs])
        if arm == "real":
            return _labelled(plain, get, ds)
        if arm == "random":
            return _labelled(rand, get, ds)
        keep = [i for i, r in enumerate(plain) if r["clip"]["label"] == 0]
        Xa, Xb = get(plain)[keep], get(cj)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(plain[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in cj])
        return np.concatenate([Xa, Xb]), y, g

    if fam == "h3":
        plain, rand, cj, _ = arms_h3(seed)
        slot = 0 if layer == 16 else 1      # extractor stored [L16, Lrel] in that order
        get = lambda recs: np.stack([_flat(r, layer, seed, slot=slot) for r in recs])
        if arm == "real":
            return _labelled(plain, get, ds)
        if arm == "random":
            return _labelled(rand, get, ds)
        keep = [i for i, r in enumerate(plain) if r["clip"]["label"] == 0]
        Xa, Xb = get(plain)[keep], get(cj)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([str(plain[i]["clip"]["scene_index"]) for i in keep]
                     + [str(r["clip"]["scene_index"]) for r in cj])
        return np.concatenate([Xa, Xb]), y, g

    raise ValueError(fam)


def _cell_task(spec: dict) -> dict:
    """Fit the three arms of one cell for one feature seed; return OOF scores."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=spec["threads"]):
        out = {"id": spec["id"], "seed": spec["seed"], "arms": {}}
        for arm in ("real", "random", "appearance"):
            cached = _load_oof(spec["id"], spec["seed"], arm)
            if cached is not None:
                out["arms"][arm] = cached
                continue
            try:
                d = build(spec, arm, spec["seed"])
            except (AssertionError, FileNotFoundError) as exc:
                out["arms"][arm] = {"ok": False, "why": str(exc)[:120]}
                continue
            if d is None:
                out["arms"][arm] = {"ok": False, "why": "unavailable"}
                continue
            X, y, g = d
            if len(np.unique(y)) < 2:
                out["arms"][arm] = {"ok": False, "why": "single class"}
                continue
            # Appearance arm keeps E8's 2000-based probe seed, matching E20 ARM_SEED.
            seed = 2000 + spec["seed"] if arm == "appearance" else 1000 + spec["seed"]
            oof = E20.oof_scores(X, y, g, seed)
            _save_oof(spec["id"], spec["seed"], arm, y, oof, g)
            out["arms"][arm] = {"ok": True, "y": y, "oof": oof, "groups": g}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these cell ids")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    OUT.mkdir(parents=True, exist_ok=True)
    spec_list = registry()
    if args.only:
        spec_list = [s for s in spec_list if any(o in s["id"] for o in args.only)]

    payload = {"task": "E20-ext: cluster-bootstrap CI for the 22 cells E20 did not cover",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "prereg": "results/E20_bootstrap_ci/E20_PREREG.md",
               "method_note": ("Same out-of-fold score bootstrap as E20 (see E20_REPORT.md §2): "
                               "the frozen probe is fit once per (cell, seed) and the resampling "
                               "acts on the held-out scores. Interval omits the refit component "
                               "(E16: 2.51 pp)."),
               "protocol": {"B": E20.B, "draw_seed": E20.DRAW_SEED, "feature_seeds": SEEDS, "seeds_by_family": SEEDS_BY_FAMILY,
                            "threads": args.threads, "ci": [E20.CI_LO, E20.CI_HI],
                            "resample_unit": "cluster (scene_index / scenario-subgroup)"},
               "cells": {}, "self_check": {}}

    jobs = [{"id": s["id"], "spec": s, "seed": sd, "threads": args.threads}
            for s in spec_list
            for sd in SEEDS_BY_FAMILY.get(s["family"], SEEDS)]
    print(f"{len(spec_list)} cells x {len(SEEDS)} seeds = {len(jobs)} fits, "
          f"{args.workers} workers", flush=True)

    t0 = time.time()
    by_cell = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_cell_task, {"id": j["id"], "seed": j["seed"],
                                       "threads": j["threads"], **j["spec"]}) for j in jobs]
        for i, f in enumerate(futs):
            r = f.result()
            by_cell.setdefault(r["id"], []).append(r)
            bad = [a for a, v in r["arms"].items() if not v.get("ok")]
            if bad:
                print(f"  [{r['id']}|s{r['seed']}] unavailable: {bad}", flush=True)
            if (i + 1) % 6 == 0:
                print(f"    {i+1}/{len(jobs)} fits, {(time.time()-t0)/60:.1f} min", flush=True)

    fam_of = {s["id"]: s["family"] for s in spec_list}
    for cid, rows in sorted(by_cell.items()):
        rows.sort(key=lambda r: r["seed"])
        good = [r for r in rows if all(v.get("ok") for v in r["arms"].values())]
        want = len(SEEDS_BY_FAMILY.get(fam_of[cid], SEEDS))
        if len(good) < want:
            print(f"  [skip] {cid}: only {len(good)}/{want} seeds complete", flush=True)
            continue
        arms_res = {a: {"y": [r["arms"][a]["y"] for r in good],
                        "oof": [r["arms"][a]["oof"] for r in good],
                        "groups": [r["arms"][a]["groups"] for r in good]}
                    for a in ("real", "random", "appearance")}
        n_cl = {a: len(np.unique(arms_res[a]["groups"][0])) for a in arms_res}
        assert len(set(n_cl.values())) == 1, f"{cid}: cluster counts differ {n_cl}"
        draws = E20.cluster_draws(arms_res["real"]["groups"][0], E20.B, E20.DRAW_SEED)
        # compose() keys on `{bb}|{dataset}|D{1,2}`; pass the cell id as `bb` and a fixed tag as
        # `dataset` so the resulting key is `<cell>|<tag>|D1`, then rename to `<cell>|D1`.
        cells = E20.compose(cid, "ext", arms_res, draws)
        renamed = {}
        for k, v in cells.items():
            suffix = k.rsplit("|", 1)[-1]
            v["cell"] = f"{cid}|{suffix}"
            v["source_cell"] = cid
            renamed[f"{cid}|{suffix}"] = v
        payload["cells"].update(renamed)
        d1 = renamed[f"{cid}|D1"]
        d2 = renamed[f"{cid}|D2"]
        print(f"  {cid:44s} D1 {d1['d1_pct_point']:6.2f}% "
              f"[{d1['d1_pct_ci95'][0]:6.2f},{d1['d1_pct_ci95'][1]:6.2f}] "
              f"frag={d1['verdict_fragile']:d} | gap {d2['gap_point']:+.4f} "
              f"frag={d2['verdict_fragile']:d}", flush=True)

    payload["elapsed_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "E20ext_cells.json").write_text(json.dumps(payload, indent=1,
                                                      default=lambda o: o.tolist()
                                                      if hasattr(o, "tolist") else None))
    print(f"\nwrote {OUT/'E20ext_cells.json'} in {payload['elapsed_min']} min", flush=True)
    tally(payload)


def tally(payload):
    cells = payload["cells"]
    d1 = [c for c in cells.values() if c["cell"].endswith("D1")]
    d2 = [c for c in cells.values() if c["cell"].endswith("D2")]
    t = {"task": payload["task"], "created": payload["created"], "B": payload["protocol"]["B"],
         "method": payload["method_note"],
         "d1": {"cells": len(d1), "robust": sum(not c["verdict_fragile"] for c in d1),
                "fragile": sum(c["verdict_fragile"] for c in d1),
                "fragile_cells": [{"cell": c["cell"], "d1_pct_point": c["d1_pct_point"],
                                   "ci95": c["d1_pct_ci95"]}
                                  for c in d1 if c["verdict_fragile"]]},
         "d2": {"cells": len(d2), "robust": sum(not c["verdict_fragile"] for c in d2),
                "fragile": sum(c["verdict_fragile"] for c in d2),
                "fragile_cells": [{"cell": c["cell"], "gap_point": c["gap_point"],
                                   "ci95": c["gap_ci95"]}
                                  for c in d2 if c["verdict_fragile"]]},
         "why": ("fragile = the 95% interval straddles the decision boundary. The point-estimate "
                 "verdicts in D1_TALLY/D2_TALLY are NOT retracted by this file.")}
    (OUT / "E20ext_verdict.json").write_text(json.dumps(t, indent=1))
    print(f"EXT D1 robust {t['d1']['robust']}/{t['d1']['cells']}  "
          f"D2 robust {t['d2']['robust']}/{t['d2']['cells']}", flush=True)


if __name__ == "__main__":
    main()
