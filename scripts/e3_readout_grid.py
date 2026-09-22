"""E3 -- the fourth axis: does the *readout method* change what you conclude?

Background.  E0/E1 established (SPEC §1.0d) that at t=600 the two readouts disagree on the
appearance control: denoising error is immune to `color_change` (-1.7%, p=0.13) while the
linear probe is not (AUC 1.0000, higher than the physics task itself).  That single cell
is the whole basis of diagnostic D5, so it needs to hold across the grid, not at one
timestep with one lower bound.

CP-1 C1a asks exactly that: the readout-sensitivity difference must reproduce on
>=2 timesteps and >=2 control pairs.

This script fills the grid now that `random`/`real` features exist at every t and seed:

  for each t in {200,400,600,800,950}:
    probe AUC for real_dit / random_dit / vae_latent / raw_pixel
    on three tasks:  valid_vs_violation | valid_vs_color_change | valid_vs_temporal_disorder
    plus the seed-spread noise floor from the 3 feature seeds

Cache layout note: the completion run wrote to separate prefixes (`real_grid`,
`random_grid`) so the delivered `real`/`random` arrays were never touched.  This script
transparently resolves a (backbone, t, seed) cell to whichever prefix holds it.

Run: python scripts/e3_readout_grid.py --layer 16 --workers 12 --threads 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_caliper import CONS, fit_probe
from wan_probe_lib import env_stamp, preflight

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E3_readout_grid"
T_GRID = [200, 400, 600, 800, 950]
SEEDS = [0, 1, 2]

TASKS = {
    "valid_vs_violation": ("valid", "violation"),
    "valid_vs_color_change": ("valid", "color_change"),
    "valid_vs_temporal_disorder": ("valid", "temporal_disorder"),
}


def resolve(backbone: str, t: int, seed: int):
    """Return (npy_path, index_path) for a cell, from whichever prefix holds it."""
    for prefix in (backbone, f"{backbone}_grid"):
        p = CONS / f"{prefix}_t{t}_s{seed}.npy"
        if p.exists():
            idx = CONS / f"{prefix}_index.json"
            if not idx.exists():
                idx = CONS / f"{backbone}_index.json"
            return p, idx
    return None, None


def load_cell(backbone: str, t: int, seed: int, layer: int):
    p, ip = resolve(backbone, t, seed)
    if p is None:
        return None, None
    idx = json.loads(ip.read_text())
    mm = np.load(p, mmap_mode="r")
    X = np.asarray(mm[:, layer], dtype=np.float32).reshape(len(idx), -1)
    return X, idx


def rows_for_task(idx, task: str):
    """Select rows + labels for one control pair.

    `valid` clips live in the physics group with label 0 (label 1 == violation);
    `color_change` is the appearance group and `temporal_disorder` the temporal group,
    both carry label=None and are excluded from the physics probe by design
    (PREREG §1.3).  Here they are the *positive* class against valid.
    """
    pos_kind = TASKS[task][1]
    sel, y = [], []
    for i, r in enumerate(idx):
        kind = r.get("kind", "")
        grp = r.get("group", "")
        if grp == "physics" and r.get("label") == 0:
            sel.append(i)
            y.append(0)
        elif pos_kind == "violation" and grp == "physics" and r.get("label") == 1:
            sel.append(i)
            y.append(1)
        elif pos_kind != "violation" and kind.startswith(pos_kind):
            sel.append(i)
            y.append(1)
    return np.array(sel), np.array(y)


def job(spec):
    from threadpoolctl import threadpool_limits

    src = spec["source"]
    if src in ("real_dit", "random_dit"):
        bb = "real" if src == "real_dit" else "random"
        X, idx = load_cell(bb, spec["t"], spec["seed"], spec["layer"])
        if X is None:
            return {**spec, "missing": True}
    else:
        name = "real_vae_latent.npy" if src == "vae_latent" else "real_raw_pixel.npy"
        idx = json.loads((CONS / "real_index.json").read_text())
        X = np.asarray(np.load(CONS / name), dtype=np.float32).reshape(len(idx), -1)

    sel, y = rows_for_task(idx, spec["task"])
    if len(np.unique(y)) < 2:
        return {**spec, "missing": True}
    groups = np.array([f"{idx[i]['scenario']}/{idx[i]['subgroup']}" for i in sel])
    with threadpool_limits(limits=spec.get("threads", 8)):
        res = fit_probe(X[sel], y, groups, mode="innerCV")
    return {**spec, **res, "missing": False}


def main():
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    pf = preflight("cuda:0")
    env = env_stamp()
    OUT.mkdir(parents=True, exist_ok=True)

    specs = []
    for task in TASKS:
        for t in T_GRID:
            for seed in args.seeds:
                for src in ("real_dit", "random_dit"):
                    specs.append({"task": task, "t": t, "seed": seed, "source": src,
                                  "layer": args.layer, "threads": args.threads})
        for src in ("vae_latent", "raw_pixel"):
            specs.append({"task": task, "t": -1, "seed": -1, "source": src,
                          "layer": -1, "threads": args.threads})

    print(f"{len(specs)} probe configs", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.workers, verbose=5)(delayed(job)(s) for s in specs)
    res = [r for r in res if not r.get("missing")]
    print(f"done in {time.time()-t0:.0f}s, {len(res)} usable", flush=True)

    (OUT / "E3_probe_raw.json").write_text(json.dumps(
        {"args": vars(args), "env": env, "preflight": pf, "results": res},
        indent=2, ensure_ascii=False))

    # aggregate over seeds -> mean AUC + seed spread (the D4 floor definition used in E1)
    agg = {}
    for task in TASKS:
        agg[task] = {}
        for t in T_GRID:
            cell = {}
            for src in ("real_dit", "random_dit"):
                v = [r["auc_mean"] for r in res
                     if r["task"] == task and r["t"] == t and r["source"] == src]
                if v:
                    cell[src] = {"auc_mean": float(np.mean(v)),
                                 "auc_seed_spread": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                 "n_seeds": len(v), "per_seed": v}
            if "real_dit" in cell and "random_dit" in cell:
                cell["margin_real_minus_random"] = (cell["real_dit"]["auc_mean"]
                                                    - cell["random_dit"]["auc_mean"])
            agg[task][str(t)] = cell
        for src in ("vae_latent", "raw_pixel"):
            v = [r["auc_mean"] for r in res if r["task"] == task and r["source"] == src]
            if v:
                agg[task][src] = {"auc_mean": float(np.mean(v))}

    (OUT / "E3_summary.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    print(f"\n{'task':30s} {'t':>5s} {'real':>8s} {'random':>8s} {'margin':>8s} {'spread':>8s}")
    for task in TASKS:
        for t in T_GRID:
            c = agg[task].get(str(t), {})
            if "real_dit" not in c:
                continue
            sp = c["real_dit"]["auc_seed_spread"]
            print(f"{task:30s} {t:5d} {c['real_dit']['auc_mean']:8.4f} "
                  f"{c['random_dit']['auc_mean']:8.4f} "
                  f"{c['margin_real_minus_random']:+8.4f} "
                  f"{(sp if sp is not None else float('nan')):8.4f}")
    print(f"\nwrote {OUT/'E3_summary.json'}")


if __name__ == "__main__":
    main()
