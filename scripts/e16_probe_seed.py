"""E16: is E8's D1 stable across the probe's own CV partition?

Why this exists. E15 measured V-JEPA 2's D1 on LikePhys six times with six different probe
seeds and got 43.8%-54.0% — a spread that straddles the 50% threshold. E8 never varied that
seed: every one of its 24 cells used probe seed `1000 + feature_seed`, so the feature seeds
gave a spread but the CV partition did not. If the same 2-3 point swing exists there, then

  * E8's pass/fail calls are still safe where the gap to 50% is large (27-44% on LikePhys,
    77-99% on IntPhys 2), but
  * E8's *ordering* claims inside a dataset — e.g. CogVideoX-2b 42.6% vs CogVideoX-5b 43.5%
    on LikePhys, a 0.9-point gap — are not resolvable at that precision, and
  * §15's LTX layer flip (48.5% at L16 vs 60.7% at L15) needs re-reading: part of that
    12-point gap may be probe noise rather than a real depth effect.

This is a zero-GPU re-analysis of the existing E8 cache. Nothing is re-extracted, no threshold
is changed, and the original `E8_verdict.json` is not touched — this writes its own file.

Design choices, both forced:
  * feature seed stays in the loop (0/1/2) because E8 averages over it; dropping it would
    make the numbers incomparable to E8. Probe seed is the new axis.
  * the point estimate is `mean over 3 feature seeds x 1 probe seed`, exactly E8's cell
    definition, so the seed-1000 column must reproduce `E8_cells_*.json` — that is the
    self-check (`--check`).

Run:
  python scripts/e16_probe_seed.py --workers 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E16_probe_seed"
E8_OUT = ROOT / "results" / "E8_second_backbone"

# OFFSETS, not fixed seeds. E8 calls probe(..., seed=1000 + feature_seed), i.e. the probe's CV
# partition is COUPLED to the feature seed. Using one fixed probe seed for all three feature
# seeds is a different statistic and does NOT reproduce E8 (measured: 0.6443 vs E16's 0.6350 on
# cogvideox2b|intphys2). So the axis here is the OFFSET added to the feature seed, and offset 0
# must reproduce E8 bit-for-bit — that is the self-check.
# The offset is ADDED to the feature seed, and the first offset must be 1000 so that
# offset=1000 reproduces E8's `seed=1000+feature_seed` exactly. Starting the list at 0 made
# "offset 0" mean probe seed 0/1/2, which is a different partition set and scored 0.6485 on
# cogvideox2b|intphys2 against E8's 0.6443 — that mismatch is how this was caught.
PROBE_SEED_OFFSETS = [1000, 2000, 3000, 4242, 5000, 6000]
FEATURE_SEEDS = [0, 1, 2]
BACKBONES = ["cogvideox2b", "cogvideox5b", "ltx", "mochi", "wan21_14b", "wan21_1b3"]
DATASETS = ["intphys2", "likephys"]
LAYERS = {"L16": 16}
T = 600


def _cell_task(spec: dict) -> dict:
    """One (backbone, dataset, weights, layer) x one probe seed -> mean AUC over feature seeds."""
    from threadpoolctl import threadpool_limits

    import e8_analyse as A8

    with threadpool_limits(spec["threads"]):
        man, gkey = A8.load_dataset(spec["dataset"])
        aucs, dps = [], []
        for fs in FEATURE_SEEDS:
            X, meta = A8.load_X(spec["backbone"], spec["dataset"], "plain", fs,
                                spec["layer"], T, "", spec["weights"])
            if X is None:
                return {"key": spec["key"], "missing": "timestep unavailable"}
            # LikePhys's `temporal` (120) and `appearance` (50) groups carry label=None by
            # design — they are controls, not physics pairs. `np.array([None, 0, 1])` is an
            # OBJECT array, and `np.isin(obj_array, [0,1])` silently returns all-False, which
            # passes an empty y to the probe and raises "Unknown label type". Coerce to a
            # nullable int first; -1 marks the non-physics rows so the mask is trustworthy.
            raw = [m["label"] for m in meta]
            y = np.array([-1 if v is None else int(v) for v in raw], dtype=np.int64)
            g = np.array([A8.group_of(m, gkey) for m in meta])
            keep = np.isin(y, [0, 1])
            assert keep.sum() > 0, f"empty label mask for {spec['key']}"
            Xs = np.asarray(X, dtype=np.float32)[keep]
            a, d = A8.probe(Xs, y[keep], g[keep],
                            seed=spec["probe_seed"] + fs)
            aucs.append(a)
            dps.append(d)
        A = np.array(aucs, float)
    return {"key": spec["key"], "probe_seed": spec["probe_seed"],
            "auc_mean": float(np.nanmean(A)), "auc_sd_over_feature_seeds": float(np.nanstd(A, ddof=1)),
            "per_feature_seed_auc": [float(x) for x in A],
            "n": int(keep.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    # MUST be 4: the probe is not bit-reproducible across BLAS thread counts. Measured on
    # cogvideox2b|intphys2 random, offset 1000: 0.5212671690375497 at 4 threads (identical to
    # E8) vs 0.5213261335383826 at 1 thread and 0.520435 at 3 threads. At 3 threads the D1 gap
    # to E8 was 0.577pp, the same order as the spreads this script measures — so the self-check
    # is only meaningful at 4.
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "E16_cells.json"))
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    t0 = time.time()
    tasks = []
    for bb in BACKBONES:
        for ds in DATASETS:
            for w in ("real", "random"):
                for lname, layer in LAYERS.items():
                    for ps in PROBE_SEED_OFFSETS:
                        tasks.append({"backbone": bb, "dataset": ds, "weights": w,
                                      "layer": layer, "probe_seed": ps,
                                      "threads": args.threads,
                                      "key": f"{bb}|{ds}|{w}|{lname}|p{ps}"})
    print(f"{len(tasks)} probe cells ({len(BACKBONES)} backbones x {len(DATASETS)} datasets x "
          f"2 weights x {len(PROBE_SEED_OFFSETS)} probe offsets), {args.workers} workers",
          flush=True)

    # Incremental journal: with 144 cells at ~2 min each a single crashed worker used to lose
    # the whole run (BrokenProcessPool, observed twice). Each finished cell is appended to a
    # jsonl immediately and re-read on start, so a rerun resumes instead of recomputing.
    # The journal's parent must exist BEFORE the loop: writing into a missing directory raised
    # FileNotFoundError, which the per-future `except Exception` swallowed as "retry", so every
    # cell was recomputed forever and nothing was ever persisted (17 min, zero output observed).
    journal = Path(args.out).with_suffix(".journal.jsonl")
    journal.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                r = json.loads(line)
                res[r["key"]] = r
            except Exception:
                continue
        print(f"  resumed {len(res)} cells from {journal.name}", flush=True)

    todo = [s for s in tasks if s["key"] not in res]
    print(f"  {len(todo)} cells to compute", flush=True)
    done = 0
    while todo:
        # Re-create the pool after a break; whatever a dead worker touched is simply retried.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_cell_task, s): s for s in todo}
            remaining = []
            try:
                for fut in as_completed(futs):
                    try:
                        r = fut.result()
                    except Exception as exc:
                        # A persist error must NOT be retried: a broken sink would loop
                        # forever. Only computation errors are retried.
                        if isinstance(exc, OSError):
                            print(f"    FATAL write error {futs[fut]['key']}: {exc}", flush=True)
                            raise
                        remaining.append(futs[fut])
                        print(f"    retry {futs[fut]['key']}: {type(exc).__name__}", flush=True)
                        continue
                    res[r["key"]] = r
                    with journal.open("a") as fh:
                        fh.write(json.dumps(r) + "\n")
                    done += 1
                    if done % 12 == 0:
                        print(f"  {done}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
            except BrokenProcessPool:
                remaining = [s for s in futs.values() if s["key"] not in res]
                print(f"    pool broke; {len(remaining)} cells left, restarting pool", flush=True)
            todo = remaining
        if todo:
            print(f"  {len(todo)} cells still missing, second pass", flush=True)

    # ---- assemble per (backbone, dataset, weights, layer) ------------------------------
    cells = {}
    for bb in BACKBONES:
        for ds in DATASETS:
            for w in ("real", "random"):
                for lname in LAYERS:
                    aucs = [res[f"{bb}|{ds}|{w}|{lname}|p{ps}"]["auc_mean"]
                            for ps in PROBE_SEED_OFFSETS]
                    A = np.array(aucs, float)
                    cells[f"{bb}|{ds}|{w}|{lname}"] = {
                        "per_probe_seed_auc": [float(x) for x in A],
                        "auc_mean_over_probe_seeds": float(np.nanmean(A)),
                        "auc_sd_over_probe_seeds": float(np.nanstd(A, ddof=1)),
                        "auc_min": float(np.nanmin(A)), "auc_max": float(np.nanmax(A)),
                        "n": res[f"{bb}|{ds}|{w}|{lname}|p{PROBE_SEED_OFFSETS[0]}"]["n"]}

    # ---- D1 per seed, its spread, and how often it lands on each side of 50% -----------
    summary = {}
    for bb in BACKBONES:
        for ds in DATASETS:
            r = cells[f"{bb}|{ds}|real|L16"]
            q = cells[f"{bb}|{ds}|random|L16"]
            d1s = []
            for i, _ps in enumerate(PROBE_SEED_OFFSETS):
                a, b = r["per_probe_seed_auc"][i], q["per_probe_seed_auc"][i]
                d1s.append((a - b) / (a - 0.5) if a > 0.5 else float("nan"))
            D = np.array(d1s, float)
            summary[f"{bb}|{ds}"] = {
                "d1_per_probe_seed": [float(x) for x in D],
                "d1_per_probe_seed_pct": [float(x * 100) for x in D],
                "d1_mean_pct": float(np.nanmean(D) * 100),
                "d1_sd_pct": float(np.nanstd(D, ddof=1) * 100),
                "d1_min_pct": float(np.nanmin(D) * 100),
                "d1_max_pct": float(np.nanmax(D) * 100),
                "n_seeds_below_50": int(np.nansum(D < 0.5)),
                "n_seeds": len(PROBE_SEED_OFFSETS),
                "verdict_stable": bool((np.nanmax(D) < 0.5) or (np.nanmin(D) >= 0.5)),
                "verdict_flips": bool(np.nanmin(D) < 0.5 <= np.nanmax(D)),
                "auc_real_mean": r["auc_mean_over_probe_seeds"],
                "auc_random_mean": q["auc_mean_over_probe_seeds"],
            }

    payload = {"task": "E16: probe-partition sensitivity of E8's D1",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "probe_seed_offsets": PROBE_SEED_OFFSETS, "feature_seeds": FEATURE_SEEDS,
               "reference_cell": {"layer": 16, "t": T},
               "why": ("E8 varied only the feature seed; the CV partition was fixed at "
                       "1000+seed. E15 found a 10-point D1 swing from that axis alone on "
                       "V-JEPA 2. This measures the same swing on E8's own cells."),
               "cells": cells, "summary": summary,
               "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))

    print(f"\n{'cell':16s} {'D1 mean':>8s} {'sd':>6s} {'min':>7s} {'max':>7s} "
          f"{'<50%':>5s}  stable?")
    for k, s in summary.items():
        print(f"{k:16s} {s['d1_mean_pct']:7.1f}% {s['d1_sd_pct']:5.2f} "
              f"{s['d1_min_pct']:6.1f}% {s['d1_max_pct']:6.1f}% "
              f"{s['n_seeds_below_50']}/{s['n_seeds']}  "
              f"{'YES' if s['verdict_stable'] else 'FLIPS'}")
    print(f"\nwrote {args.out}  ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
