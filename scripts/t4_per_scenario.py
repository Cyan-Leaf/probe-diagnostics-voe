"""T4 addendum: per-scenario probes (SPEC §6.3 — per-scenario distribution over global mean).

Trains the same linear-probe protocol *within each scenario* (GroupKFold over that
scenario's 10 subgroups) for real_dit and random_dit at a given layer/timestep, plus the
two non-DiT lower bounds.

Run: uv run --no-project python scripts/t4_per_scenario.py --layer 16 --t 600 --workers 12
"""

import argparse
import json
from pathlib import Path

import numpy as np

from t4_caliper import CONS, fit_probe

OUT = Path(__file__).resolve().parents[1] / "results" / "T4_caliper"


def _load(backbone: str, source: str, layer: int, t: int, seed: int):
    idx = json.loads((CONS / f"{backbone}_index.json").read_text())
    if source in ("real_dit", "random_dit"):
        mm = np.load(CONS / f"{backbone}_t{t}_s{seed}.npy", mmap_mode="r")
        X = np.asarray(mm[:, layer], dtype=np.float32)
    elif source == "vae_latent":
        X = np.asarray(np.load(CONS / "real_vae_latent.npy"), dtype=np.float32)
    else:
        X = np.asarray(np.load(CONS / "real_raw_pixel.npy"), dtype=np.float32)
    return X.reshape(len(idx), -1), idx


def _job(spec: dict) -> dict:
    from threadpoolctl import threadpool_limits

    X, idx = _load(spec["backbone"], spec["source"], spec["layer"], spec["t"], spec["seed"])
    sel = [i for i, r in enumerate(idx)
           if r["group"] == "physics" and r["scenario"] == spec["scenario"]]
    y = np.array([idx[i]["label"] for i in sel])
    groups = np.array([idx[i]["subgroup"] for i in sel])
    with threadpool_limits(limits=spec.get("threads", 8)):
        res = fit_probe(X[sel], y, groups, mode="innerCV")
    return {**spec, **res}


def main() -> None:
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--t", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    idx = json.loads((CONS / "real_index.json").read_text())
    scenarios = sorted({r["scenario"] for r in idx})

    specs = []
    for sc in scenarios:
        specs.append({"source": "real_dit", "backbone": "real", "scenario": sc,
                      "layer": args.layer, "t": args.t, "seed": args.seed,
                      "threads": args.threads})
        if (CONS / "random_index.json").exists():
            specs.append({"source": "random_dit", "backbone": "random", "scenario": sc,
                          "layer": args.layer, "t": args.t, "seed": args.seed,
                          "threads": args.threads})
        specs.append({"source": "vae_latent", "backbone": "real", "scenario": sc,
                      "layer": -1, "t": -1, "seed": -1, "threads": args.threads})
        specs.append({"source": "raw_pixel", "backbone": "real", "scenario": sc,
                      "layer": -1, "t": -1, "seed": -1, "threads": args.threads})

    print(f"{len(specs)} per-scenario configs", flush=True)
    res = Parallel(n_jobs=args.workers, verbose=5)(delayed(_job)(s) for s in specs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E1_per_scenario.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n{'scenario':16s} {'real':>16s} {'random':>16s} {'vae':>8s} {'pixel':>8s} {'real-rand':>10s}")
    table = {}
    for sc in scenarios:
        row = {r["source"]: r for r in res if r["scenario"] == sc}
        rl = row.get("real_dit", {}).get("auc_mean", float("nan"))
        rd = row.get("random_dit", {}).get("auc_mean", float("nan"))
        va = row.get("vae_latent", {}).get("auc_mean", float("nan"))
        px = row.get("raw_pixel", {}).get("auc_mean", float("nan"))
        rls = row.get("real_dit", {}).get("auc_std", float("nan"))
        rds = row.get("random_dit", {}).get("auc_std", float("nan"))
        table[sc] = {"real": rl, "real_std": rls, "random": rd, "random_std": rds,
                     "vae": va, "pixel": px, "delta": rl - rd,
                     "n": row.get("real_dit", {}).get("n"),
                     "n_pos": row.get("real_dit", {}).get("n_pos")}
        print(f"{sc:16s} {rl:.4f}±{rls:.3f} {rd:.4f}±{rds:.3f} {va:8.4f} {px:8.4f} {rl-rd:>10.4f}")
    (OUT / "E1_per_scenario_table.json").write_text(json.dumps(table, indent=2))
    print(f"\nwrote {OUT/'E1_per_scenario.json'}")


if __name__ == "__main__":
    main()
