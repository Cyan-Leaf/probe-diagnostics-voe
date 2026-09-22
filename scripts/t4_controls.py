"""T4 addendum 2: what is the probe actually reading?

Two extra label definitions on the *same* features and the *same* protocol:

  A. valid vs color_change      -> appearance-only edit, physics untouched.
     A high AUC means the probe reads appearance, not physics.
  B. valid vs temporal_disorder -> frame order destroyed, appearance untouched.
     (cf. the temporal controls in Punzo et al. 2606.09646)

Both are run for real_dit, random_dit, vae_latent and raw_pixel.

Run: uv run --no-project python scripts/t4_controls.py --layer 16 --t 600 --workers 12
"""

import argparse
import json
from pathlib import Path

import numpy as np

from t4_caliper import CONS, fit_probe

OUT = Path(__file__).resolve().parents[1] / "results" / "T4_caliper"

TASKS = {
    "valid_vs_color_change": ("valid", "color_change"),
    "valid_vs_temporal_disorder": ("valid", "temporal_disorder"),
    "valid_vs_violation": ("valid", "__violation__"),
}


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
    pos_kind, neg_kind = TASKS[spec["task"]]

    # restrict to the scenarios where the negative class exists, so the comparison is paired
    if neg_kind == "__violation__":
        scen = sorted({r["scenario"] for r in idx})
    else:
        scen = sorted({r["scenario"] for r in idx if r["kind"] == neg_kind})

    sel, y = [], []
    for i, r in enumerate(idx):
        if r["scenario"] not in scen:
            continue
        if r["kind"] == pos_kind:
            sel.append(i)
            y.append(0)
        elif neg_kind == "__violation__":
            if r["group"] == "physics" and r["label"] == 1:
                sel.append(i)
                y.append(1)
        elif r["kind"] == neg_kind:
            sel.append(i)
            y.append(1)

    y = np.array(y)
    groups = np.array([f"{idx[i]['scenario']}/{idx[i]['subgroup']}" for i in sel])
    with threadpool_limits(limits=spec.get("threads", 8)):
        res = fit_probe(X[sel], y, groups, mode="innerCV")
    return {**spec, **res, "n_scenarios": len(scen)}


def main() -> None:
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--t", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    specs = []
    for task in TASKS:
        for source, backbone in (("real_dit", "real"), ("random_dit", "random"),
                                 ("vae_latent", "real"), ("raw_pixel", "real")):
            if backbone == "random" and not (CONS / "random_index.json").exists():
                continue
            specs.append({"task": task, "source": source, "backbone": backbone,
                          "layer": args.layer, "t": args.t, "seed": args.seed,
                          "threads": args.threads})

    res = Parallel(n_jobs=args.workers, verbose=5)(delayed(_job)(s) for s in specs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E1_controls.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n{'task':28s} {'source':12s} {'n':>5s} {'n_pos':>6s} {'AUC':>16s} {'acc':>8s}")
    for r in res:
        print(f"{r['task']:28s} {r['source']:12s} {r['n']:>5d} {r['n_pos']:>6d} "
              f"{r['auc_mean']:.4f}±{r['auc_std']:.3f} {r['acc_mean']:8.4f}")
    print(f"\nwrote {OUT/'E1_controls.json'}")


if __name__ == "__main__":
    main()
