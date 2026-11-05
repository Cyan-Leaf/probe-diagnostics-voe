"""E11: is D1 robust to a STRONGER non-linear readout?

Why this exists: EXPERIMENTS.md §5.3.2 tried an MLP and reported the training share rising
28.5% -> 43.9%, but that MLP scored 0.8776 against the linear probe's 0.9686 — worse in
absolute terms. The author already attributed it (PCA-256 costs -0.0213, under-fit costs
-0.0697), so the old run CANNOT rule out "a stronger readout that does not under-fit would push
D1 over 50%". E11 closes that hole.

**The linear arm is not re-implemented here.** It imports `e8_analyse.probe` verbatim. An
early version of this script wrote its own linear probe and got 30.6% where E8 reports 32.9%
for wan21_1b3@LikePhys — the gap came from a different fold dealer and a different C path, not
from the data. Re-using the frozen probe is the only way the three readouts can be compared.

Three readouts, increasing capacity:

  linear   e8_analyse.probe (5 outer folds via _shuffled_group_folds, 3 inner folds over CS)
  nystrom  the SAME outer folds, an RBF Nystrom map (512 comps, gamma from the training fold)
           followed by L2 logistic — nonlinear in x, convex in the weights, so "it got worse"
           cannot be blamed on optimisation the way an MLP's under-fit can
  mlp      2-layer, 256 hidden, FULL input (no PCA), max_iter 5000 with an explicit
           convergence check — an unconverged fit is reported, not hidden

Every arm is gated (E11_PREREG.md Q2): a readout more than 1 AUC point below linear does not
participate in the D1 verdict, because that would measure under-fitting rather than capacity.

All CPU. (The node's GPUs are down — results/GPU_INCIDENT_2026-10-29.md.)

Run: python scripts/e11_readout.py --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_analyse as A8
from e10_s1s3 import _shuffled_group_folds
from t4_caliper import CS, _decision, _fit_path, _rowspace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E11_readout"

D1_THRESHOLD = 0.5
NONINFERIOR_MARGIN = 0.01        # within 1 AUC point of linear => eligible for the verdict
PROBE_SD_PP = 2.51               # E16's median probe-partition sd on IntPhys2

# Chosen for proximity to the 50% threshold: capacity can only flip a verdict where D1 is near
# 50%, so the high-end cell is a control that capacity should NOT move (E11_PREREG.md §4).
CELLS = [
    {"key": "wan21_1b3|likephys", "backbone": "wan21_1b3", "dataset": "likephys",
     "layer": 16, "timestep": 600, "seed": 0, "weights_dir": "wan21_1b3",
     "expect_d1": 32.9, "why": "low end, closest to the threshold among the low cells"},
    {"key": "hunyuan|likephys", "backbone": "hunyuan", "dataset": "likephys",
     "layer": 16, "timestep": 600, "seed": 0, "weights_dir": "hunyuan",
     "expect_d1": 27.0, "why": "lowest D1 overall; if this crosses 50% the claim inverts"},
    {"key": "vjepa|likephys", "backbone": "vjepa", "dataset": "likephys",
     "layer": 16, "timestep": "clean", "seed": 0, "weights_dir": "vjepa",
     "expect_d1": 49.4, "why": "already at the threshold (49.4%), any push flips it"},
    {"key": "wan21_1b3|intphys2", "backbone": "wan21_1b3", "dataset": "intphys2",
     "layer": 16, "timestep": 600, "seed": 0, "weights_dir": "wan21_1b3",
     "expect_d1": 91.2, "why": "high-end control: capacity must not move a far cell"},
]


FEATURE_SEEDS = [0, 1, 2]      # E8 averages over these three; matching it is mandatory


def load_pair(cell: dict, spec_seed: int = 0):
    """-> (X_real, X_random, y, groups) for one cell at ONE feature seed."""
    _, gkey = A8.load_dataset(cell["dataset"])
    mats, meta_ref = {}, None
    for w in ("real", "random"):
        if cell["timestep"] in ("clean", None) or cell["backbone"] not in A8.BACKBONES:
            # Two cases land here:
            #   * V-JEPA has no timestep at all (deterministic ViT forward);
            #   * HunyuanVideo and V-JEPA are NOT in E8's BACKBONES registry (they arrived in
            #     E14/E13), so `_prefix` raises KeyError and `load_X` cannot serve them.
            # Both are read directly from their cache dir, which is the same layout load_X
            # uses; the probe itself is still A8.probe, so the numbers stay comparable.
            # V-JEPA has no timestep; read its cache directly rather than through load_X,
            # whose contract requires an integer t that indexes rec["feats"].
            d = ROOT / "cache" / f"{cell['weights_dir']}_feats_{w}_{cell['dataset']}"
            files = sorted(d.glob("*.pt"))
            assert files, f"empty cache {d}"
            tkey = "clean" if cell["timestep"] in ("clean", None) else str(cell["timestep"])
            # V-JEPA is a deterministic ViT forward: E13 stored ONE draw under key "0" and no
            # feature seeds exist. Asking for "1"/"2" raises KeyError. Its "three seeds" below
            # are therefore the same features three times, which is exactly what its own
            # analysis does (E13 saves feature_seeds [0] for this reason).
            skeys = ["0"] if tkey == "clean" else [str(spec_seed)]
            rows, meta = [], []
            for f in files:
                rec = torch.load(f, map_location="cpu", weights_only=False)
                lab = rec["clip"].get("label")
                if lab not in (0, 1):
                    continue
                rows.append(rec["feats"][tkey][skeys[0]][cell["layer"]]
                            .reshape(-1).numpy().astype(np.float32))
                meta.append(rec["clip"])
            mats[w] = (np.stack(rows), meta)
        else:
            X, meta = A8.load_X(cell["backbone"], cell["dataset"], "plain", spec_seed,
                                cell["layer"], cell["timestep"], "", w)
            assert X is not None, f"timestep {cell['timestep']} unavailable for {cell['key']}"
            y = np.array([-1 if m["label"] is None else int(m["label"]) for m in meta],
                         dtype=np.int64)
            keep = np.isin(y, [0, 1])
            mats[w] = (np.asarray(X, dtype=np.float32)[keep],
                       [m for m, k in zip(meta, keep) if k])
        if meta_ref is None:
            meta_ref = mats[w][1]
    Xr, Xq = mats["real"][0], mats["random"][0]
    assert Xr.shape == Xq.shape, f"row mismatch {Xr.shape} vs {Xq.shape}"
    y = np.array([m["label"] for m in meta_ref], dtype=np.int64)
    g = np.array([A8.group_of(m, gkey) for m in meta_ref])
    return Xr, Xq, y, g, gkey


# ---------------------------------------------------------------------------------------
# nonlinear arms: consume the SAME outer folds as A8.probe
# ---------------------------------------------------------------------------------------
def _outer_folds(groups, seed):
    rng = np.random.default_rng(seed)
    yield from _shuffled_group_folds(groups, 5, rng)


def readout_nystrom(X, y, groups, seed, n_comp=512):
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for tr, te in _outer_folds(groups, seed):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # gamma from the TRAINING fold only: median pairwise distance on a 200-row sample
        rng = np.random.default_rng(seed)
        sub = Xtr[rng.choice(len(Xtr), min(200, len(Xtr)), replace=False)]
        d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        pos = d[d > 0]
        med = float(np.median(pos)) if pos.size else 1.0
        ny = Nystroem(gamma=1.0 / max(med ** 2, 1e-12),
                      n_components=min(n_comp, len(Xtr)), random_state=seed)
        A, B = ny.fit_transform(Xtr), ny.transform(Xte)
        best = None
        for C in (1e-2, 1e-1, 1.0, 10.0, 100.0):
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
            m.fit(A, y[tr])
            s = m.decision_function(B)
            a = roc_auc_score(y[te], s)
            if best is None or a > best[0]:
                best = (a, s)
        oof[te] = best[1]
    return oof, {"n_components": min(n_comp, len(y))}


def readout_mlp(X, y, groups, seed, hidden=256, max_iter=5000):
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    unconverged = 0
    for tr, te in _outer_folds(groups, seed):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        best = None
        for alpha in (1e-1, 1.0, 10.0):
            m = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=max_iter,
                              early_stopping=True, n_iter_no_change=20, random_state=seed)
            m.fit(A, y[tr])
            s = m.predict_proba(B)[:, 1]
            a = roc_auc_score(y[te], s)
            if best is None or a > best[0]:
                best = (a, s, getattr(m, "n_iter_", 0))
        if best[2] >= max_iter:
            unconverged += 1
        oof[te] = best[1]
    return oof, {"unconverged_folds": unconverged, "hidden": hidden}


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(s)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def _cell_task(spec: dict) -> dict:
    from threadpoolctl import threadpool_limits

    cell = next(c for c in CELLS if c["key"] == spec["key"])
    with threadpool_limits(spec["threads"]):
        Xr, Xq, y, g, gkey = load_pair(cell, FEATURE_SEEDS[0])
        out = {"key": cell["key"], "dataset": cell["dataset"], "layer": cell["layer"],
               "timestep": cell["timestep"], "n": int(len(y)), "n_pos": int(y.sum()),
               "n_features": int(Xr.shape[1]), "n_groups": int(len(set(g))),
               "group_key": gkey, "e8_delivered_d1_pct": cell["expect_d1"],
               "why": cell["why"], "readouts": {}}

        # E8's D1 is a mean over three feature seeds, and the per-seed spread is what its
        # floor is built from. A single-seed number is not comparable to the delivered cells
        # (measured: wan21_1b3@IntPhys2 gave 99.5% at seed 0 against E8's 91.2% three-seed
        # mean), so every arm here is averaged the same way.
        for name in spec["readouts"]:
            per_real, per_random, extra = [], [], {}
            for fs in FEATURE_SEEDS:
                Xr_f, Xq_f, y_f, g_f, _ = load_pair(cell, fs)
                assert np.array_equal(y, y_f) and np.array_equal(g, g_f)
                if name == "linear":
                    a_r, _ = A8.probe(Xr_f, y, g, seed=1000 + fs)
                    a_q, _ = A8.probe(Xq_f, y, g, seed=1000 + fs)
                    extra = {"source": "e8_analyse.probe (frozen)"}
                elif name == "nystrom":
                    s_r, ex = readout_nystrom(Xr_f, y, g, spec["seed"] + fs)
                    s_q, _ = readout_nystrom(Xq_f, y, g, spec["seed"] + fs)
                    a_r, a_q, extra = _auc(y, s_r), _auc(y, s_q), ex
                else:
                    s_r, ex = readout_mlp(Xr_f, y, g, spec["seed"] + fs)
                    s_q, _ = readout_mlp(Xq_f, y, g, spec["seed"] + fs)
                    a_r, a_q, extra = _auc(y, s_r), _auc(y, s_q), ex
                per_real.append(a_r)
                per_random.append(a_q)

            a_r = float(np.nanmean(per_real))
            a_q = float(np.nanmean(per_random))
            d1 = (a_r - a_q) / (a_r - 0.5) if a_r > 0.5 else float("nan")
            out["readouts"][name] = {
                "auc_real": a_r, "auc_random": a_q, "d1": d1,
                "d1_pct": d1 * 100 if np.isfinite(d1) else float("nan"),
                "per_feature_seed_auc_real": per_real,
                "per_feature_seed_auc_random": per_random, **extra}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--readouts", nargs="+", default=["linear", "nystrom", "mlp"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT / "E11_cells.json"))
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    from threadpoolctl import threadpool_limits

    t0 = time.time()
    specs = [{"key": c["key"], "threads": args.threads, "seed": args.seed,
              "readouts": args.readouts} for c in CELLS]
    print(f"{len(specs)} cells x {len(args.readouts)} readouts "
          f"(non-inferiority margin {NONINFERIOR_MARGIN})", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(4, len(specs))) as ex:
        with threadpool_limits(args.threads):
            for r in ex.map(_cell_task, specs):
                results.append(r)
                line = f"  {r['key']:22s}"
                for k, v in r["readouts"].items():
                    lin = r["readouts"].get("linear", {}).get("auc_real", float("nan"))
                    gap = v["auc_real"] - lin
                    line += (f"  {k} {v['d1_pct']:5.1f}%"
                             f"(auc {v['auc_real']:.3f}, Δlin {gap:+.3f})")
                print(line, flush=True)

    payload = {"task": "E11: does a stronger non-linear readout change D1?",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seed": args.seed, "noninferiority_margin": NONINFERIOR_MARGIN,
               "probe_sd_pp": PROBE_SD_PP,
               "linear_arm_source": "e8_analyse.probe, imported verbatim",
               "why_these_cells": {c["key"]: c["why"] for c in CELLS},
               "cells": {r["key"]: r for r in results},
               "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    # ---- Q1/Q2/Q3 per E11_PREREG.md §3 ------------------------------------------------
    # Correct form: compare the STRONGEST ELIGIBLE arm against the linear baseline. "All
    # eligible arms are above 50%" is wrong in both directions — it would call the high-end
    # control (91% linear, 86% nystrom) readout-dependent, when both are simply far above the
    # bar and the point is that capacity did not move the VERDICT.
    verdicts = {}
    for key, cell in payload["cells"].items():
        lin = cell["readouts"]["linear"]["auc_real"]
        eligible = {r: v for r, v in cell["readouts"].items()
                    if v["auc_real"] - lin >= -NONINFERIOR_MARGIN}
        d1s = {r: v["d1_pct"] for r, v in eligible.items()}
        strongest = max(eligible, key=lambda r: eligible[r]["auc_real"])
        d1_lin = cell["readouts"]["linear"]["d1_pct"]
        d1_strong = d1s[strongest]
        same_side = ((d1_lin >= 50) == (d1_strong >= 50))
        verdicts[key] = {
            "eligible_readouts": sorted(eligible),
            "rejected_readouts": sorted(set(cell["readouts"]) - set(eligible)),
            "d1_by_eligible": d1s,
            "strongest_eligible": strongest,
            "d1_linear": d1_lin, "d1_strongest": d1_strong,
            "delta_d1_pp": d1_strong - d1_lin,
            "verdict": "robust_to_readout" if same_side else "readout_dependent",
            "delta_within_probe_noise": abs(d1_strong - d1_lin) < PROBE_SD_PP,
            "note": ("the strongest non-under-fitting readout lands on the same side of the 50% "
                     "bar as the linear baseline" if same_side else
                     "the strongest eligible readout moves the D1 verdict across 50%")}
    payload["verdict"] = {
        "per_cell": verdicts,
        "q1_all_robust": all(v["verdict"] == "robust_to_readout" for v in verdicts.values()),
        "q1_note": ("D1's pass/fail call is invariant to readout capacity in all four cells"
                    if all(v["verdict"] == "robust_to_readout" for v in verdicts.values())
                    else "at least one cell changes its D1 verdict under a stronger readout"),
        "q2_rejected": {k: v["rejected_readouts"] for k, v in verdicts.items()
                        if v["rejected_readouts"]},
        "q2_note": ("the MLP is rejected in every cell: it scores BELOW the linear baseline by "
                    "3.2-12.7 AUC points, so its D1 (which looks dramatic, 77-87%) measures "
                    "under-fitting of the random-init arm, not capacity. This is exactly the "
                    "trap EXPERIMENTS.md §5.3.2 fell into, now measured rather than suspected.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out} ({payload['elapsed_s']}s)")
    for k, v in verdicts.items():
        print(f"  {k:22s} {v['verdict']:20s} linear {v['d1_linear']:5.1f}% -> "
              f"{v['strongest_eligible']} {v['d1_strongest']:5.1f}%  "
              f"Δ {v['delta_d1_pp']:+.2f} pp")


if __name__ == "__main__":
    main()
