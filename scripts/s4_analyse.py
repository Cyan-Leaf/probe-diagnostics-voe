"""S4 analysis: real AUC per scenario per camera-motion level.

Protocol is deliberately a *copy* of the S3 protocol, because the criterion in
PREREG §4.2 is stated against the S3 baseline (real AUC = 1.0000, sd = 0.0000) and the
floor (permutation sd = 0.1243) is defined for the n=50 / 10-subgroup design:

  layer 16, t = 600, feature seed 0
  downsample to 4 positives per subgroup + all 10 negatives  -> n = 50
  25 resamples with the same s3_seed sequence
  randomised group->fold partition (5 outer folds), inner 3-fold GroupKFold for C

Reusing `e10_s1s3._probe_auc` / `_match_design` verbatim guarantees there is no protocol
drift between the baseline and the levels; anything else would make the comparison
meaningless.

Zero level is judged FIRST and on its own (hard rule 1): the identity transform still goes
through the crop + resample pipeline, so if that alone moves real AUC by more than
2 x 0.1243 = 0.2486, the experiment is measuring the resampler and S4 is void.

Run: python scripts/s4_analyse.py --levels 0
     python scripts/s4_analyse.py --levels 0 1 2 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_s1s3 import _match_design, _probe_auc, _bootstrap_ci
from wan_probe_lib import env_stamp, preflight

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E10_S4"
SCENARIOS = ["ball_drop", "cloth_drape", "faucet", "fluid", "pendulum", "pyramid", "river"]

LAYER = 16
T = 600
SEED = 0
N_POS_PER_SUB = 4
N_RESAMPLE = 25
S3_SEED = 2026
PERM_FLOOR_SD = 0.1243          # PREREG D4, n=50 / 10-subgroup design
THRESHOLD = 2 * PERM_FLOOR_SD   # 0.2486


def load_level(backbone: str, level: int, scenario: str):
    """Read the per-clip .pt files for one (backbone, level, scenario).

    The S4 cache was never consolidated into .npy: 8 dirs x 450 clips x 2.4 MB is small
    enough to read directly, and skipping the consolidate step removes a place where the
    level-to-array mapping could silently go wrong.
    """
    d = ROOT / "cache" / f"feats_{backbone}_cam{level}"
    rows, y, groups, cams = [], [], [], []
    for f in sorted(d.glob("*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        c = rec["clip"]
        if c["scenario"] != scenario or c["group"] != "physics":
            continue
        rows.append(rec["feats"][str(T)][str(SEED)][LAYER].reshape(-1).float().numpy())
        y.append(int(c["label"]))
        groups.append(c["subgroup"])
        cams.append(rec["camera"]["realised"])
    if not rows:
        return None
    return (np.stack(rows), np.array(y), np.array(groups), cams)


def scenario_auc(backbone: str, level: int, scenario: str):
    got = load_level(backbone, level, scenario)
    if got is None:
        return None
    X, y, g, cams = got
    aucs = []
    for r in range(N_RESAMPLE):
        rng = np.random.default_rng(S3_SEED + 7919 * r)
        keep = _match_design(y, g, N_POS_PER_SUB, rng)
        if len(np.unique(y[keep])) < 2:
            continue
        a = _probe_auc(X[keep], y[keep], g[keep], seed=S3_SEED + 104729 * r)["auc_mean"]
        if not np.isnan(a):
            aucs.append(a)
    a = np.asarray(aucs, dtype=float)
    lo, hi = _bootstrap_ci(a) if a.size > 1 else (float("nan"), float("nan"))
    return {
        "n_clips": int(len(y)),
        "n_after_downsample": int(len(keep)),
        "n_resamples": int(a.size),
        "auc_mean": float(a.mean()) if a.size else None,
        "auc_sd": float(a.std(ddof=1)) if a.size > 1 else None,
        "auc_ci95": [lo, hi],
        "auc_min": float(a.min()) if a.size else None,
        "auc_max": float(a.max()) if a.size else None,
        "realised_motion": {
            "translation_px_excursion_mean": float(np.mean(
                [c["translation_px_excursion"] for c in cams])),
            "rot_deg_excursion_mean": float(np.mean(
                [c["rot_deg_excursion"] for c in cams])),
            "max_abs_grid_coord": float(np.max([c["max_abs_grid_coord"] for c in cams])),
            "padding_pixels": int(np.sum([c["padding_pixels"] for c in cams])),
        },
    }


def main():
    import s4_camera as CAM

    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    pf = preflight("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)

    from threadpoolctl import threadpool_limits

    payload = {
        "args": vars(args),
        "env": env_stamp(),
        "preflight": pf,
        "protocol": {
            "layer": LAYER, "t": T, "feature_seed": SEED,
            "downsample": f"{N_POS_PER_SUB} positives per subgroup + all negatives -> n=50",
            "n_resample": N_RESAMPLE, "s3_seed": S3_SEED,
            "identical_to": "e10_s1s3.s3_downsample (verbatim reuse of _match_design/_probe_auc)",
            "permutation_floor_sd": PERM_FLOOR_SD,
            "threshold_2x_floor": THRESHOLD,
        },
        "camera_levels": CAM.LEVELS,
        "crop": CAM.crop_geometry(3),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
    }

    with threadpool_limits(limits=args.threads):
        for lvl in args.levels:
            print(f"=== level {lvl} ===", flush=True)
            payload["levels"][str(lvl)] = {}
            for bb in ("real", "random"):
                d = ROOT / "cache" / f"feats_{bb}_cam{lvl}"
                if not d.exists():
                    print(f"  {bb}: cache missing, skipped", flush=True)
                    continue
                per = {}
                for sc in SCENARIOS:
                    r = scenario_auc(bb, lvl, sc)
                    if r is None:
                        continue
                    per[sc] = r
                    print(f"  {bb:7s} {sc:12s} n={r['n_after_downsample']:3d} "
                          f"AUC {r['auc_mean']:.4f} sd {(r['auc_sd'] or 0):.4f} "
                          f"[{r['auc_min']:.4f},{r['auc_max']:.4f}] "
                          f"exc {r['realised_motion']['translation_px_excursion_mean']:6.1f}px "
                          f"pad {r['realised_motion']['padding_pixels']}", flush=True)
                payload["levels"][str(lvl)][bb] = per
            # margin per scenario
            lv = payload["levels"][str(lvl)]
            if "real" in lv and "random" in lv:
                lv["margin"] = {sc: (lv["real"][sc]["auc_mean"] - lv["random"][sc]["auc_mean"])
                                for sc in lv["real"] if sc in lv["random"]}

    payload["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- zero-level gate (hard rule 1) -------------------------------------------------
    if "0" in payload["levels"] and "real" in payload["levels"]["0"]:
        z = payload["levels"]["0"]["real"]
        deltas = {sc: 1.0 - z[sc]["auc_mean"] for sc in z}
        worst = max(deltas.values()) if deltas else float("nan")
        payload["zero_level_gate"] = {
            "baseline_from_S3": 1.0,
            "per_scenario_drop": deltas,
            "max_drop": worst,
            "mean_drop": float(np.mean(list(deltas.values()))) if deltas else None,
            "threshold": THRESHOLD,
            "passed": bool(worst <= THRESHOLD),
            "meaning": ("identity transform through the crop+resample pipeline; a drop "
                        "beyond 2x the permutation floor would mean S4 measures the "
                        "resampler, not camera motion -> S4 void (A.3 item 1)"),
        }
        print(f"\n=== zero-level gate ===")
        for sc, dv in sorted(deltas.items(), key=lambda kv: -kv[1]):
            print(f"  {sc:12s} drop {dv:+.4f}")
        print(f"  max drop {worst:.4f} vs threshold {THRESHOLD:.4f} -> "
              f"{'PASS' if worst <= THRESHOLD else 'FAIL -> S4 VOID'}")

    (OUT / f"S4_levels_{'_'.join(map(str, args.levels))}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}/S4_levels_{'_'.join(map(str, args.levels))}.json")


if __name__ == "__main__":
    main()
