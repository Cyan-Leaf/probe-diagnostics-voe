"""E15 addendum: D2 on LikePhys for the two new models (E15_D2_ADDENDUM.md).

Compares the SAME thing E8 did on LikePhys — manifest-provided `color_change` clips vs
physics-legal clips — so the two are comparable. Zero GPU: all features are already cached.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_analyse import auc_of, probe_oof
from e8_verdict import interpretable

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "results" / "E15_likephys"
LAYER = 16


def load(dirname):
    return [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted((CACHE / dirname).glob("*.pt"))]


def pick(recs):
    """physics-legal (label 0) as class 0, color_change as class 1 — exactly E8's mask."""
    sel = [r for r in recs if (r["clip"].get("group") == "physics" and r["clip"].get("label") == 0)
           or r["clip"].get("kind") == "color_change"]
    y = np.array([0 if (r["clip"].get("group") == "physics") else 1 for r in sel])
    g = np.array([f"{r['clip']['scenario']}/{r['clip']['subgroup']}" for r in sel])
    return sel, y, g


def vjepa_matrix(recs, L):
    return np.stack([r["feats"]["clean"]["0"][L].reshape(-1).numpy().astype(np.float32)
                     for r in recs])


def hunyuan_matrix(recs, L, fs):
    return np.stack([r["feats"]["600"][str(fs)][L].reshape(-1).numpy().astype(np.float32)
                     for r in recs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    from threadpoolctl import threadpool_limits
    import e13_analyse as E

    t0 = time.time()
    out = {"criterion": "AUC(legal vs color_change) >= AUC(legal vs physics violation) -> fail",
           "dataset": "likephys", "layer": LAYER,
           "why_color_change": "manifest-provided group, same one E8 used; a fresh jitter "
                               "would not be comparable to E8 §14.2",
           "caveat": "color_change has a temporal component per the LikePhys paper (mid-flight), "
                     "so it is not homogeneous with IntPhys2's frame-constant cj",
           "cells": {}}

    with threadpool_limits(args.threads):
        # ---- V-JEPA 2 ViT-L: permutation floor, as in E13 ----
        recs = load("vjepa_feats_real_likephys")
        sel, y, g = pick(recs)
        X = vjepa_matrix(sel, LAYER)
        a_app = auc_of(probe_oof(X, y, g, seed=2000))
        floor = E.perm_floor(X, y, g, seed=9100 + LAYER, draws=10)
        phys = json.loads((OUT / "E15_cells_vjepa.json").read_text())["cells"]["vjepa|likephys|L16"]
        out["cells"]["vjepa2_vitl"] = {
            "auc_appearance_vs_legal": a_app, "auc_physics_vs_legal": phys["auc_real"],
            "gap": a_app - phys["auc_real"], "floor_perm": floor,
            "n_legal": int((y == 0).sum()), "n_color_change": int((y == 1).sum()),
            "verdict": "fail" if a_app >= phys["auc_real"] else "pass",
            "legend": "appearance AUC uses the same 120 legal + 50 color_change clips as E8"}
        print(f"V-JEPA 2   L16: appearance {a_app:.4f}  physics {phys['auc_real']:.4f} "
              f"gap {a_app-phys['auc_real']:+.4f}  -> "
              f"{out['cells']['vjepa2_vitl']['verdict']}", flush=True)

        # ---- HunyuanVideo: seed-spread floor, as in E14 ----
        recs = load("hunyuan_feats_real_likephys")
        sel, y, g = pick(recs)
        per = []
        for fs in (0, 1, 2):
            per.append(auc_of(probe_oof(hunyuan_matrix(sel, LAYER, fs), y, g, seed=2000 + fs)))
        A = np.array(per, float)
        hy = json.loads((OUT / "E15_cells_hunyuan.json").read_text())["cells"]["hunyuan|likephys|L16"]
        out["cells"]["hunyuan"] = {
            "auc_appearance_vs_legal": float(np.nanmean(A)),
            "per_feature_seed": [float(x) for x in A],
            "auc_physics_vs_legal": hy["auc_real"], "gap": float(np.nanmean(A)) - hy["auc_real"],
            "floor_seed_spread": float(np.nanstd(A, ddof=1)),
            "n_legal": int((y == 0).sum()), "n_color_change": int((y == 1).sum()),
            "verdict": "fail" if np.nanmean(A) >= hy["auc_real"] else "pass"}
        print(f"Hunyuan    L16: appearance {np.nanmean(A):.4f}  physics {hy['auc_real']:.4f} "
              f"gap {np.nanmean(A)-hy['auc_real']:+.4f}  -> "
              f"{out['cells']['hunyuan']['verdict']}", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "E15_D2_likephys.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT/'E15_D2_likephys.json'} ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
