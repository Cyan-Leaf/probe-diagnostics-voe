"""E7 addendum: is D2's failure an artefact of how strong the recolour was?

The pre-registered D2 control uses hue +0.35 turn / saturation x1.4, which changes RGB by
0.304 on average -- clearly visible.  An obvious objection is that AUC 1.0000 on that task
only shows the probe can see a large low-level change, which would be unremarkable.

This runs the same task with a perceptually subtle recolour (hue +0.05 / sat x1.05, RGB
change 0.043, one seventh of the registered level).  If that still separates better than the
physics task, the D2 conclusion does not depend on the magnitude choice.

EXPLORATORY: added after seeing the registered result, labelled per hard rule 4.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e7_analyse import LAYER, SEEDS, load, probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E7_intphys2"


def appearance_task(bb: str, variant: str):
    per = []
    for seed in SEEDS:
        Xp, mp = load(bb, "plain", seed)
        Xc, mc = load(bb, variant, seed)
        keep = [i for i, m in enumerate(mp) if m["label"] == 0]
        Xa = np.asarray(Xp[keep], dtype=np.float32)
        Xb = np.asarray(Xc, dtype=np.float32)
        X = np.concatenate([Xa, Xb], axis=0)
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        g = np.array([mp[i]["scene_index"] for i in keep] + [m["scene_index"] for m in mc])
        per.append(probe(X, y, g, seed=3000 + seed))
    A = np.array([p[0] for p in per], float)
    D = np.array([p[1] for p in per], float)
    return {"auc_mean": float(np.nanmean(A)), "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "dprime_mean": float(np.nanmean(D)),
            "dprime_seed_spread": float(np.nanstd(D, ddof=1)),
            "per_seed_auc": [float(x) for x in A], "n": int(len(y))}


def main():
    from threadpoolctl import threadpool_limits

    verdict = json.loads((OUT / "E7_verdict.json").read_text())
    phys_real = verdict["D1"]["rows"][0]["real_auc"]
    phys_dp = verdict["D1"]["rows"][0]["real_dprime"]

    res = {"physics_real_auc": phys_real, "physics_real_dprime": phys_dp,
           "levels": {}}
    with threadpool_limits(limits=8):
        for variant, rgb in (("cj", 0.3038), ("cjweak", 0.0425)):
            res["levels"][variant] = {"rgb_mean_abs_change": rgb}
            for bb in ("real", "random"):
                d = (ROOT / "cache" / f"feats_{bb}_intphys2_{variant}")
                if not d.exists():
                    continue
                r = appearance_task(bb, variant)
                res["levels"][variant][bb] = r
                print(f"  {variant:7s} (RGB Δ {rgb:.4f}) {bb:7s} n={r['n']:4d} "
                      f"AUC {r['auc_mean']:.4f}±{r['auc_seed_spread']:.4f}  "
                      f"d' {r['dprime_mean']:6.3f}", flush=True)

    weak = res["levels"]["cjweak"]["real"]["auc_mean"]
    strong = res["levels"]["cj"]["real"]["auc_mean"]
    res["verdict"] = {
        "status": "EXPLORATORY -- added after the registered result",
        "weak_appearance_auc": weak,
        "strong_appearance_auc": strong,
        "physics_auc": phys_real,
        "weak_still_beats_physics": bool(weak > phys_real),
        "reading": (
            f"A recolour {res['levels']['cj']['rgb_mean_abs_change']/res['levels']['cjweak']['rgb_mean_abs_change']:.0f}x "
            f"weaker than the registered one (RGB change 0.043 vs 0.304) still separates at "
            f"AUC {weak:.4f}, against {phys_real:.4f} for the physics task. "
            + ("D2's failure therefore does not depend on the magnitude chosen: the probe is "
               "more sensitive to a subtle physics-legal recolour than to the physics "
               "violation it is supposed to measure."
               if weak > phys_real else
               "At this magnitude the appearance task no longer beats the physics task, so "
               "D2's failure IS magnitude-dependent and must be reported with the "
               "magnitude stated.")),
    }
    (OUT / "E7_d2_sensitivity.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n  {res['verdict']['reading']}")
    print(f"\nwrote {OUT/'E7_d2_sensitivity.json'}")


if __name__ == "__main__":
    main()
