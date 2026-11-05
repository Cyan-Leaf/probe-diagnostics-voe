"""E8 verdict: turn the cells into the four lower bounds, D1 and D2, per backbone.

Two questions, kept separate:

* Does the *ordering* reproduce — trained DiT > random DiT > raw pixel / VAE latent?
  This is the pre-registered target (E8_PREREG.md §4); magnitudes are allowed to differ.
* Do the D1 and D2 *verdicts* point the same way as on Wan?

D1 is reported only where it is interpretable.  PREREG.md's T10 addendum says the ratio
`(AUC_real − AUC_random) / (AUC_real − 0.5)` degenerates as `AUC_real → 0.5`, and the known
failure case (a purely-appearance representation) reported "111% training contribution" and
passed.  The gate applied here is the frozen one, and it bites hard on CogVideoX: at t=600
that backbone sits at σ_eff=0.977, so real AUC is expected near chance and the ratio will
refuse to be read at all.  That is why E8_PREREG.md §2 pre-registered t=298 alongside.

Run: python scripts/e8_verdict.py --backbones cogvideox2b --dataset intphys2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_analyse import BACKBONES, TIMESTEPS, layers_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E8_second_backbone"

# Wan2.2-TI2V-5B reference values, for "does the ordering reproduce".
# IntPhys2: results/E7_intphys2/E7_verdict.json.  LikePhys: results/E3_readout_grid/E3_summary.json.
WAN_REF = {
    "intphys2": {"real": 0.6458, "random": 0.5168, "vae_latent": 0.5358, "raw_pixel": 0.5476,
                 "d1_pct": 88.5, "appearance": 0.9980, "d1_verdict": "pass",
                 "d2_verdict": "fail"},
    "likephys": {"real": 0.9686, "random": 0.8353, "vae_latent": 0.8200,
                 "raw_pixel": 0.7854, "d1_pct": 28.5, "appearance": 1.0000,
                 "d1_verdict": "fail", "d2_verdict": "fail"},
}

D1_THRESHOLD = 50.0


def d1_ratio(real, rand):
    """(AUC_real − AUC_random) / (AUC_real − 0.5), as a FRACTION (0.852 = 85.2%).

    Returning a fraction keeps the arithmetic in one unit; everything user-facing converts
    to percent at the print/serialise step. Getting this backwards silently divides every
    reported share by 100 and turns a pass into a fail.
    """
    if not np.isfinite(real) or not np.isfinite(rand):
        return float("nan")
    denom = real - 0.5
    return (real - rand) / denom if denom > 1e-9 else float("nan")


def interpretable(real, rand, floor):
    """PREREG.md T10 gate: effect > 2x floor AND real − 0.5 > 2x floor, both in AUC units."""
    if not np.isfinite(real) or not np.isfinite(rand) or not np.isfinite(floor) or floor <= 0:
        return False, "floor unavailable"
    effect = real - rand
    if effect <= 2 * floor:
        return False, f"effect {effect:.4f} <= 2x floor {2*floor:.4f}"
    if (real - 0.5) <= 2 * floor:
        return False, f"real-0.5 {real-0.5:.4f} <= 2x floor {2*floor:.4f} (ratio degenerates)"
    return True, "effect and denominator both clear 2x floor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["cogvideox2b"])
    ap.add_argument("--dataset", default="intphys2", choices=["intphys2", "likephys"])
    args = ap.parse_args()

    cells = json.loads((OUT / f"E8_cells_{args.dataset}.json").read_text())["cells"]
    ref = WAN_REF[args.dataset]
    verdicts = {"dataset": args.dataset, "wan_reference": ref,
                "d1_threshold": D1_THRESHOLD,
                "note_on_timesteps": (
                    "Both timesteps are reported. t=600 is PREREG.md's reference cell verbatim; "
                    "t=298 has the SNR that Wan's t=600 has (E8_PREREG.md §2). On a VP schedule "
                    "the two differ by ~20x in SNR, so agreement between them is evidence and "
                    "disagreement is a finding, not a bug."),
                "backbones": {}}

    for bb in args.backbones:
        meta = BACKBONES[bb]
        L = layers_for(bb)
        entry = {"label": meta["label"], "params_b": meta["params_b"], "n_layers": L["n_layers"],
                 "cells": {}}

        for lname in ("L16", "Lrel"):
            for t in TIMESTEPS:
                real = cells.get(f"{bb}|{args.dataset}|t{t}|{lname}|real|physics")
                rand = cells.get(f"{bb}|{args.dataset}|t{t}|{lname}|random|physics")
                if not real or not rand:
                    continue
                r_auc, q_auc = real["auc_mean"], rand["auc_mean"]
                floor = max(real["auc_seed_spread"] or 0.0, rand["auc_seed_spread"] or 0.0)
                ok, why = interpretable(r_auc, q_auc, floor)
                pct = d1_ratio(r_auc, q_auc)
                entry["cells"][f"{lname}|t{t}"] = {
                    "layer": real["layer"], "timestep": t,
                    "real_auc": r_auc, "random_auc": q_auc,
                    "real_dprime": real["dprime_mean"], "random_dprime": rand["dprime_mean"],
                    "floor": floor,
                    "d1_training_share": pct if np.isfinite(pct) else None,
                    "d1_pct": 100.0 * pct if np.isfinite(pct) else None,
                    "interpretable": ok, "interpretable_because": why,
                    "verdict": ("NOT interpretable" if not ok else
                                ("pass" if 100.0 * pct >= D1_THRESHOLD else "fail")),
                    "n": real["n"],
                    "ordering": {
                        "trained_gt_random": bool(r_auc > q_auc),
                        "random_gt_chance": bool(q_auc > 0.5),
                    },
                }

        # D2: appearance control vs physics, real backbone, best available timestep
        for lname in ("L16", "Lrel"):
            app = cells.get(f"{bb}|{args.dataset}|{lname}|real|appearance")
            phys = cells.get(f"{bb}|{args.dataset}|t600|{lname}|real|physics")
            if not app or not phys:
                continue
            entry["cells"][f"{lname}|d2"] = {
                "physics_auc": phys["auc_mean"], "appearance_auc": app["auc_mean"],
                "gap": app["auc_mean"] - phys["auc_mean"],
                "passed": bool(app["auc_mean"] < phys["auc_mean"]),
                "verdict": "pass" if app["auc_mean"] < phys["auc_mean"] else "fail",
                "n_appearance": app["n"],
            }

        # the two non-DiT bounds (backbone-independent, but read from this backbone's cache
        # so the frames/preprocessing are identical)
        for name in ("vae_latent", "raw_pixel"):
            c = cells.get(f"{bb}|{args.dataset}|{name}")
            if c:
                entry["cells"][f"lb|{name}"] = {"auc_mean": c["auc_mean"], "n": c["n"]}

        # ---- ordering verdict, the pre-registered question -----------------------------
        pick = entry["cells"].get(f"Lrel|t298") or entry["cells"].get(f"L16|t600") or {}
        va = entry["cells"].get("lb|vae_latent", {}).get("auc_mean")
        px = entry["cells"].get("lb|raw_pixel", {}).get("auc_mean")
        if pick and va is not None and px is not None:
            seq = [pick.get("real_auc"), pick.get("random_auc"), max(va, px)]
            # The Wan reference is computed from its own numbers, never asserted. The full
            # ordering trained > random > non-DiT does NOT hold on Wan either (IntPhys2:
            # random 0.5168 < raw-pixel 0.5476), so claiming it reproduces would be false.
            w = [ref["real"], ref["random"], max(ref["vae_latent"], ref["raw_pixel"])]
            entry["ordering_check"] = {
                "cell_used": "Lrel|t298" if "Lrel|t298" in entry["cells"] else "L16|t600",
                "trained": seq[0], "random": seq[1], "non_dit": seq[2],
                "trained_gt_random": bool(seq[0] > seq[1]),
                "random_gt_nondit": bool(seq[1] > seq[2]),
                "trained_gt_nondit": bool(seq[0] > seq[2]),
                "full_ordering_reproduces": bool(seq[0] > seq[1] > seq[2]),
                "wan_trained": w[0], "wan_random": w[1], "wan_nondit": w[2],
                "wan_random_gt_nondit": bool(w[1] > w[2]),
                "wan_full_ordering": bool(w[0] > w[1] > w[2]),
                "same_pattern_as_wan": bool((seq[0] > seq[1] > seq[2])
                                            == (w[0] > w[1] > w[2])),
            }
        verdicts["backbones"][bb] = entry

    outp = OUT / f"E8_verdict_{args.dataset}.json"
    outp.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))

    # ---- console summary ---------------------------------------------------------------
    for bb, e in verdicts["backbones"].items():
        print(f"\n=== {e['label']} ({e['params_b']}B, {e['n_layers']} layers) / {args.dataset} ===")
        print(f"{'cell':12s} {'real':>8s} {'rand':>8s} {'D1%':>7s}  verdict")
        for k, c in e["cells"].items():
            if "d1_pct" in c and c["d1_pct"] is not None:
                print(f"{k:12s} {c['real_auc']:8.4f} {c['random_auc']:8.4f} "
                      f"{c['d1_pct']:7.1f}  {c['verdict']}  ({c['interpretable_because']})")
            elif k.endswith("|d2"):
                print(f"{k:12s} phys {c['physics_auc']:.4f} app {c['appearance_auc']:.4f}"
                      f"  -> D2 {c['verdict']}")
            elif k.startswith("lb|"):
                print(f"{k:12s} AUC {c['auc_mean']:.4f}")
        oc = e.get("ordering_check")
        if oc:
            print(f"  ordering: trained {oc['trained']:.4f} / random {oc['random']:.4f} / "
                  f"non-DiT {oc['non_dit']:.4f}")
            print(f"            full ordering here = {oc['full_ordering_reproduces']}, "
                  f"on Wan = {oc['wan_full_ordering']} "
                  f"(Wan random {oc['wan_random']:.4f} vs non-DiT {oc['wan_nondit']:.4f}) -> "
                  f"same pattern = {oc['same_pattern_as_wan']}")
    print(f"\nWan reference: D1 {ref['d1_pct']}% ({ref['d1_verdict']}), "
          f"D2 {ref['d2_verdict']}, real {ref['real']:.4f} / random {ref['random']:.4f}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
