"""E13 verdict: Q1/Q2 from the cells, per E13_PREREG.md §5.

Kept separate from the analysis so the JSON that gets quoted is produced by a script from
another JSON — the same rule every other task in this repo follows (never hand-edit a
deliverable).  The only inputs are `E13_cells.json` and the thresholds written into the
pre-registration before any extraction ran.

Q1 asks whether "training contributes only a little" survives on a non-generative encoder.
The comparison it needs is <b>V-JEPA 2 on clean video</b> against <b>CogVideoX-2b on clean
latents (t=0)</b> — the only pair in this project that differs in training objective while
holding "clean input" fixed.  E8's own D1 numbers come from noised latents and are NOT
part of this comparison (E13_PREREG.md §3.1).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E13_non_generative"
D1_THRESHOLD = 0.5          # frozen by PREREG.md; must match E8
TOL = 0.05                  # Q2's "same ballpark" band, fixed in E13_PREREG.md §5


def main():
    cells = json.loads((OUT / "E13_cells.json").read_text())["cells"]

    # ---- pick the pre-registered cells: L16 reference, flat caliber --------------------
    vj = cells["vjepa|intphys2|L16|flat|physics"]
    vj_alt = cells["vjepa|intphys2|L13|flat|physics"]
    vj_tavg = cells["vjepa|intphys2|L16|timeavg|physics"]
    cog = cells["cogvideox2b|intphys2|L16|flat|t0"]
    raw = cells["vjepa|intphys2|raw_pixel"]

    # ---- Q1 ---------------------------------------------------------------------------
    d1_v = vj["d1"]
    d1_c = cog["d1"]
    both_low = d1_v < D1_THRESHOLD and d1_c < D1_THRESHOLD
    both_high = d1_v >= D1_THRESHOLD and d1_c >= D1_THRESHOLD
    # Labels name the two cells' values, not a judgement about E8, because "consistent" was
    # ambiguous: it could mean "consistent with E8" or "the two cells agree with each other".
    if both_low:
        q1_verdict = "both_below_threshold"
        q1_note = ("non-generative encoder and clean-latent diffusion both put < 50% of the "
                   "readable signal on training")
    elif both_high:
        q1_verdict = "both_above_threshold"
        q1_note = ("both put >= 50% on training: on CLEAN input, training carries most of "
                   "the readable signal in both a generative and a non-generative encoder. "
                   "Whatever makes the share drop below 50% elsewhere, it is not "
                   "'generative training' and not 'clean input'")
    else:
        q1_verdict = "mixed"
        q1_note = ("the two disagree, which is itself the informative outcome: how much "
                   "training contributes depends on the training objective")

    # ---- Q2 ---------------------------------------------------------------------------
    # Same criterion form as E8: appearance AUC >= physics AUC means the probe cannot
    # separate physics from generic editing traces.
    gap = vj["auc_appearance"] - vj["auc_real"]
    q2_verdict = "fail" if gap >= 0 else "pass"

    verdict = {
        "model": "facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34d3255ef3547f27c7b226aefab26f",
        "reference_cell": {"layer": 16, "t": None,
                           "why_null": "non-generative: no noising step"},
        "geometry": {"height": 256, "width": 256, "frames": 64, "tokens": 8192},
        "lower_bounds": {
            "raw_pixel": raw["auc"],
            "random_vit": vj["auc_random"],
            "trained_vit": vj["auc_real"],
            "missing": "vae_latent — model has no VAE (E13_PREREG.md §3.3)",
        },
        "q1": {
            "d1_vjepa": d1_v, "d1_pct_vjepa": d1_v * 100,
            "d1_cogvideox_t0": d1_c, "d1_pct_cogvideox_t0": d1_c * 100,
            "threshold": D1_THRESHOLD,
            "verdict": q1_verdict, "note": q1_note,
        },
        "q2": {
            "auc_physics": vj["auc_real"], "auc_appearance": vj["auc_appearance"],
            "gap": gap, "verdict": q2_verdict,
            "threshold": "appearance >= physics -> fail (PREREG.md D2)",
        },
        "sensitivity": {
            "L13_flat": {"d1_pct": vj_alt["d1_pct"], "auc_real": vj_alt["auc_real"],
                         "verdict_same": (vj_alt["d1"] >= D1_THRESHOLD) ==
                                         (d1_v >= D1_THRESHOLD)},
            "L16_timeavg": {"d1_pct": vj_tavg["d1_pct"], "auc_real": vj_tavg["auc_real"],
                            "verdict_same": (vj_tavg["d1"] >= D1_THRESHOLD) ==
                                            (d1_v >= D1_THRESHOLD)},
        },
        "comparability": (
            "D1 is NOT numerically comparable to E8's D1: E8 reads a noised latent at t=600, "
            "this reads clean video, and the floor definition differs (permutation within "
            "scenes here, seed-spread there). D2 IS comparable — it depends only on the "
            "perturbation and the readout."),
        "floor_definition": "permutation within scenes (PREREG.md D4 definition B)",
        "note": (
            "Geometry is the model's own (256x256, 64 frames) because the position encoding "
            "is built for it; that is a smaller field of view than E8's 512, so AUC_real here "
            "is not the same quantity as E8's even before the noising difference."),
    }

    dst = OUT / "E13_verdict.json"
    dst.write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
