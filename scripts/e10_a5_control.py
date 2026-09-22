"""Control for the A.5 motion measurement: run the identical estimator on scenarios whose
camera is static.

`shadow_camera` is the only LikePhys scenario documented as having camera motion.  If the
same optical-flow + affine estimator reports a comparable per-frame translation on
static-camera scenarios, then the estimator is tracking *content* motion (moving shadows,
falling balls, cloth) rather than the camera, and the A.5 numbers are not usable as a
camera-motion ruler.

This control has to pass before the A.5 strengths can be frozen.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_a5_measure import N_FRAMES, SIZE, per_frame_motion, read_frames, detrend_std

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E10_shadow_camera"
SCENARIOS = ["shadow_camera", "shadow", "pendulum", "ball_drop", "block_slide",
             "cloth_drape", "fluid", "flag"]
N_CLIPS = 5


def main():
    manifest = json.loads((ROOT / "results/T3_null_floor/manifest.json").read_text())
    table = {}
    for scen in SCENARIOS:
        clips = [c for c in manifest
                 if c["scenario"] == scen and c["kind"] == "valid"][:N_CLIPS]
        meds, excs, inls, jit = [], [], [], []
        for c in clips:
            f = read_frames(c["path"], N_FRAMES, SIZE)
            tr, ro, ok, dx, dy, inl = per_frame_motion(f)
            tx = np.nancumsum(np.nan_to_num(dx))
            ty = np.nancumsum(np.nan_to_num(dy))
            meds.append(float(np.nanmedian(tr)))
            excs.append(float(np.hypot(tx, ty).max()))
            inls.append(float(np.nanmedian(inl)))
            jit.append(detrend_std(tr))
        table[scen] = {
            "n_clips": len(clips),
            "trans_px_per_frame_median": round(float(np.median(meds)), 3),
            "trans_px_per_frame_range": [round(min(meds), 3), round(max(meds), 3)],
            "max_excursion_px_median": round(float(np.median(excs)), 2),
            "jitter_px_std_median": round(float(np.median(jit)), 3),
            "ransac_inlier_median": round(float(np.median(inls)), 3),
        }
        t = table[scen]
        print(f"  {scen:15s} trans/f {t['trans_px_per_frame_median']:6.2f} "
              f"{str(t['trans_px_per_frame_range']):18s} "
              f"excursion {t['max_excursion_px_median']:7.1f}px  "
              f"jitter {t['jitter_px_std_median']:5.2f}  "
              f"inlier {t['ransac_inlier_median']:.2f}", flush=True)

    anomaly = table["shadow_camera"]["trans_px_per_frame_median"]
    others = {k: v["trans_px_per_frame_median"] for k, v in table.items()
              if k != "shadow_camera"}
    worst_name = max(others, key=others.get)
    worst = others[worst_name]
    ratio_median = anomaly / float(np.median(list(others.values())))
    ratio_worst = anomaly / worst if worst else float("inf")

    # The ruler is only usable if shadow_camera stands clearly ABOVE EVERY static-camera
    # scenario.  Comparing against the median hides a single confounded scenario, which is
    # exactly the failure mode that matters: if any static scene reaches the same apparent
    # motion, then "1x = real camera motion" is not distinguishable from "1x = a scene with
    # fast-moving content".
    separates = bool(ratio_worst > 1.5)
    verdict = {
        "shadow_camera_trans_px_per_frame": anomaly,
        "static_scenarios": others,
        "static_median": round(float(np.median(list(others.values()))), 3),
        "worst_static_scenario": worst_name,
        "worst_static_value": worst,
        "ratio_vs_static_median": round(ratio_median, 2),
        "ratio_vs_worst_static": round(ratio_worst, 2),
        "estimator_is_camera_specific": separates,
        "conclusion": (
            "estimator cleanly separates shadow_camera from every static-camera scenario; "
            "the A.5 numbers can be read as camera motion"
            if separates else
            f"NOT USABLE as a camera-motion ruler: the static-camera scenario "
            f"'{worst_name}' reports {worst} px/frame vs shadow_camera's {anomaly}, i.e. "
            f"the estimator is dominated by CONTENT motion in at least one scenario. "
            f"A.5 strengths must not be frozen from this measurement."),
    }
    print("\n" + json.dumps(verdict, indent=2))
    p = OUT / "A5_motion_control.json"
    p.write_text(json.dumps({"per_scenario": table, "verdict": verdict}, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
