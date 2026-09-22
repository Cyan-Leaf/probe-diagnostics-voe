"""PREREG appendix A.5 steps M1-M3: measure the *actual* camera motion in `shadow_camera`.

Why this exists (PREREG.md §A.5): H4 claims that scenes with camera motion are the only
honest test of a physics probe.  Its sole empirical anchor is `shadow_camera`, the one
LikePhys scenario with a moving camera.  If the three synthetic-motion strengths used in
S4 are not calibrated against that scenario's real motion, then "margin falls with motion
strength" only shows "the picture got scrambled" and cannot support H4.

  M1  per-frame global motion of the 10 `shadow_camera` valid clips (first 49 frames)
      via cv2.estimateAffinePartial2D on tracked corners
  M2  aggregate to median / p90, separating steady translation from high-frequency
      jitter (jitter = std of the de-trended residual)
  M3  strength 2 (1x) = M2 median; strength 1 = 0.5x; strength 3 = 2x, then validate
      against the two geometric rails in §A.1:
        - strength 1 cumulative displacement must exceed 32 px (one token), else
          rescale to 1x/2x/4x
        - strength 3 must keep frame overlap >= ~70%

Pure CPU, no GPU, reads the mp4s directly.  Writes results/E10_shadow_camera/A5_motion.json.
The numbers still have to be pasted into PREREG.md §A.5 and frozen before S4 runs.

Run: python scripts/e10_a5_measure.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E10_shadow_camera"
N_FRAMES = 49
SIZE = 512
TOKEN_PX = 32


def read_frames(path: str, n: int, size: int):
    """Decode with decord (project-wide choice, see PITFALLS #14) and resize to 512."""
    import decord

    vr = decord.VideoReader(path, width=size, height=size)
    idx = list(range(min(n, len(vr))))
    return vr.get_batch(idx).asnumpy()  # [F, H, W, 3] uint8


def per_frame_motion(frames):
    """Estimate frame-to-frame similarity transform.

    Returns per-frame translation magnitude, rotation, the *signed* dx/dy (needed to
    tell a steady pan apart from an oscillation: summing magnitudes gives path length,
    which for a back-and-forth camera vastly overstates how far the content actually
    travels) and the RANSAC inlier ratio (low ratio => the estimate is being dragged by
    moving content rather than the camera, which matters here because shadow scenes have
    large moving shadows).
    """
    import cv2

    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    trans, rot, ok, dxs, dys, inl = [], [], [], [], [], []
    for a, b in zip(grey[:-1], grey[1:]):
        pts = cv2.goodFeaturesToTrack(a, maxCorners=600, qualityLevel=0.01,
                                      minDistance=8, blockSize=7)
        if pts is None or len(pts) < 12:
            trans.append(np.nan); rot.append(np.nan); dxs.append(np.nan)
            dys.append(np.nan); inl.append(np.nan); ok.append(False)
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None,
                                                  winSize=(21, 21), maxLevel=3)
        good_a = pts[status.ravel() == 1].reshape(-1, 2)
        good_b = nxt[status.ravel() == 1].reshape(-1, 2)
        if len(good_a) < 12:
            trans.append(np.nan); rot.append(np.nan); dxs.append(np.nan)
            dys.append(np.nan); inl.append(np.nan); ok.append(False)
            continue
        M, inliers = cv2.estimateAffinePartial2D(good_a, good_b, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0)
        if M is None:
            trans.append(np.nan); rot.append(np.nan); dxs.append(np.nan)
            dys.append(np.nan); inl.append(np.nan); ok.append(False)
            continue
        dx, dy = float(M[0, 2]), float(M[1, 2])
        dxs.append(dx)
        dys.append(dy)
        trans.append(float(np.hypot(dx, dy)))
        rot.append(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        inl.append(float(inliers.mean()) if inliers is not None else float("nan"))
        ok.append(True)
    return (np.array(trans), np.array(rot), np.array(ok),
            np.array(dxs), np.array(dys), np.array(inl))


def detrend_std(x):
    """Jitter = std of the residual after removing a linear trend (the steady pan)."""
    v = x[~np.isnan(x)]
    if v.size < 3:
        return float("nan")
    t = np.arange(v.size)
    coef = np.polyfit(t, v, 1)
    return float(np.std(v - np.polyval(coef, t), ddof=1))


def main() -> None:
    manifest = json.loads((ROOT / "results/T3_null_floor/manifest.json").read_text())
    clips = [c for c in manifest
             if c["scenario"] == "shadow_camera" and c["kind"] == "valid"]
    assert clips, "no shadow_camera valid clips in manifest"

    per_clip = []
    for c in clips:
        frames = read_frames(c["path"], N_FRAMES, SIZE)
        tr, ro, ok, dx, dy, inl = per_frame_motion(frames)
        # signed trajectory: where the content actually is relative to frame 0
        traj_x = np.nancumsum(np.nan_to_num(dx))
        traj_y = np.nancumsum(np.nan_to_num(dy))
        excursion = np.hypot(traj_x, traj_y)
        net_disp = float(excursion[-1])
        max_exc = float(excursion.max())
        path_len = float(np.nansum(tr))
        per_clip.append({
            "clip_id": c["clip_id"],
            "n_frames": int(len(frames)),
            "n_pairs_ok": int(ok.sum()),
            "ransac_inlier_ratio_median": float(np.nanmedian(inl)),
            "trans_px_per_frame_median": float(np.nanmedian(tr)),
            "trans_px_per_frame_p90": float(np.nanpercentile(tr, 90)),
            "trans_px_per_frame_max": float(np.nanmax(tr)),
            "rot_deg_per_frame_median": float(np.nanmedian(np.abs(ro))),
            "rot_deg_per_frame_p90": float(np.nanpercentile(np.abs(ro), 90)),
            "jitter_px_std": detrend_std(tr),
            "path_length_px": path_len,
            "net_displacement_px": net_disp,
            "max_excursion_px": max_exc,
            "oscillation_ratio": float(path_len / max_exc) if max_exc > 1e-6 else float("inf"),
        })
        p = per_clip[-1]
        print(f"  {c['clip_id'][-34:]:34s} med {p['trans_px_per_frame_median']:6.2f} "
              f"inlier {p['ransac_inlier_ratio_median']:.2f}  "
              f"path {p['path_length_px']:7.1f}  net {p['net_displacement_px']:6.1f}  "
              f"maxexc {p['max_excursion_px']:6.1f}  osc {p['oscillation_ratio']:5.1f}x",
              flush=True)

    def agg(key):
        v = np.array([p[key] for p in per_clip], dtype=float)
        v = v[np.isfinite(v)]
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "min": float(v.min()), "max": float(v.max())}

    m2 = {
        "trans_px_per_frame": agg("trans_px_per_frame_median"),
        "rot_deg_per_frame": agg("rot_deg_per_frame_median"),
        "jitter_px_std": agg("jitter_px_std"),
        "path_length_px": agg("path_length_px"),
        "net_displacement_px": agg("net_displacement_px"),
        "max_excursion_px": agg("max_excursion_px"),
        "oscillation_ratio": agg("oscillation_ratio"),
        "ransac_inlier_ratio": agg("ransac_inlier_ratio_median"),
    }

    # M3: 1x = the measured median, then 0.5x / 1x / 2x
    base_t = m2["trans_px_per_frame"]["median"]
    base_r = m2["rot_deg_per_frame"]["median"]
    base_j = m2["jitter_px_std"]["median"]
    base_exc = m2["max_excursion_px"]["median"]

    def build(scales):
        return {f"level_{i+1}": {
            "scale": s,
            "trans_px_per_frame": round(base_t * s, 4),
            "rot_deg_per_frame": round(base_r * s, 4),
            "jitter_px_std": round(base_j * s, 4),
            "path_length_px": round(base_t * s * (N_FRAMES - 1), 2),
            "max_excursion_px": round(base_exc * s, 2),
        } for i, s in enumerate(scales)}

    scales = [0.5, 1.0, 2.0]
    levels = build(scales)

    # §A.1 rail 1: level 1 must move content by more than one token (32 px) to be visible
    # at all.  Judged on max excursion, not path length: a camera that jitters back and
    # forth covers a long path while the content stays put.
    lvl1_exc = levels["level_1"]["max_excursion_px"]
    rail1_ok = lvl1_exc > TOKEN_PX
    if not rail1_ok:
        scales = [1.0, 2.0, 4.0]
        levels = build(scales)
        lvl1_exc = levels["level_1"]["max_excursion_px"]
        rail1_ok = lvl1_exc > TOKEN_PX

    # §A.1 rail 2: the top level must still keep >= ~70% frame overlap.  Overlap is set
    # by how far the content is displaced at its furthest point, i.e. max excursion.
    top_exc = levels["level_3"]["max_excursion_px"]
    overlap = max(0.0, (SIZE - top_exc) / SIZE)
    rail2_ok = overlap >= 0.70
    # what per-frame rate would rail 2 actually allow?
    max_exc_allowed = SIZE * 0.30
    rate_cap = max_exc_allowed / (base_exc / base_t) if base_t > 0 else float("nan")

    out = {
        "source": "shadow_camera valid clips, first 49 frames, resized to 512x512",
        "method": ("goodFeaturesToTrack + calcOpticalFlowPyrLK + "
                   "estimateAffinePartial2D(RANSAC); jitter = std of de-trended "
                   "per-frame translation"),
        "M1_per_clip": per_clip,
        "M2_aggregate": m2,
        "M3_levels": levels,
        "M3_rails": {
            "rail1_level1_excursion_gt_32px": {
                "value_px": levels["level_1"]["max_excursion_px"],
                "threshold_px": TOKEN_PX, "passed": bool(rail1_ok),
                "action": ("kept 0.5x/1x/2x" if scales[0] == 0.5 else
                           "rescaled to 1x/2x/4x per §A.5 M3"),
            },
            "rail2_level3_overlap_ge_70pct": {
                "max_excursion_px": top_exc,
                "overlap_fraction": round(overlap, 4),
                "threshold": 0.70, "passed": bool(rail2_ok),
                "rate_cap_px_per_frame_implied_by_rail2": round(rate_cap, 3),
                "measured_rate_px_per_frame": round(base_t, 3),
                "measured_over_cap": (round(base_t / rate_cap, 2)
                                      if rate_cap and np.isfinite(rate_cap) else None),
            },
        },
        "conflict": (
            None if rail2_ok else
            "PREREG A.5 M3 (calibrate to shadow_camera's real motion) and A.1 rail 2 "
            "(keep >=70% frame overlap) cannot both be satisfied: the measured motion "
            f"is {round(base_t,2)} px/frame but rail 2 caps it at "
            f"{round(rate_cap,2)} px/frame. Owner decision required before S4."),
        "status": ("MEASURED -- paste M3_levels into PREREG.md §A.5 and freeze before S4"
                   if rail2_ok else
                   "MEASURED BUT RAILS CONFLICT -- do not freeze A.5 yet, see 'conflict'"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "A5_motion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== M2 aggregate ===")
    print(json.dumps(m2, indent=2))
    print("\n=== M3 levels ===")
    print(json.dumps(levels, indent=2))
    print("\n=== rails ===")
    print(json.dumps(out["M3_rails"], indent=2))
    print(f"\nwrote {OUT/'A5_motion.json'}")


if __name__ == "__main__":
    main()
