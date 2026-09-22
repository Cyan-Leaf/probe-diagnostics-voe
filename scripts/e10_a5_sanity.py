"""Independent sanity check on the A.5 motion measurement.

The A.5 result (median 12.7 px/frame, net excursion 479 px over 49 frames) implies the
`shadow_camera` camera sweeps roughly a full frame width during a clip.  That conclusion
blocks S4, so verify it without optical flow: if the camera really moves that far, frame 0
and frame 48 should share almost no content, while a static-camera scenario's first and
last frames should still overlap heavily.

Compares `shadow_camera` against `shadow` (same scene family, static camera) as a control.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 49
SIZE = 512


def frames(path):
    import decord
    vr = decord.VideoReader(path, width=SIZE, height=SIZE)
    return vr.get_batch(list(range(min(N, len(vr))))).asnumpy().astype(np.float32) / 255.0


def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-9 else 0.0


def best_shift_ncc(a, b, max_shift=240, step=8):
    """Crude search for the horizontal shift that best re-aligns frame 0 onto frame 48."""
    best = (-2.0, 0)
    h, w, _ = a.shape
    for s in range(-max_shift, max_shift + 1, step):
        if s >= 0:
            aa, bb = a[:, s:, :], b[:, : w - s, :]
        else:
            aa, bb = a[:, : w + s, :], b[:, -s:, :]
        if aa.shape[1] < 64:
            continue
        v = ncc(aa, bb)
        if v > best[0]:
            best = (v, s)
    return best


def main():
    manifest = json.loads((ROOT / "results/T3_null_floor/manifest.json").read_text())
    out = {}
    for scen in ("shadow_camera", "shadow"):
        clips = [c for c in manifest if c["scenario"] == scen and c["kind"] == "valid"][:4]
        rows = []
        for c in clips:
            f = frames(c["path"])
            first, last = f[0], f[min(N, len(f)) - 1]
            raw = ncc(first, last)
            aligned, shift = best_shift_ncc(first, last)
            rows.append({"clip_id": c["clip_id"],
                         "ncc_frame0_vs_frame48_unaligned": round(raw, 4),
                         "ncc_after_best_horizontal_shift": round(aligned, 4),
                         "best_shift_px": shift,
                         "ncc_frame0_vs_frame1": round(ncc(f[0], f[1]), 4)})
            print(f"  {scen:14s} {c['clip_id'][-28:]:28s} "
                  f"ncc(0,48)={raw:6.3f}  best_shift={shift:+5d}px -> {aligned:6.3f}  "
                  f"ncc(0,1)={rows[-1]['ncc_frame0_vs_frame1']:6.3f}", flush=True)
        out[scen] = rows

    sc = np.mean([r["ncc_frame0_vs_frame48_unaligned"] for r in out["shadow_camera"]])
    st = np.mean([r["ncc_frame0_vs_frame48_unaligned"] for r in out["shadow"]])
    shifts = [abs(r["best_shift_px"]) for r in out["shadow_camera"]]
    out["summary"] = {
        "mean_ncc_0_48_shadow_camera": round(float(sc), 4),
        "mean_ncc_0_48_shadow_static": round(float(st), 4),
        "mean_abs_best_shift_shadow_camera_px": float(np.mean(shifts)),
        "interpretation": (
            "If shadow_camera's first/last frames decorrelate strongly while the static "
            "`shadow` control stays correlated, the large measured camera motion is real "
            "and not an optical-flow artefact."),
    }
    print("\n" + json.dumps(out["summary"], indent=2))
    p = ROOT / "results/E10_shadow_camera/A5_motion_sanity.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
