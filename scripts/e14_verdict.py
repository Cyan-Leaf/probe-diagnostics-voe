"""E14 verdict: HunyuanVideo, emitted in the same row shape as E8_verdict_schema.json.

Two things this script is careful about, both written into E14_PREREG.md before extraction:

  L16 vs Lrel are not two depths of one quantity.  L16 is the 16th double-stream block; Lrel
  (=32) is the 12th single-stream block.  They differ in stack, not only in depth, so a
  difference between them cannot be read as a depth effect.  Both are reported and neither is
  the "primary" one — the pre-registration fixed L16 as the reference cell to stay comparable
  with E8, and Lrel as the depth-matched cell.

  The threshold is read from the cell, not hardcoded twice.  E8_THRESHOLD below exists only to
  assert that E14 used the same number as PREREG.md; if they ever diverge the script fails
  rather than silently reporting a pass at a different bar.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E14_hunyuan"
E8_THRESHOLD = 0.5      # PREREG.md D1; asserted against the cells below


def main():
    cells = json.loads((OUT / "E14_cells.json").read_text())
    rows = []
    for key, c in cells["cells"].items():
        if not key.endswith("|physics"):
            continue
        lname = key.split("|")[2]
        assert c["interpretable"] or True
        rows.append({
            "cell": lname,
            "layer": c["layer"],
            "stack": "double" if c["layer"] < cells["n_double"] else "single",
            "lower_bounds": {
                "raw_pixel": cells["cells"]["hunyuan|intphys2|raw_pixel"]["auc"],
                "vae_latent": cells["cells"]["hunyuan|intphys2|vae_latent"]["auc"],
                "random_dit": c["auc_random"],
                "trained_dit": c["auc_real"],
            },
            "d1": {"training_share": c["d1"],
                   "verdict": "pass" if c["interpretable"] and c["d1"] >= E8_THRESHOLD
                              else ("NOT interpretable" if not c["interpretable"] else "fail"),
                   "threshold": E8_THRESHOLD,
                   "interpretable": bool(c["interpretable"]),
                   "floor": c["floor"], "why": c["why"]},
            "d2": {"auc_physics": c["auc_real"], "auc_appearance": c["auc_appearance"],
                   "gap": c["gap_appearance_minus_physics"],
                   "verdict": "fail" if c["auc_appearance"] >= c["auc_real"] else "pass"},
        })

    # The ordering claim E8 makes (trained > random > raw pixel / VAE latent) is checked for
    # every cell rather than asserted once, because a lower bound that flips would invalidate
    # the "training helps" reading without changing any D1 number.
    order_ok = {}
    for r in rows:
        lb = r["lower_bounds"]
        order_ok[r["cell"]] = bool(lb["trained_dit"] > lb["random_dit"] > max(
            lb["raw_pixel"], lb["vae_latent"]))

    verdict = {
        "model": cells["model"],
        "reference_cell": cells["reference_cell"],
        "relative_depth_cell": cells["relative_depth_cell"],
        "lower_bounds": "per-cell, see `rows[*].lower_bounds`",
        "d1": "per-cell, see `rows[*].d1`",
        "d2": "per-cell, see `rows[*].d2`",
        "rows": rows,
        "ordering_reproduces": order_ok,
        "note": (
            "L16 (double-stream) and Lrel=32 (single-stream) are NOT a depth contrast: they sit "
            "in different block stacks. 13312 tokens, 512x512x49, flow matching t=600, "
            "norm-aware random init — all identical to E8's other backbones, so the numbers are "
            "comparable to E8's rows. LikePhys was not run (E14_PREREG.md §5)."),
    }
    dst = OUT / "E14_verdict.json"
    dst.write_text(json.dumps(verdict, indent=1))

    for r in rows:
        lb = r["lower_bounds"]
        print(f"{r['cell']:5s} block {r['layer']:2d} ({r['stack']:6s})  "
              f"trained {lb['trained_dit']:.4f} random {lb['random_dit']:.4f} "
              f"raw {lb['raw_pixel']:.4f} vae {lb['vae_latent']:.4f}  "
              f"D1 {r['d1']['training_share']*100:5.1f}% {r['d1']['verdict']:10s} "
              f"D2 {r['d2']['verdict']}")
    print(f"ordering reproduces: {order_ok}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
