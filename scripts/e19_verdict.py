"""E19 verdict: does the D1/D2 direction survive 14B -> 33B, a new text encoder, and a new VAE?

`E19_PREREG.md` §3 fixes three questions and their thresholds before any number existed:

  Q1  D1 > 50%?              -> `pass` / `fail` / `NOT interpretable` (T10 gate first)
  Q2  D2 fail?               -> fail when the appearance arm beats the physics arm
  Q3  lower-bound order      -> does "trained > random > pixel" reproduce on 33B?

Two disciplines carried over from earlier tasks, both of which the report restates:

  * **A cell that fails the T10 gate reports no D1 at all** -- only its raw AUC. E18 showed what
    happens otherwise: a collapsed denominator manufactures a large D1 out of nothing.
  * **`AUC_real` is reported beside every D1** (E8 §15.2): below 0.6 the ratio is not comparable
    across cells, so a bare D1 there would invite exactly the wrong reading.

Nothing is recomputed here except the presentations of numbers already in `E19_cells.json`; the
thresholds are read from `E19_PREREG.md`'s constants, which are copied from `PREREG.md`.

Run: .venv_h3/bin/python scripts/e19_verdict.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E19_h3_scale"

D1_THRESHOLD = 0.5
WEAK_SIGNAL_AUC = 0.6      # E8 §15.2


def main():
    cells = json.loads((OUT / "E19_cells.json").read_text())
    q1, q2, q3 = {}, {}, {}

    for name, c in cells["cells"].items():
        ok = bool(c["interpretable"]) and c["d1"] >= D1_THRESHOLD
        q1[name] = {
            "d1_pct": c["d1_pct"], "auc_real": c["auc_real"], "auc_random": c["auc_random"],
            "interpretable": bool(c["interpretable"]),
            "verdict": ("pass" if ok else "fail") if c["interpretable"] else "NOT interpretable",
            "why": c["why"],
            "weak_signal": bool(c["auc_real"] < WEAK_SIGNAL_AUC),
            "comparable_across_cells": bool(c["auc_real"] >= WEAK_SIGNAL_AUC),
        }
        q2[name] = {
            "auc_physics": c["auc_real"], "auc_appearance": c["auc_appearance"],
            "gap": c["gap_appearance_minus_physics"],
            "verdict": "fail" if c["auc_appearance"] >= c["auc_real"] else "pass",
        }

    lb = cells["lower_bounds"]
    CANON = ["trained_dit", "raw_pixel", "vae_latent", "random_dit"]
    order = sorted(lb, key=lambda k: -lb[k])
    q3 = {
        "values": lb,
        "order_by_auc": order,
        "canonical_order": CANON,
        "canonical_source": (
            "Read from results/E8_second_backbone/E8_cells_intphys2.json, not from the task "
            "sheet. Across E8's six models: **trained beats raw_pixel 6/6 and beats vae_latent "
            "6/6**, while the vae-vs-random order flips by model (cogvideox2b/5b and "
            "wan21_14b/1b3 put vae above random, ltx and mochi put random above vae). So the "
            "load-bearing claim is 'trained beats both non-DiT arms 6/6'; the three-way order "
            "the task sheet sketched ('trained > random > pixel') is NOT what E8 measured -- "
            "random_dit never clears raw_pixel in any of the six. E19 is judged on the two "
            "claims E8 actually established."),
        "order_reproduces": order == CANON,
        "two_way_claims": {
            "reference": "E8 achieved 6/6 on both",
            "trained_beats_both_non_dit_arms": bool(
                lb["trained_dit"] > lb["raw_pixel"] and lb["trained_dit"] > lb["vae_latent"]),
            "trained_above_random": bool(lb["trained_dit"] > lb["random_dit"]),
            "trained_above_pixel": bool(lb["trained_dit"] > lb["raw_pixel"]),
            "trained_above_vae": bool(lb["trained_dit"] > lb["vae_latent"]),
        },
        "note": ("PREREG-equivalent judgement is the ORDER, not the values: each backbone is "
                 "measured on its own features and absolute AUC is not comparable across "
                 "models (E8 §14). A break in the order is a finding, not noise."),
    }

    verdict = {
        "model": cells["model"],
        "reference_cell": cells["reference_cell"],
        "frame_geometry": cells["frame_geometry"],
        "threads": cells.get("threads"),
        "d1_threshold": D1_THRESHOLD,
        "weak_signal_auc": WEAK_SIGNAL_AUC,
        "q1_d1": q1,
        "q2_d2": q2,
        "q3_lower_bounds": q3,
        "summary": {
            "d1_pass": sum(1 for v in q1.values() if v["verdict"] == "pass"),
            "d1_fail": sum(1 for v in q1.values() if v["verdict"] == "fail"),
            "d1_not_interpretable": sum(1 for v in q1.values()
                                        if v["verdict"] == "NOT interpretable"),
            "d2_fail": sum(1 for v in q2.values() if v["verdict"] == "fail"),
            "d2_pass": sum(1 for v in q2.values() if v["verdict"] == "pass"),
            "order_reproduces": q3["order_reproduces"],
        },
        "not_done": [
            "LikePhys: not run (consistent with E17/E18, IntPhys 2 first)",
            "D3-D6: explicitly out of scope (NEXT_STEPS_AND_DIVISION.md forbids all six)",
            "t=298 SNR-matched cell: a single-model cell would break comparability",
            "audio rows: passed to forward at the right length and tag, never read back",
        ],
        "carried_forward": [
            "E18's cross-sigma confound does NOT apply here: every E19 cell is at t=600 and no "
            "cross-timestep comparison is made",
            "E16's threads rule: all probe calls above ran at threads=%s" % cells.get("threads"),
        ],
    }

    p = OUT / "E19_verdict.json"
    p.write_text(json.dumps(verdict, indent=1))

    for name, v in q1.items():
        w = " (weak-signal: D1 not comparable)" if v["weak_signal"] else ""
        print(f"{name:10s} real {v['auc_real']:.4f} rand {v['auc_random']:.4f} "
              f"D1 {v['d1_pct']:6.2f}%  {v['verdict']}{w}")
    for name, v in q2.items():
        print(f"{name:10s} D2 physics {v['auc_physics']:.4f} appearance {v['auc_appearance']:.4f} "
              f"gap {v['gap']:+.4f}  {v['verdict']}")
    print(f"lower bounds: {q3['order_by_auc']}  order reproduces: {q3['order_reproduces']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
