"""E17 verdict: Q1/Q2/Q3 from the two-expert cells, per E17_PREREG.md §3.

The one thing that must not go wrong here: the two experts are NOT interchangeable rows. The
low-noise expert (`transformer_2`) is the in-range one at t=600 (sigma 0.600 < boundary 0.875);
the high-noise expert (`transformer`) is being run outside the noise band it was trained on.
Every number from the high-noise expert is therefore labelled, in the JSON itself, not just in
the prose — a reader who opens only the verdict must still see it.

Q2's threshold is the probe-partition sd measured by E16 (median 2.51pp on IntPhys 2), not a
round number. Using anything smaller would treat E16's own noise as signal.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E17_moe_experts"
E16 = ROOT / "results" / "E16_probe_seed" / "E16_cells.json"
D1_THRESHOLD = 0.5


def probe_sd_median() -> float:
    """E16's IntPhys 2 probe-partition sd median, in percentage points."""
    cells = json.loads(E16.read_text())["summary"]
    vals = [v["d1_sd_pct"] for k, v in cells.items()
            if "intphys2" in k and not v["verdict_flips"]]
    vals.sort()
    n = len(vals)
    return float(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)


def main():
    cells = json.loads((OUT / "E17_cells.json").read_text())["cells"]
    sd = probe_sd_median()

    lo16 = cells["a14b|low|L16"]
    lo21 = cells["a14b|low|Lrel"]
    hi16 = cells["a14b|high|L16"]
    hi21 = cells["a14b|high|Lrel"]

    # ---- Q1: both experts >= 50%? -------------------------------------------------------
    lo_pass = lo16["interpretable"] and lo16["d1"] >= D1_THRESHOLD
    hi_pass = hi16["interpretable"] and hi16["d1"] >= D1_THRESHOLD
    if lo_pass and hi_pass:
        q1 = "both_above_threshold"
    elif not lo_pass and not hi_pass:
        q1 = "both_below_threshold"
    else:
        q1 = "mixed"

    # ---- Q2: is the expert-to-expert difference bigger than probe noise? ----------------
    deltas = {lname: cells[f"a14b|low|{lname}"]["d1_pct"] - cells[f"a14b|high|{lname}"]["d1_pct"]
              for lname in ("L16", "Lrel")}
    mean_delta = sum(deltas.values()) / len(deltas)
    q2 = "exceeds_noise" if abs(mean_delta) >= sd else "within_noise"

    verdict = {
        "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers@5be7df9619b5",
        "reference_cell": {"layer": 16, "t": 600},
        "experts": {
            "low": {"subfolder": "transformer_2", "in_range_at_t600": True},
            "high": {"subfolder": "transformer", "in_range_at_t600": False,
                     "caveat": "E17_PREREG.md §4.1: the high-noise expert only runs above "
                               "sigma 0.875; at t=600 (sigma 0.600) it is OUT OF RANGE, so its "
                               "AUC describes 'a 14B denoiser trained on another noise band', "
                               "not this expert's competence in its own regime"},
        },
        "threshold": D1_THRESHOLD,
        "cells": {
            "low_expert_L16": {k: lo16[k] for k in
                               ("auc_real", "auc_random", "auc_appearance", "d1", "d1_pct",
                                "floor", "interpretable", "n", "in_range_at_t600")
                               if k in lo16} | {"in_range_at_t600": True},
            "low_expert_Lrel": {"layer": lo21["layer"], "d1_pct": lo21["d1_pct"],
                                "auc_real": lo21["auc_real"], "interpretable": lo21["interpretable"]},
            "high_expert_L16": {"auc_real": hi16["auc_real"], "auc_random": hi16["auc_random"],
                                "auc_appearance": hi16["auc_appearance"],
                                "d1": hi16["d1"], "d1_pct": hi16["d1_pct"],
                                "floor": hi16["floor"], "interpretable": hi16["interpretable"],
                                "in_range_at_t600": False},
            "high_expert_Lrel": {"layer": hi21["layer"], "d1_pct": hi21["d1_pct"],
                                 "auc_real": hi21["auc_real"],
                                 "interpretable": hi21["interpretable"],
                                 "in_range_at_t600": False},
        },
        "q1": {"verdict": q1, "low_pass": bool(lo_pass), "high_pass": bool(hi_pass),
               "note": "both experts are trained 14B denoisers on the same data; a split "
                       "verdict would mean the D1 call depends on which noise band the expert "
                       "specialises in"},
        "q2": {"delta_low_minus_high_L16": deltas["L16"],
               "delta_low_minus_high_Lrel": deltas["Lrel"],
               "mean_delta": mean_delta,
               "probe_sd_threshold_pp": sd,
               "threshold_source": "E16 probe-partition sd, median over IntPhys2 cells",
               "verdict": q2,
               "note": "|delta| < threshold => the expert-to-expert difference is inside the "
                       "noise this probe already has; the model cannot be used to argue that "
                       "noise-band specialisation changes D1"},
        "q3": {
            "low_expert_d2": "fail" if lo16["auc_appearance"] >= lo16["auc_real"] else "pass",
            "high_expert_d2": "fail" if hi16["auc_appearance"] >= hi16["auc_real"] else "pass",
            "both_low_expert_layers_fail": bool(
                cells["a14b|low|Lrel"]["auc_appearance"] >= cells["a14b|low|Lrel"]["auc_real"]),
        },
        "note": ("Q2's delta mixes two things that this design cannot separate: how much each "
                 "expert specialises in its noise band, and how well either generalises to a "
                 "band it did not train on (E17_PREREG.md §4.2). Reported as a bound, not as "
                 "an attribution."),
    }

    (OUT / "E17_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(f"probe-noise threshold (E16 median sd): {sd:.2f} pp")
    print(f"low  expert L16 D1 {lo16['d1_pct']:.1f}%  "
          f"{'pass' if lo_pass else 'fail/not-interp'}   (in range at t=600)")
    print(f"high expert L16 D1 {hi16['d1_pct']:.1f}%  "
          f"{'pass' if hi_pass else 'fail/not-interp'}   (OUT of range at t=600)")
    print(f"Q1: {q1}")
    print(f"Q2: {q2}  (delta {mean_delta:+.2f} pp vs threshold {sd:.2f})")
    print(f"Q3: low {verdict['q3']['low_expert_d2']}, high {verdict['q3']['high_expert_d2']}")
    print(f"wrote {OUT / 'E17_verdict.json'}")


if __name__ == "__main__":
    main()
