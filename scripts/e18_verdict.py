"""E18 verdict: does noise-band specialisation change D1 once each expert is at home?

E17_PREREG.md §... no: E18_PREREG.md §3 pre-registers one question and one threshold.

  Δ_own = D1(low @ t=300) − D1(high @ t=900)      <- both experts in their own band
  |Δ_own| <  probe-sd (E16 IntPhys2 median 2.51pp) -> experts_match_in_own_band
                                                      E17's null survives; Q2's confound is closed
  |Δ_own| >= probe-sd                              -> experts_differ_in_own_band
                                                      E17's null was an artefact of out-of-range use

Δ_cross (= E17's t=600 delta) is read from E17's delivered JSON and reported beside Δ_own.
If |Δ_own| > |Δ_cross| the home-band comparison is the MORE sensitive one, which is itself the
finding, so it is reported as a flag rather than buried.

Gate discipline (E18_PREREG.md §3, "附带问题"): a cell that fails the T10 gate reports no D1 --
only its raw AUC and `interpretable: false`. This is why the threshold check happens per cell
before any delta is computed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E18_own_band"
E17 = ROOT / "results" / "E17_moe_experts"
E16 = ROOT / "results" / "E16_probe_seed" / "E16_cells.json"

D1_THRESHOLD = 0.5
OWN_BAND = {"low": 300, "high": 900}


def probe_sd_median() -> float:
    """E16's IntPhys 2 probe-partition sd median, in percentage points (same source as E17)."""
    cells = json.loads(E16.read_text())["summary"]
    vals = sorted(v["d1_sd_pct"] for k, v in cells.items()
                  if "intphys2" in k and not v["verdict_flips"])
    n = len(vals)
    return float(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)


def main():
    cells = json.loads((OUT / "E18_cells.json").read_text())["cells"]
    e17 = json.loads((E17 / "E17_cells.json").read_text())["cells"]
    sd = probe_sd_median()

    per_layer = {}
    for lname in ("L16", "Lrel"):
        lo = cells[f"a14b|low@300|{lname}"]
        hi = cells[f"a14b|high@900|{lname}"]
        lo_ok = lo["interpretable"] and lo["d1"] >= D1_THRESHOLD
        hi_ok = hi["interpretable"] and hi["d1"] >= D1_THRESHOLD
        d_own = ((lo["d1_pct"] - hi["d1_pct"]) if (lo_ok and hi_ok) else None)

        # E17's cross comparison at t=600, read not recomputed.
        e17_lo = e17[f"a14b|low|{lname}"]
        e17_hi = e17[f"a14b|high|{lname}"]
        d_cross = e17_lo["d1_pct"] - e17_hi["d1_pct"]

        per_layer[lname] = {
            "d_own": d_own, "d_cross": d_cross,
            "low_own_band": {k: lo[k] for k in
                             ("t", "sigma", "auc_real", "auc_random", "auc_appearance",
                              "d1_pct", "floor", "interpretable", "why", "n")},
            "high_own_band": {k: hi[k] for k in
                              ("t", "sigma", "auc_real", "auc_random", "auc_appearance",
                               "d1_pct", "floor", "interpretable", "why", "n")},
            "both_above_threshold": bool(lo_ok and hi_ok),
        }

    valid = [v["d_own"] for v in per_layer.values() if v["d_own"] is not None]
    mean_own = sum(valid) / len(valid) if valid else None
    mean_cross = sum(v["d_cross"] for v in per_layer.values()) / len(per_layer)

    if mean_own is None:
        q = "undetermined_not_interpretable"
        why = ("at least one own-band cell fails the T10 gate, so E18_PREREG.md §3 forbids "
               "reporting D1 for it; the comparison cannot be made on this data")
    elif abs(mean_own) < sd:
        q = "experts_match_in_own_band"
        why = ("both experts read physics equally well in their own band; the expert-to-expert "
               "difference stays inside probe noise, so E17's null is not an out-of-range "
               "artefact and Q2's confound is closed")
    else:
        q = "experts_differ_in_own_band"
        why = ("the two experts differ once each is in its own band; E17's t=600 null was "
               "produced by measuring the high-noise expert out of range, and E17_REPORT §5 "
               "must be retracted")

    verdict = {
        "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers@5be7df9619b5",
        "design": "diagonal: low@t=300 (sigma 0.300) and high@t=900 (sigma 0.900)",
        "own_band": OWN_BAND,
        "boundary_ratio": 0.875,
        "threshold": D1_THRESHOLD,
        "probe_sd_threshold_pp": sd,
        "threshold_source": "E16 probe-partition sd, median over IntPhys2 cells (same as E17)",
        "q": {
            "verdict": q,
            "delta_own_pp": mean_own,
            "delta_cross_pp": mean_cross,
            "per_layer": per_layer,
            "note": why,
        },
        "home_band_more_sensitive": (None if mean_own is None
                                     else bool(abs(mean_own) > abs(mean_cross))),
        "d2": {lname: ("fail" if v["low_own_band"]["auc_appearance"] >= v["low_own_band"]["auc_real"]
                       else "pass")
               for lname, v in per_layer.items()},
        "not_done": [
            "off-diagonal cells at t=300/900 (E18_PREREG.md §2: one point per expert cannot "
            "describe generalisation)",
            "no trend claim across t: two points, and sigma differs 3x (E18_PREREG.md §6)",
            "LikePhys: not run here, consistent with E17",
        ],
    }

    (OUT / "E18_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(f"probe-noise threshold (E16 median sd): {sd:.2f} pp")
    for lname, v in per_layer.items():
        lo, hi = v["low_own_band"], v["high_own_band"]
        fo = f"{v['d_own']:+.2f}" if v["d_own"] is not None else "n/a"
        print(f"  {lname:4s} low@300 D1 {lo['d1_pct']:5.1f}% "
              f"{'ok ' if lo['interpretable'] else 'NI '} | high@900 D1 {hi['d1_pct']:5.1f}% "
              f"{'ok ' if hi['interpretable'] else 'NI '} | delta_own {fo:>7s} pp "
              f"(cross {v['d_cross']:+.2f})")
    print(f"Q: {q}  (delta_own {mean_own if mean_own is None else round(mean_own, 2)} vs "
          f"threshold {sd:.2f})")
    print(f"wrote {OUT / 'E18_verdict.json'}")


if __name__ == "__main__":
    main()
