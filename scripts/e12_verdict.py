"""E12 verdict: Q1/Q2/Q3 from the two ladder JSONs, per E12_PREREG.md §3.

Reads `E12_cells_{ltx,cogvideox2b}.json` and applies the pre-registered criteria literally.
Nothing is recomputed here: the AUCs, floors and bootstrap CIs are the analysis script's
output; this script only decides what they mean.

The one place judgement enters is Q1's PASS rule, and it is fixed in advance: PASS-token needs
BOTH (a) the real-slope 95% CI to exclude 0 with a rise, AND (b) that rise to be at least 0.05
AUC from the lowest to the highest rung. (a) alone would call a statistically detectable but
practically nil drift a success — the same "significant but tiny" trap this project flagged in
T12. Both calibers must agree, because the time ladder confounds token count with feature
dimensionality (E12_PREREG.md §4.4).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E12_token_budget"
MIN_RISE = 0.05          # AUC units, lowest rung -> highest rung; fixed in §3 Q1


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def rise(ladder: dict, key: str = "L16") -> float:
    """AUC_real at the highest-token rung minus at the lowest, flat caliber."""
    pts = sorted(ladder["ladders"].items(), key=lambda kv: kv[1]["tokens"])
    lo = pts[0][1][f"flat|{key}"]["auc_real"]
    hi = pts[-1][1][f"flat|{key}"]["auc_real"]
    return hi - lo


def q1_block(cells: dict) -> dict:
    """Q1: does AUC_real rise with the token budget, above the random-init control?"""
    out = {"calibers": {}}
    for caliber in ("flat", "timeavg"):
        q = cells["q1"][caliber]
        sr = q["slope_real"]
        d = q["delta_slope_real_minus_random"]
        low, high = sr["ci95"]
        out["calibers"][caliber] = {
            "slope_real": sr["slope"], "ci95": sr["ci95"],
            "slope_random": q["slope_random"]["slope"],
            "slope_raw_pixel": q["slope_raw_pixel"]["slope"],
            "slope_vae_latent": q["slope_vae_latent"]["slope"],
            "delta_slope": d["slope"], "delta_ci95": d["ci95"],
            "slope_positive": bool(low > 0),
            "slope_excludes_rise": bool(high < 0.01),
            "delta_positive": bool(d["ci95"][0] > 0),
        }
    out["rise_lowest_to_highest_flat"] = rise(cells)
    out["min_rise_required"] = MIN_RISE
    flat = out["calibers"]["flat"]
    tavg = out["calibers"]["timeavg"]
    pass_flat = flat["slope_positive"] and out["rise_lowest_to_highest_flat"] >= MIN_RISE
    pass_tavg = tavg["slope_positive"]
    if pass_flat and pass_tavg:
        out["verdict"] = "pass_token"
    elif flat["slope_excludes_rise"] and tavg["slope_excludes_rise"]:
        out["verdict"] = "fail_token"
    else:
        out["verdict"] = "inconclusive"
    out["note"] = (
        "Both calibers must pass: the temporal ladder changes F_lat, so the probe's input "
        "dimension moves with the token count (E12_PREREG.md §4.4). A rise only in `flat` "
        "would be a capacity effect, not a budget effect.")
    return out


def q2_block(ltx: dict, cog: dict) -> dict:
    """Q2: at matched token budgets, is LTX's real AUC distinguishable from CogVideoX's?"""
    def at(cells: dict, lo: int, hi: int, key: str = "L16") -> tuple[int, float]:
        best = min(cells["ladders"].items(),
                   key=lambda kv: abs(kv[1]["tokens"] - (lo + hi) / 2))
        return best[1]["tokens"], best[1][f"flat|{key}"]["auc_real"]

    l1800 = at(ltx, 1792, 1872)
    c1800 = at(cog, 1792, 1872)
    l13300 = at(ltx, 12800, 13312)
    c13300 = at(cog, 12800, 13312)
    gap_lo = l1800[1] - c1800[1]
    gap_hi = l13300[1] - c13300[1]
    both_small = abs(gap_lo) < 0.05 and abs(gap_hi) < 0.05
    both_neg = gap_lo < -0.05 and gap_hi < -0.05
    return {
        "at_1800": {"ltx_tokens": l1800[0], "ltx": l1800[1],
                    "cog_tokens": c1800[0], "cog": c1800[1], "gap": gap_lo},
        "at_13300": {"ltx_tokens": l13300[0], "ltx": l13300[1],
                     "cog_tokens": c13300[0], "cog": c13300[1], "gap": gap_hi},
        "tolerance": 0.05,
        "verdict": ("budget" if both_small else
                    "architecture" if both_neg else "mixed"),
        "note": ("`budget` would mean matching the token count is enough to match the AUC; "
                 "`architecture` means LTX stays below CogVideoX at equal budget, so the "
                 "32x-spatial VAE / patch-1 design contributes independently."),
    }


def q3_block(cells: dict) -> dict:
    """Q3: from which token budget does D1 become interpretable?"""
    pts = []
    for tag, e in sorted(cells["ladders"].items(), key=lambda kv: kv[1]["tokens"]):
        c = e.get("flat|L16")
        if not c:
            continue
        pts.append({"tokens": e["tokens"], "tag": tag,
                    "d1_pct": c["d1_pct"], "interpretable": c["interpretable"],
                    "auc_real": c["auc_real"]})
    first = next((p["tokens"] for p in pts if p["interpretable"]), None)
    n_interp = sum(1 for p in pts if p["interpretable"])
    return {
        "points": pts,
        "first_interpretable_token": first,
        "n_interpretable": n_interp,
        "n_total": len(pts),
        "note": ("Interpretability is the frozen T10 gate (effect > 2x floor AND "
                 "AUC-0.5 > 2x floor). Where it fails the D1 ratio is not read at all, "
                 "so a 'D1 >= 50%' does not appear there — that is not a pass."),
    }


def main():
    ltx = load("E12_cells_ltx.json")
    cog = load("E12_cells_cogvideox2b.json")

    verdict = {
        "question": "does the token budget explain LTX's weak real AUC on IntPhys 2?",
        "reference_cell": {"layer": 16, "t": 600},
        "ladders": {
            "ltx_temporal": [
                {"tag": t, "tokens": e["tokens"], "frames": e["tokens"] // 256 * 8 - 7,
                 "auc_real": e["flat|L16"]["auc_real"],
                 "auc_random": e["flat|L16"]["auc_random"],
                 "d1_pct": e["flat|L16"]["d1_pct"],
                 "floor": e["flat|L16"]["floor"],
                 "interpretable": e["flat|L16"]["interpretable"],
                 "raw_pixel": e["flat|raw_pixel"]["auc"]}
                for t, e in sorted(ltx["ladders"].items(), key=lambda kv: kv[1]["tokens"])],
            "cog_spatial": [
                {"tag": t, "tokens": e["tokens"],
                 "size": int(t.replace("BASE", "S512F49").split("F")[0][1:]),
                 "auc_real": e["flat|L16"]["auc_real"],
                 "auc_random": e["flat|L16"]["auc_random"],
                 "d1_pct": e["flat|L16"]["d1_pct"],
                 "floor": e["flat|L16"]["floor"],
                 "interpretable": e["flat|L16"]["interpretable"],
                 "raw_pixel": e["flat|raw_pixel"]["auc"]}
                for t, e in sorted(cog["ladders"].items(), key=lambda kv: kv[1]["tokens"])],
        },
        "q1": q1_block(ltx),
        "q2": q2_block(ltx, cog),
        "q3": q3_block(ltx),
        "crosscheck": {
            "BASE_ltx_L16_flat_auc_real": ltx["ladders"]["BASE"]["flat|L16"]["auc_real"],
            "BASE_ltx_matches_E8": abs(ltx["ladders"]["BASE"]["flat|L16"]["auc_real"]
                                       - 0.5506) < 5e-4,
            "BASE_cog_L16_flat_auc_real": cog["ladders"]["BASE"]["flat|L16"]["auc_real"],
            "BASE_cog_matches_E8": abs(cog["ladders"]["BASE"]["flat|L16"]["auc_real"]
                                       - 0.6443) < 5e-4,
            "note": "E12_PREREG.md §7 requires the two baseline cells to reproduce E8; both "
                    "are read from E8's own cache (tag BASE), so a mismatch would mean the "
                    "analysis path, not the extraction, is wrong.",
        },
        "note": ("LikePhys is not covered: its clips are 56-61 frames, so LTX's 8x temporal "
                 "compression caps F_lat at 8 and the temporal ladder cannot be built "
                 "(E12_PREREG.md §4.4). The LikePhys LTX cell stands as E8 reported it."),
    }

    dst = OUT / "E12_verdict.json"
    dst.write_text(json.dumps(verdict, indent=1))

    print("LTX temporal ladder (flat L16):")
    for r in verdict["ladders"]["ltx_temporal"]:
        print(f"  {r['tag']:10s} tok {r['tokens']:6d} real {r['auc_real']:.4f} "
              f"rand {r['auc_random']:.4f} D1 {r['d1_pct']:5.1f}% "
              f"{'OK' if r['interpretable'] else 'not interp'}")
    print("\nCogVideoX spatial ladder (flat L16):")
    for r in verdict["ladders"]["cog_spatial"]:
        print(f"  {r['tag']:10s} tok {r['tokens']:6d} S={r['size']:4d} real {r['auc_real']:.4f} "
              f"rand {r['auc_random']:.4f} D1 {r['d1_pct']:5.1f}%")
    print(f"\nQ1: {verdict['q1']['verdict']}  (rise {verdict['q1']['rise_lowest_to_highest_flat']:+.4f})")
    print(f"Q2: {verdict['q2']['verdict']}")
    print(f"Q3: first interpretable at {verdict['q3']['first_interpretable_token']} tokens")
    print(f"crosscheck: ltx {verdict['crosscheck']['BASE_ltx_matches_E8']}, "
          f"cog {verdict['crosscheck']['BASE_cog_matches_E8']}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
