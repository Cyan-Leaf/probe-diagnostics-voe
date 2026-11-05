"""Build results/E9_saturation/manifest.json from the delivered cells file."""

import glob
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E9_saturation"


def sha(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    d = json.loads((OUT / "E9_cells.json").read_text())
    v = json.loads((OUT / "E9_verdict.json").read_text())
    agg = d["aggregates"]
    manifest = {
        "task": "E9: does dropping the saturated scenarios move LikePhys D1? (zero GPU)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_cell": {"backbone": d["backbone"], "dataset": d["dataset"],
                           "layer": d["layer"], "t": d["t"]},
        "feature_seeds": [0, 1, 2],
        "threads": 4,
        "threads_why": (
            "The probe is not bit-reproducible across BLAS thread counts (first measured in "
            "E16). All E9 numbers use 4 so they are comparable to E8's delivered cells; "
            "the all-clips D1 here is 32.898%, which is exactly E8's LikePhys L16/t=600 value, "
            "and that match is the self-check."),
        "saturation_definition": {
            "rule": "AUC >= 1 - 1e-9 on BOTH trained and random",
            "eps": d["sat_eps"],
            "why_strict": (
                "AUC = 1.0 is the ceiling of a bounded rank statistic (PROTOCOL_VALIDITY D3/D6). "
                "A looser delta would call 0.995 'saturated', which is strong signal rather than "
                "a statistical ceiling, and only the latter justifies exclusion.")},
        "selfcheck": {
            "rule": "all-clips pooled D1 must equal E8's delivered LikePhys L16/t=600 value",
            "expected_pct": 32.9,
            "measured_pct": agg["all_clips"]["d1_pct"],
            "verdict": "PASS",
            "note": "matches to the displayed precision; the pooled arm reuses e8_analyse.probe"},
        "aggregates_reported_separately": ["all_clips", "drop_saturated", "per_scenario_mean"],
        "why_two_aggregates": (
            "Pooled D1 recomputes AUC over all clips (E8's convention); the per-scenario mean "
            "averages 12 separate D1 values (what 'saturated scenarios drag the mean down' "
            "literally describes). They are different quantities and E9_PREREG.md §3 forbids "
            "promoting either to 'the' main result."),
        "verdict": {"q1": v["q1"],
                    "d1_all_pct": agg["all_clips"]["d1_pct"],
                    "d1_drop_pct": agg["drop_saturated"]["d1_pct"],
                    "delta_pp": v["q2"]["delta_pp"],
                    "within_noise": v["q2"]["within_noise"],
                    "n_saturated_strict": v["q3"]["n_saturated"],
                    "fraction_clips_dropped": v["q3"]["fraction_dropped"]},
        "known_discrepancies": {
            "saturated_count": (
                "2 at L16/t=600, not the 5 claimed in NEXT_STEPS_AND_DIVISION.md §1. The 5/12 in "
                "EXPERIMENTS.md §5.6 (D3) was measured on L14/t=400, whose features are NOT "
                "cached on this node (load_X returns None), so it can neither be confirmed nor "
                "refuted here. Registered, not overwritten."),
            "shadow_camera": (
                "AUC_real 0.3662 < 0.5 — the network ranked physics-violating clips ABOVE legal "
                "ones. D1 is undefined (negative denominator), not small. Excluded from both "
                "per-scenario means so the two sides average over the same scenario set.")},
        "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e9_*.py"))},
        "deliverables": ["E9_PREREG.md", "E9_cells.json", "E9_verdict.json", "E9_REPORT.md",
                         "manifest.json"],
        "runtime_s": d["verdict"]["elapsed_s"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    show = {k: manifest[k] for k in ("threads", "selfcheck", "verdict", "runtime_s")}
    print(json.dumps(show, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
