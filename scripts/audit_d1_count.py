"""Count every D1 cell in the delivered JSONs. Same rationale as audit_d2_count.py:
this number has been mis-stated by hand, so it is computed and exported instead of remembered."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "D1_TALLY.json"


def main():
    rows = []
    e8 = json.loads((ROOT / "results/E8_second_backbone/E8_verdict.json").read_text())
    for ds, d in e8["datasets"].items():
        for bb, e in d["backbones"].items():
            c = e.get("L16|t600")
            if not c:
                continue
            ok = bool(c["interpretable"])
            verdict = ("pass" if c["d1_training_share"] >= 0.5 else "fail") if ok else "NOT interpretable"
            rows.append({"cell": f"E8|{ds}|{bb}", "dataset": ds, "source": "E8",
                         "d1_pct": c["d1_pct"], "interpretable": ok, "verdict": verdict})

    e13 = json.loads((ROOT / "results/E13_non_generative/E13_cells.json").read_text())["cells"]
    for k, v in e13.items():
        if "auc_real" not in v:
            continue
        rows.append({"cell": f"E13|intphys2|{k}", "dataset": "intphys2", "source": "E13",
                     "d1_pct": v["d1_pct"], "interpretable": bool(v["interpretable"]),
                     "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                if v["interpretable"] else "NOT interpretable"})

    e13h = json.loads((ROOT / "results/E13_non_generative/E13_cells_vith.json").read_text())["cells"]
    for k, v in e13h.items():
        rows.append({"cell": f"E13h|intphys2|L{v['layer']}", "dataset": "intphys2",
                     "source": "E13", "d1_pct": v["d1_pct"],
                     "interpretable": bool(v["interpretable"]),
                     "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                if v["interpretable"] else "NOT interpretable"})

    e14 = json.loads((ROOT / "results/E14_hunyuan/E14_cells.json").read_text())["cells"]
    for k, v in e14.items():
        if not k.endswith("physics"):
            continue
        rows.append({"cell": f"E14|intphys2|{k.split('|')[2]}", "dataset": "intphys2",
                     "source": "E14", "d1_pct": v["d1_pct"],
                     "interpretable": bool(v["interpretable"]),
                     "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                if v["interpretable"] else "NOT interpretable"})

    e17 = json.loads((ROOT / "results/E17_moe_experts/E17_cells.json").read_text())["cells"]
    for k, v in e17.items():
        if not (k.endswith("L16") or k.endswith("Lrel")):
            continue
        rows.append({"cell": f"E17|intphys2|{k}", "dataset": "intphys2", "source": "E17",
                     "d1_pct": v["d1_pct"], "interpretable": bool(v["interpretable"]),
                     "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                if v["interpretable"] else "NOT interpretable"})

    # E15's LikePhys cells live in their own files. V-JEPA's D1 there is NOT a single number:
    # E15 showed it moves 43.8-54.0% across probe partitions, so a single-seed pass/fail would
    # be the exact "pick the seed that looks good" error E16 was written to prevent. Its
    # verdict therefore comes from the SIX-SEED MEAN against a +-1sd band.
    lps = ROOT / "results/E15_likephys"
    seeds = json.loads((lps / "E15_vjepa_probe_seeds.json").read_text())["layers"]
    vj = json.loads((lps / "E15_cells_vjepa.json").read_text())["cells"]
    for k, v in vj.items():
        L = k.split("|")[-1]        # "L16" / "L13"
        sd = seeds.get(L[1:], {})
        mean = sd.get("d1_mean")
        if mean is None:
            verdict = "pass" if v["d1"] >= 0.5 else "fail"
        else:
            lo, hi = mean - sd["d1_sd"], mean + sd["d1_sd"]
            verdict = "at_threshold" if lo < 0.5 <= hi else ("pass" if mean >= 0.5 else "fail")
            mean = mean * 100
        rows.append({"cell": f"E15-vjepa|likephys|{L}", "dataset": "likephys",
                     "source": "E15", "d1_pct": mean * 100 if mean is not None and mean < 1.5
                     else mean,
                     "d1_mean_over_6_probe_seeds_pct": (mean * 100 if mean is not None
                                                        and mean < 1.5 else mean),
                     "single_seed_d1_pct": v["d1_pct"],
                     "interpretable": bool(v.get("interpretable")),
                     "verdict": verdict,
                     "why": "D1 straddles 50% across probe partitions; judged on the 6-seed mean "
                            "and its +-1sd band, not on seed 1000"})

    hun = json.loads((lps / "E15_cells_hunyuan.json").read_text())["cells"]
    for k, v in hun.items():
        rows.append({"cell": f"E15-hunyuan|likephys|{k.split('|')[-1]}", "dataset": "likephys",
                     "source": "E15", "d1_pct": v.get("d1_pct"),
                     "interpretable": bool(v.get("interpretable")),
                     "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                if v.get("interpretable") else "NOT interpretable"})

    # E18's own-band cells. Counted, but flagged: E18_confound_check.json shows D1 in these
    # cells is dominated by where the random arm sits (denominator 0.0955-0.1326 amplifies a
    # 0.0116 AUC gap into ~10pp), and one cell sits below the real-AUC 0.6 weak-signal line
    # from E8 §15.2. The verdict is recorded for the tally, but `d1_comparable_across_t` is
    # false -- quoting these beside E17's t=600 numbers as one scale is what E18_REPORT §3.1
    # forbids.
    e18 = ROOT / "results/E18_own_band/E18_cells.json"
    if e18.exists():
        for k, v in json.loads(e18.read_text())["cells"].items():
            rows.append({"cell": f"E18-ownband|intphys2|{k}", "dataset": "intphys2",
                         "source": "E18", "d1_pct": v["d1_pct"],
                         "auc_real": v["auc_real"], "auc_random": v["auc_random"],
                         "interpretable": bool(v["interpretable"]),
                         "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                    if v["interpretable"] else "NOT interpretable",
                         "d1_comparable_across_t": False,
                         "why": "not comparable across t: both the real-arm AUC and the "
                                "random-arm level shift with sigma "
                                "(see E18_confound_check.json)"})

    # E19: MiniMax-H3 (33B). Counted like every other D1 cell; the L16 cell sits below the
    # real-AUC 0.6 weak-signal line from E8 §15.2, so it is flagged not-comparable and must not
    # be quoted next to Wan's 0.65-0.67 cells as if on one scale.
    e19 = ROOT / "results/E19_h3_scale/E19_cells.json"
    if e19.exists():
        for k, v in json.loads(e19.read_text())["cells"].items():
            rows.append({"cell": f"E19-h3|intphys2|{k}", "dataset": "intphys2",
                         "source": "E19", "d1_pct": v["d1_pct"],
                         "auc_real": v["auc_real"], "auc_random": v["auc_random"],
                         "interpretable": bool(v["interpretable"]),
                         "verdict": ("pass" if v["d1"] >= 0.5 else "fail")
                                    if v["interpretable"] else "NOT interpretable",
                         "comparable_across_cells": bool(v["auc_real"] >= 0.6),
                         "why": "L16 real AUC 0.574 < 0.6 -> weak signal (E8 §15.2)"
                                if v["auc_real"] < 0.6 else "above the 0.6 weak-signal line"})

    by_ds = {}
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        by_ds[ds] = {"cells": len(sub), **dict(Counter(r["verdict"] for r in sub))}

    tally = {
        "cells": len(rows),
        "verdicts_all": dict(Counter(r["verdict"] for r in rows)),
        "by_dataset": by_ds,
        "by_source": dict(Counter(r["source"] for r in rows)),
        "why": ("Counted from the delivered JSONs, not by hand. A cell is one (model, dataset, "
                "layer) D1 evaluation; 'NOT interpretable' means the frozen T10 gate refused "
                "the ratio, so it is neither a pass nor a fail."),
        "rows": rows,
    }
    OUT.write_text(json.dumps(tally, indent=1))
    print(json.dumps({k: v for k, v in tally.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
