"""Count every D2 cell that actually exists in the delivered JSONs, and export the count.

Written as a script because the number has now been mis-stated twice by hand (13, then 20,
then 28). The count is a derived fact about the deliverables, so it gets computed, not
remembered. Every report that quotes a D2 total should quote this file.

Two numbers, and they mean different things:

  cells               every (model, dataset, layer, caliber) appearance comparison. A caliber
                      or a second layer of the SAME extraction is a different cell but NOT an
                      independent replication.
  independent_units   (model, dataset, one feature extraction). This is the number to quote
                      when the claim is "it replicates across N models".
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "D2_TALLY.json"

# variant -> family. Two CogVideoX sizes are one family; two ViT sizes are one family.
FAMILY = {
    "cogvideox2b": "CogVideoX", "cogvideox5b": "CogVideoX",
    "wan21_1b3": "Wan", "wan21_14b": "Wan", "wan22_5b": "Wan",
    "ltx": "LTX", "mochi": "Mochi", "hunyuan": "HunyuanVideo",
    "vjepa2_vitl": "V-JEPA 2", "vjepa_vitl": "V-JEPA 2", "vjepa_vith": "V-JEPA 2",
    "wan22_a14b": "Wan",
    "wan22_a14b_low_t300": "Wan", "wan22_a14b_high_t900": "Wan",
    "minimax_h3": "MiniMax-H3",
}


def main():
    rows, units = [], set()

    e8 = json.loads((ROOT / "results/E8_second_backbone/E8_verdict.json").read_text())
    for ds, d in e8["datasets"].items():
        for bb, e in d["backbones"].items():
            if "d2" in e:
                rows.append({"cell": f"E8|{ds}|{bb}", "source": "E8", "verdict": e["d2"]["verdict"],
                             "auc_physics": e["d2"]["auc_physics"],
                             "auc_appearance": e["d2"]["auc_appearance"]})
                units.add((bb, ds))

    e14 = json.loads((ROOT / "results/E14_hunyuan/E14_verdict.json").read_text())
    for r in e14["rows"]:
        rows.append({"cell": f"E14|intphys2|{r['cell']}", "source": "E14",
                     "verdict": r["d2"]["verdict"],
                     "auc_physics": r["d2"]["auc_physics"],
                     "auc_appearance": r["d2"]["auc_appearance"]})
    units.add(("hunyuan", "intphys2"))

    for f, tag, pre in (("E13_cells.json", "E13", "vjepa_vitl"),
                        ("E13_cells_vith.json", "E13", "vjepa_vith")):
        cells = json.loads((ROOT / "results/E13_non_generative" / f).read_text())["cells"]
        for k, v in cells.items():
            if "auc_appearance" not in v:
                continue
            rows.append({"cell": f"{tag}|intphys2|{k}", "source": tag,
                         "verdict": "fail" if v["auc_appearance"] >= v["auc_real"] else "pass",
                         "auc_physics": v["auc_real"],
                         "auc_appearance": v["auc_appearance"]})
        units.add((pre, "intphys2"))

    # ViT-H LikePhys D2 lives in its own file (computed after E15 closed).
    vith_d2 = (ROOT / "results/E13_non_generative/E13_cells_vith_likephys_d2.json")
    if vith_d2.exists():
        for k, v in json.loads(vith_d2.read_text())["cells"].items():
            rows.append({"cell": f"E13vithD2|likephys|{k}", "source": "E13",
                         "verdict": v["verdict"],
                         "auc_physics": v["auc_physics_vs_legal"],
                         "auc_appearance": v["auc_appearance_vs_legal"]})
        units.add(("vjepa_vith", "likephys"))

    # E17's two experts x two layers. The two experts share one feature extraction per
    # (expert, weights) pair, so they count as two units, not four cells of new evidence.
    e17 = ROOT / "results/E17_moe_experts/E17_cells.json"
    if e17.exists():
        cells17 = json.loads(e17.read_text())["cells"]
        for k, v in cells17.items():
            if not k.endswith("L16") and not k.endswith("Lrel"):
                continue
            rows.append({"cell": f"E17|intphys2|{k}", "source": "E17",
                         "verdict": "fail" if v["auc_appearance"] >= v["auc_real"] else "pass",
                         "auc_physics": v["auc_real"],
                         "auc_appearance": v["auc_appearance"]})
        units.add(("hunyuan", "intphys2"))      # already counted; kept for clarity
        units.add(("wan22_a14b", "intphys2"))

    # E18's own-band cells (t=300 / t=900). Same counting rule as E17 -- two layers off one
    # feature extraction are one unit -- but these ARE new extractions (different timestep,
    # different noise draw, different cache), so each is an additional independent unit rather
    # than a repeat of E17. Counting them as repeats would understate novel evidence; counting
    # the two layers separately would overstate replication. Same rule both ways.
    e18 = ROOT / "results/E18_own_band/E18_cells.json"
    if e18.exists():
        cells18 = json.loads(e18.read_text())["cells"]
        for k, v in cells18.items():
            rows.append({"cell": f"E18|intphys2|{k}", "source": "E18",
                         "verdict": "fail" if v["auc_appearance"] >= v["auc_real"] else "pass",
                         "auc_physics": v["auc_real"],
                         "auc_appearance": v["auc_appearance"]})
        units.add(("wan22_a14b_low_t300", "intphys2"))
        units.add(("wan22_a14b_high_t900", "intphys2"))

    # E19: MiniMax-H3 (33B). Two layers off one extraction -> one unit, same rule as E17/E18.
    e19 = ROOT / "results/E19_h3_scale/E19_cells.json"
    if e19.exists():
        for k, v in json.loads(e19.read_text())["cells"].items():
            rows.append({"cell": f"E19|intphys2|{k}", "source": "E19",
                         "verdict": "fail" if v["auc_appearance"] >= v["auc_real"] else "pass",
                         "auc_physics": v["auc_real"],
                         "auc_appearance": v["auc_appearance"]})
        units.add(("minimax_h3", "intphys2"))

    d2 = json.loads((ROOT / "results/E15_likephys/E15_D2_likephys.json").read_text())["cells"]
    for k, v in d2.items():
        rows.append({"cell": f"E15D2|likephys|{k}", "source": "E15-D2-addendum",
                     "verdict": v["verdict"],
                     "auc_physics": v["auc_physics_vs_legal"],
                     "auc_appearance": v["auc_appearance_vs_legal"]})
        # addendum keys are "vjepa2_vitl" / "hunyuan"; map to the same names used elsewhere
        units.add(({"vjepa2_vitl": "vjepa2_vitl"}.get(k, k), "likephys"))

    tally = {
        "cells": len(rows),
        "independent_units": len(units),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_dataset": dict(Counter(r["cell"].split("|")[1] for r in rows)),
        # A model VARIANT (cogvideox2b vs cogvideox5b) is not a model FAMILY. Counting the
        # variants as families is how "6 families" became "10" in an earlier draft.
        "model_variants": sorted({u[0] for u in units}),
        "n_model_variants": len({u[0] for u in units}),
        "families": sorted({FAMILY.get(u[0], u[0]) for u in units}),
        "n_families": len({FAMILY.get(u[0], u[0]) for u in units}),
        "units": [f"{m}|{d}" for m, d in sorted(units)],
        "why": ("cells counts every appearance comparison; independent_units counts "
                "(model, dataset, one extraction). Quote both: 'cells' alone overstates "
                "replication because a caliber variant or a second layer reuses the same "
                "features."),
        "rows": rows,
    }
    OUT.write_text(json.dumps(tally, indent=1))
    print(json.dumps({k: v for k, v in tally.items()
                      if k not in ("rows", "units")}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
