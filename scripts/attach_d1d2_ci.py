"""Attach E20's cluster-bootstrap intervals to the D1/D2 tallies, without changing a verdict.

Why a separate script rather than editing the two count scripts: those scripts exist to
*derive* the cell counts, and the project forbids hand-editing delivered numbers. This script
is additive — it reads `D1_TALLY.json` / `D2_TALLY.json`, leaves every numeric field untouched,
and adds one `ci95` block per cell that E20 covered. The E8 cells are the only ones E20
measured, so the other models keep no interval and say so explicitly (`ci95: null`,
`ci95_source: "not measured in E20"`) rather than silently looking as solid as the rest.

**Run order matters.** `audit_d1_count.py` / `audit_d2_count.py` rebuild their tally from the
delivered JSONs and therefore DROP this annotation. The order is always:

    python scripts/audit_d1_count.py && python scripts/audit_d2_count.py
    python scripts/attach_d1d2_ci.py

Re-running the count after this script silently removes every interval; that was observed
(12 rows with CI -> 0) and is why the order is written here rather than left to memory.

Run: python scripts/attach_d1d2_ci.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E20 = ROOT / "results" / "E20_bootstrap_ci" / "E20_cells.json"
EXT = ROOT / "results" / "E20_bootstrap_ci" / "E20ext_cells.json"

D1 = ROOT / "results" / "D1_TALLY.json"
D2 = ROOT / "results" / "D2_TALLY.json"

# E20 cell key -> the tally's (source, dataset, backbone) triple. E20 measured E8's six
# backbones on both datasets, and nothing else, so this table is exhaustive by construction.
SOURCE = "E8"

# E20-ext (the extension run) writes its own cells; its ids differ from the tally's for three
# families, so the mapping is written out rather than guessed. Everything not listed matches
# exactly.
EXT_MAP = {
    # tally id                                   -> E20-ext source_cell
    "E13|intphys2|vjepa|intphys2|L16|flat|physics": "E13|intphys2|vjepa|intphys2|L16|flat",
    "E13|intphys2|vjepa|intphys2|L13|flat|physics": "E13|intphys2|vjepa|intphys2|L13|flat",
    "E13|intphys2|vjepa|intphys2|L16|timeavg|physics": "E13|intphys2|vjepa|intphys2|L16|timeavg",
    "E13|intphys2|vjepa|intphys2|L13|timeavg|physics": "E13|intphys2|vjepa|intphys2|L13|timeavg",
    # E13's ViT-H rows are stored in the tally as `E13h|intphys2|L{layer}`
    "E13h|intphys2|L16": "E13|intphys2|vith|intphys2|L16",
    "E13h|intphys2|L17": "E13|intphys2|vith|intphys2|L17",
    # E18: the D2 tally writes `E18|`, the D1 tally and E20-ext write `E18-ownband|`
    "E18|intphys2|a14b|low@300|L16": "E18-ownband|intphys2|a14b|low@300|L16",
    "E18|intphys2|a14b|low@300|Lrel": "E18-ownband|intphys2|a14b|low@300|Lrel",
    "E18|intphys2|a14b|high@900|L16": "E18-ownband|intphys2|a14b|high@900|L16",
    "E18|intphys2|a14b|high@900|Lrel": "E18-ownband|intphys2|a14b|high@900|Lrel",
    # E19: `E19|` in the D2 tally, `E19-h3|` elsewhere
    "E19|intphys2|h3|L16": "E19-h3|intphys2|h3|L16",
    "E19|intphys2|h3|Lrel": "E19-h3|intphys2|h3|Lrel",
    # E15's D2 addendum rows name the model, not the E15 cell id
    "E15D2|likephys|vjepa2_vitl": "E15-vjepa|likephys|L16",
    "E15D2|likephys|hunyuan": "E15-hunyuan|likephys|L16",
}


def main():
    e20 = json.loads(E20.read_text())["cells"]
    ext = json.loads(EXT.read_text())["cells"] if EXT.exists() else {}
    d1_ci, d2_ci = {}, {}
    for key, v in e20.items():
        k = (v["dataset"], v["backbone"])
        if key.endswith("|D1"):
            d1_ci[k] = {"d1_pct_point_e20": round(v["d1_pct_point"], 2),
                        "ci95_pct": [round(x, 2) for x in v["d1_pct_ci95"]],
                        "ci_width_pct": round(v["d1_pct_ci_width"], 2),
                        "verdict_point_e20": v["verdict_point"],
                        "verdict_fragile": v["verdict_fragile"]}
        else:
            d2_ci[k] = {"gap_point_e20": round(v["gap_point"], 4),
                        "ci95_gap": [round(x, 4) for x in v["gap_ci95"]],
                        "ci_width": round(v["gap_ci_width"], 4),
                        "verdict_point_e20": v["verdict_point"],
                        "verdict_fragile": v["verdict_fragile"]}

    # --- E20-ext: keyed by the tally's own cell string (after EXT_MAP)
    ext_by_id = {}
    for key, v in ext.items():
        cid = key.rsplit("|", 1)[0]
        ext_by_id.setdefault(cid, {})[key.rsplit("|", 1)[-1]] = v
    ext_d1, ext_d2 = {}, {}
    for cid, pair in ext_by_id.items():
        if "D1" in pair:
            v = pair["D1"]
            ext_d1[cid] = {"d1_pct_point_e20ext": round(v["d1_pct_point"], 2),
                           "ci95_pct": [round(x, 2) for x in v["d1_pct_ci95"]],
                           "ci_width_pct": round(v["d1_pct_ci_width"], 2),
                           "verdict_point_e20ext": v["verdict_point"],
                           "verdict_fragile": v["verdict_fragile"]}
        if "D2" in pair:
            v = pair["D2"]
            ext_d2[cid] = {"gap_point_e20ext": round(v["gap_point"], 4),
                           "ci95_gap": [round(x, 4) for x in v["gap_ci95"]],
                           "ci_width": round(v["gap_ci_width"], 4),
                           "verdict_point_e20ext": v["verdict_point"],
                           "verdict_fragile": v["verdict_fragile"]}

    for path, table, field in ((D1, d1_ci, "D1"), (D2, d2_ci, "D2")):
        tally = json.loads(path.read_text())
        n_with, n_without = 0, 0
        for r in tally["rows"]:
            # Tally rows key on (source, dataset, backbone); only E8's are in E20.
            src = r.get("source")
            ds = r.get("dataset")
            # D2 rows do not carry `dataset` in every source, so fall back to the cell string.
            if ds is None:
                parts = r["cell"].split("|")
                ds = parts[1] if len(parts) > 1 else None
            bb = r["cell"].split("|")[-1]
            ci = table.get((ds, bb)) if src == SOURCE else None
            if ci:
                r["ci95"] = ci
                r["ci95_source"] = "results/E20_bootstrap_ci (cluster bootstrap, B=300)"
                n_with += 1
                continue
            # fall through to the extension run, matching on the tally's own cell string
            lookup = EXT_MAP.get(r["cell"], r["cell"])
            ext_ci = (ext_d1 if field == "D1" else ext_d2).get(lookup)
            if ext_ci:
                r["ci95"] = ext_ci
                r["ci95_source"] = ("results/E20_bootstrap_ci/E20ext_cells.json "
                                    "(cluster bootstrap, B=300)")
                n_with += 1
            else:
                r["ci95"] = None
                r["ci95_source"] = "not measured in E20 or E20-ext"
                n_without += 1
        tally[f"ci95_{field}_note"] = (
            f"{n_with} of {len(tally['rows'])} rows carry a 95% cluster-bootstrap interval "
            f"from E20; {n_without} do not. A row without an interval is not weaker evidence — "
            "it simply was not resampled. Intervals condition on the fitted probe (see "
            "E20_REPORT.md §2). The point-estimate verdicts are unchanged.")
        path.write_text(json.dumps(tally, indent=1))
        print(f"{path.name}: {n_with} rows with CI, {n_without} without")


if __name__ == "__main__":
    main()
