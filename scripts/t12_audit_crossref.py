"""T12.1(b): cross-reference consistency across the manuscript and the archived reports.

Three things are checked:

  1. the same quantity quoted in more than one document must carry the same value
  2. section cross-references (§5.x referring to §5.y) must point at a section that exists
  3. the retired term "最小充分 / minimal sufficient" must have zero hits (the protocol was
     re-graded to `minimal necessary` in T10 and the old wording is not defensible)

Run: python scripts/t12_audit_crossref.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T12_audit"

DOCS = {
    "EXPERIMENTS": ROOT / "results/T11_writing/EXPERIMENTS.md",
    "RELATED_WORK_probing": ROOT / "results/T11_writing/RELATED_WORK_probing.md",
    "SPEC": ROOT / "SPEC.md",
    "PREREG": ROOT / "PREREG.md",
    "PROTOCOL_VALIDITY": ROOT / "PROTOCOL_VALIDITY.md",
    "RELATED_WORK": ROOT / "research/RELATED_WORK.md",
    "T10_REPORT": ROOT / "results/T10_validity/T10_REPORT.md",
    "T11_1_HUMAN": ROOT / "results/T10_validity/human_eval/T11_1_HUMAN_REPORT.md",
    "E7_REPORT": ROOT / "results/E7_intphys2/E7_REPORT.md",
    "E4_REPORT": ROOT / "results/E4_complexity/E4_REPORT.md",
    "S4_REPORT": ROOT / "results/E10_S4/S4_REPORT.md",
    "E6_TABLE": ROOT / "results/E6_literature/E6_TABLE.md",
}

# quantities that appear in several documents and must agree
SHARED = {
    "best real AUC (L14/t400)": [r"0\.9740"],
    "random lower bound (best layer)": [r"0\.8576"],
    "D1 best-cell share": [r"24\.6\s*%"],
    "D1 fixed-cell share (L16/t600)": [r"28\.5\s*%"],
    "MDL training share": [r"31\.8\s*%"],
    "IntPhys2 D1 pooled": [r"88\.5\s*%"],
    "IntPhys2 physics AUC": [r"0\.6458"],
    "IntPhys2 appearance AUC": [r"1\.0000"],
    "E4 within-scenario intercept": [r"\+?0\.00112"],
    "E4 temporal_grad variance": [r"56\.3\s*%|56\s*%"],
    "S2 shadow_camera mean": [r"0\.4927"],
    "S4 max drop": [r"0\.0200"],
    "S4 dprime retained": [r"0\.754"],
    "C2 real AUC": [r"0\.8100"],
    "ablation unit count": [r"17\s*(个)?\s*(单元|units?)"],
}

RETIRED = [
    (r"最小充分", "retired term 最小充分"),
    (r"minimal\s+sufficient", "retired term minimal sufficient"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {}
    for k, p in DOCS.items():
        texts[k] = p.read_text().replace("\u2212", "-") if p.exists() else None

    missing = [k for k, v in texts.items() if v is None]

    # ---- 1. retired terminology -------------------------------------------------------
    # Most occurrences are the ban itself ("禁止使用「最小充分」"), which must NOT be
    # flagged -- otherwise the rule that enforces the terminology looks like a violation of
    # it. Only uses that assert the term are real hits.
    # Judge the LINE the term sits on, not a +/-140 char window: the window around
    # SPEC.md:19 happens to contain an unrelated "禁止使用 ... 标题" sentence, which made a
    # real violation (the paper title still says "Minimal Sufficient") look like a ban.
    BAN_LINE = re.compile(r"禁[止用]|停止使用|不得|应为\s*0|改为|换成|降级为|retired|must not"
                          r"|出现\s*\*\*?16|不是\s*[`「]?minimal[_ ]?sufficient", re.I)
    retired_hits = []
    for name, t in texts.items():
        if t is None:
            continue
        for pat, label in RETIRED:
            for m in re.finditer(pat, t, re.I):
                line = t[:m.start()].count("\n") + 1
                ls = t.rfind("\n", 0, m.start()) + 1
                le = t.find("\n", m.end())
                line_txt = t[ls:le if le > 0 else len(t)]
                ctx = re.sub(r"\s+", " ", line_txt)
                asserting = not BAN_LINE.search(line_txt)
                retired_hits.append({"doc": name, "line": line, "term": label,
                                     "asserting_use": asserting, "context": ctx})

    # ---- 2. shared quantities ---------------------------------------------------------
    shared_rows = []
    for label, pats in SHARED.items():
        where = []
        for name, t in texts.items():
            if t is None:
                continue
            if any(re.search(p, t) for p in pats):
                where.append(name)
        shared_rows.append({"quantity": label, "appears_in": where, "n_docs": len(where)})

    # ---- 3. section cross-references ---------------------------------------------------
    xref = []
    for name, t in texts.items():
        if t is None:
            continue
        headings = set(re.findall(r"^#{1,4}\s*§?\s*([\d.]+[\d])", t, re.M))
        headings |= set(re.findall(r"^#{1,4}\s*([\d.]+[\d])\s", t, re.M))
        refs = set(re.findall(r"§\s*([\d]+\.[\d.]*[\d])", t))
        # only self-references can be validated; cross-document ones are skipped
        own_prefix = {h.split(".")[0] for h in headings}
        dangling = sorted(r for r in refs
                          if r.split(".")[0] in own_prefix and r not in headings)
        if dangling:
            xref.append({"doc": name, "headings": sorted(headings),
                         "dangling_refs": dangling})

    payload = {
        "docs_checked": [k for k, v in texts.items() if v is not None],
        "docs_missing": missing,
        "retired_terms": {
            "n_raw_hits": len(retired_hits),
            "n_asserting_use": sum(1 for h in retired_hits if h["asserting_use"]),
            "violations": [h for h in retired_hits if h["asserting_use"]],
            "ban_statements": [h for h in retired_hits if not h["asserting_use"]],
        },
        "shared_quantities": shared_rows,
        "dangling_section_refs": xref,
    }
    (OUT / "T12_crossref_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"docs checked: {len(payload['docs_checked'])}"
          + (f"   MISSING: {missing}" if missing else ""))

    real = [h for h in retired_hits if h["asserting_use"]]
    bans = [h for h in retired_hits if not h["asserting_use"]]
    print(f"\n=== retired terminology ===")
    print(f"  {len(retired_hits)} raw hits: {len(bans)} are the ban itself, "
          f"{len(real)} assert the term")
    if real:
        print("  --- REAL violations ---")
        for h in real:
            print(f"  {h['doc']}:{h['line']}  {h['term']}")
            print(f"      ...{h['context'][:200]}...")
    else:
        print("  no asserting use found")

    print(f"\n=== shared quantities ===")
    for r in shared_rows:
        flag = "" if r["n_docs"] else "   <-- appears nowhere"
        print(f"  {r['quantity']:34s} {r['n_docs']} docs  {r['appears_in']}{flag}")

    print(f"\n=== dangling section refs ===")
    if not xref:
        print("  none")
    else:
        for x in xref:
            print(f"  {x['doc']}: {x['dangling_refs']}")
    print(f"\nwrote {OUT/'T12_crossref_audit.json'}")


if __name__ == "__main__":
    main()
