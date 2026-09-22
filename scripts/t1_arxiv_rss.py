"""C1c complement: pull the arXiv listing RSS directly.

Why this is needed on top of OpenAlex: OpenAlex's `publication_date` is an indexing /
version date, not the arXiv submission date -- the 2026-09-16..18 window returned works
whose arXiv ids are from 2024-2026 and contained no 2609.* postings at all.  So OpenAlex
cannot reproduce arXiv's `submittedDate` window.

The arXiv listing RSS (`rss.arxiv.org/rss/<cat>`) *is* reachable from this box and carries
exactly the newly announced papers, which is precisely the "since the last rescan" window
that C1c asks for.  Its limitation is that it only covers the most recent announcement
cycle -- that is stated in the output rather than papered over.

Run: python scripts/t1_arxiv_rss.py --out results/T1_rescan_20260917
"""

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "cogreason-voe-T1/0.1"}
CATS = ["cs.CV", "cs.LG", "cs.AI"]

DOMAIN = re.compile(
    r"video diffusion|video generation|world model|text-to-video|video model|"
    r"diffusion transformer|video foundation", re.I)
METHOD = re.compile(
    r"\bprob(e|es|ing)\b|linearly decodable|linear probe|internal representation|"
    r"hidden state|verifier|reward guidance|denoising error|readout|plausibilit|"
    r"intuitive physics|physical (understanding|reasoning|plausibility)", re.I)
PHYS = re.compile(r"physic|plausibilit|intuitive", re.I)


def fetch(cat):
    url = f"https://rss.arxiv.org/rss/{cat}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    items, per_cat = {}, {}
    for cat in CATS:
        xml = fetch(cat)
        (out / "raw" / f"rss_{cat}.xml").write_text(xml)
        root = ET.fromstring(xml)
        n = 0
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            desc = (it.findtext("description") or "").strip()
            link = (it.findtext("link") or "").strip()
            aid = link.rsplit("/", 1)[-1]
            n += 1
            if aid in items:
                items[aid]["cats"].append(cat)
                continue
            items[aid] = {"arxiv_id": aid, "title": re.sub(r"\s+", " ", title),
                          "abstract": re.sub(r"<[^>]+>", " ", desc)[:1500],
                          "link": link, "cats": [cat]}
        per_cat[cat] = n
        print(f"  {cat}: {n} items", flush=True)

    # mechanical domain x method screen (rule 1 applied to the fresh listing)
    flagged = []
    for it in items.values():
        blob = it["title"] + " " + it["abstract"]
        d, m, p = bool(DOMAIN.search(blob)), bool(METHOD.search(blob)), bool(PHYS.search(blob))
        it["hit_domain"], it["hit_method"], it["hit_physics"] = d, m, p
        if d and (m or p):
            flagged.append(it)

    known = set()
    prev = ROOT / "results" / "T1_collision_rescan"
    for f in list(prev.glob("*.json")) + [ROOT / "research" / "RELATED_WORK.md"]:
        try:
            known.update(re.findall(r"\d{4}\.\d{4,5}", f.read_text()))
        except Exception:
            pass
    for it in flagged:
        it["already_known"] = it["arxiv_id"].split("v")[0] in known

    payload = {
        "source": "https://rss.arxiv.org/rss/<cat>",
        "cats": CATS,
        "items_per_cat": per_cat,
        "n_unique_items": len(items),
        "n_flagged": len(flagged),
        "screen": {"domain": DOMAIN.pattern, "method": METHOD.pattern,
                   "physics": PHYS.pattern,
                   "rule": "domain AND (method OR physics)"},
        "limitation": ("RSS carries only the latest announcement cycle, so this covers "
                       "the most recent listing, not an arbitrary date range."),
        "flagged": flagged,
    }
    (out / "arxiv_rss_screen.json").write_text(json.dumps(payload, indent=2,
                                                          ensure_ascii=False))
    print(f"\n{len(items)} unique new items, {len(flagged)} flagged by domain x method\n")
    for it in flagged:
        tag = "KNOWN" if it.get("already_known") else "**NEW**"
        print(f"  [{tag}] {it['arxiv_id']}  {it['title'][:92]}")
        print(f"          {it['abstract'][:200]}")
    print(f"\nwrote {out/'arxiv_rss_screen.json'}")


if __name__ == "__main__":
    main()
