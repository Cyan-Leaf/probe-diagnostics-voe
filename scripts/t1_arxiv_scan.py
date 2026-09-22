"""T1 collision rescan: query the arXiv API and dump raw + parsed results.

Run: uv run --no-project python scripts/t1_arxiv_scan.py
Requires http_proxy/https_proxy to be exported (dev box has no direct egress).
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "T1_collision_rescan"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

WINDOW = "submittedDate:[202606160000+TO+202609170000]"

QUERIES = {
    "q01_videodiff_physics_window": f'cat:cs.CV+AND+abs:%22video+diffusion%22+AND+abs:physics+AND+{WINDOW}',
    "q02_physics_probe_window": f'cat:cs.CV+AND+abs:physics+AND+%28abs:probe+OR+abs:probing%29+AND+{WINDOW}',
    "q03_videogen_physics_guidance_window": f'cat:cs.CV+AND+abs:%22video+generation%22+AND+abs:physics+AND+abs:guidance+AND+{WINDOW}',
    "q04_intuitive_physics_window": f'cat:cs.CV+AND+abs:%22intuitive+physics%22+AND+{WINDOW}',
    "q05_worldmodel_probe_window": f'cat:cs.CV+AND+abs:%22world+model%22+AND+%28abs:probe+OR+abs:probing%29+AND+{WINDOW}',
    "q06_selfguidance_video": 'abs:%22self-guidance%22+AND+abs:video',
    "q07_plausibility_linear_probe": 'all:%22physical+plausibility%22+AND+all:%22linear+probe%22',
    "q08_knowledge_action_gap": 'abs:%22knowledge-action%22+OR+abs:%22know-do+gap%22+OR+abs:%22knowing-doing+gap%22',
    "q09_internal_repr_videodiff": 'abs:%22video+diffusion%22+AND+%28abs:%22internal+representation%22+OR+abs:%22hidden+states%22+OR+abs:%22intermediate+representation%22%29',
    "q10_probe_guided_sampling": 'abs:%22probe%22+AND+%28abs:%22guided+sampling%22+OR+abs:%22guidance%22%29+AND+abs:diffusion+AND+cat:cs.CV',
    "q11_discrim_gen_gap_video": 'abs:%22generation%22+AND+abs:%22discriminative%22+AND+abs:gap+AND+abs:video+AND+cat:cs.CV',
    "q12_voe_video_model": 'abs:%22violation+of+expectation%22+AND+cat:cs.CV',
    "q13_physics_video_window_broad": f'cat:cs.CV+AND+abs:physics+AND+abs:video+AND+{WINDOW}',
    "q14_inference_time_physics": 'abs:%22inference-time%22+AND+abs:physics+AND+abs:video',
    "q15_linear_decodable_diffusion": 'abs:%22linearly+decodable%22+OR+abs:%22linear+decodability%22',
}

KNOWN = {
    "LikePhys": "2510.11512",
    "WMReward": "2601.10553",
    "InvisibleHand": "2606.05328",
}


def fetch(query: str, max_results: int = 200) -> str:
    url = (
        f"{API}?search_query={query}&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "cogreason-voe-T1/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def parse(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        def g(tag):
            node = e.find(f"a:{tag}", NS)
            return (node.text or "").strip() if node is not None else ""

        aid = g("id")
        out.append(
            {
                "arxiv_id": aid.rsplit("/", 1)[-1],
                "title": re.sub(r"\s+", " ", g("title")),
                "published": g("published"),
                "updated": g("updated"),
                "summary": re.sub(r"\s+", " ", g("summary")),
                "authors": [
                    (a.find("a:name", NS).text or "").strip()
                    for a in e.findall("a:author", NS)
                ],
                "primary_cat": (
                    e.find("{http://arxiv.org/schemas/atom}primary_category").get("term")
                    if e.find("{http://arxiv.org/schemas/atom}primary_category") is not None
                    else ""
                ),
                "link": aid,
            }
        )
    return out


def main() -> None:
    import argparse

    global OUT, RAW

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to results/T1_collision_rescan. Monthly "
                         "rescans MUST pass a new dir -- the original is delivered "
                         "Preliminary results and is read-only (HANDOFF rule 2).")
    ap.add_argument("--window", default=None,
                    help="submittedDate window YYYYMMDD-YYYYMMDD, e.g. 20260916-20260918")
    args = ap.parse_args()

    queries = dict(QUERIES)
    if args.out:
        OUT = Path(args.out)
        RAW = OUT / "raw"
        RAW.mkdir(parents=True, exist_ok=True)
    if args.window:
        lo, hi = args.window.split("-")
        new_window = f"submittedDate:[{lo}0000+TO+{hi}0000]"
        queries = {k: v.replace(WINDOW, new_window) for k, v in queries.items()}
        print(f"window -> {new_window}", flush=True)
    print(f"out -> {OUT}", flush=True)

    assert os.environ.get("https_proxy"), "export the oversea proxy first"
    all_papers: dict[str, dict] = {}
    log = {"_window": args.window or WINDOW, "_out": str(OUT)}
    for name, q in queries.items():
        try:
            xml_text = fetch(q)
        except Exception as exc:  # noqa: BLE001
            log[name] = f"ERROR {exc}"
            print(f"[{name}] ERROR {exc}", flush=True)
            time.sleep(3)
            continue
        (RAW / f"{name}.atom").write_text(xml_text)
        papers = parse(xml_text)
        log[name] = {"query": urllib.parse.unquote(q), "n": len(papers)}
        print(f"[{name}] {len(papers)} hits", flush=True)
        for p in papers:
            key = p["arxiv_id"].split("v")[0]
            p.setdefault("found_by", [])
            if key in all_papers:
                all_papers[key]["found_by"].append(name)
            else:
                p["found_by"] = [name]
                all_papers[key] = p
        time.sleep(3)

    for tag, aid in KNOWN.items():
        try:
            xml_text = fetch(f"id_list=&search_query=all:{aid}")
            papers = parse(xml_text)
        except Exception as exc:  # noqa: BLE001
            papers = []
            print(f"[known:{tag}] ERROR {exc}", flush=True)
        log[f"known_{tag}"] = [
            {"id": p["arxiv_id"], "title": p["title"], "updated": p["updated"]}
            for p in papers
        ]
        print(f"[known:{tag}] {[p['arxiv_id'] for p in papers]}", flush=True)
        time.sleep(3)

    (OUT / "arxiv_query_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    papers_sorted = sorted(all_papers.values(), key=lambda p: p["published"], reverse=True)
    (OUT / "arxiv_hits.json").write_text(
        json.dumps(papers_sorted, indent=2, ensure_ascii=False)
    )
    print(f"\ntotal unique papers: {len(papers_sorted)} -> {OUT/'arxiv_hits.json'}")


if __name__ == "__main__":
    main()
