"""C1c rescan via OpenAlex, replacing the arXiv API (blocked at the proxy, see
results/T1_rescan_20260917/C1c_STATUS.md).

Reproduces the two search hard-rules from research/RELATED_WORK.md that the arXiv API
normally serves:

  rule 1  mechanical cartesian product of DOMAIN x METHOD terms -- not concept-word
          search.  The two P0 collisions of the previous round were only caught this
          way (`2603.14294` says "verifier"/"reward-gradient guidance", none of the
          concept words).
  rule 2  full paginated pull of the target window, then read every title by hand --
          not top-k by relevance.

OpenAlex is used because it supports both (`from_publication_date` + cursor paging) and
is reachable from this box.  Results are cross-checked against what the previous arXiv
rescan already recorded, so only genuinely new items need triage.

Run:
  python scripts/t1_openalex_scan.py --from 2026-09-16 --to 2026-09-18 \
      --out results/T1_rescan_20260917
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.openalex.org/works"
UA = {"User-Agent": "cogreason-voe-T1/0.1 (mailto:research@example.org)"}

# rule 1: domain x method, expanded mechanically
DOMAIN = [
    "video diffusion", "video generation", "world model", "video foundation model",
    "text-to-video", "diffusion transformer",
]
METHOD = [
    "probe", "probing", "linearly decodable", "linear probe", "internal representation",
    "hidden states", "verifier", "reward guidance", "denoising error", "readout",
    "plausibility", "intuitive physics",
]


def get(url: str, tries: int = 4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def paged(params: dict, cap: int = 2000):
    """rule 2: full pull via cursor paging, not top-k."""
    out, cursor = [], "*"
    while True:
        p = dict(params)
        p["cursor"] = cursor
        p["per-page"] = 200
        d = get(f"{API}?{urllib.parse.urlencode(p)}")
        out.extend(d.get("results", []))
        cursor = d.get("meta", {}).get("next_cursor")
        if not cursor or not d.get("results") or len(out) >= cap:
            break
        time.sleep(0.4)
    return out


def norm(w: dict) -> dict:
    ids = w.get("ids", {}) or {}
    arx = ""
    for loc in (w.get("locations") or []):
        url = ((loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or ""))
        if "arxiv.org/abs/" in url:
            arx = url.split("arxiv.org/abs/")[1].split()[0].strip("v0123456789.").strip()
            arx = url.split("arxiv.org/abs/")[1].split()[0]
            break
    inv = w.get("abstract_inverted_index")
    abstract = ""
    if inv:
        pos = {}
        for word, idxs in inv.items():
            for i in idxs:
                pos[i] = word
        abstract = " ".join(pos[k] for k in sorted(pos))
    return {
        "openalex_id": w.get("id", "").rsplit("/", 1)[-1],
        "arxiv_id": arx,
        "doi": ids.get("doi", ""),
        "title": w.get("title") or "",
        "date": w.get("publication_date", ""),
        "abstract": abstract[:1200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", required=True)
    ap.add_argument("--to", dest="dto", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    seen, log = {}, {}
    combos = [(d, m) for d in DOMAIN for m in METHOD]
    print(f"rule 1: {len(DOMAIN)} domain x {len(METHOD)} method = {len(combos)} queries",
          flush=True)

    for i, (d, m) in enumerate(combos):
        q = f'"{d}" "{m}"'
        params = {
            "search": q,
            "filter": f"from_publication_date:{args.dfrom},to_publication_date:{args.dto}",
        }
        try:
            rows = paged(params)
        except Exception as exc:
            log[q] = f"ERROR {exc}"
            print(f"  [{i+1}/{len(combos)}] {q:52s} ERROR {exc}", flush=True)
            continue
        log[q] = len(rows)
        new = 0
        for w in rows:
            r = norm(w)
            k = r["openalex_id"]
            if k not in seen:
                r["found_by"] = [q]
                seen[k] = r
                new += 1
            else:
                seen[k]["found_by"].append(q)
        if rows:
            print(f"  [{i+1}/{len(combos)}] {q:52s} {len(rows):4d} hits (+{new} new)",
                  flush=True)
        time.sleep(0.3)

    # rule 2: unfiltered window sweep -- catches papers whose abstract uses none of our words
    print("rule 2: full window sweep (concept-free)", flush=True)
    for topic in ("video", "diffusion model", "physics video"):
        params = {
            "search": topic,
            "filter": f"from_publication_date:{args.dfrom},to_publication_date:{args.dto}",
        }
        try:
            rows = paged(params, cap=3000)
        except Exception as exc:
            print(f"  sweep {topic}: ERROR {exc}", flush=True)
            continue
        print(f"  sweep {topic!r}: {len(rows)} works", flush=True)
        for w in rows:
            r = norm(w)
            k = r["openalex_id"]
            if k not in seen:
                r["found_by"] = [f"sweep:{topic}"]
                seen[k] = r
            else:
                seen[k]["found_by"].append(f"sweep:{topic}")
        time.sleep(0.4)

    papers = sorted(seen.values(), key=lambda p: p["date"], reverse=True)

    # cross-check against everything the previous arXiv rescan already knows
    known = set()
    prev = ROOT / "results" / "T1_collision_rescan"
    for f in list(prev.glob("*.json")) + [ROOT / "research" / "RELATED_WORK.md"]:
        try:
            known.update(__import__("re").findall(r"\d{4}\.\d{4,5}", f.read_text()))
        except Exception:
            pass
    for p in papers:
        base = (p["arxiv_id"] or "").split("v")[0]
        p["already_known"] = bool(base and base in known)

    fresh = [p for p in papers if not p["already_known"]]
    (out / "openalex_hits.json").write_text(json.dumps(papers, indent=2, ensure_ascii=False))
    (out / "openalex_query_log.json").write_text(json.dumps(
        {"window": [args.dfrom, args.dto], "n_queries": len(combos),
         "per_query": log, "n_unique": len(papers), "n_new": len(fresh)},
        indent=2, ensure_ascii=False))

    print(f"\nunique works in window: {len(papers)}")
    print(f"not already in T1 records: {len(fresh)}")
    for p in fresh[:60]:
        print(f"  {p['date']}  {(p['arxiv_id'] or p['doi'])[:28]:28s} {p['title'][:88]}")
    print(f"\nwrote {out/'openalex_hits.json'}")


if __name__ == "__main__":
    main()
