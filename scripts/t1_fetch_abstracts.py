"""Fetch arXiv abstracts for an explicit id list (T1 manual-review helper).

Run: uv run --no-project python scripts/t1_fetch_abstracts.py <id> [<id> ...] --tag batch2
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from t1_arxiv_scan import RAW, parse

OUT = Path(__file__).resolve().parents[1] / "results" / "T1_collision_rescan"
API = "https://export.arxiv.org/api/query"


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "cogreason-voe-T1/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {attempt} after {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--tag", default="batch")
    ap.add_argument("--chars", type=int, default=1100)
    args = ap.parse_args()

    xml_text = get(f"{API}?id_list={','.join(args.ids)}&max_results=200")
    (RAW / f"{args.tag}.atom").write_text(xml_text)
    papers = parse(xml_text)
    (OUT / f"arxiv_{args.tag}.json").write_text(json.dumps(papers, indent=2, ensure_ascii=False))
    for p in papers:
        print(f"\n### {p['arxiv_id']} ({p['published'][:10]} / upd {p['updated'][:10]} / {p['primary_cat']})")
        print(f"    {p['title']}")
        print(f"    {p['summary'][:args.chars]}")
    print(f"\nfetched {len(papers)} -> results/T1_collision_rescan/arxiv_{args.tag}.json")


if __name__ == "__main__":
    main()
