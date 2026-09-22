"""T1 follow-up: paginate the broad in-window query and pull abstracts for named candidates.

Run: uv run --no-project python scripts/t1_arxiv_followup.py
"""

import json
import time
import urllib.request
from pathlib import Path

from t1_arxiv_scan import RAW, parse  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "T1_collision_rescan"
API = "https://export.arxiv.org/api/query"

WINDOW = "submittedDate:[202606160000+TO+202609170000]"
BROAD = [
    ("w1_physics_video", f"cat:cs.CV+AND+abs:physics+AND+abs:video+AND+{WINDOW}"),
    ("w2_videogen_any", f"cat:cs.CV+AND+abs:%22video+generation%22+AND+{WINDOW}"),
    ("w3_videodiffusion_any", f"cat:cs.CV+AND+abs:%22video+diffusion%22+AND+{WINDOW}"),
    ("w4_worldmodel_any", f"cat:cs.CV+AND+abs:%22world+model%22+AND+{WINDOW}"),
]

CANDIDATES = [
    "2606.05328",  # Invisible Hand (our foundation)
    "2510.11512",  # LikePhys
    "2601.10553",  # WMReward
    "2602.12218",  # Observer Effect in World Models
    "2603.17825",  # Steering Video DiT with Massive Activations
    "2512.13290",  # LINA interventions physical alignment
    "2609.08250",  # CALIPER
    "2608.07077",  # Transformers struggle to use emergent world models
    "2606.00499",  # OptiWorld
    "2601.18577",  # Self-Refining Video Sampling
    "2603.13770",  # PhysAlign
    "2603.19607",  # Physion-Eval
    "2605.14815",  # Probing camera control
    "2607.01869",  # QWERTY
    "2412.05827",  # Self-Guidance (image)
    "2602.13294",  # VisPhyWorld
]


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
    all_papers: dict[str, dict] = {}
    for name, q in BROAD:
        for start in range(0, 600, 100):
            url = (
                f"{API}?search_query={q}&start={start}&max_results=100"
                "&sortBy=submittedDate&sortOrder=descending"
            )
            xml_text = get(url)
            (RAW / f"{name}_s{start}.atom").write_text(xml_text)
            got = parse(xml_text)
            print(f"[{name}] start={start} -> {len(got)}", flush=True)
            for p in got:
                all_papers.setdefault(p["arxiv_id"].split("v")[0], p)
            time.sleep(4)
            if len(got) < 100:
                break
    (OUT / "arxiv_window_broad.json").write_text(
        json.dumps(sorted(all_papers.values(), key=lambda p: p["published"], reverse=True),
                   indent=2, ensure_ascii=False)
    )
    print(f"broad in-window unique: {len(all_papers)}")

    ids = ",".join(CANDIDATES)
    xml_text = get(f"{API}?id_list={ids}&max_results=100")
    (RAW / "candidates.atom").write_text(xml_text)
    cands = parse(xml_text)
    (OUT / "arxiv_candidates.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False))
    print(f"candidates fetched: {len(cands)}")
    for c in cands:
        print(f"\n### {c['arxiv_id']}  ({c['published'][:10]} / upd {c['updated'][:10]} / {c['primary_cat']})")
        print(f"    {c['title']}")
        print(f"    {c['summary'][:1200]}")


if __name__ == "__main__":
    main()
