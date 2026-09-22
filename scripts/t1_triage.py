"""T1 triage: score arXiv hits for collision risk with the Show Gap proposal.

Run: uv run --no-project python scripts/t1_triage.py
"""

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "T1_collision_rescan"

WINDOW = ("2026-06-16", "2026-09-17")

# collision axes -> (weight, regexes)
AXES = {
    "A_gap_paired": (
        3,
        [
            r"know[\w\- ]{0,12}(but|yet)[\w\- ]{0,12}(not|cannot|fail)",
            r"(discriminat\w+|understanding|perception|recognit\w+)[^.]{0,60}\bgap\b[^.]{0,60}(generat\w+|synthes\w+)",
            r"(generat\w+)[^.]{0,60}\bgap\b[^.]{0,60}(discriminat\w+|understanding|perception)",
            r"knowledge[- ]action",
            r"know(ing)?[- ]doing gap",
            r"generation[- ]verification gap",
            r"understanding[- ]generation gap",
        ],
    ),
    "B_self_probe_guidance": (
        3,
        [
            r"self[- ]guidance",
            r"(internal|intermediate|hidden)[\w ]{0,20}(state|representation|feature)s?[^.]{0,80}guid",
            r"probe[^.]{0,60}(guid|steer|interven|control)",
            r"(guid|steer)[^.]{0,60}\bprobe",
            r"linear probe[^.]{0,80}(sampling|denois|generat)",
            r"activation steering",
            r"representation engineering",
        ],
    ),
    "C_probe_physics": (
        2,
        [
            r"linear(ly)? (probe|decod|read)",
            r"probing[^.]{0,60}(diffusion|video model|world model)",
            r"(physical|physics)[^.]{0,40}(plausibility|realism)[^.]{0,60}(decod|probe|linear)",
        ],
    ),
    "D_inference_time_physics": (
        2,
        [
            r"(inference|test)[- ]time[^.]{0,60}(physic|plausib)",
            r"(reward|verifier)[^.]{0,60}(denois|sampl)[^.]{0,60}(physic|plausib)",
            r"physics[^.]{0,40}(guidance|guided) (sampling|denoising|generation)",
        ],
    ),
    "E_voe_cogsci": (
        1,
        [
            r"violation[- ]of[- ]expectation",
            r"\bVoE\b",
            r"looking[- ]time",
            r"A-not-B",
            r"developmental psycholog",
            r"infant",
        ],
    ),
}

MUST_DOMAIN = re.compile(
    r"video (diffusion|generat|model)|diffusion transformer|world model|text-to-video|"
    r"image-to-video|\bDiT\b|video synthes",
    re.I,
)


def score(p: dict) -> tuple[int, dict]:
    text = f"{p['title']} . {p['summary']}"
    hits = {}
    total = 0
    for axis, (w, pats) in AXES.items():
        found = [pat for pat in pats if re.search(pat, text, re.I)]
        if found:
            hits[axis] = found
            total += w * len(found)
    if MUST_DOMAIN.search(text):
        total += 1
        hits["domain"] = True
    return total, hits


def main() -> None:
    papers = json.loads((OUT / "arxiv_hits.json").read_text())
    scored = []
    for p in papers:
        s, h = score(p)
        p["_score"], p["_hits"] = s, h
        scored.append(p)
    scored.sort(key=lambda p: (-p["_score"], p["published"]))

    in_window = [
        p for p in scored if WINDOW[0] <= p["published"][:10] <= WINDOW[1]
    ]
    print(f"papers total={len(scored)}  in 3-month window={len(in_window)}")

    lines = ["# T1 arXiv triage (auto-scored)\n"]
    for label, subset in (("A. In window (2026-06-16..2026-09-16)", in_window),
                          ("B. All hits (any date), score>=4", [p for p in scored if p["_score"] >= 4])):
        lines.append(f"\n## {label}  — n={len(subset)}\n")
        for p in subset:
            if p["_score"] < 2:
                continue
            axes = ",".join(k for k in p["_hits"] if k != "domain")
            lines.append(
                f"- **[{p['_score']}]** `{p['arxiv_id']}` ({p['published'][:10]}, {p['primary_cat']}) "
                f"[{axes}] {p['title']}\n  - {p['summary'][:400]}...\n"
            )
    (OUT / "triage.md").write_text("\n".join(lines))

    print("\n=== TOP 40 by score (any date) ===")
    for p in scored[:40]:
        axes = ",".join(k for k in p["_hits"] if k != "domain")
        print(f"[{p['_score']:2d}] {p['arxiv_id']:14s} {p['published'][:10]} {axes:45s} {p['title'][:95]}")

    print("\n=== IN-WINDOW, score>=3 ===")
    for p in in_window:
        if p["_score"] >= 3:
            axes = ",".join(k for k in p["_hits"] if k != "domain")
            print(f"[{p['_score']:2d}] {p['arxiv_id']:14s} {p['published'][:10]} {axes:45s} {p['title'][:95]}")

    print(f"\nwrote {OUT/'triage.md'}")


if __name__ == "__main__":
    main()
