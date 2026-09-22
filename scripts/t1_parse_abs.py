"""Parse arXiv /abs pages saved by T1 for venue/comments/code metadata."""

import html
import re
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "results" / "T1_collision_rescan" / "raw"


def grab(text: str, pat: str) -> str:
    m = re.search(pat, text, re.S)
    if not m:
        return "-"
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", m.group(1)))).strip()


def main() -> None:
    for f in sorted(RAW.glob("abs_*.html")):
        t = f.read_text(errors="ignore")
        print(f"\n=== {f.name}")
        print("  title    :", grab(t, r'class="title mathjax">(.*?)</h1>'))
        print("  authors  :", grab(t, r'class="authors">(.*?)</div>')[:400])
        print("  subjects :", grab(t, r'class="tablecell subjects">(.*?)</td>'))
        print("  comments :", grab(t, r'class="tablecell comments[^"]*">(.*?)</td>'))
        print("  journal  :", grab(t, r'class="tablecell jref">(.*?)</td>'))
        print("  versions :", grab(t, r'(<b>\[v1\].*?)</div>')[:300])
        gh = sorted(set(re.findall(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", t)))
        print("  github   :", gh[:8])


if __name__ == "__main__":
    main()
