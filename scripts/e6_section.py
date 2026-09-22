"""E6 helper: pull a named section, or keyword context, out of a fetched paper text.

The audit has to cite where in the paper a claim lives, so grepping ad hoc is not enough --
this returns the section span so the D1-D6 table can carry a locator.

Run:
  python scripts/e6_section.py 2606.09646 --section "4.1 IntPhys2" --chars 3000
  python scripts/e6_section.py 2606.09646 --grep "random init" --ctx 400
  python scripts/e6_section.py 2606.09646 --toc
"""

import argparse
import re
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "results" / "E6_literature" / "raw"

# arXiv HTML renderings start with a nav TOC that repeats every heading; the body copy is
# the *second* occurrence, so section lookup deliberately searches from the end.
HEAD = re.compile(r"^\s*((?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+[A-Z][^\n]{0,90})\s*$", re.M)


def load(pid: str) -> str:
    p = RAW / f"{pid}.txt"
    if not p.exists():
        raise SystemExit(f"no text for {pid}; run scripts/e6_fetch.py first")
    return p.read_text(errors="ignore")


def toc(text: str):
    seen = []
    for m in HEAD.finditer(text):
        h = re.sub(r"\s+", " ", m.group(1)).strip()
        if h not in seen:
            seen.append(h)
    return seen


def section(text: str, name: str, chars: int) -> str:
    """Last occurrence of the heading, then run to the next heading of same-or-higher level."""
    pat = re.compile(r"^\s*" + re.escape(name) + r"\s*$", re.M)
    hits = list(pat.finditer(text))
    if not hits:
        pat = re.compile(re.escape(name), re.I)
        hits = list(pat.finditer(text))
    if not hits:
        return ""
    start = hits[-1].end()
    nxt = HEAD.search(text, start + 50)
    end = nxt.start() if nxt else len(text)
    return text[start:min(end, start + chars)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--section")
    ap.add_argument("--grep")
    ap.add_argument("--ctx", type=int, default=350)
    ap.add_argument("--chars", type=int, default=2500)
    ap.add_argument("--toc", action="store_true")
    ap.add_argument("--max-hits", type=int, default=8)
    args = ap.parse_args()

    text = load(args.pid)

    if args.toc:
        for h in toc(text):
            print("  ", h)
        return

    if args.section:
        body = section(text, args.section, args.chars)
        if not body:
            print(f"[section {args.section!r} not found]")
        else:
            print(re.sub(r"\n{3,}", "\n\n", body).strip())
        return

    if args.grep:
        n = 0
        for m in re.finditer(args.grep, text, re.I):
            a, b = max(0, m.start() - args.ctx), min(len(text), m.end() + args.ctx)
            print(f"--- hit @{m.start()} ---")
            print(re.sub(r"\s+", " ", text[a:b]))
            print()
            n += 1
            if n >= args.max_hits:
                break
        if n == 0:
            print("[no hits]")


if __name__ == "__main__":
    main()
