"""Dump readable text + outbound links from a saved HTML page (T1 helper)."""

import argparse
import html
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--chars", type=int, default=2000)
    args = ap.parse_args()
    raw = Path(args.path).read_text(errors="ignore")
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))
    print(text[: args.chars])
    print("\n=== links ===")
    for link in sorted(set(re.findall(r'href="([^"]+)"', raw))):
        if link.startswith("http") or "git" in link.lower() or link.endswith((".zip", ".pdf")):
            print("  ", link)


if __name__ == "__main__":
    main()
