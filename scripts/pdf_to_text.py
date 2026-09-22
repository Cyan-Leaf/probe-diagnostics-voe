"""Extract text from a PDF into a .txt next to it (T1 full-text review helper).

Run: uv run --no-project python scripts/pdf_to_text.py <pdf_path>
"""

import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    args = ap.parse_args()
    src = Path(args.pdf)
    reader = PdfReader(str(src))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        parts.append(f"\n\n===== PAGE {i} =====\n")
        parts.append(page.extract_text() or "")
    out = src.with_suffix(".txt")
    out.write_text("".join(parts))
    print(f"pages={len(reader.pages)} chars={sum(len(p) for p in parts)} -> {out}")


if __name__ == "__main__":
    main()
