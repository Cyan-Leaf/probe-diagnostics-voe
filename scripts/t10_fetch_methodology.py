"""T10.4: fetch the three probing-methodology papers so the genealogy claims can be
verified against the originals (hard rule 4: original text only, else 无法核实).

The claim that needs the most care is the Voita & Titov quote, which PROTOCOL_VALIDITY.md
and RELATED_WORK.md §0.5 both lean on:

    probe accuracy "does not substantially favour pretrained representations over
    randomly initialized ones"

If that sentence is not in the paper as written, or means something narrower in context,
the D1 genealogy claim has to be rewritten.  So this fetches the full text and the audit
step quotes the surrounding paragraph rather than the fragment.

arXiv ids:
  1909.03368  Hewitt & Liang, EMNLP 2019, Designing and Interpreting Probes with Control Tasks
  2003.12298  Voita & Titov, EMNLP 2020, Information-Theoretic Probing with MDL
  2006.00995  Elazar et al., TACL 2020, Amnesic Probing

Run: python scripts/t10_fetch_methodology.py
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "T10_validity" / "raw"
UA = {"User-Agent": "cogreason-voe-T10/0.1"}

PAPERS = {
    "1909.03368": ("Hewitt & Liang 2019", "Designing and Interpreting Probes with Control Tasks"),
    "2003.12298": ("Voita & Titov 2020", "Information-Theoretic Probing with MDL"),
    "2006.00995": ("Elazar et al. 2020", "Amnesic Probing"),
}


def get(url, timeout=90):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=timeout).read()


def html_to_text(raw: bytes) -> str:
    import html as h
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"</(p|div|h1|h2|h3|h4|li|tr|section)>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = h.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", s).strip()


def pdf_to_text(p: Path) -> str:
    from pypdf import PdfReader
    r = PdfReader(str(p))
    out = []
    for i, pg in enumerate(r.pages, 1):
        out.append(f"\n\n===== PAGE {i} =====\n")
        out.append(pg.extract_text() or "")
    return "".join(out)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    status = {}
    for pid, (short, title) in PAPERS.items():
        st = {"short": short, "title": title, "sources": {}}
        print(f"=== {pid}  {short}: {title}", flush=True)

        try:
            b = get(f"https://arxiv.org/abs/{pid}")
            (RAW / f"{pid}.abs.html").write_bytes(b)
            txt = html_to_text(b)
            vers = re.findall(r"\[v(\d+)\]\s*([A-Za-z]{3},\s*\d+\s+\w+\s+\d{4})", txt)
            st["sources"]["abs"] = {"ok": True, "versions": [f"v{v} {d}" for v, d in vers]}
            print(f"  abs  ok  versions={st['sources']['abs']['versions']}", flush=True)
        except Exception as exc:
            st["sources"]["abs"] = {"ok": False, "error": str(exc)}
            print(f"  abs  FAIL {exc}", flush=True)
        time.sleep(1)

        text = None
        for suf in ("", "v1", "v2", "v3"):
            try:
                b = get(f"https://arxiv.org/html/{pid}{suf}")
                if len(b) < 20000:
                    continue
                (RAW / f"{pid}.html").write_bytes(b)
                text = html_to_text(b)
                st["sources"]["html"] = {"ok": True, "bytes": len(b),
                                         "variant": suf or "latest",
                                         "text_chars": len(text)}
                print(f"  html ok {len(b)}B -> {len(text)} chars", flush=True)
                break
            except Exception:
                continue
        if "html" not in st["sources"]:
            st["sources"]["html"] = {"ok": False}
            print("  html unavailable (older papers often have no arXiv HTML)", flush=True)
        time.sleep(1)

        try:
            b = get(f"https://arxiv.org/pdf/{pid}")
            p = RAW / f"{pid}.pdf"
            p.write_bytes(b)
            st["sources"]["pdf"] = {"ok": True, "bytes": len(b)}
            print(f"  pdf  ok {len(b)}B", flush=True)
            if text is None:
                text = pdf_to_text(p)
                st["sources"]["pdf"]["text_chars"] = len(text)
                print(f"       -> pdf text {len(text)} chars", flush=True)
        except Exception as exc:
            st["sources"]["pdf"] = {"ok": False, "error": str(exc)}
            print(f"  pdf  FAIL {exc}", flush=True)
        time.sleep(1)

        if text:
            (RAW / f"{pid}.txt").write_text(text)
            st["auditable"] = True
            st["text_chars"] = len(text)
            st["text_source"] = "html" if st["sources"]["html"].get("ok") else "pdf"
        else:
            st["auditable"] = False
        status[pid] = st

    (RAW.parent / "methodology_fetch_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False))
    print("\n=== summary ===")
    for pid, st in status.items():
        print(f"  {pid} {st['short']:22s} auditable={st['auditable']} "
              f"src={st.get('text_source','-')} chars={st.get('text_chars','-')}")


if __name__ == "__main__":
    main()
