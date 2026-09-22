"""E6: fetch the full text of the four papers the diagnostic table is built from.

The arXiv *API* is still 406 through this proxy (verified again 2026-09-18), but the
paper-serving routes `arxiv.org/abs`, `/html` and `/pdf` are reachable.  E6's hard rule is
"only the original text, otherwise write 无法核实", so the whole audit depends on getting
these files -- hence a dedicated fetch step with the retrieval status recorded per paper.

Saves to results/E6_literature/raw/:
    <id>.abs.html    the abstract page (metadata, version history)
    <id>.html        the full HTML rendering, when arXiv has one
    <id>.pdf         the PDF, as the fallback and as the citable artefact
    <id>.txt         plain text extracted from whichever source worked

Run: python scripts/e6_fetch.py
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "E6_literature" / "raw"
UA = {"User-Agent": "cogreason-voe-E6/0.1"}

PAPERS = {
    "2606.05328": "Invisible Hand of Physics (When Video Diffusion Models Know More Than They Show)",
    "2603.14294": "Seeking Physics in Diffusion Noise",
    "2606.09646": "Punzo et al. (IntPhys2 + MVP probing)",
    "2510.11512": "LikePhys",
}


def get(url, timeout=90):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
    return r.read()


def html_to_text(raw: bytes) -> str:
    import html as htmlmod
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    # keep block structure so section headings survive
    s = re.sub(r"</(p|div|h1|h2|h3|h4|li|tr|section)>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmlmod.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, 1):
        out.append(f"\n\n===== PAGE {i} =====\n")
        out.append(page.extract_text() or "")
    return "".join(out)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    status = {}

    for pid, title in PAPERS.items():
        st = {"title": title, "sources": {}}
        print(f"=== {pid}  {title}", flush=True)

        # abstract page: gives the version list, needed for RELATED_WORK rule 4
        try:
            b = get(f"https://arxiv.org/abs/{pid}")
            (RAW / f"{pid}.abs.html").write_bytes(b)
            txt = html_to_text(b)
            vers = re.findall(r"\[v(\d+)\]\s*([A-Za-z]{3},\s*\d+\s+\w+\s+\d{4})", txt)
            st["sources"]["abs"] = {"ok": True, "bytes": len(b),
                                   "versions": [f"v{v} {d}" for v, d in vers]}
            print(f"  abs  ok {len(b)}B  versions={st['sources']['abs']['versions']}",
                  flush=True)
        except Exception as exc:
            st["sources"]["abs"] = {"ok": False, "error": str(exc)}
            print(f"  abs  FAIL {exc}", flush=True)
        time.sleep(1)

        # full HTML (best for section-level citation)
        text = None
        for suffix in ("", "v1", "v2", "v3", "v4"):
            try:
                b = get(f"https://arxiv.org/html/{pid}{suffix}")
                if len(b) < 20000:
                    continue
                (RAW / f"{pid}.html").write_bytes(b)
                text = html_to_text(b)
                st["sources"]["html"] = {"ok": True, "bytes": len(b),
                                        "variant": suffix or "latest",
                                        "text_chars": len(text)}
                print(f"  html ok {len(b)}B ({suffix or 'latest'}) -> {len(text)} chars",
                      flush=True)
                break
            except Exception:
                continue
        if text is None:
            st["sources"]["html"] = {"ok": False}
            print("  html unavailable", flush=True)
        time.sleep(1)

        # PDF: fallback text source and the artefact to cite pages from
        local_pdf = ROOT / "results/T1_collision_rescan/raw" / f"{pid}v3.pdf"
        dst_pdf = RAW / f"{pid}.pdf"
        try:
            if local_pdf.exists():
                dst_pdf.write_bytes(local_pdf.read_bytes())
                st["sources"]["pdf"] = {"ok": True, "from": "local T1 copy",
                                       "bytes": dst_pdf.stat().st_size}
            else:
                b = get(f"https://arxiv.org/pdf/{pid}")
                dst_pdf.write_bytes(b)
                st["sources"]["pdf"] = {"ok": True, "from": "arxiv", "bytes": len(b)}
            print(f"  pdf  ok {st['sources']['pdf']['bytes']}B "
                  f"({st['sources']['pdf']['from']})", flush=True)
            if text is None:
                text = pdf_to_text(dst_pdf)
                st["sources"]["pdf"]["text_chars"] = len(text)
                print(f"       -> pdf text {len(text)} chars", flush=True)
        except Exception as exc:
            st["sources"]["pdf"] = {"ok": False, "error": str(exc)}
            print(f"  pdf  FAIL {exc}", flush=True)
        time.sleep(1)

        if text:
            (RAW / f"{pid}.txt").write_text(text)
            st["text_source"] = ("html" if st["sources"].get("html", {}).get("ok")
                                 else "pdf")
            st["text_chars"] = len(text)
            st["auditable"] = True
        else:
            st["auditable"] = False
        status[pid] = st

    (RAW.parent / "fetch_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False))
    print("\n=== summary ===")
    for pid, st in status.items():
        print(f"  {pid}  auditable={st['auditable']:<5} "
              f"source={st.get('text_source','-'):5s} chars={st.get('text_chars','-')}")
    print(f"\nwrote {RAW.parent/'fetch_status.json'}")


if __name__ == "__main__":
    main()
