"""T14.2(a): assemble the public release tree.

The release is a *copy*, never a move: `results/` and the caches stay untouched (hard rule),
and the public tree is built next to them so the two can be diffed.

What decides inclusion:
  - code and configs: yes
  - result JSONs: yes, but manifests are rewritten because they embed absolute paths of the
    dev box for every clip (2064 + 1840 hits in the scan)
  - fetched paper HTML/PDF: no -- third-party copyrighted text, and it is what produced the
    297 false-positive "IP addresses" in the scan
  - .pt / .npy / videos: no
  - documents that describe the internal machine (MACHINE_HANDOFF, T1/T2 reports with the
    proxy host): sanitised, with the removal recorded rather than silent

Run: python scripts/t14_build_release.py --out release
"""

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- what to copy ----------------------------------------------------------------------
CODE = ["scripts", "humaneval/serve.py", "humaneval/serve.sh", "humaneval/index.html",
        "humaneval/transcode.py", "humaneval/README.md"]
DOCS = ["SPEC.md", "PREREG.md", "PROTOCOL_VALIDITY.md", "EXPERIMENTS.md",
        "research/RELATED_WORK.md", "handoff/PITFALLS.md"]
RESULT_GLOBS = ["results/**/*.json", "results/**/*.md", "results/T13_figures/*.png",
                "results/T13_figures/*.pdf"]

# --- what never ships ------------------------------------------------------------------
EXCLUDE_DIRS = {"raw", "superseded", "export", "clips", "filled"}
EXCLUDE_SUFFIX = {".pt", ".npy", ".mp4", ".html", ".pdf", ".db", ".log", ".bak"}
# The sensitive-info scan is itself a catalogue of every internal path in the repo, and the
# rating answer key would invalidate the human-eval package. Neither ships.
EXCLUDE_NAMES = {"ANSWER_KEY.json", "T14_sensitive_scan.json"}
KEEP_PDF_UNDER = {"results/T13_figures"}      # our own figures are fine

# --- sanitisation ----------------------------------------------------------------------
SUBS = [
    (re.compile(r"$REPO"), "$REPO"),
    (re.compile(r"$HF_HOME"), "$HF_HOME"),
    (re.compile(r"$WORKDIR"), "$WORKDIR"),
    (re.compile(r"$WORKDIR[\w/.-]*"), "$WORKDIR"),
    (re.compile(r"aiplatform-[\w.-]+"), "<dev-host>"),
    (re.compile(r"<http-proxy>\d*\.[\w.]+:\d+"), "<http-proxy>"),
    (re.compile(r"<http-proxy>\d*[\w.-]*"), "<http-proxy>"),
    (re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<internal-ip>"),
    (re.compile(r"[\w-]+\.idchb\d\w*\.hb\d\.kwaidc\.com"), "<dev-host>"),
    (re.compile(r"\b[\w.-]*corp\.kuaishou\.com\b"), "<corp-domain>"),
    (re.compile(r"\b[\w.-]*\.txyun\b"), "<internal-domain>"),
    (re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
     "GPU-<uuid>"),
    (re.compile(r"\b0000:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d\b"), "<pci-addr>"),
    (re.compile(r"<user>\w*"), "<user>"),
    (re.compile(r"/home/[\w.-]+"), "$HOME"),
    (re.compile(r"/root/(?!GSB)[\w./-]+"), "$HOME"),
]
TEXT_SUFFIX = {".md", ".py", ".sh", ".json", ".txt", ".yaml", ".yml", ".toml", ".html"}


def sanitise(text: str):
    n = 0
    for rx, rep in SUBS:
        text, k = rx.subn(rep, text)
        n += k
    return text, n


def skip(rel: Path) -> bool:
    if any(p in EXCLUDE_DIRS for p in rel.parts):
        return True
    if rel.name in EXCLUDE_NAMES:
        return True
    if rel.suffix.lower() in EXCLUDE_SUFFIX:
        if rel.suffix.lower() == ".pdf" and any(
                str(rel).startswith(k) for k in KEEP_PDF_UNDER):
            return False
        return True
    return False


def rewrite_manifest(obj):
    """Manifests carry one absolute path per clip. Replace with a repo-relative key plus the
    dataset id, so a user can rebuild them with the download script instead of receiving our
    filesystem layout."""
    if isinstance(obj, list):
        return [rewrite_manifest(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "path" and isinstance(v, str):
                out[k] = "$DATA/" + "/".join(Path(v).parts[-3:])
            else:
                out[k] = rewrite_manifest(v)
        return out
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release")
    args = ap.parse_args()
    dst_root = ROOT / args.out
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    copied, sanitised_files, total_subs, skipped = [], [], 0, []

    def put(src: Path, rel: Path):
        nonlocal total_subs
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in TEXT_SUFFIX:
            text = src.read_text(errors="ignore")
            if src.name.startswith("manifest") and src.suffix == ".json":
                try:
                    text = json.dumps(rewrite_manifest(json.loads(text)),
                                      indent=1, ensure_ascii=False)
                except Exception:
                    pass
            text, n = sanitise(text)
            if n:
                sanitised_files.append({"file": str(rel), "substitutions": n})
                total_subs += n
            out.write_text(text)
        else:
            shutil.copy2(src, out)
        copied.append(str(rel))

    for item in CODE + DOCS:
        p = ROOT / item
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not skip(f.relative_to(ROOT)):
                    put(f, f.relative_to(ROOT))
        elif p.is_file():
            put(p, p.relative_to(ROOT))

    for pat in RESULT_GLOBS:
        for f in sorted(ROOT.glob(pat)):
            rel = f.relative_to(ROOT)
            if skip(rel):
                skipped.append(str(rel))
                continue
            put(f, rel)

    report = {
        "n_copied": len(copied),
        "n_files_sanitised": len(sanitised_files),
        "n_substitutions": total_subs,
        "sanitised": sorted(sanitised_files, key=lambda x: -x["substitutions"])[:25],
        "excluded_dirs": sorted(EXCLUDE_DIRS),
        "excluded_suffixes": sorted(EXCLUDE_SUFFIX),
        "excluded_names": sorted(EXCLUDE_NAMES),
        "placeholders": {"$REPO": "repository root", "$WORKDIR": "parent working dir",
                         "$HF_HOME": "HuggingFace cache", "$DATA": "dataset root",
                         "<dev-host>": "development machine hostname",
                         "<http-proxy>": "outbound HTTP proxy",
                         "<internal-ip>": "internal IP", "<user>": "OS user"},
    }
    (dst_root / "RELEASE_MANIFEST.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    print(f"copied {len(copied)} files into {dst_root.relative_to(ROOT)}/")
    print(f"sanitised {len(sanitised_files)} files, {total_subs} substitutions")
    print("\ntop sanitised:")
    for s in report["sanitised"][:10]:
        print(f"  {s['substitutions']:5d}  {s['file']}")
    print(f"\nwrote {dst_root.relative_to(ROOT)}/RELEASE_MANIFEST.json")


if __name__ == "__main__":
    main()
