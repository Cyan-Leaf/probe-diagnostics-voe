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
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- what to copy ----------------------------------------------------------------------
CODE = ["scripts", "humaneval/serve.py", "humaneval/serve.sh", "humaneval/index.html",
        "humaneval/transcode.py", "humaneval/README.md",
        # The smoke path and its dependency list are the entry point of the release; leaving
        # them out of this list silently DELETED them from the published tree on rebuild
        # (found 2026-11-05: repro/l0_smoke.py, requirements.txt and .gitignore were all being
        # removed). Anything tracked in the release repo must appear here or under DOCS.
        "repro/l0_smoke.py", "T14_REPRO.md", "PROPOSAL_v1.0_en.md"]
DOCS = ["SPEC.md", "PREREG.md", "PROTOCOL_VALIDITY.md", "EXPERIMENTS.md",
        "research/RELATED_WORK.md", "handoff/PITFALLS.md",
        # Added 2026-11-05: the two new tasks' preregistrations are part of the argument
        # (they fix the criteria before the runs), so they ship with the code.
        "results/E20_bootstrap_ci/E20_PREREG.md",
        "results/E21_physics_iq_real/E21_PREREG.md"]
RESULT_GLOBS = ["results/**/*.json", "results/**/*.md", "results/T13_figures/*.png",
                "results/T13_figures/*.pdf"]

# --- what never ships ------------------------------------------------------------------
# "__pycache__" is not cosmetic: .pyc files embed the absolute source path of the build box,
# so shipping them leaked $WORKDIR through 33 files that no text scan of the
# .py sources would ever have caught (found 2026-11-05 by grepping the built tree, not the source).
EXCLUDE_DIRS = {"raw", "superseded", "export", "clips", "filled", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_SUFFIX = {".pt", ".npy", ".mp4", ".html", ".pdf", ".db", ".log", ".bak",
                  ".pyc", ".pyo"}
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
    # `release/` is a checked-out git repository, so a blind rmtree deletes `.git` and with it
    # the history of the already-published release. Build into a staging directory instead and
    # swap the contents in, leaving `.git` untouched.
    staged = dst_root.with_name(dst_root.name + ".staging")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    # Some files exist ONLY in the published tree: README.md (written for the public audience,
    # not copied from the repo root), .gitignore, requirements.txt, the submitted proposal and
    # RELEASE_MANIFEST.json. They are tracked in git but have no source file here, so a rebuild
    # that mirrors only the repo would delete them — which is exactly what happened once.
    # Carry them forward instead of relying on anyone remembering.
    # The authoritative copies now live in `release_src/` in the working repo, so the public
    # README is editable and version-controlled instead of existing only inside the published
    # tree. When a source copy is missing, fall back to the published file so an older checkout
    # does not lose it.
    SRC = {"README.md": "release_src/README.md",
           "requirements.txt": "release_src/requirements.txt",
           ".gitignore": "release_src/gitignore.txt"}
    RELEASE_ONLY = {".gitignore", "README.md", "requirements.txt",
                    "PROPOSAL_v1.0_zh.md", "PROPOSAL_v1.0_zh_submitted.md",
                    "RELEASE_MANIFEST.json"}
    if dst_root.exists():
        for entry in dst_root.iterdir():
            if entry.name == ".git":
                continue
            if entry.name in RELEASE_ONLY and entry.is_file():
                shutil.copy2(entry, staged / entry.name)
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    for name, src in SRC.items():
        if (ROOT / src).exists():
            shutil.copy2(ROOT / src, staged / name)
    dst_root = staged

    # Carry-forward files count as copied, otherwise the integrity guard reports a false loss
    # for exactly the files it just preserved.
    copied, sanitised_files, total_subs, skipped = sorted(RELEASE_ONLY), [], 0, []

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

    # Swap the freshly built tree into place, keeping the git metadata that lives in it.
    final = ROOT / args.out
    for entry in sorted(dst_root.iterdir()):
        target = final / entry.name
        if entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(entry), str(target))
        else:
            shutil.move(str(entry), str(target))
    shutil.rmtree(dst_root)

    # Regression guard: the rebuild DELETES everything not in CODE/DOCS/RESULT_GLOBS, so a
    # file dropped from those lists disappears from the public repo without any error. Check
    # the previous commit's tracked files against what exists now and fail loudly on loss.
    try:
        # Against HEAD, not the index: the rebuild has already replaced the working tree by
        # the time this runs, so `git ls-files` would report the NEW files and never notice a loss.
        prev = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=final,
                              capture_output=True, text=True, timeout=30).stdout.split()
    except Exception:
        prev = []
    present = set(copied)
    lost = sorted(set(prev) - present)
    if lost:
        print(f"\n!! {len(lost)} previously published files are MISSING after this rebuild:")
        for f in lost[:20]:
            print(f"   {f}")
        print("   Add them to CODE/DOCS or fix the exclusion rules. Do not commit a release "
              "that silently drops files.")
    else:
        print(f"\nintegrity: all {len(prev)} files from the last commit are still present")

    print(f"copied {len(copied)} files into {args.out}/")
    print(f"sanitised {len(sanitised_files)} files, {total_subs} substitutions")
    print("\ntop sanitised:")
    for s in report["sanitised"][:10]:
        print(f"  {s['substitutions']:5d}  {s['file']}")
    print(f"\nwrote {args.out}/RELEASE_MANIFEST.json")


if __name__ == "__main__":
    main()
