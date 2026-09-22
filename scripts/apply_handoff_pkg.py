"""Extract the v0.3 handoff docs over the working tree, leaving results/, cache/, scripts/ alone.

Run: uv run --no-project python scripts/apply_handoff_pkg.py \
        --tar $REPO_handoff_v0.3_20260917.tar.gz
"""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {"results", "cache", "scripts", ".venv", "prompt.md"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.tar) as tf:
            names = tf.getnames()
            tf.extractall(td)
        src_root = Path(td) / "cogreason-voe"
        assert src_root.is_dir(), f"unexpected layout: {names[:3]}"
        for src in sorted(src_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(src_root)
            if rel.parts[0] in PROTECTED:
                print(f"SKIP (protected) {rel}")
                continue
            dst = ROOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            existed = dst.exists()
            old = dst.stat().st_size if existed else 0
            shutil.copy2(src, dst)
            print(f"{'overwrite' if existed else 'new      '} {str(rel):34s} "
                  f"{old:6d} -> {dst.stat().st_size:6d} bytes")


if __name__ == "__main__":
    main()
