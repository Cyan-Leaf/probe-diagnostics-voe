"""Verify a downloaded HF snapshot against the repo API json (size-level check)."""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("api_json")
    ap.add_argument("snapshot_dir")
    args = ap.parse_args()
    d = json.load(open(args.api_json))
    root = Path(args.snapshot_dir)
    ok = miss = bad = 0
    total_ok = 0
    for s in d.get("siblings", []):
        name, want = s["rfilename"], (s.get("size") or 0)
        p = root / name
        if not p.exists():
            if name.startswith(("assets/", "examples/", ".git")):
                continue
            print(f"MISSING  {name}")
            miss += 1
            continue
        got = p.stat().st_size
        if want and got != want:
            print(f"SIZE-MISMATCH {name}: got {got} want {want}")
            bad += 1
        else:
            ok += 1
            total_ok += got
    print(f"\nok={ok} missing={miss} mismatch={bad} bytes_ok={total_ok/1e9:.2f} GB")
    raise SystemExit(1 if (miss or bad) else 0)


if __name__ == "__main__":
    main()
