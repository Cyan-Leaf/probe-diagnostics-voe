"""Summarise an HF model API json (with ?blobs=true) saved by curl."""

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    d = json.load(open(args.path))
    sib = d.get("siblings", [])
    tot = sum((s.get("size") or 0) for s in sib)
    print(f"repo={d.get('id')} sha={d.get('sha')} files={len(sib)} total={tot/1e9:.2f} GB")
    for s in sorted(sib, key=lambda s: -(s.get("size") or 0)):
        sz = s.get("size") or 0
        flag = "" if sz > 1e8 else "  (small)"
        print(f"  {sz/1e9:7.3f} GB  {s['rfilename']}{flag}")


if __name__ == "__main__":
    main()
