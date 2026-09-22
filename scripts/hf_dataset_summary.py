"""Summarise an HF *dataset* API json (with ?blobs=true)."""

import argparse
import json
import re
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    d = json.load(open(args.path))
    sib = d.get("siblings", [])
    tot = sum((s.get("size") or 0) for s in sib)
    print(f"repo={d.get('id')} sha={d.get('sha')} private={d.get('private')} gated={d.get('gated')}")
    print(f"files={len(sib)} total={tot/1e9:.2f} GB  downloads={d.get('downloads')} likes={d.get('likes')}")
    print(f"tags={d.get('tags')}")
    groups: dict[str, list[int]] = defaultdict(list)
    for s in sib:
        name = s["rfilename"]
        key = re.sub(r"\d+", "#", name.split("/")[0]) if "/" in name else "(root)"
        groups[key].append(s.get("size") or 0)
    print("\ntop-level layout:")
    for k, v in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {sum(v)/1e9:8.3f} GB  n={len(v):5d}  {k}")
    print(f"\nfirst {args.top} files:")
    for s in sib[: args.top]:
        print(f"  {(s.get('size') or 0)/1e6:9.2f} MB  {s['rfilename']}")


if __name__ == "__main__":
    main()
