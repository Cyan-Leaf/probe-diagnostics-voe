"""T2: download the datasets needed for T3/T4 (+ Physion-Eval metadata for format check).

Run:
  export http_proxy=... https_proxy=... HF_HOME=$HF_HOME
  uv run --no-project python scripts/t2_download_data.py --which likephys intphys2 physion_meta
"""

import argparse
import os
import time

from huggingface_hub import snapshot_download

JOBS = {
    # (repo_id, revision pinned 2026-09-16, allow_patterns)
    "likephys": ("JianhaoDYDY/LikePhys-Benchmark", "30fc7a194a7c4eb1de9102f49c7355121ba79b1f", None),
    "intphys2": ("facebook/IntPhys2", "a077a2f94e25889016fc6e5983cf21e2ddc25fb2", None),
    "physion_meta": (
        "PhysionLabs/Physion-Eval",
        "1c53fd7a5ecdca220976deffb5535cb22c60244e",
        ["*.json", "README.md"],
    ),
    "physion_videos": (
        "PhysionLabs/Physion-Eval",
        "1c53fd7a5ecdca220976deffb5535cb22c60244e",
        ["videos.zip"],
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", required=True, choices=sorted(JOBS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=12)
    args = ap.parse_args()
    assert os.environ.get("https_proxy"), "export the oversea proxy first"
    assert os.environ.get("HF_HOME"), "set HF_HOME"
    for name in args.which:
        repo, rev, pats = JOBS[name]
        t0 = time.time()
        print(f"[{name}] downloading {repo}@{rev[:8]} patterns={pats}", flush=True)
        for attempt in range(args.retries):
            try:
                path = snapshot_download(
                    repo_id=repo,
                    revision=rev,
                    repo_type="dataset",
                    allow_patterns=pats,
                    max_workers=args.workers,
                )
                print(f"[{name}] done in {time.time()-t0:.0f}s -> {path}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                wait = min(60 * (attempt + 1), 300)
                print(f"[{name}] attempt {attempt} failed ({type(exc).__name__}: {exc!s:.180}); "
                      f"sleep {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"[{name}] GAVE UP after {args.retries} attempts", flush=True)


if __name__ == "__main__":
    main()
