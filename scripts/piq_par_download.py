"""Parallel chunked downloader for large HF files over a bandwidth-limited proxy.

Why this exists: single-stream throughput through the corporate proxy to HF's Xet CDN is
~65 KB/s, so a 62 GB shard would take 11 days. Measured scaling with concurrency:

    1 stream   -> 0.065 MB/s
    8 streams  -> 0.7  MB/s
    32 streams -> 2.7  MB/s
    96 streams -> 5.5  MB/s

So total throughput is per-connection limited, not link limited: opening many small range
requests multiplies it. `hf download` cannot do this (xet bridges whole files), hence this
script.

Design:
  * each file is cut into fixed-size chunks; a thread pool pulls them as range requests
  * chunks land as `part` files, then are concatenated — so an interrupted run resumes at
    chunk granularity (a finished chunk is never refetched)
  * size and SHA256 are verified against the HF API metadata before the file is accepted
  * retries per chunk, because the proxy drops connections under load

Usage:
  python scripts/piq_par_download.py --repo MiniMaxAI/MiniMax-H3 \
      --dest data/MiniMax-H3 --include FL2VA/transformer FL2VA/video_vae \
      --workers 96 --chunk-mb 8
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK_LOCK = threading.Lock()
STATS = {"bytes": 0, "chunks": 0, "retries": 0}


def api_tree(repo: str, path: str) -> list:
    url = f"https://huggingface.co/api/models/{repo}/tree/main/{path}".rstrip("/")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_files(repo: str, includes: list) -> list:
    """Expand each --include into concrete files.

    An include may be a directory (recurse) or a single file. The tree API returns 404 for a
    file path, so a failed tree call is retried as a direct file lookup before giving up --
    without this, mixing `FL2VA/transformer` and `FL2VA/model_index.json` kills the run.
    """
    out = []
    seen = set()

    def walk(path):
        try:
            entries = api_tree(repo, path)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            size = file_size(repo, path)
            if path not in seen:
                seen.add(path)
                out.append({"path": path, "size": size, "sha256": None})
            return
        for ent in entries:
            p = ent["path"]
            if ent["type"] == "directory":
                walk(p)
            elif ent["type"] == "file" and p not in seen:
                seen.add(p)
                out.append({"path": p, "size": ent["size"],
                            "sha256": (ent.get("lfs") or {}).get("oid")})

    for inc in includes:
        walk(inc.rstrip("/"))
    return out


def file_size(repo: str, path: str) -> int:
    req = urllib.request.Request(resolve_url(repo, path), method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["content-length"])


def resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def fetch_chunk(url: str, start: int, end: int, dst: Path, tries: int = 6) -> int:
    """Download [start, end] inclusive. Returns bytes written."""
    if dst.exists() and dst.stat().st_size == end - start + 1:
        n = dst.stat().st_size
        with CHUNK_LOCK:
            STATS["bytes"] += n
        return n
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            dst.write_bytes(data)
            with CHUNK_LOCK:
                STATS["bytes"] += len(data)
                STATS["chunks"] += 1
            return len(data)
        except Exception as exc:  # noqa: BLE001
            with CHUNK_LOCK:
                STATS["retries"] += 1
            if attempt == tries - 1:
                raise IOError(f"chunk {start}-{end} failed after {tries}: {exc}") from exc
            time.sleep(min(2 ** attempt, 20))


def sha256_of(p: Path, bs: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(bs), b""):
            h.update(blk)
    return h.hexdigest()


def download_file(repo: str, ent: dict, dest: Path, workers: int, chunk: int) -> dict:
    rel = ent["path"]
    final = dest / rel
    final.parent.mkdir(parents=True, exist_ok=True)
    size = ent["size"]
    url = resolve_url(repo, rel)

    if final.exists() and final.stat().st_size == size:
        print(f"[skip] {rel} already complete ({size/2**30:.2f} GB)", flush=True)
        return {"path": rel, "status": "skipped"}

    pdir = dest / ".parts" / rel.replace("/", "__")
    pdir.mkdir(parents=True, exist_ok=True)
    bounds = [(s, min(s + chunk - 1, size - 1)) for s in range(0, size, chunk)]
    todo = [(s, e) for s, e in bounds if not (pdir / f"{s}-{e}.part").exists()]
    print(f"[file] {rel} {size/2**30:.2f} GB, {len(bounds)} chunks, "
          f"{len(todo)} to fetch, {workers} workers", flush=True)

    t0 = time.time()
    base = STATS["bytes"]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_chunk, url, s, e, pdir / f"{s}-{e}.part"): (s, e)
                for s, e in todo}
        for fut in as_completed(futs):
            fut.result()
            if STATS["chunks"] % 200 == 0:
                done = STATS["bytes"] - base
                el = time.time() - t0
                print(f"  {done/2**20:.0f} MB @ {done/2**20/el:.2f} MB/s, "
                      f"{STATS['retries']} retries", flush=True)

    tmp = final.with_suffix(final.suffix + ".cat")
    with open(tmp, "wb") as out:
        for s, e in bounds:
            out.write((pdir / f"{s}-{e}.part").read_bytes())
    got = tmp.stat().st_size
    if got != size:
        raise IOError(f"{rel}: concatenated {got} != expected {size}")

    want = ent.get("sha256")
    if want:
        print(f"  verifying sha256 ...", flush=True)
        h = sha256_of(tmp)
        if h != want:
            raise IOError(f"{rel}: sha256 mismatch\n got {h}\n want {want}")
    os.replace(tmp, final)
    for s, e in bounds:
        (pdir / f"{s}-{e}.part").unlink(missing_ok=True)
    pdir.rmdir()
    el = time.time() - t0
    print(f"[done] {rel} {size/2**30:.2f} GB in {el/60:.1f} min "
          f"({size/2**20/el:.2f} MB/s avg)", flush=True)
    return {"path": rel, "status": "done", "size": size,
            "seconds": round(el, 1), "sha256": want}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--include", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--manifest", default=None, help="append per-file results as JSONL")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = list_files(args.repo, args.include)
    files.sort(key=lambda f: f["size"])
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total/2**30:.2f} GB total, {args.workers} workers, "
          f"{args.chunk_mb} MB chunks", flush=True)

    results = []
    for ent in files:
        try:
            res = download_file(args.repo, ent, dest, args.workers, args.chunk_mb << 20)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {ent['path']}: {exc}", flush=True)
            res = {"path": ent["path"], "status": "failed", "error": str(exc)}
        res["repo"] = args.repo
        results.append(res)
        if args.manifest:
            with open(args.manifest, "a") as f:
                f.write(json.dumps(res) + "\n")

    bad = [r for r in results if r["status"] == "failed"]
    print(f"\nfinished: {len(results)-len(bad)} ok, {len(bad)} failed, "
          f"total retries {STATS['retries']}", flush=True)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
