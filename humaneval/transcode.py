#!/usr/bin/env python3
"""Transcode the evaluation clips to browser-playable H.264.

The LikePhys clips are `mpeg4` (Simple Profile, fourcc mp4v).  decord and ffmpeg read that
happily, which is why every server-side check passed, but **no current browser decodes it**:
Chrome/Edge/Safari/Firefox ship H.264, VP8/9, AV1 and HEVC only.  So the platform served a
valid 200 with a valid mp4 and the `<video>` element silently refused it.

Fix: pre-transcode to H.264 + yuv420p + faststart once, into a cache next to the database,
and serve those.  Transcoding on the fly would add a second of latency per clip and would
re-encode the same file for every rater.

Only the 96 clips referenced by the answer key are converted (48 pairs x 2 sides), not the
whole 920-clip corpus.

Run: python humaneval/transcode.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
PKG = PROJ / "results" / "T10_validity" / "human_eval"
CACHE = HERE / "data" / "clips"


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    key = json.loads((PKG / "ANSWER_KEY.json").read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg()

    jobs = {}
    for p in key:
        for side in ("left", "right"):
            src = Path(p[f"{side}_path"])
            # one output per source clip, keyed by pair+side so the server can find it
            # without needing to know the original path
            jobs[(p["pair_id"], side)] = src

    done = skipped = failed = 0
    for (pid, side), src in sorted(jobs.items()):
        dst = CACHE / f"{pid}_{side}.mp4"
        if dst.exists() and dst.stat().st_size > 1000:
            skipped += 1
            continue
        if not src.exists():
            print(f"  MISSING {src}", flush=True)
            failed += 1
            continue
        cmd = [ff, "-y", "-loglevel", "error", "-i", str(src),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p",          # browsers reject yuv444 / 10-bit
               "-movflags", "+faststart",      # moov atom first, so it plays while loading
               "-an",                          # no audio track in the source anyway
               str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            print(f"  FAILED {pid}/{side}: {r.stderr.strip()[:200]}", flush=True)
            failed += 1
        else:
            done += 1
            if done % 20 == 0:
                print(f"  transcoded {done}...", flush=True)

    print(f"\ntranscoded {done}, already present {skipped}, failed {failed}")
    print(f"cache -> {CACHE}")
    total = sum(f.stat().st_size for f in CACHE.glob('*.mp4'))
    print(f"{len(list(CACHE.glob('*.mp4')))} files, {total/2**20:.1f} MB")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
