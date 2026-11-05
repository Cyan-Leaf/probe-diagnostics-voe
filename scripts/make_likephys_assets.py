"""Build presentation clips from the LikePhys human-eval set (the ones actually rated).

Why this source and not IntPhys 2: the humaneval clips are what the team already watched and
vetted, one law per clip on a plain grey background with a single object. That reads instantly on
a projector. IntPhys 2 is photo-real but every clip is a full cluttered scene, so the audience
spends its attention finding the violation instead of seeing it.

Source: `humaneval/data/clips/<PAIR>_{left,right}.mp4` -- the browser-playable H.264 versions of
`results/T10_validity/human_eval/ANSWER_KEY.json`. The answer key decides which side violates
something, and it is NOT always the same side, so every output normalises to
**left = valid, right = violated**; otherwise the slide would teach the audience the wrong
association and the same left/right mistake would appear in two consecutive pairs.

Outputs (under `assets/presentation/likephys/`):
  pair_<pair>_<kind>.mp4    valid | invalid, side by side
  cj_<pair>_<kind>.mp4      original | colour-jittered (D2's manipulation), same valid clip
  stills/...png             one 4-frame montage each, for slides without video

Run: python3 scripts/make_likephys_assets.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_extract import HUE_SHIFT, SAT_SCALE, colour_jitter

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "humaneval" / "data" / "clips"
KEY = ROOT / "results" / "T10_validity" / "human_eval" / "ANSWER_KEY.json"
OUT = ROOT / "assets" / "presentation" / "likephys"
FFMPEG = "/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

# Hand-picked after rendering every candidate pair and looking at it. Criteria, in order:
#   1. the violation is visible without explanation (no counting, no timing judgement)
#   2. the valid clip is unambiguously normal, so "which one is wrong" is not a coin flip
#   3. one object, one law
CHOSEN = [
    ("P047", "shadow",       "无阴影：物体在，影子没了"),
    ("P033", "block_slide",  "悬浮：方块滑下来后没有落地，停在空中"),
    ("P048", "shadow",       "物体凭空消失：只剩影子"),
    ("P004", "ball_drop",    "穿透：球穿过地面继续下落"),
    ("P011", "fluid",        "液体相变：流动的水中途变成另一种形态"),
    ("P032", "ball_collision", "穿透：两球相撞时互相穿过"),
]


def read_all(path: Path) -> np.ndarray:
    import decord

    vr = decord.VideoReader(str(path))
    return vr.get_batch(list(range(len(vr)))).asnumpy()


def write_mp4(frames: np.ndarray, dst: Path, fps: int = 30) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", str(dst)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    p.communicate(frames.tobytes())
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {dst}")


def jitter_np(frames_u8: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
    x = x * 2.0 - 1.0
    px = x.permute(1, 0, 2, 3).unsqueeze(0)
    out = colour_jitter(px)
    y = (out[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    return (y.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def hstack(a: np.ndarray, b: np.ndarray, gap: int = 8) -> np.ndarray:
    n, h, w, _ = a.shape
    sep = np.full((n, h, gap, 3), 245, np.uint8)
    return np.concatenate([a, sep, b], axis=2)


def montage_row(frames: np.ndarray, k: int = 4) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return np.concatenate([frames[i] for i in idx], axis=1)


def main():
    import imageio.v2 as imageio

    key = {x["pair_id"]: x for x in json.loads(KEY.read_text())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stills").mkdir(exist_ok=True)

    lines = ["# LikePhys 演示片段（人评时评过的那批）", "",
             f"色相 `hue={HUE_SHIFT}`、饱和 `sat={SAT_SCALE}`，与 D2 的 `cj` 同一函数。", "",
             "**所有片段已统一为「左 = 物理正常 / 右 = 违反」** —— 原始答案键里违反侧左右不定。", ""]

    for pid, scenario, zh in CHOSEN:
        x = key[pid]
        viol_side = x["_answer_violation_side"]
        valid_side = "left" if viol_side == "right" else "right"
        v = read_all(CLIPS / f"{pid}_{valid_side}.mp4")
        i = read_all(CLIPS / f"{pid}_{viol_side}.mp4")
        n = min(len(v), len(i))
        v, i = v[:n], i[:n]

        name = f"{pid}_{scenario}"
        write_mp4(hstack(v, i), OUT / f"pair_{name}.mp4")
        imageio.imwrite(OUT / "stills" / f"pair_{name}.png",
                        np.concatenate([montage_row(v), montage_row(i)], axis=0))

        cj = jitter_np(v)
        write_mp4(hstack(v, cj), OUT / f"cj_{name}.mp4")
        imageio.imwrite(OUT / "stills" / f"cj_{name}.png",
                        np.concatenate([montage_row(v), montage_row(cj)], axis=0))

        lines += [f"## {pid} · {scenario}", "",
                  f"- 违反类型：**{x['_violation_kind']}** —— {zh}",
                  f"- `pair_{name}.mp4`：左 = `{x['left_clip'] if valid_side == 'left' else x['right_clip'].split('/')[-1]}`"
                  f"（valid），右 = 违反片段",
                  f"- `cj_{name}.mp4`：左 = 原片，右 = 只改色相/饱和度（physics-preserving）",
                  f"- 静帧：`stills/pair_{name}.png`、`stills/cj_{name}.png`", ""]
        print(f"{pid} {scenario:16s} valid_side={valid_side:5s} kind={x['_violation_kind']}")

    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")
    for f in sorted(OUT.glob("*.mp4")):
        print(f"  {f.name:34s} {f.stat().st_size/2**20:5.2f} MB")


if __name__ == "__main__":
    main()
