"""Build presentation material: side-by-side videos and stills for the milestone talk.

Selection rule, and why it is not "pick the first Easy+Fixed scene":
the scenes were ranked by the mean absolute pixel difference between the possible and the
impossible clip of the SAME scene, then each candidate was rendered and looked at. That
matters because the largest raw difference in the whole dataset (scene 236, diff 40.98) is a
handheld-camera walk-through where nothing physically impossible happens on screen -- the
difference is camera motion. Two scenes survived the visual check:

  * scene 1  (solidity,     Easy, Fixed) -- a box slides down a ramp and either settles tilted
                                           on the slope or floats above it perfectly level
  * scene 17 (permanence,   Easy, Fixed) -- a black object is visible through the cup's handle;
                                           it either stays or vanishes and reappears

Outputs (all under `assets/presentation/`):

  pair_<name>.mp4       possible on the left, impossible on the right, same scene
  cj_<name>.mp4         original on the left, hue+0.35/sat x1.4 on the right (D2's control)
  stills/pair_<name>.png   6-panel montage, P top row / I bottom row
  stills/cj_<name>.png     6-panel montage, original top row / jittered bottom row

The jitter is `e8_extract.colour_jitter`, imported rather than re-implemented: the slide has to
show the *same* perturbation D2 measured, or someone may reasonably ask whether the picture and
the AUC came from the same manipulation.

Run: python3 scripts/make_presentation_assets.py
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
OUT = ROOT / "assets" / "presentation"
MANIFEST = ROOT / "results" / "E7_intphys2" / "manifest.json"
FFMPEG = "/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

# Hand-verified pairs (see module docstring).
PAIRS = {
    "solidity": {
        "scene": "1",
        "possible": "intphys2/1/1_Possible/099631878708",
        "impossible": "intphys2/1/1_Impossible/0412d75b071c",
        "zh": "固体性 solidity — 箱子沿斜面下滑，最后要么顺着斜面停住，要么悬空且水平",
        "cue": "看最后 1 秒的箱底与斜面之间有没有缝",
    },
    "permanence": {
        "scene": "17",
        "possible": "intphys2/17/1_Possible/69cf2d9aa1fd",
        "impossible": "intphys2/17/1_Impossible/88123fcc3ef5",
        "zh": "持久性 permanence — 杯柄里的黑色物体，要么一直在，要么消失又出现",
        "cue": "盯住杯柄那个黑色区域",
    },
}


def read_all(path: str) -> np.ndarray:
    import decord

    vr = decord.VideoReader(path)
    return vr.get_batch(list(range(len(vr)))).asnumpy()


def write_mp4(frames: np.ndarray, dst: Path, fps: int = 24) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(dst)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    p.communicate(frames.tobytes())
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {dst}")


def jitter_np(frames_u8: np.ndarray) -> np.ndarray:
    """Apply the project's own colour_jitter to uint8 [N,H,W,3]."""
    x = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
    x = x * 2.0 - 1.0
    px = x.permute(1, 0, 2, 3).unsqueeze(0)
    out = colour_jitter(px)
    y = (out[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    return (y.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def hstack(a: np.ndarray, b: np.ndarray, gap: int = 6) -> np.ndarray:
    n, h, w, _ = a.shape
    sep = np.full((n, h, gap, 3), 250, np.uint8)
    return np.concatenate([a, sep, b], axis=2)


def montage2x3(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Two rows of three evenly-spaced frames; row order matches the on-screen layout."""
    k = 3
    ti = np.linspace(0, len(top) - 1, k).round().astype(int)
    bi = np.linspace(0, len(bottom) - 1, k).round().astype(int)
    rows = [np.concatenate([top[i] for i in ti], axis=1),
            np.concatenate([bottom[i] for i in bi], axis=1)]
    return np.concatenate(rows, axis=0)


def main():
    import imageio.v2 as imageio

    by_id = {c["clip_id"]: c for c in json.loads(MANIFEST.read_text())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stills").mkdir(exist_ok=True)

    report = ["# 演示材料清单", "",
              f"色相偏移 `hue={HUE_SHIFT}`、饱和度 `sat={SAT_SCALE}` —— 与 D2 的 `cj` 变体"
              "调用同一个函数（`e8_extract.colour_jitter`）。", "",
              f"源数据 manifest：`{MANIFEST}`", ""]

    for name, spec in PAIRS.items():
        p = by_id[spec["possible"]]
        q = by_id[spec["impossible"]]
        print(f"[{name}] scene {spec['scene']}: {p['clip_id']} vs {q['clip_id']}")

        a = read_all(p["path"])
        b = read_all(q["path"])
        n = min(len(a), len(b))
        step = 2                                   # 60 fps source -> keep every 2nd frame
        a2, b2 = a[:n:step], b[:n:step]

        write_mp4(hstack(a2, b2), OUT / f"pair_{name}.mp4")
        imageio.imwrite(OUT / "stills" / f"pair_{name}.png", montage2x3(a2, b2))

        cj = jitter_np(a2)
        write_mp4(hstack(a2, cj), OUT / f"cj_{name}.mp4")
        imageio.imwrite(OUT / "stills" / f"cj_{name}.png", montage2x3(a2, cj))

        report += [f"## {name}", "",
                   f"- 中文说明：{spec['zh']}",
                   f"- 观众提示：{spec['cue']}",
                   f"- `pair_{name}.mp4` — **左 = 物理可能 / 右 = 物理不可能**（同一场景、同一相机）",
                   f"  - possible   `{spec['possible']}`",
                   f"  - impossible `{spec['impossible']}`",
                   f"  - 源 `{p['path']}`",
                   f"  - 源 `{q['path']}`",
                   f"- `cj_{name}.mp4` — **左 = 原片 / 右 = 只改色相与饱和度**，"
                   f"物体运动完全不变。这就是 D2 里 `cj` 变体对每一帧做的同一个变换。",
                   f"- 静帧 `stills/pair_{name}.png`（上=可能，下=不可能）、"
                   f"`stills/cj_{name}.png`（上=原片，下=染色）", ""]

    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\nwrote", OUT)
    for f in sorted(OUT.glob("*.mp4")):
        print(f"  {f.name:24s} {f.stat().st_size/2**20:5.1f} MB")


if __name__ == "__main__":
    main()
