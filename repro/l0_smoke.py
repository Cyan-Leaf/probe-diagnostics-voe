#!/usr/bin/env python3
"""L0 smoke reproduction: one GPU, under 30 minutes, from nothing to one comparable number.

What it reproduces
------------------
The IntPhys 2 D1/D2 contrast -- the example the proposal opens with. On a subset:

    D1  physics task:     probe on trained DiT  vs  probe on randomly initialised DiT
    D2  appearance task:  same trained features, labels swapped for a colour-jitter control

The point being reproduced is not a specific decimal, it is the ordering:

    AUC(appearance) >> AUC(physics)  and  AUC(physics, trained) only modestly above random

Why a subset changes the numbers
--------------------------------
The paper's numbers use all 253 scenes (1012 clips). L0 uses `--scenes 24` by default, so the
probe sees ~96 clips. Small-sample AUC on this task is noisy, and the tolerance below was
measured, not guessed: `--repeat 5` over five different scene draws.

Anything that cannot be given a tolerance is reported as a range instead. See T14_REPRO.md.

Usage
-----
    python repro/l0_smoke.py --scenes 24            # default, ~20 min on one 4090
    python repro/l0_smoke.py --scenes 24 --repeat 5 # measure the tolerance yourself
    python repro/l0_smoke.py --dry-run              # print the plan and exit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

EXPECTED = {
    # filled in from the full-set run; L0 checks the ORDERING, the values are context
    "full_set": {
        "physics_trained": 0.6458,
        "physics_random": 0.5168,
        "appearance_trained": 1.0000,
        "d1_share": 0.885,
    },
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_intphys2() -> Path | None:
    """Locate IntPhys 2 without assuming our layout.

    Order: explicit env var, then the HF cache, then a local directory. Returns None so the
    caller can print a download instruction rather than crashing.
    """
    env = os.environ.get("INTPHYS2_ROOT")
    if env and (Path(env) / "Main" / "metadata.csv").exists():
        return Path(env)
    hf = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for base in [Path(hf) / "hub", Path(hf)]:
        if not base.exists():
            continue
        for p in base.glob("datasets--facebook--IntPhys2/snapshots/*"):
            if (p / "Main" / "metadata.csv").exists():
                return p
    local = REPO / "data" / "IntPhys2"
    return local if (local / "Main" / "metadata.csv").exists() else None


DOWNLOAD_HINT = """
IntPhys 2 not found. Get it with either:

  huggingface-cli download facebook/IntPhys2 --repo-type dataset --local-dir data/IntPhys2

  # or the direct archive (~12 GB):
  # https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip

then re-run, or point INTPHYS2_ROOT at the directory containing Main/metadata.csv.

L0 only reads the `Main` split and only the scenes it samples, but the download itself is
the whole dataset -- budget for that.
"""


def build_subset(root: Path, n_scenes: int, seed: int):
    rows = list(csv.DictReader((root / "Main" / "metadata.csv").open()))
    scenes = sorted({r["SceneIndex"] for r in rows})
    import random
    picked = set(random.Random(seed).sample(scenes, min(n_scenes, len(scenes))))
    sel = [r for r in rows if r["SceneIndex"] in picked]
    return [{
        "clip_id": f"intphys2/{r['SceneIndex']}/{r['type']}",
        "path": str(root / "Main" / r["file_name"]),
        "scene_index": r["SceneIndex"],
        "label": 1 if r["type"].endswith("Impossible") else 0,
        "condition": r["condition"], "camera": r["Camera"],
    } for r in sel]


def extract(clips, backbone: str, jitter: bool, device: str):
    """Features for one (backbone, variant). Held in memory: L0 writes no cache on purpose,
    so a second run cannot silently pass by reusing the first run's artefacts."""
    import numpy as np
    import torch
    import wan_probe_lib as W
    from e7_extract import JITTER, colour_jitter, read_spanning

    pre = W.preflight(device)
    assert pre["healthy"], f"preflight failed: {pre}"

    vae = W.load_vae(device, torch.float32)
    tf = W.load_transformer(device, torch.bfloat16, random_init=(backbone == "random"))
    pe, _ = W.load_prompt_embeds(str(REPO / "results/T2_env/prompt_embeds_empty.pt"),
                                 device, torch.bfloat16)
    feats, keep = [], []
    t0 = time.time()
    for i, c in enumerate(clips):
        try:
            frames = read_spanning(c["path"], 49)
        except Exception as exc:
            log(f"  skip {c['clip_id']}: {exc}")
            continue
        px = W.preprocess_frames(frames, 512, 512, device)
        if jitter:
            j = JITTER["cj"]
            px = colour_jitter(px, hue=j["hue"], sat=j["sat"])
        z0 = W.encode_latents(vae, px, normalize=True)
        grid = W.latent_grid(z0)
        z_t, _, _ = W.flow_noise(z0, 600, seed=0)
        ts = W.per_token_timestep(z_t, 600)
        with W.capture_blocks(tf, [16], grid) as hs:
            W.forward_v(tf, z_t, ts, pe)
        feats.append(hs.pooled[16].reshape(-1).float().cpu().numpy())
        keep.append(c)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            log(f"  {backbone}{'+cj' if jitter else ''} {i+1}/{len(clips)} "
                f"({el/(i+1):.2f}s/clip)")
    del vae, tf
    torch.cuda.empty_cache()
    return np.stack(feats), keep


def probe(X, y, groups, seed=0):
    import numpy as np
    from e10_s1s3 import _shuffled_group_folds
    from t4_caliper import CS, _decision, _fit_path, _rowspace
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    aucs = []
    for tr, te in _shuffled_group_folds(groups, 5, rng):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        scores = {c: [] for c in CS}
        for itr, ite in GroupKFold(n_splits=3).split(Xtr, y[tr], groups[tr]):
            if len(np.unique(y[tr][ite])) < 2:
                continue
            s2 = StandardScaler().fit(Xtr[itr])
            A, B = _rowspace(s2.transform(Xtr[itr]), s2.transform(Xtr[ite]))
            for c, coef, b in _fit_path(A, y[tr][itr], CS):
                scores[c].append(roc_auc_score(y[tr][ite], _decision(B, coef, b)))
        best = max(CS, key=lambda c: np.mean(scores[c]) if scores[c] else -1)
        Ztr, Zte = _rowspace(Xtr, Xte)
        _, coef, b = _fit_path(Ztr, y[tr], [best])[-1]
        aucs.append(roc_auc_score(y[te], _decision(Zte, coef, b)))
    return float(np.mean(aucs)) if aucs else float("nan")


def one_run(root, n_scenes, seed, device):
    import numpy as np
    clips = build_subset(root, n_scenes, seed)
    log(f"subset: {len(clips)} clips from {n_scenes} scenes (seed {seed})")

    Xr, kr = extract(clips, "real", False, device)
    Xq, kq = extract(clips, "random", False, device)
    pos = [c for c in clips if c["label"] == 0]
    Xc, kc = extract(pos, "real", True, device)

    y = np.array([c["label"] for c in kr])
    g = np.array([c["scene_index"] for c in kr])
    phys_real = probe(Xr, y, g)
    phys_rand = probe(Xq, np.array([c["label"] for c in kq]),
                      np.array([c["scene_index"] for c in kq]))

    keep = [i for i, c in enumerate(kr) if c["label"] == 0]
    Xa = np.concatenate([Xr[keep], Xc])
    ya = np.concatenate([np.zeros(len(keep), int), np.ones(len(Xc), int)])
    ga = np.array([kr[i]["scene_index"] for i in keep] + [c["scene_index"] for c in kc])
    appear = probe(Xa, ya, ga)

    share = ((phys_real - phys_rand) / (phys_real - 0.5)
             if phys_real > 0.5 else float("nan"))
    return {"seed": seed, "n_clips": len(kr),
            "physics_trained": phys_real, "physics_random": phys_rand,
            "appearance_trained": appear, "d1_share": share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="different scene draws, to measure the tolerance yourself")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "results" / "L0_smoke.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = find_intphys2()
    if root is None:
        print(DOWNLOAD_HINT)
        sys.exit(2)
    log(f"IntPhys 2 at {root}")

    if args.dry_run:
        n_fwd = args.scenes * 4 * 2 + args.scenes * 2
        print(f"plan: {args.scenes} scenes -> {args.scenes*4} clips\n"
              f"      ~{n_fwd} DiT forward passes, 3 model loads\n"
              f"      expect ~15-25 min on one 24 GB GPU\n"
              f"      writes {args.out} and no cache")
        return

    runs = [one_run(root, args.scenes, args.seed + k, args.device)
            for k in range(args.repeat)]

    import numpy as np
    def agg(k):
        v = np.array([r[k] for r in runs], float)
        return {"mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                "min": float(v.min()), "max": float(v.max())}

    summary = {k: agg(k) for k in
               ("physics_trained", "physics_random", "appearance_trained", "d1_share")}
    checks = {
        "appearance_above_physics": bool(
            summary["appearance_trained"]["mean"] > summary["physics_trained"]["mean"]),
        "physics_above_chance": bool(summary["physics_trained"]["mean"] > 0.5),
        "random_below_trained": bool(
            summary["physics_random"]["mean"] < summary["physics_trained"]["mean"]),
    }
    out = {"subset": {"scenes": args.scenes, "repeat": args.repeat, "seed": args.seed},
           "runs": runs, "summary": summary,
           "qualitative_checks": checks,
           "all_checks_pass": all(checks.values()),
           "full_set_reference": EXPECTED["full_set"],
           "note": ("L0 reproduces the ORDERING, not the decimals. Subset AUCs differ from "
                    "the full-set values; see T14_REPRO.md for the measured tolerance.")}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 62)
    print(f"{'quantity':22s} {'subset':>10s} {'full set':>10s}")
    for k in ("physics_trained", "physics_random", "appearance_trained"):
        print(f"{k:22s} {summary[k]['mean']:10.4f} {EXPECTED['full_set'][k]:10.4f}")
    print(f"{'d1_share':22s} {summary['d1_share']['mean']:10.3f} "
          f"{EXPECTED['full_set']['d1_share']:10.3f}")
    if args.repeat > 1:
        print(f"\nobserved spread over {args.repeat} draws:")
        for k, v in summary.items():
            print(f"  {k:22s} sd={v['sd']:.4f}  range [{v['min']:.4f}, {v['max']:.4f}]")
    print("\nqualitative checks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
