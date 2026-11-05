"""E21 analysis: is the take contrast (two legal recordings of one scene) readable?

Design and verdict rules frozen in `results/E21_physics_iq_real/E21_PREREG.md` §3-§4.
`probe` is imported from e8_analyse verbatim (E11 measured a 2.3 pp bias in a rewrite), and the
grouping is the SCENE, so both takes and all three perspectives of a scene stay in one fold.

Three arms, all on the same features:
  take          take-1 vs take-2  — both physically legal; a physics-irrelevant contrast
  perspective   left vs right only — positive control, must be learnable
  random / vae / pixel             — the lower bounds that make 'take' interpretable

Run: python scripts/e21_analyse.py --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_analyse as A8

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E21_physics_iq_real"
CACHE = {"real": ROOT / "cache" / "piq_feats_real",
         "random": ROOT / "cache" / "piq_feats_random"}
SEEDS = [0, 1, 2]
LAYER = 16
T = 600
SMALL_UNIT = 100

# PREREG §4, fixed before the run. The 0.05 margin is E8 §15.2's weak-signal margin.
TAKE_ABS, TAKE_MARGIN, PERSP = 0.60, 0.05, 0.95
# Read-only reference: Wan2.1-1.3B physics AUC on IntPhys 2, from the delivered E8 JSON.
# E21 does NOT recompute it; it is loaded and quoted so the two are on the same scale.
E8_INT = ROOT / "results" / "E8_second_backbone" / "E8_cells_intphys2.json"
E8_REF_KEY = "wan21_1b3|intphys2|t600|L16|real|physics"


def load(backbone: str, field: str | None, seed: int):
    d = CACHE[backbone]
    files = sorted(d.glob("*.pt"))
    assert files, f"empty cache {d} — run e21_extract.py first"
    rows, meta = [], []
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        if field is None:
            rows.append(rec["feats"][str(T)][str(seed)][LAYER].reshape(-1).numpy().astype(np.float32))
        else:
            rows.append(rec[field].reshape(-1).numpy().astype(np.float32))
        c = rec["clip"]
        meta.append({"clip_id": c["clip_id"], "scene": c["scene"],
                     "perspective": c["perspective"], "take": c["take"],
                     "label_take": c["label_take"], "label_persp_lr": c["label_persp_lr"]})
    return np.stack(rows), meta


def _labels(meta: list, task: str):
    g = np.array([m["scene"] for m in meta])
    if task == "take":
        y, sel = np.array([m["label_take"] for m in meta]), np.arange(len(meta))
    elif task == "perspective":
        sel = np.array([i for i, m in enumerate(meta) if m["label_persp_lr"] is not None])
        y = np.array([meta[i]["label_persp_lr"] for i in sel])
        g = g[sel]
    else:
        raise ValueError(task)
    return sel, y, g


def cell(backbone: str, task: str, field: str | None = None) -> dict | None:
    """Mean AUC over feature seeds, with the seed spread as the floor (E8's convention)."""
    per = []
    for seed in SEEDS:
        X, meta = load(backbone, field, seed)
        sel, y, g = _labels(meta, task)
        if len(np.unique(y)) < 2:
            return None
        a, dp = A8.probe(np.asarray(X[sel], dtype=np.float32), y, g, seed=1000 + seed)
        per.append((a, dp))
    A = np.array([p[0] for p in per], float)
    D = np.array([p[1] for p in per], float)
    return {"backbone": backbone, "task": task, "field": field or "dit_L16",
            "layer": LAYER if field is None else -1, "timestep": T if field is None else -1,
            "n": int(len(sel)), "n_pos": int(y.sum()),
            "n_groups": int(len(np.unique(g))),
            "auc_mean": float(np.nanmean(A)), "auc_seed_spread": float(np.nanstd(A, ddof=1)),
            "dprime_mean": float(np.nanmean(D)),
            "per_seed_auc": [float(x) for x in A],
            "small_unit": bool(len(sel) < SMALL_UNIT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    from threadpoolctl import threadpool_limits

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"task": "E21: real-world video (Physics-IQ) take contrast",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "prereg": "results/E21_physics_iq_real/E21_PREREG.md",
               "protocol": {"seeds": SEEDS, "layer": LAYER, "t": T, "folds": 5,
                            "grouping": "scene (both takes + 3 perspectives stay in one fold)",
                            "threads": args.threads, "readout": "AUC and d'"},
               "cells": {}}

    has_random = bool(list(CACHE["random"].glob("*.pt"))) if CACHE["random"].exists() else False

    with threadpool_limits(limits=args.threads):
        for task in ("take", "perspective"):
            for field in (None, "vae_latent_pooled", "raw_pixel_32"):
                r = cell("real", task, field)
                if r is None:
                    continue
                key = f"real|{task}|{r['field']}"
                payload["cells"][key] = r
                print(f"  {key:34s} n={r['n']:4d} AUC {r['auc_mean']:.4f}"
                      f"±{r['auc_seed_spread']:.4f}", flush=True)
            if has_random:
                r = cell("random", task, None)
                if r:
                    key = f"random|{task}|dit_L16"
                    payload["cells"][key] = r
                    print(f"  {key:34s} n={r['n']:4d} AUC {r['auc_mean']:.4f}"
                          f"±{r['auc_seed_spread']:.4f}", flush=True)

    (OUT / "E21_cells.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT/'E21_cells.json'}", flush=True)
    _verdict(payload, has_random)


def _verdict(payload: dict, has_random: bool):
    """Verdicts on the SEPARATION magnitude, not the raw AUC.

    AMENDED 2026-11-05 after the first analysis, and the reason is a genuine design error in
    PREREG §4 rather than a post-hoc convenience: the pre-registered rules were written as
    `AUC_take >= 0.60`, which silently assumes the probe can only separate the takes in the
    take-1 -> take-2 direction. Nothing fixes that direction. An AUC of 0.40 separates the two
    takes as well as 0.60 does, so a lower-tail result would have been scored as "refuted"
    while the data in fact showed a *stronger* take signal. The measured value was 0.4891
    (seeds 0.5087 / 0.5052 / 0.4535), i.e. it lands near chance either way, so this amendment
    does not change what E21 concludes — but it would have mattered, and a rule that can
    mislabel a result must be fixed before it is relied on.

    Separation is therefore `max(AUC, 1 - AUC)`, computed per feature seed and averaged,
    0.5 = no separation. This is the same correction D2's gap needed (a difference, not a
    signed AUC), and it is registered here rather than applied silently.
    """
    c = payload["cells"]

    def sep(bb="real", task="take", field="dit_L16"):
        cell_ = c.get(f"{bb}|{task}|{field}")
        if not cell_:
            return float("nan")
        return float(np.mean([max(a, 1.0 - a) for a in cell_["per_seed_auc"]]))

    def auc(bb="real", task="take", field="dit_L16"):
        cell_ = c.get(f"{bb}|{task}|{field}")
        return cell_["auc_mean"] if cell_ else float("nan")

    e8 = json.loads(E8_INT.read_text())["cells"]
    ref = e8.get(E8_REF_KEY)
    ref_auc = float(ref["per_seed_auc"][0]) if ref else float("nan")

    take_auc, take_sep = auc(), sep()
    persp = auc(task="perspective")
    take_rand = sep("random") if has_random else float("nan")
    take_vae, take_pix = sep(field="vae_latent_pooled"), sep(field="raw_pixel_32")

    v = {
        "task": "E21 verdict",
        "created": payload["created"],
        "prereg": payload["prereg"],
        "rule_amendment": {
            "what": "verdicts use separation max(AUC, 1-AUC) instead of the raw AUC",
            "why": "PREREG §4 wrote the rule as AUC >= threshold, which assumes a fixed "
                   "direction between take-1 and take-2. Nothing fixes that direction, so a "
                   "lower-tail AUC would have been scored 'refuted' while showing a stronger "
                   "take signal. Registered as an amendment; it does not change E21's outcome "
                   "because the measured AUC is near chance either way.",
            "when": "2026-11-05, after the first analysis run, before the report was written",
        },
        "reference": {"e8_intphys2_wan21_1b3_physics_auc": ref_auc,
                      "source": "results/E8_second_backbone/E8_cells_intphys2.json",
                      "note": "quoted read-only; E21 does not recompute it"},
        "d_take_1": {"rule": "separation(take) >= 0.60",
                     "separation": take_sep, "raw_auc": take_auc, "threshold": TAKE_ABS,
                     "verdict": "hold" if take_sep >= TAKE_ABS else "refuted"},
        "d_take_2": {"rule": "separation(take) >= AUC_physics_E8 - 0.05",
                     "separation": take_sep, "reference": ref_auc,
                     "threshold": ref_auc - TAKE_MARGIN,
                     "verdict": ("hold" if take_sep >= ref_auc - TAKE_MARGIN else "refuted")
                                if np.isfinite(ref_auc) else "unavailable"},
        "d_take_3": {"rule": f"AUC_perspective >= {PERSP}", "value": persp,
                     "threshold": PERSP,
                     "verdict": "hold" if persp >= PERSP else "refuted",
                     "why": "positive control: an obvious real-world difference must be learnable"},
        "lower_bounds_separation": {"random_init_dit": take_rand, "vae_latent": take_vae,
                                    "raw_pixel": take_pix},
        "interpretation_guard": (
            "take-1/take-2 are two legal recordings of one scene, NOT a possible/impossible "
            "pair. This measures whether a physics-irrelevant real-world contrast is linearly "
            "readable; it does NOT measure violation detection and must not be reported as "
            "physics understanding on real video."),
    }
    (OUT / "E21_verdict.json").write_text(json.dumps(v, indent=1, ensure_ascii=False))
    print(json.dumps({k: v[k] for k in ("d_take_1", "d_take_2", "d_take_3",
                                        "lower_bounds_separation")}, indent=1))


if __name__ == "__main__":
    main()
