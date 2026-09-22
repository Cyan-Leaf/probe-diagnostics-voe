"""E4 -- is the E0 main effect a complexity artefact?

The effect under test (E0, t=600): violation clips have *lower* denoising error than the
matched valid clip, mean paired diff -0.00966 (p = 1.7e-4), and it is carried by four
fluid / soft-body scenarios (fluid -0.0709, faucet -0.0335, cloth_drape -0.0143,
river -0.0081) while the remaining eight sit near zero.

Candidate explanation (PREREG §5 keeps it a *candidate* until this experiment):
the fluid violation edits -- sudden_freeze, phase_transition, teleporting_fluid,
rubber_cloth and friends -- remove motion and texture from the scene, and the flow-matching
denoising error is strongly driven by exactly those quantities.  If so, the direction of
the E0 effect says nothing about physics and must be reported as a complexity artefact.

Design
------
Pairing follows E0: within each (scenario, subgroup) the single valid clip is paired with
each violation clip, so every pair shares a scene layout.

For each pair we have
    d_err  = mse(violation) - mse(valid)          (negative = the E0 effect)
    d_cplx = complexity(violation) - complexity(valid)

and ask whether d_err is explained by d_cplx:

  1. correlation of d_err with each complexity proxy
  2. OLS  d_err ~ d_cplx ; the intercept is the effect that survives complexity matching.
     If the intercept loses significance while the slope is significant, the E0 direction
     is a complexity artefact.
  3. the same, per scenario (PREREG §5: per-scenario distribution beats the global mean)

Complexity proxies are deliberately content-level and physics-agnostic:
    flow_energy    mean Farneback flow magnitude      (how much stuff moves)
    temporal_grad  mean |I_t - I_{t-1}|               (motion, estimator-free)
    spatial_grad   mean |grad I| (Sobel)              (edge / texture density)
    hf_frac        fraction of FFT energy above 1/4 Nyquist  (fine texture)

> Note the A.5 lesson: these estimators conflate content motion with camera motion.  Here
> content motion is precisely what we want, so the confound does not apply -- but these
> numbers must never be reused to infer camera motion.

Run: python scripts/e4_complexity.py --workers 12
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E4_complexity"
CACHE = ROOT / "cache" / "feats_real"
T = "600"
PROXIES = ["flow_energy", "temporal_grad", "spatial_grad", "hf_frac"]
FLOW_SIZE = 128     # dense flow at 512 would cost ~1 h; complexity ranks are scale-stable
N_FRAMES = 49


def complexity(path: str) -> dict:
    import cv2
    import decord

    vr = decord.VideoReader(str(path), width=FLOW_SIZE, height=FLOW_SIZE)
    n = min(N_FRAMES, len(vr))
    rgb = vr.get_batch(list(range(n))).asnumpy().astype(np.float32) / 255.0
    grey = rgb.mean(axis=3)

    g8 = (grey * 255).astype(np.uint8)
    flows = []
    for a, b in zip(g8[:-1], g8[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(float(np.hypot(f[..., 0], f[..., 1]).mean()))

    tg = float(np.abs(np.diff(grey, axis=0)).mean())

    sob = []
    for g in grey:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        sob.append(float(np.hypot(gx, gy).mean()))

    hf = []
    cy = cx = FLOW_SIZE // 2
    yy, xx = np.mgrid[0:FLOW_SIZE, 0:FLOW_SIZE]
    rad = np.hypot(yy - cy, xx - cx)
    mask = rad > (FLOW_SIZE / 4.0)
    for g in grey:
        sp = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
        tot = sp.sum()
        hf.append(float(sp[mask].sum() / tot) if tot > 0 else 0.0)

    return {
        "flow_energy": float(np.mean(flows)) * (512.0 / FLOW_SIZE),
        "temporal_grad": tg,
        "spatial_grad": float(np.mean(sob)),
        "hf_frac": float(np.mean(hf)),
    }


def _job(clip):
    try:
        return clip["clip_id"], complexity(clip["path"])
    except Exception as exc:
        return clip["clip_id"], {"error": str(exc)}


def load_errors():
    import torch
    out = {}
    for f in sorted(CACHE.glob("*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        c = rec["clip"]
        if c["group"] != "physics":
            continue
        seeds = rec["errors"].get(T, {})
        if not seeds:
            continue
        out[c["clip_id"]] = float(np.mean([seeds[s]["mse"] for s in sorted(seeds)]))
    return out


def ols(x, y):
    """slope, intercept and their t-stats for y ~ x (intercept = effect at matched cplx)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3 or np.std(x) < 1e-12:
        return None
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    from scipy import stats
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {"intercept": float(beta[0]), "intercept_se": float(se[0]),
            "intercept_t": float(t[0]), "intercept_p": float(p[0]),
            "slope": float(beta[1]), "slope_se": float(se[1]),
            "slope_t": float(t[1]), "slope_p": float(p[1]),
            "r2": float(r2), "n": int(n)}


def main():
    from joblib import Parallel, delayed
    from scipy import stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "results/T3_null_floor/manifest.json").read_text())
    clips = [c for c in manifest if c["group"] == "physics"]
    print(f"{len(clips)} physics clips", flush=True)

    cpath = OUT / "complexity_per_clip.json"
    if cpath.exists():
        cplx = json.loads(cpath.read_text())
        print(f"reusing {len(cplx)} cached complexity records", flush=True)
    else:
        res = Parallel(n_jobs=args.workers, verbose=5)(delayed(_job)(c) for c in clips)
        cplx = {k: v for k, v in res}
        cpath.write_text(json.dumps(cplx, indent=1))
    bad = [k for k, v in cplx.items() if "error" in v]
    if bad:
        print(f"WARNING {len(bad)} clips failed: {bad[:3]}", flush=True)

    err = load_errors()
    print(f"{len(err)} denoising errors at t={T}", flush=True)

    by = {}
    for c in clips:
        by.setdefault((c["scenario"], c["subgroup"]), []).append(c)

    pairs = []
    for (sc, sg), group in sorted(by.items()):
        valid = [c for c in group if c["label"] == 0]
        viol = [c for c in group if c["label"] == 1]
        if len(valid) != 1 or not viol:
            continue
        v = valid[0]
        if v["clip_id"] not in err or "error" in cplx.get(v["clip_id"], {"error": 1}):
            continue
        for w in viol:
            if w["clip_id"] not in err or "error" in cplx.get(w["clip_id"], {"error": 1}):
                continue
            row = {"scenario": sc, "subgroup": sg, "valid_id": v["clip_id"],
                   "violation_id": w["clip_id"], "kind": w["kind"],
                   "d_err": err[w["clip_id"]] - err[v["clip_id"]]}
            for p in PROXIES:
                row[f"d_{p}"] = cplx[w["clip_id"]][p] - cplx[v["clip_id"]][p]
            pairs.append(row)

    print(f"{len(pairs)} (valid, violation) pairs", flush=True)
    d_err = np.array([r["d_err"] for r in pairs])

    raw = {
        "n_pairs": len(pairs),
        "mean_d_err": float(d_err.mean()),
        "median_d_err": float(np.median(d_err)),
        "wilcoxon_p": float(stats.wilcoxon(d_err).pvalue),
        "frac_negative": float((d_err < 0).mean()),
    }
    print(f"\nraw effect: mean {raw['mean_d_err']:+.5f}  median "
          f"{raw['median_d_err']:+.5f}  p={raw['wilcoxon_p']:.2e}  "
          f"frac<0 {raw['frac_negative']:.3f}", flush=True)

    corr, models = {}, {}
    for p in PROXIES:
        dx = np.array([r[f"d_{p}"] for r in pairs])
        r, pv = stats.pearsonr(dx, d_err)
        rs, ps = stats.spearmanr(dx, d_err)
        corr[p] = {"pearson_r": float(r), "pearson_p": float(pv),
                   "spearman_r": float(rs), "spearman_p": float(ps),
                   "mean_d_cplx": float(dx.mean()),
                   "frac_cplx_reduced": float((dx < 0).mean())}
        models[p] = ols(dx, d_err)
        m = models[p]
        print(f"  {p:14s} r={r:+.3f} (p={pv:.1e})  d_cplx mean={dx.mean():+.4f} "
              f"frac<0={float((dx<0).mean()):.2f} | OLS slope p={m['slope_p']:.1e} "
              f"intercept {m['intercept']:+.5f} (p={m['intercept_p']:.1e}) R2={m['r2']:.3f}",
              flush=True)

    # all proxies together
    Xall = np.column_stack([[r[f"d_{p}"] for r in pairs] for p in PROXIES])
    Xd = np.column_stack([np.ones(len(pairs)), Xall])
    beta, *_ = np.linalg.lstsq(Xd, d_err, rcond=None)
    resid = d_err - Xd @ beta
    dof = len(pairs) - Xd.shape[1]
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)))
    t0 = beta[0] / se[0]
    p0 = 2 * (1 - stats.t.cdf(abs(t0), dof))
    ss_tot = float(((d_err - d_err.mean()) ** 2).sum())
    multi = {"intercept": float(beta[0]), "intercept_p": float(p0),
             "coefs": {p: float(b) for p, b in zip(PROXIES, beta[1:])},
             "r2": float(1 - float(resid @ resid) / ss_tot),
             "n": len(pairs)}
    print(f"\n  all proxies: intercept {multi['intercept']:+.5f} (p={multi['intercept_p']:.1e})"
          f"  R2={multi['r2']:.3f}", flush=True)

    # per scenario
    per = {}
    for sc in sorted({r["scenario"] for r in pairs}):
        sel = [r for r in pairs if r["scenario"] == sc]
        de = np.array([r["d_err"] for r in sel])
        dx = np.array([r["d_flow_energy"] for r in sel])
        per[sc] = {"n": len(sel), "mean_d_err": float(de.mean()),
                   "mean_d_flow": float(dx.mean()),
                   "frac_cplx_reduced": float((dx < 0).mean()),
                   "ols_flow": ols(dx, de)}

    print(f"\n{'scenario':16s} {'n':>4s} {'d_err':>9s} {'d_flow':>9s} "
          f"{'frac cplx down':>15s} {'intercept':>11s} {'int p':>9s}")
    for sc, v in sorted(per.items(), key=lambda kv: kv[1]["mean_d_err"]):
        m = v["ols_flow"]
        print(f"{sc:16s} {v['n']:4d} {v['mean_d_err']:+9.5f} {v['mean_d_flow']:+9.4f} "
              f"{v['frac_cplx_reduced']:15.2f} "
              f"{(m['intercept'] if m else float('nan')):+11.5f} "
              f"{(m['intercept_p'] if m else float('nan')):9.1e}")

    explained = bool(models["flow_energy"]["slope_p"] < 0.05
                     and multi["intercept_p"] > 0.05)
    verdict = {
        "raw_effect": raw,
        "correlations": corr,
        "ols_single": models,
        "ols_all_proxies": multi,
        "per_scenario": per,
        "complexity_explains_effect": explained,
        "reading": (
            "complexity difference predicts the denoising difference AND the effect does "
            "not survive controlling for it -> the E0 main-effect *direction* must be "
            "reported as a complexity artefact (PREREG §5)"
            if explained else
            "the effect survives complexity matching -> the direction is not fully "
            "explained by the complexity of the edit"),
    }
    (OUT / "E4_complexity.json").write_text(json.dumps(
        {"verdict": verdict, "pairs": pairs}, indent=2, ensure_ascii=False))
    print(f"\n  complexity_explains_effect = {explained}")
    print(f"  {verdict['reading']}")
    print(f"\nwrote {OUT/'E4_complexity.json'}")


if __name__ == "__main__":
    main()
