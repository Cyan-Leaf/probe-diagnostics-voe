"""T3 / E0 — null floor: how much of the metric moves when only the noise seed changes.

Reads cache/feats_real/*.pt (denoising errors, 5 seeds x 5 timesteps) and reports
  1. clip-level seed noise
  2. the valid-vs-violation paired effect (Wilcoxon + bootstrap CI + effect sizes)
  3. the seed-induced spread of that *effect estimate*  <- the error bar the factorial design needs
  4. variance decomposition (seed noise vs between-clip variance)
plus the temporal / appearance control groups.

Run: uv run --no-project python scripts/t3_null_floor.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T3_null_floor"
CACHE = ROOT / "cache" / "feats_real"
BOOT = 10_000
BOOT_SEED = 12345


def load_errors() -> tuple[list[dict], dict]:
    rows = []
    env = {}
    for f in sorted(CACHE.glob("*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        env = env or {"env": rec["env"], "preflight": rec["preflight"], "config": rec["config"]}
        c = rec["clip"]
        for t, per_seed in rec["errors"].items():
            for seed, e in per_seed.items():
                rows.append(
                    {
                        "clip_id": c["clip_id"],
                        "scenario": c["scenario"],
                        "subgroup": c["subgroup"],
                        "kind": c["kind"],
                        "group": c["group"],
                        "label": c["label"],
                        "t": int(t),
                        "seed": int(seed),
                        "mse": float(e["mse"]),
                    }
                )
    return rows, env


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def bootstrap_ci(vals: np.ndarray, n: int = BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    from scipy import stats

    rows, env = load_errors()
    assert rows, f"no cached errors in {CACHE}"
    timesteps = sorted({r["t"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    clips = sorted({r["clip_id"] for r in rows})
    print(f"loaded {len(rows)} rows: {len(clips)} clips x {len(timesteps)} t x {len(seeds)} seeds")

    by = defaultdict(dict)  # (clip_id, t) -> {seed: mse}
    meta = {}
    for r in rows:
        by[(r["clip_id"], r["t"])][r["seed"]] = r["mse"]
        meta[r["clip_id"]] = r

    report: dict = {
        "provenance": env,
        "n_clips": len(clips),
        "timesteps": timesteps,
        "seeds": seeds,
        "bootstrap": {"n": BOOT, "seed": BOOT_SEED},
        "clip_level_seed_noise": {},
        "effect_valid_vs_violation": {},
        "seed_spread_of_effect": {},
        "variance_decomposition": {},
        "controls": {},
        "per_scenario": {},
    }

    # ---- 1. clip-level seed noise ----------------------------------------------------
    for t in timesteps:
        rel_ranges, stds, means = [], [], []
        for cid in clips:
            v = np.array([by[(cid, t)][s] for s in seeds if s in by[(cid, t)]], dtype=np.float64)
            if len(v) < 2:
                continue
            rel_ranges.append((v.max() - v.min()) / v.mean())
            stds.append(v.std(ddof=1))
            means.append(v.mean())
        rel_ranges, stds, means = map(np.array, (rel_ranges, stds, means))
        report["clip_level_seed_noise"][str(t)] = {
            "n_clips": int(len(rel_ranges)),
            "median_rel_range_pct": float(np.median(rel_ranges) * 100),
            "p90_rel_range_pct": float(np.percentile(rel_ranges, 90) * 100),
            "median_seed_std": float(np.median(stds)),
            "median_clip_mean": float(np.median(means)),
            "median_cv_pct": float(np.median(stds / means) * 100),
        }

    # ---- 2/3. paired effect + its seed-induced spread --------------------------------
    subgroups = sorted({(meta[c]["scenario"], meta[c]["subgroup"]) for c in clips})

    def paired_diffs(t: int, seed_set, group="physics") -> tuple[np.ndarray, list]:
        diffs, keys = [], []
        for sc, sg in subgroups:
            pos = [c for c in clips if meta[c]["scenario"] == sc and meta[c]["subgroup"] == sg
                   and meta[c]["kind"] == "valid"]
            neg = [c for c in clips if meta[c]["scenario"] == sc and meta[c]["subgroup"] == sg
                   and meta[c]["group"] == group and meta[c]["label"] == 1]
            if group != "physics":
                neg = [c for c in clips if meta[c]["scenario"] == sc and meta[c]["subgroup"] == sg
                       and meta[c]["group"] == group]
            if not pos or not neg:
                continue

            def avg(cs):
                return float(np.mean([np.mean([by[(c, t)][s] for s in seed_set if s in by[(c, t)]])
                                      for c in cs]))

            diffs.append(avg(neg) - avg(pos))
            keys.append((sc, sg))
        return np.array(diffs), keys

    for t in timesteps:
        d, keys = paired_diffs(t, seeds)
        lo, hi = bootstrap_ci(d)
        w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        pos_pairs = np.array([np.mean([by[(c, t)][s] for s in seeds])
                              for c in clips if meta[c]["kind"] == "valid"])
        neg_pairs = np.array([np.mean([by[(c, t)][s] for s in seeds])
                              for c in clips if meta[c]["group"] == "physics" and meta[c]["label"] == 1])
        report["effect_valid_vs_violation"][str(t)] = {
            "n_pairs": int(len(d)),
            "mean_paired_diff": float(d.mean()),
            "median_paired_diff": float(np.median(d)),
            "boot_ci95": [lo, hi],
            "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "cohens_d_paired": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
            "cliffs_delta_unpaired": float(cliffs_delta(neg_pairs, pos_pairs)),
            "frac_pairs_positive": float((d > 0).mean()),
            "mean_valid_mse": float(pos_pairs.mean()),
            "mean_violation_mse": float(neg_pairs.mean()),
            "relative_effect_pct": float(d.mean() / pos_pairs.mean() * 100),
        }

        per_seed_means = []
        for s in seeds:
            ds, _ = paired_diffs(t, [s])
            per_seed_means.append(float(ds.mean()))
        report["seed_spread_of_effect"][str(t)] = {
            "per_seed_mean_effect": per_seed_means,
            "spread_abs": float(max(per_seed_means) - min(per_seed_means)),
            "std_across_seeds": float(np.std(per_seed_means, ddof=1)),
            "effect_over_seed_spread": (
                float(abs(np.mean(per_seed_means)) / (max(per_seed_means) - min(per_seed_means)))
                if max(per_seed_means) != min(per_seed_means) else None
            ),
        }

        # ---- 4. variance decomposition ----------------------------------------------
        within = []
        clip_means = []
        for cid in clips:
            v = np.array([by[(cid, t)][s] for s in seeds if s in by[(cid, t)]])
            within.append(v.var(ddof=1))
            clip_means.append(v.mean())
        within_var = float(np.mean(within))
        between_var = float(np.var(clip_means, ddof=1))
        report["variance_decomposition"][str(t)] = {
            "within_clip_seed_var": within_var,
            "between_clip_var": between_var,
            "icc_like_between_over_total": between_var / (between_var + within_var),
            "seed_noise_share_pct": within_var / (between_var + within_var) * 100,
        }

        # ---- controls ----------------------------------------------------------------
        ctl = {}
        for grp in ("temporal", "appearance"):
            dg, kg = paired_diffs(t, seeds, group=grp)
            if len(dg) >= 5:
                lo2, hi2 = bootstrap_ci(dg)
                w2 = stats.wilcoxon(dg, alternative="two-sided", zero_method="wilcox")
                ctl[grp] = {
                    "n_pairs": int(len(dg)),
                    "mean_paired_diff": float(dg.mean()),
                    "boot_ci95": [lo2, hi2],
                    "wilcoxon_p": float(w2.pvalue),
                    "relative_effect_pct": float(dg.mean() / pos_pairs.mean() * 100),
                }
        report["controls"][str(t)] = ctl

        # ---- per scenario ------------------------------------------------------------
        per_sc = {}
        for sc in sorted({m["scenario"] for m in meta.values()}):
            sel = [i for i, k in enumerate(keys) if k[0] == sc]
            if not sel:
                continue
            dv = d[sel]
            per_sc[sc] = {
                "n_pairs": int(len(dv)),
                "mean_paired_diff": float(dv.mean()),
                "frac_positive": float((dv > 0).mean()),
            }
        report["per_scenario"][str(t)] = per_sc

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E0_null_floor.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ---- console digest ---------------------------------------------------------------
    print("\n=== E0 null floor digest ===")
    print(f"{'t':>5} {'seedCV%':>8} {'relRange%':>10} {'effect':>10} {'effect%':>9} "
          f"{'seedSpread':>11} {'eff/spread':>11} {'p':>10} {'seedShare%':>11}")
    for t in timesteps:
        a = report["clip_level_seed_noise"][str(t)]
        b = report["effect_valid_vs_violation"][str(t)]
        c = report["seed_spread_of_effect"][str(t)]
        v = report["variance_decomposition"][str(t)]
        ratio = c["effect_over_seed_spread"]
        print(f"{t:>5} {a['median_cv_pct']:>8.2f} {a['median_rel_range_pct']:>10.2f} "
              f"{b['mean_paired_diff']:>10.5f} {b['relative_effect_pct']:>8.2f}% "
              f"{c['spread_abs']:>11.6f} {ratio if ratio is None else round(ratio,2):>11} "
              f"{b['wilcoxon_p']:>10.2e} {v['seed_noise_share_pct']:>11.3f}")
    print(f"\nwrote {OUT/'E0_null_floor.json'}")


if __name__ == "__main__":
    main()
