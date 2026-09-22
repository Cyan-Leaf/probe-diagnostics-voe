"""Plots for T3 (E0 null floor) and T4 (E1 CALIPER / layer x timestep scan).

Run after the corresponding analysis jsons exist:
  uv run --no-project python scripts/make_plots.py --which e0 e1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T3 = ROOT / "results" / "T3_null_floor"
T4 = ROOT / "results" / "T4_caliper"


def plot_e0() -> None:
    rep = json.loads((T3 / "E0_null_floor.json").read_text())
    ts = [int(t) for t in rep["timesteps"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    eff = [rep["effect_valid_vs_violation"][str(t)]["mean_paired_diff"] for t in ts]
    lo = [rep["effect_valid_vs_violation"][str(t)]["boot_ci95"][0] for t in ts]
    hi = [rep["effect_valid_vs_violation"][str(t)]["boot_ci95"][1] for t in ts]
    spread = [rep["seed_spread_of_effect"][str(t)]["spread_abs"] for t in ts]
    ax.errorbar(ts, eff, yerr=[np.array(eff) - np.array(lo), np.array(hi) - np.array(eff)],
                marker="o", capsize=4, label="valid→violation paired effect (95% boot CI)")
    ax.bar(ts, spread, width=40, alpha=0.35, color="crimson",
           label="seed-induced spread of the effect (null floor)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("diffusion timestep t")
    ax.set_ylabel("Δ denoising error")
    ax.set_title("E0: effect vs seed noise floor")
    ax.legend(fontsize=8)

    ax = axes[1]
    cv = [rep["clip_level_seed_noise"][str(t)]["median_cv_pct"] for t in ts]
    rr = [rep["clip_level_seed_noise"][str(t)]["median_rel_range_pct"] for t in ts]
    share = [rep["variance_decomposition"][str(t)]["seed_noise_share_pct"] for t in ts]
    ax.plot(ts, cv, "o-", label="median per-clip seed CV %")
    ax.plot(ts, rr, "s-", label="median per-clip seed range %")
    ax.plot(ts, share, "^-", label="seed share of total variance %")
    ax.set_xlabel("diffusion timestep t")
    ax.set_ylabel("percent")
    ax.set_title("E0: clip-level seed noise")
    ax.legend(fontsize=8)

    ax = axes[2]
    t_ref = 600 if 600 in ts else ts[len(ts) // 2]
    per_sc = rep["per_scenario"][str(t_ref)]
    names = sorted(per_sc, key=lambda k: per_sc[k]["mean_paired_diff"])
    vals = [per_sc[k]["mean_paired_diff"] for k in names]
    floor = rep["seed_spread_of_effect"][str(t_ref)]["spread_abs"]
    colors = ["tab:green" if v > floor else "tab:orange" if v > 0 else "tab:red" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(floor, color="crimson", ls="--", lw=1, label=f"null floor ({floor:.4g})")
    ax.set_xlabel(f"mean paired Δ at t={t_ref}")
    ax.set_title("E0: per-scenario effect")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(T3 / "E0_null_floor.png", dpi=160)
    print(f"wrote {T3/'E0_null_floor.png'}")


def plot_e1() -> None:
    rep = json.loads((T4 / "E1_summary.json").read_text())
    raw = json.loads((T4 / "E1_probe_raw.json").read_text())

    real = [r for r in raw if r["source"] == "real_dit"]
    rand = [r for r in raw if r["source"] == "random_dit"]
    t_ref = rep["reference_timestep"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    axes = axes.ravel()

    ax = axes[0]
    for src, rows, style in (("real_dit", real, "o-"), ("random_dit (CALIPER lower bound)", rand, "s--")):
        sel = sorted([r for r in rows if r["t"] == t_ref and r["seed"] == 0], key=lambda r: r["layer"])
        if not sel:
            continue
        ax.errorbar([r["layer"] for r in sel], [r["auc_mean"] for r in sel],
                    yerr=[r["auc_std"] for r in sel], fmt=style, capsize=3, label=src)
    for src, key, col in (("vae_latent", "vae_latent", "tab:green"),
                          ("raw_pixel", "raw_pixel", "tab:red")):
        v = rep["lower_bounds"].get(key)
        if v:
            ax.axhline(v["auc_mean"], color=col, ls=":", label=f"{src} ({v['auc_mean']:.3f})")
    ax.axhline(0.5, color="k", lw=0.8, label="chance")
    ax.set_xlabel("DiT block index")
    ax.set_ylabel("AUC (GroupKFold-5 mean ± std)")
    ax.set_title(f"E1: layer scan at t={t_ref} (pooled, n=750)")
    ax.legend(fontsize=7)

    ax = axes[1]
    layers = sorted({r["layer"] for r in real})
    ts = sorted({r["t"] for r in real})
    grid = np.full((len(layers), len(ts)), np.nan)
    for r in real:
        if r["seed"] != 0:
            continue
        grid[layers.index(r["layer"]), ts.index(r["t"])] = r["auc_mean"]
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[-0.5, len(ts) - 0.5, min(layers) - 0.5, max(layers) + 0.5])
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(ts)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("DiT block index")
    ax.set_title("E1: layer × timestep AUC (real DiT, seed 0)")
    plt.colorbar(im, ax=ax, label="AUC")

    ax = axes[2]
    labels, vals, errs = [], [], []
    best = rep["best_real"]
    labels.append(f"real_dit\nL{best['layer']} t{best['t']}")
    vals.append(best["auc_mean"])
    errs.append(best["auc_std"])
    for k, v in rep["lower_bounds"].items():
        labels.append(k.replace("(best layer)", "\n(best layer)"))
        vals.append(v["auc_mean"])
        errs.append(v["auc_std"])
    ax.bar(labels, vals, yerr=errs, capsize=4,
           color=["tab:blue"] + ["tab:gray"] * (len(vals) - 1))
    ax.axhline(0.5, color="k", lw=0.8)
    floor = rep["probe_null_floor"]["auc_seed_spread"]
    ax.axhspan(max(vals[1:]) - floor, max(vals[1:]) + floor, color="crimson", alpha=0.15,
               label=f"±probe null floor ({floor:.4f})")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, max(1.02, max(vals) + 0.06))
    ax.set_title("E1: CALIPER discriminability verdict (pooled)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[3]
    ctl_path = T4 / "E1_controls.json"
    if ctl_path.exists():
        ctl = json.loads(ctl_path.read_text())
        tasks = ["valid_vs_violation", "valid_vs_color_change", "valid_vs_temporal_disorder"]
        srcs = ["real_dit", "random_dit", "vae_latent", "raw_pixel"]
        w = 0.2
        xs = np.arange(len(tasks))
        for j, s in enumerate(srcs):
            v = [next((r["auc_mean"] for r in ctl if r["task"] == t and r["source"] == s), np.nan)
                 for t in tasks]
            e = [next((r["auc_std"] for r in ctl if r["task"] == t and r["source"] == s), np.nan)
                 for t in tasks]
            ax.bar(xs + j * w - 1.5 * w, v, w, yerr=e, capsize=2, label=s)
        ax.set_xticks(xs)
        ax.set_xticklabels(["physics\nviolation", "appearance only\n(color_change)",
                            "frame order\n(temporal_disorder)"], fontsize=8)
        ax.axhline(0.5, color="k", lw=0.8)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("AUC")
        ax.set_title("E1 controls: what does the probe actually read?")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(T4 / "E1_caliper.png", dpi=160)
    print(f"wrote {T4/'E1_caliper.png'}")

    # per-scenario figure
    ps_path = T4 / "E1_per_scenario_table.json"
    if ps_path.exists():
        ps = json.loads(ps_path.read_text())
        names = sorted(ps, key=lambda k: ps[k]["delta"])
        x = np.arange(len(names))
        fig2, ax2 = plt.subplots(figsize=(11, 4.6))
        for j, (key, lbl) in enumerate((("real", "real_dit"), ("random", "random_dit"),
                                        ("vae", "vae_latent"), ("pixel", "raw_pixel"))):
            ax2.bar(x + j * 0.2 - 0.3, [ps[n][key] for n in names], 0.2, label=lbl)
        ax2.axhline(0.5, color="k", lw=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("within-scenario AUC")
        ax2.set_ylim(0.4, 1.05)
        ax2.set_title("E1: within-scenario probes (test folds are tiny — see report caveat)")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(T4 / "E1_per_scenario.png", dpi=160)
        print(f"wrote {T4/'E1_per_scenario.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", choices=["e0", "e1"], required=True)
    args = ap.parse_args()
    if "e0" in args.which:
        plot_e0()
    if "e1" in args.which:
        plot_e1()


if __name__ == "__main__":
    main()
