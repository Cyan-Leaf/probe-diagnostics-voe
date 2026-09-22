"""T11.2: generate the figures for the Experiments section.

F1-F7 correspond to the placeholders in SPEC.md §6-8.  The ones that depend on the human
baseline (F-human, exact numbering TBD) are left for T11.1 after the ratings come back.

Priority:
  F1 (two-dataset lower bounds)  ............ P0, the headline finding
  F7 (protocol validity 4-panel) ............ P0, answers the instructor's question
  F2 (layer×t heatmaps) ..................... P1
  F4 (per-scenario ceiling) ................. P1
  F5 (E4 complexity attribution) ............ P1
  F6 (D6 proof: AUC vs d' decay) ............ P1
  F3 (D1-D6 × 4 papers audit table) ......... P2, can be typeset rather than plotted

Output directory: results/T11_figures/
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T12_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ======================================================================================= F1
def f1_lower_bounds():
    """Two-dataset lower-bound comparison: the headline evidence that edit-based benchmarks
    let everything through."""
    e7lb = json.loads((ROOT / "results/E7_intphys2/E7_lower_bounds.json").read_text())
    like = {"raw_pixel": 0.7886, "vae_latent": 0.8200, "random_dit": 0.8353,
            "trained_dit": 0.9686, "appearance": 1.0000}
    intp = {"raw_pixel": e7lb["sources"]["raw_pixel"]["auc_mean"],
            "vae_latent": e7lb["sources"]["vae_latent"]["auc_mean"],
            "random_dit": e7lb["comparison"]["random_dit_auc"],
            "trained_dit": e7lb["comparison"]["real_dit_auc"],
            "appearance": 1.0000}

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(5)
    w = 0.35
    labels = ["Raw pixel", "VAE latent", "Random DiT", "Trained DiT", "Appearance\ncontrol"]
    lv = [like[k] for k in ["raw_pixel", "vae_latent", "random_dit", "trained_dit", "appearance"]]
    iv = [intp[k] for k in ["raw_pixel", "vae_latent", "random_dit", "trained_dit", "appearance"]]

    ax.bar(x - w/2, lv, w, label="LikePhys (edit-based)", color="#d29922", edgecolor="#000", linewidth=0.6)
    ax.bar(x + w/2, iv, w, label="IntPhys 2 (rendered)", color="#4c8dff", edgecolor="#000", linewidth=0.6)
    ax.axhline(0.5, color="#666", linestyle="--", linewidth=0.8, zorder=0, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probe AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="upper left", frameon=True, edgecolor="#ccc")
    ax.set_title("Figure 1. Lower bounds: edit-based vs rendered violations", fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.5)

    # value labels instead of connector lines: the lines drew through the bars and read
    # as plotting errors rather than as a same-row comparison
    for i, (l, ip) in enumerate(zip(lv, iv)):
        ax.text(i - w/2, l + 0.008, f"{l:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, ip + 0.008, f"{ip:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "F1_lower_bounds.png")
    fig.savefig(OUT / "F1_lower_bounds.pdf")
    plt.close(fig)
    print(f"F1 saved: {OUT / 'F1_lower_bounds.png'}")


# ======================================================================================= F2
def f2_profiles():
    """Timestep and layer profiles.

    Only 1-D sweeps exist (E1 stored profiles, not the full layer x timestep grid), so this
    is two line plots rather than the heatmap triptych originally sketched. The random-init
    bound is drawn as a horizontal reference because it was only computed at its best layer.
    """
    e1 = json.loads((ROOT / "results/T4_caliper/E1_summary.json").read_text())
    rand = e1["lower_bounds"]["random_dit(best layer)"]["auc_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    tp = e1["timestep_profile"]
    ts = sorted(int(k) for k in tp)
    ax1.plot(ts, [tp[str(t)] for t in ts], marker="o", linewidth=2,
             color="#4c8dff", label="Trained DiT")
    ax1.axhline(rand, color="#9aa3b2", linestyle="--", linewidth=1.5,
                label=f"Random DiT ({rand:.3f})")
    ax1.axhline(0.5, color="#666", linestyle=":", linewidth=0.8, alpha=0.6, label="Chance")
    ax1.set_xlabel("Denoising timestep t")
    ax1.set_ylabel("Probe AUC")
    ax1.set_ylim(0.45, 1.02)
    ax1.set_title("Timestep profile (layer 16)", fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, edgecolor="#ccc", fontsize=7)

    lp = e1["layer_profile_at_reference_t"]
    ax2.plot([r["layer"] for r in lp], [r["auc_mean"] for r in lp], marker="o",
             linewidth=2, color="#4c8dff", label="Trained DiT")
    ax2.axhline(rand, color="#9aa3b2", linestyle="--", linewidth=1.5, label="Random DiT")
    ax2.axhline(0.5, color="#666", linestyle=":", linewidth=0.8, alpha=0.6, label="Chance")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Probe AUC")
    ax2.set_ylim(0.45, 1.02)
    ax2.set_title(f"Layer profile (t={e1['reference_timestep']})", fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, edgecolor="#ccc", fontsize=7)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linestyle=":", linewidth=0.5)

    fig.suptitle("Figure 2. Probe AUC profiles (LikePhys)", fontweight="bold",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "F2_profiles.png")
    fig.savefig(OUT / "F2_profiles.pdf")
    plt.close(fig)
    print(f"F2 saved: {OUT / 'F2_profiles.png'}")


# ======================================================================================= F4
def f4_ceiling():
    """Per-scenario AUC with the saturated units flagged (D3)."""
    t = json.loads((ROOT / "results/T4_caliper/E1_per_scenario_table.json").read_text())
    floor2x = 0.0208  # 2 x probe null floor, E1_summary.verdict
    scens = sorted(t)
    real = [t[s]["real"] for s in scens]
    rnd = [t[s]["random"] for s in scens]
    sat = [t[s]["delta"] < floor2x for s in scens]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(scens))
    ax.bar(x, real, color=["#f85149" if s else "#4c8dff" for s in sat],
           edgecolor="#000", linewidth=0.6)
    ax.scatter(x, rnd, color="#111", s=44, marker="_", linewidths=2, zorder=3,
               label="Random DiT")
    ax.axhline(0.5, color="#666", linestyle=":", linewidth=0.8, alpha=0.6, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=45, ha="right")
    ax.set_ylabel("Probe AUC")
    ax.set_ylim(0.45, 1.04)
    ax.set_title("Figure 4. Per-scenario probe AUC "
                 f"({sum(sat)}/{len(scens)} saturated, in red)",
                 fontweight="bold", pad=10)
    handles = [mpatches.Patch(facecolor="#f85149", edgecolor="#000", linewidth=0.6,
                              label=f"Saturated (real−random < {floor2x})"),
               mpatches.Patch(facecolor="#4c8dff", edgecolor="#000", linewidth=0.6,
                              label="Resolvable")]
    ax.legend(handles=handles + list(ax.get_legend_handles_labels()[0]),
              loc="lower left", frameon=True, edgecolor="#ccc", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "F4_ceiling.png")
    fig.savefig(OUT / "F4_ceiling.pdf")
    plt.close(fig)
    print(f"F4 saved: {OUT / 'F4_ceiling.png'}")


# ======================================================================================= F5
def f5_complexity():
    """E4: the denoising effect tracks complexity, and reverses once matched within scenario."""
    e4 = json.loads((ROOT / "results/E4_complexity/E4_decomposition.json").read_text())
    rows = e4["per_scenario"]
    v = e4["verdict"]
    x = np.array([r["mean_d_flow_energy"] for r in rows])
    y = np.array([r["raw_mean_d_err"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    ax1.scatter(x, y, s=46, alpha=0.85, edgecolor="#000", linewidth=0.6, color="#4c8dff")
    ax1.axhline(0, color="#666", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.axvline(0, color="#666", linestyle=":", linewidth=0.8, alpha=0.6)
    for r in rows:
        if abs(r["raw_mean_d_err"]) > 0.012:
            ax1.annotate(r["scenario"], (r["mean_d_flow_energy"], r["raw_mean_d_err"]),
                         fontsize=6.5, ha="right", va="bottom", alpha=0.75)
    ax1.set_xlabel("Δ optical-flow energy (violation − valid)")
    ax1.set_ylabel("Δ denoising error")
    ax1.set_title("Complexity tracks the effect", fontweight="bold", fontsize=9.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.22, linestyle=":", linewidth=0.5)

    labels = ["Raw\npooled", "Pooled\nregression", "Within-scenario\nmatched"]
    vals = [v["global_mean_d_err"], v["pooled_intercept"],
            v["within_scenario_weighted_intercept"]]
    cols = ["#9aa3b2", "#d29922", "#3fb950"]
    ax2.bar(labels, vals, color=cols, edgecolor="#000", linewidth=0.6, width=0.6)
    ax2.axhline(0, color="#111", linewidth=0.9)
    ax2.set_ylabel("Mean Δ denoising error")
    ax2.set_title("Sign reverses once matched", fontweight="bold", fontsize=9.5)
    for i, val in enumerate(vals):
        ax2.text(i, val + (0.0006 if val >= 0 else -0.0006), f"{val:+.5f}",
                 ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5,
                 fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.5)

    fig.suptitle("Figure 5. The denoising readout is a complexity meter, not a physics meter",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "F5_complexity.png")
    fig.savefig(OUT / "F5_complexity.pdf")
    plt.close(fig)
    print(f"F5 saved: {OUT / 'F5_complexity.png'}")


# ======================================================================================= F6
def f6_d6_proof():
    """D6 in one picture: the same folds, AUC blind and d' not."""
    s4v = json.loads((ROOT / "results/E10_S4/S4_verdict.json").read_text())
    s4dp = json.loads((ROOT / "results/E10_S4/S4_exploratory_dprime.json").read_text())
    levels = ["0", "1", "2", "3"]
    px = [0, 32, 64, 128]
    per_sc = {r["scenario"]: r for r in s4v["per_scenario"]}
    dpr = s4dp["verdict"]["dprime_relative_to_level0"]["real"]
    scens = sorted(dpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
    cmap = plt.get_cmap("tab10")

    for i, s in enumerate(scens):
        ax1.plot(px, [per_sc[s]["real_auc"][l] for l in levels], marker="o",
                 linewidth=1.4, alpha=0.85, color=cmap(i), label=s)
        ax2.plot(px, [dpr[s][l] for l in levels], marker="o",
                 linewidth=1.4, alpha=0.85, color=cmap(i), label=s)

    ax1.axhline(1.0, color="#d29922", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(4, 0.9993, "ceiling", fontsize=7, color="#d29922", va="top")
    ax1.set_ylim(0.974, 1.003)
    ax1.set_ylabel("Probe AUC")
    ax1.set_xlabel("Camera-motion excursion (px)")
    ax1.set_title("AUC: max drop 0.0200 → “no decay”", fontweight="bold", fontsize=9.5)

    ax2.axhline(1.0, color="#666", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_ylim(0.6, 1.04)
    ax2.set_ylabel("d′ relative to level 0")
    ax2.set_xlabel("Camera-motion excursion (px)")
    ax2.set_title("d′: declines in 7/7", fontweight="bold", fontsize=9.5)
    ax2.legend(loc="lower left", frameon=True, edgecolor="#ccc", fontsize=6.5, ncol=2)

    for ax in (ax1, ax2):
        ax.set_xticks(px)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22, linestyle=":", linewidth=0.5)

    fig.suptitle("Figure 6. Why D6 exists: identical folds, one statistic blind to the effect",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "F6_d6_proof.png")
    fig.savefig(OUT / "F6_d6_proof.pdf")
    plt.close(fig)
    print(f"F6 saved: {OUT / 'F6_d6_proof.png'}")


# ======================================================================================= F3
def f3_audit():
    """D1-D6 x 4 papers: which diagnostics the existing literature reports.

    Rendered as a marker grid rather than a heatmap: the cells are categorical judgements
    (reported / approximated / not reported / not applicable), and a colour ramp would
    imply an ordering between them that does not exist.
    """
    m = json.loads((ROOT / "results/E6_literature/E6_audit_matrix.json").read_text())
    papers = m["papers"]
    checks = m["checks"]

    style = {
        "yes":     ("#3fb950", "✓", "Reported"),
        "partial": ("#d29922", "~", "Approximated"),
        "no":      ("#f85149", "✗", "Not reported"),
        "na":      ("#8b949e", "–", "N/A (postdates paper)"),
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    nrow, ncol = len(checks), len(papers)

    for r, chk in enumerate(checks):
        for c, pap in enumerate(papers):
            v = chk["verdicts"][pap["id"]]
            col, mark, _ = style[v]
            ax.add_patch(mpatches.Rectangle((c - 0.45, nrow - 1 - r - 0.42), 0.9, 0.84,
                                            facecolor=col, edgecolor="#000", linewidth=0.7))
            ax.text(c, nrow - 1 - r, mark, ha="center", va="center",
                    fontsize=15, color="#fff", fontweight="bold")

    ax.set_xlim(-0.6, ncol - 0.4)
    ax.set_ylim(-0.6, nrow - 0.4)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([p["short"] for p in papers], fontsize=8.5)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f'{c["id"]}  {c["name"]}' for c in reversed(checks)], fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [mpatches.Patch(facecolor=col, edgecolor="#000", linewidth=0.6, label=lab)
               for col, _, lab in style.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=7.5, handlelength=1.2)

    ax.set_title("Figure 3. Which diagnostics the existing literature reports",
                 fontweight="bold", pad=12)
    fig.savefig(OUT / "F3_audit.png")
    fig.savefig(OUT / "F3_audit.pdf")
    plt.close(fig)
    print(f"F3 saved: {OUT / 'F3_audit.png'}")


# ======================================================================================= F7
def f7_validity():
    """Protocol validity: the four independent lines of evidence.

    Laid out with constrained_layout and short axis labels: the first version let panel A's
    xlabel collide with panel C's title and clipped panel B's tick labels.
    """
    abl = json.loads((ROOT / "results/T10_validity/T10_1_ablation.json").read_text())
    con = json.loads((ROOT / "results/T10_validity/T10_2_construct.json").read_text())
    mdl = json.loads((ROOT / "results/T10_validity/T10_5_mdl.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), layout="constrained")
    checks = ["D1", "D2", "D3", "D4", "D5", "D6"]

    # --- A: minimality ablation -------------------------------------------------------
    ax = axes[0, 0]
    missed = [abl["ablation"][c]["n_missed"] for c in checks]
    ax.barh(checks, missed, color="#f85149", edgecolor="#000", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Units missed if removed", fontsize=8)
    ax.set_xlim(0, max(missed) * 1.18)
    ax.set_title("A. Minimality — none removable", fontweight="bold", loc="left", fontsize=9.5)
    for i, v in enumerate(missed):
        ax.text(v + max(missed) * 0.02, i, str(v), va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.text(0.98, 0.04, f"n = {abl['summary']['n_units']} units", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#666")

    # --- B: construct validity --------------------------------------------------------
    ax = axes[0, 1]
    names = ["A1", "A2", "A3", "A4"]
    sub = ["positive", "negative", "mixed", "saturated"]
    fired = ["\n".join(con["cases"][c]["failed_checks"]) or "none" for c in names]
    ok = [con["cases"][c]["protocol_correct"] for c in names]
    ax.bar(range(4), [1] * 4, color=["#3fb950" if o else "#f85149" for o in ok],
           edgecolor="#000", linewidth=0.6)
    for i, (o, f) in enumerate(zip(ok, fired)):
        ax.text(i, 0.62, "✓" if o else "✗", ha="center", va="center",
                fontsize=17, color="#fff", fontweight="bold")
        ax.text(i, 0.26, f, ha="center", va="center", fontsize=6.4, color="#fff",
                linespacing=1.35)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{n}\n{s}" for n, s in zip(names, sub)], fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    ax.set_title("B. Construct validity — 4/4 correct", fontweight="bold", loc="left",
                 fontsize=9.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, -0.30, "label = checks that fired", transform=ax.transAxes,
            ha="center", fontsize=7, color="#666")

    # --- C: the C2 attack -------------------------------------------------------------
    ax = axes[1, 0]
    c2 = con["cases"]["C2"]
    passed = [c not in c2["failed_checks"] for c in checks]
    ax.barh(checks, [1] * 6, color=["#f85149" if p else "#3fb950" for p in passed],
            edgecolor="#000", linewidth=0.6)
    ax.invert_yaxis()
    for i, p_ in enumerate(passed):
        ax.text(0.5, i, "passed" if p_ else "failed", ha="center", va="center",
                fontsize=8, color="#fff", fontweight="bold")
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("C. C2 attack — all six passed", fontweight="bold", loc="left", fontsize=9.5)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, -0.16, "a pipeline leak that encodes no physics\n⇒ minimal NECESSARY, not sufficient",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#333")

    # --- D: MDL cross-check -----------------------------------------------------------
    ax = axes[1, 1]
    comp = mdl["comparison"]
    vals = [comp["auc_based_D1_share"] * 100, comp["mdl_based_training_share"] * 100]
    bars = ax.bar(["AUC\n(ours)", "MDL codelength\n(Voita & Titov)"], vals,
                  color=["#4c8dff", "#9b59d0"], edgecolor="#000", linewidth=0.6, width=0.55)
    ax.axhline(50, color="#666", linestyle="--", linewidth=0.9)
    ax.text(-0.42, 51.2, "D1 threshold (50%)", va="bottom", ha="left", fontsize=7,
            color="#666")
    ax.set_ylabel("Training contribution (%)", fontsize=8)
    ax.set_ylim(0, 62)
    ax.set_title("D. Same verdict under both statistics", fontweight="bold", loc="left",
                 fontsize=9.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                va="bottom", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=7.5)

    fig.suptitle("Figure 7. Protocol validity: four independent lines of evidence",
                 fontweight="bold", fontsize=11.5)
    fig.savefig(OUT / "F7_validity.png")
    fig.savefig(OUT / "F7_validity.pdf")
    plt.close(fig)
    print(f"F7 saved: {OUT / 'F7_validity.png'}")


# ======================================================================================= F8
def f8_human():
    """Human baseline against the protocol's verdict.

    EXPLORATORY: n = 1 rater, and that rater is the project owner, so this is not the
    independent external anchor PROTOCOL_VALIDITY §5 asks for.  Plotted because the
    direction of the result matters for how D3 is worded, and flagged in the title so the
    figure cannot be read as a 5-10 rater result.
    """
    hb = json.loads((ROOT / "results/T10_validity/human_eval/"
                     "human_baseline_results.json").read_text())
    per = hb["per_scenario"]
    scens = sorted(per, key=lambda s: per[s]["probe_real_auc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    x = np.arange(len(scens))
    w = 0.38
    probe = [per[s]["probe_real_auc"] for s in scens]
    human = [per[s]["human_accuracy"] for s in scens]
    sat = [per[s]["saturated"] for s in scens]

    ax1.bar(x - w/2, probe, w, label="Probe AUC (real DiT)", color="#4c8dff",
            edgecolor="#000", linewidth=0.6)
    ax1.bar(x + w/2, human, w, label="Human accuracy (n=1)", color="#3fb950",
            edgecolor="#000", linewidth=0.6)
    ax1.axhline(0.5, color="#666", linestyle=":", linewidth=0.8, alpha=0.6, label="Chance")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}*" if t else s for s, t in zip(scens, sat)],
                        rotation=45, ha="right", fontsize=7.5)
    ax1.set_ylabel("AUC / accuracy")
    ax1.set_ylim(0, 1.08)
    ax1.legend(loc="lower right", frameon=True, edgecolor="#ccc", fontsize=7)
    ax1.set_title("Humans resolve every scenario, including the saturated ones",
                  fontweight="bold", fontsize=9.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.5)
    ax1.text(0.01, 0.03, "* protocol flags as saturated", transform=ax1.transAxes,
             fontsize=6.5, color="#666")

    # shadow_camera singled out: the cell we wanted and did not get
    sc = per["shadow_camera"]
    labels = ["Probe AUC", "Human acc"]
    vals = [sc["probe_real_auc"], sc["human_accuracy"]]
    ax2.bar(labels, vals, color=["#f85149", "#3fb950"], edgecolor="#000",
            linewidth=0.6, width=0.5)
    ax2.axhline(0.5, color="#666", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("AUC / accuracy")
    ax2.set_title("shadow_camera:\nprobe at chance, humans at 4/4",
                  fontweight="bold", fontsize=9.5)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.5)
    ax2.text(0.5, -0.30, f"mean confidence {sc['mean_confidence']:.2f}/5 "
                         f"(lowest of all scenarios)",
             transform=ax2.transAxes, ha="center", fontsize=7, color="#666")

    fig.suptitle("Figure 8. Human baseline vs the protocol's verdict "
                 "— EXPLORATORY, n=1 non-blind rater",
                 fontweight="bold", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "F8_human.png")
    fig.savefig(OUT / "F8_human.pdf")
    plt.close(fig)
    print(f"F8 saved: {OUT / 'F8_human.png'}")



def main():
    f1_lower_bounds()
    f2_profiles()
    f4_ceiling()
    f5_complexity()
    f6_d6_proof()
    f3_audit()
    f7_validity()
    f8_human()
    print(f"\nFigures written to {OUT}/")


if __name__ == "__main__":
    main()
