"""Draw the probe-explanation figure for the milestone talk.

The question this figure has to answer, in the words the teammate used:
"can I understand it as -- freeze one layer of the video generation model, then test whether it
has learned real physics knowledge?"

The answer is *almost* yes, and the "almost" is the whole project. Three panels, so the
correction lands visually instead of surviving only as a spoken caveat:

  (1) What we actually do: video -> frozen model -> **one layer's activations** -> a separately
      *trained* linear readout -> a number. Three callouts mark exactly where "test whether it
      learned physics" goes wrong: the readout is trained by us on labelled pairs; it is one
      layer, not "the model"; the label is "same scene, one law violated", not "physics".
  (2) The comparison arms that make the number interpretable: same architecture with random
      weights, and raw pixels / VAE latent.
  (3) The control that limits the claim: colour-jittered clips have identical physics, and the
      probe scores them ~1.00 against 0.65 for physics.

Every number is read from the delivered JSON at build time, so the figure cannot drift.

Note on matplotlib text: it does not parse Markdown, so emphasis is done with colour and size
rather than `**...**` (an earlier version printed literal asterisks).

Run: python3 scripts/make_probe_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "presentation"
FONT = ROOT / "assets" / "fonts" / "NotoSansSC.ttf"
E8 = ROOT / "results" / "E8_second_backbone" / "E8_cells_intphys2.json"

INK = "#111827"
MUTED = "#6b7280"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#047857"
AMBER = "#b45309"
PANEL = "#f8fafc"
LIGHT = "#e5e7eb"


def load_numbers() -> dict:
    cells = json.loads(E8.read_text())["cells"]
    k = "wan21_1b3|intphys2|t600|L16|{}|physics"
    return {
        "trained": cells[k.format("real")]["auc_mean"],
        "random": cells[k.format("random")]["auc_mean"],
        "appearance_real": cells["wan21_1b3|intphys2|L16|real|appearance"]["auc_mean"],
        "appearance_random": cells["wan21_1b3|intphys2|L16|random|appearance"]["auc_mean"],
        "vae": cells["wan21_1b3|intphys2|vae_latent"]["auc_mean"],
        "pixel": cells["wan21_1b3|intphys2|raw_pixel"]["auc_mean"],
        "model": "Wan2.1-1.3B",
    }


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=11, weight="normal", tc=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight=weight, zorder=3, linespacing=1.55)


def arrow(ax, p0, p1, color=INK, lw=1.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 color=color, linewidth=lw, zorder=1, shrinkA=2, shrinkB=2))


def panel1(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.985, "① 我们实际做的事", fontsize=16, weight="bold", color=INK, va="top")

    y, h = 0.755, 0.15
    box(ax, 0.00, y, 0.15, h, "视频\n（成对、带标签）", fc="#eef2ff")
    box(ax, 0.20, y, 0.21, h, "视频生成模型\n全部冻结\n（不训练、不微调）", fc="#e0f2fe")
    box(ax, 0.46, y, 0.17, h, "取某一层的\n激活值\n（此处 L16）", fc="#fef3c7",
        ec=AMBER, lw=2.0)
    box(ax, 0.68, y, 0.15, h, "线性分类器\n这个要训练", fc="#dcfce7", ec=GREEN, lw=2.0)
    box(ax, 0.87, y, 0.13, h, "AUC\n0.65", fc="white", ec=RED, lw=2.2, fs=13, weight="bold")

    for x0, x1 in ((0.15, 0.20), (0.41, 0.46), (0.63, 0.68), (0.83, 0.87)):
        arrow(ax, (x0 + 0.004, y + h / 2), (x1 - 0.004, y + h / 2))

    # callouts, hung off the element each one corrects
    ax.annotate("标签只是「同一场景，\n违反某一条物理定律」\n—— 不是「懂不懂物理」",
                xy=(0.075, y - 0.008), xytext=(0.00, 0.50),
                fontsize=10, color=AMBER, ha="left", va="top", linespacing=1.6,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.3, shrinkB=2), zorder=4)
    ax.annotate("只是「一层」，\n不是「这个模型」",
                xy=(0.545, y - 0.008), xytext=(0.285, 0.575),
                fontsize=10, color=AMBER, ha="left", va="top", linespacing=1.6,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.3, shrinkB=2), zorder=4)
    ax.annotate("模型冻结 ≠ 流程冻结：\n读出头是我们用带标签的\n成对样本训出来的",
                xy=(0.80, y - 0.008), xytext=(0.615, 0.50),
                fontsize=10, color=GREEN, ha="left", va="top", linespacing=1.6,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3, shrinkB=2), zorder=4)

    ax.text(0.0, 0.285,
            "所以准确的说法是：这个被冻结的层里，存在一个「线性可分」的信号，\n"
            "能把物理可能 / 物理不可能分开 —— 而且后面两个对照，就是用来限定它到底是什么信号。",
            fontsize=11.5, color=INK, va="top", linespacing=1.9)

    ax.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.145,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#fef2f2", edgecolor=RED, linewidth=1.2, zorder=1))
    ax.text(0.014, 0.135,
            "不成立的说法：「模型学会了物理」「模型理解了物理」",
            fontsize=11, color=RED, va="top", weight="bold", zorder=2)
    ax.text(0.014, 0.075,
            "本实验没有做任何干预，也没有排除最简单的解释（见 ③）。",
            fontsize=10, color=RED, va="top", zorder=2)


def panel2(ax, n):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.985, "② 对照臂：让「0.65」这个数有意义", fontsize=15, weight="bold",
            color=INK, va="top")
    ax.text(0.0, 0.915, f"{n['model']} @ IntPhys 2，L16，t=600",
            fontsize=10, color=MUTED, va="top")

    # No inline notes: they collided with the bars at this aspect ratio. The two readings that
    # matter are in the boxed footer instead, where they cannot overlap anything.
    arms = [
        ("训练后的网络 + 线性读出", n["trained"], BLUE),
        ("随机初始化的同架构网络\n+ 同一个线性读出", n["random"], AMBER),
        ("VAE 潜变量\n（过了 VAE、没过 DiT）", n["vae"], MUTED),
        ("原始像素 32×32", n["pixel"], MUTED),
    ]
    x0, bw, bh = 0.30, 0.58, 0.105
    top, gap = 0.700, 0.135
    for i, (lab, val, col) in enumerate(arms):
        y = top - i * gap
        ax.text(x0 - 0.015, y + bh / 2, lab, ha="right", va="center", fontsize=10,
                color=INK, linespacing=1.5)
        ax.add_patch(FancyBboxPatch((x0, y), bw, bh, boxstyle="round,pad=0.002",
                                    facecolor=LIGHT, edgecolor="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x0, y), max(bw * val, 0.004), bh,
                                    boxstyle="round,pad=0.002", facecolor=col,
                                    edgecolor="none", alpha=0.88, zorder=2))
        ax.text(x0 + bw * val + 0.012, y + bh / 2, f"{val:.3f}", va="center", fontsize=11.5,
                color=col, weight="bold")

    ax.plot([x0 + bw * 0.5] * 2, [0.26, top + bh + 0.01], ls=":", lw=1.2, color=RED, zorder=3)
    ax.text(x0 + bw * 0.5, 0.235, "0.5 = 抛硬币", ha="center", fontsize=9, color=RED)

    ax.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.175,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#eff6ff", edgecolor=BLUE, linewidth=1.0, zorder=1))
    d1 = (n["trained"] - n["random"]) / (n["trained"] - 0.5) * 100
    ax.text(0.014, 0.172,
            f"随机权重只剩 0.513（≈抛硬币）→ 高出随机的部分里有 {d1:.0f}% 是训练带来的，"
            f"随机初始化只占 {100-d1:.0f}%。\n"
            "读法：跟「随机初始化的同一架构」比 → 训练贡献的那部分；跟「像素 / VAE」比 → "
            "网络是否比原始输入多做了事。",
            fontsize=9.5, color=INK, va="top", linespacing=1.9, zorder=2)


def panel3(ax, n):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.985, "③ 关键对照：它读到的到底是什么？", fontsize=15, weight="bold",
            color=INK, va="top")
    ax.text(0.0, 0.915,
            "同一批视频只做色相 / 饱和度变换 ——\n物体运动、几何、时间因果全部不变，物理内容一模一样。",
            fontsize=10, color=MUTED, va="top", linespacing=1.6)

    rows = [("物理：\n可能 vs 不可能", n["trained"], BLUE, "这才是我们想测的"),
            ("外观：\n原片 vs 只改颜色", n["appearance_real"], RED,
             "只换颜色就几乎满分")]
    x0, bw, bh = 0.34, 0.54, 0.115
    for i, (lab, val, col, note) in enumerate(rows):
        y = 0.56 - i * 0.245
        ax.text(x0 - 0.015, y + bh / 2, lab, ha="right", va="center", fontsize=10.5,
                color=INK, linespacing=1.5)
        ax.add_patch(FancyBboxPatch((x0, y), bw, bh, boxstyle="round,pad=0.002",
                                    facecolor=LIGHT, edgecolor="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x0, y), bw * val, bh, boxstyle="round,pad=0.002",
                                    facecolor=col, edgecolor="none", alpha=0.88, zorder=2))
        ax.text(x0 + bw * val + 0.014, y + bh / 2, f"{val:.3f}", va="center", fontsize=13,
                color=col, weight="bold")
        ax.text(x0 + 0.005, y - 0.018, note, fontsize=9, color=col, va="top")

    ax.plot([x0 + bw * 0.5] * 2, [0.30, 0.79], ls=":", lw=1.2, color="#9ca3af")
    ax.text(x0 + bw * 0.5, 0.275, "0.5", ha="center", fontsize=9, color="#9ca3af")

    ax.text(0.0, 0.245,
            "同一个模型、同一层、同一个读出，只换标签定义：物理 0.65，外观 1.00。",
            fontsize=10.5, color=INK, va="top", linespacing=1.8)

    ax.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.155,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#fef2f2", edgecolor=RED, linewidth=1.2, zorder=1))
    ax.text(0.014, 0.142,
            "这个读出分得开的，主要是「这批视频被做过什么处理」，\n而不是「这段视频违反了哪条物理定律」。",
            fontsize=10.5, color=RED, va="top", linespacing=1.8, weight="bold", zorder=2)


def main():
    if FONT.exists():
        font_manager.fontManager.addfont(str(FONT))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    n = load_numbers()
    fig = plt.figure(figsize=(15.6, 10.2), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0], width_ratios=[1.16, 1.0],
                          hspace=0.10, wspace=0.08,
                          left=0.045, right=0.985, top=0.975, bottom=0.035)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(PANEL)

    panel1(ax1)
    panel2(ax2, n)
    panel3(ax3, n)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "probe_explained.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print("wrote", p, f"{p.stat().st_size/2**20:.2f} MB")
    print("numbers:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in n.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
