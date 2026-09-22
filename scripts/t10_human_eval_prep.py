"""Human baseline: generate the blind pairwise evaluation package.

Design is fixed by MIGRATION_PLAN.md §4.3:
  blind ............. the rater is not told which clip is the violation
  paired ............ both clips of a pair shown together, "which looks more wrong?"
  order randomised .. left/right shuffled per pair to kill position bias
  no hint ........... never say "find the physics error", only "which looks off"
  confidence ........ record a 1-5 certainty alongside the choice
  coverage .......... prioritise the 5 saturated scenarios if the budget is short

Why it matters for T10 (PROTOCOL_VALIDITY §5): the first three evidence types are all
internal consistency. Human judgement is the only *external* anchor. The decisive cell is
"protocol says the probe result is untrustworthy AND humans also cannot tell them apart",
because that is the protocol's verdict being confirmed by a source that knows nothing about
the model.

This script only prepares the package; it renders no verdict.  Scoring happens in
`t10_human_score.py` once the answer sheet comes back.

Run: python scripts/t10_human_eval_prep.py --per-scenario 4
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T10_validity" / "human_eval"

# The five LikePhys scenarios whose probe result is saturated (E1_per_scenario_table.json,
# delta < 2x the probe null floor 0.0208) -- these are the highest-value rows.
PRIORITY_NOTE = ("saturated units first: the protocol says their probe numbers carry no "
                 "resolution, so they are where human agreement or disagreement is most "
                 "informative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=4,
                    help="pairs per scenario (valid vs one violation kind)")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "results/T3_null_floor/manifest.json").read_text())
    table = json.loads((ROOT / "results/T4_caliper/E1_per_scenario_table.json").read_text())

    floor2x = 0.0208
    saturated = {sc for sc, v in table.items() if v["delta"] < floor2x}

    by = {}
    for c in manifest:
        if c["group"] != "physics":
            continue
        by.setdefault((c["scenario"], c["subgroup"]), []).append(c)

    rng = random.Random(args.seed)
    pairs = []
    per_sc = {}
    # saturated scenarios first, then the rest
    order = sorted(by.items(), key=lambda kv: (kv[0][0] not in saturated, kv[0]))
    for (sc, sg), clips in order:
        if per_sc.get(sc, 0) >= args.per_scenario:
            continue
        valid = [c for c in clips if c["label"] == 0]
        viol = [c for c in clips if c["label"] == 1]
        if not valid or not viol:
            continue
        v = valid[0]
        w = rng.choice(viol)
        flip = rng.random() < 0.5
        left, right = (w, v) if flip else (v, w)
        pairs.append({
            "pair_id": f"P{len(pairs)+1:03d}",
            "scenario": sc, "subgroup": sg,
            "left_clip": left["clip_id"], "right_clip": right["clip_id"],
            "left_path": left["path"], "right_path": right["path"],
            # withheld from the rater sheet, kept here for scoring
            "_answer_violation_side": "left" if left["label"] == 1 else "right",
            "_violation_kind": w["kind"],
            "_scenario_saturated": sc in saturated,
            "_probe_real_auc": table[sc]["real"],
            "_probe_random_auc": table[sc]["random"],
            "_probe_delta": table[sc]["delta"],
        })
        per_sc[sc] = per_sc.get(sc, 0) + 1

    rng.shuffle(pairs)
    for i, p in enumerate(pairs, 1):
        p["order"] = i

    # 1) answer key (do NOT give to raters)
    (OUT / "ANSWER_KEY.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False))

    # 2) rater sheet: no answers, no scenario hint about which is edited
    sheet = OUT / "rater_sheet.csv"
    with sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "pair_id", "left_clip", "right_clip",
                    "your_choice(left/right/cannot_tell)", "confidence(1-5)", "notes"])
        for p in sorted(pairs, key=lambda x: x["order"]):
            w.writerow([p["order"], p["pair_id"], p["left_clip"], p["right_clip"], "", "", ""])

    # 3) instructions -- wording matters, so it is generated rather than left to the rater
    (OUT / "INSTRUCTIONS.md").write_text(f"""# 人评基线 — 评分说明

> **不要先读 `ANSWER_KEY.json`。** 那是答案，读了这批数据就废了。

## 你要做什么

`rater_sheet.csv` 每行是**一对**视频。两条都看完，然后回答：

> **哪一条看起来更不对？**

- 填 `left` / `right` / `cannot_tell`
- 再填一个 **1–5 的确定度**（1 = 纯猜，5 = 非常确定）
- `notes` 可选，写你觉得哪里不对

## 三条必须遵守的

1. **不要试图找「物理错误」。** 题目只问「哪条看起来更不对」。
   一旦你抱着找物理 bug 的心态看，就不再是我们要测的那个量了。
2. **不确定就填 `cannot_tell`。** 这一栏的比例本身是结果，不是失败。
3. **按 `order` 列的顺序做**，不要跳着挑好做的。

## 怎么播放

视频路径在 `pair_paths.csv` 里（`left_path` / `right_path`）。
clip 都是 512×512、约 2 秒，建议循环播放几遍再判断。

## 样本量

- 本包共 **{len(pairs)} 对**，覆盖 {len(per_sc)} 个场景
- 目标 **5–10 人各做完全部**；人手不够时优先保证覆盖，宁可少人做全，不要多人做半
- 每人独立做，**做完之前不要互相讨论**

## 为什么要做这个

前三类 validity 证据（构造效度、最小性、充分性边界）都是**内部一致性**。
人评是**唯一的外部锚**。

最有价值的情形是：**协议说某个场景的 probe 结果不可信，而人类在同一批 clip 上也分不清** ——
那就等于协议的判定被一个**完全不知道模型长什么样**的来源印证了。

反过来如果人类轻松分清而协议说不可信，那说明协议可能过严，也是需要知道的。

**所以 `cannot_tell` 和低确定度不是噪声，恰恰是我们要的信号。**

## 本包的构成

- 优先放了**饱和场景**（{PRIORITY_NOTE}）
- 每对的左右**已随机翻转**，位置不含信息
- 顺序已打乱，相邻两行不是同一场景
""")

    # 4) paths file, separate so the sheet stays clean
    with (OUT / "pair_paths.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "pair_id", "left_path", "right_path"])
        for p in sorted(pairs, key=lambda x: x["order"]):
            w.writerow([p["order"], p["pair_id"], p["left_path"], p["right_path"]])

    n_sat = sum(1 for p in pairs if p["_scenario_saturated"])
    meta = {
        "n_pairs": len(pairs),
        "n_scenarios": len(per_sc),
        "pairs_per_scenario": per_sc,
        "n_pairs_in_saturated_scenarios": n_sat,
        "saturated_scenarios": sorted(saturated),
        "left_is_violation_count": sum(1 for p in pairs
                                       if p["_answer_violation_side"] == "left"),
        "design": {
            "blind": True, "paired": True, "order_randomised": True,
            "no_physics_hint": True, "confidence_recorded": True,
            "seed": args.seed,
        },
        "files": {
            "rater_sheet.csv": "give to raters (no answers)",
            "pair_paths.csv": "video paths for playback",
            "INSTRUCTIONS.md": "give to raters",
            "ANSWER_KEY.json": "DO NOT give to raters",
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"{len(pairs)} pairs over {len(per_sc)} scenarios "
          f"({n_sat} in saturated scenarios)")
    print(f"left-is-violation: {meta['left_is_violation_count']}/{len(pairs)} "
          f"(should be near half)")
    for sc in sorted(per_sc):
        tag = " [saturated]" if sc in saturated else ""
        print(f"  {sc:16s} {per_sc[sc]} pairs  probe real "
              f"{table[sc]['real']:.3f} / random {table[sc]['random']:.3f}{tag}")
    print(f"\npackage -> {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
