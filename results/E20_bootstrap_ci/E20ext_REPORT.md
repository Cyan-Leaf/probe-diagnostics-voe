# E20-ext — 把置信区间补到全部 69 个格

> 执行于 2026-11-05，纯 CPU（14 min）。**方法与 `E20_PREREG.md` 完全一致**（折外分数 bootstrap，
> B=300，按场景聚类，各臂配对）。
> 原始数据：`E20ext_cells.json` / `E20ext_verdict.json` / `oof_cache/`。
> 脚本：`e20_extend.py`；挂接：`attach_d1d2_ci.py`。

## 0. 一句话结论

**D1 22/25 稳健、D2 25/25 稳健。** 加上 E20 的 12 格，全项目现在是
**D1 31/37 稳健、D2 37/37 稳健**，69 个格**全部有 95% 区间**。

三个新增脆弱格全部贴着 50% 阈值，其中 **E19-H3 的 L16 是唯一一个"交付时判 pass、
但区间跨阈值"的高显著性格里最值得注意的一个**。

## 1. 为什么补这一步

E20 只测了 E8 的 6 族 × 2 数据集。而 `D1_TALLY` / `D2_TALLY` 里还有 23 个 D1 行、22 个 D2 行
**只有点估计**。这是项目自己证据表里的不一致：两个同样写作 "fail" 的格，
一个检查过能否经受重采样、另一个没检查过。**这个差别不该由读者去猜。**

本次不重算任何点估计：每格的取值仍来自各实验自己的交付 JSON，
区间用与 E20 **逐字相同**的算术（`oof_scores` / `cluster_draws` / `boot_auc` / `compose`
全部 import，未复制）。

## 2. 结果总表

### 2.1 D1（25 格，本表新增）

| cell | E20-ext 点估计 | 95% CI | 脆弱? |
|---|---|---|---|
| E13 V-JEPA ViT-L @ IntPhys2 L16 flat | 92.91% | [87.54, 97.99] | |
| E13 V-JEPA ViT-L @ IntPhys2 L13 flat | 88.11% | [81.39, 93.29] | |
| E13 V-JEPA ViT-L @ IntPhys2 L16 timeavg | 94.19% | [83.88, 102.23] | |
| E13 V-JEPA ViT-L @ IntPhys2 L13 timeavg | 88.72% | [79.80, 96.04] | |
| E13 CogVideoX-2b @ IntPhys2 L16 (t0, clean) | 75.06% | [56.30, 93.87] | |
| E13 V-JEPA ViT-H @ IntPhys2 L16 | 82.66% | [74.74, 90.27] | |
| E13 V-JEPA ViT-H @ IntPhys2 L17 | 88.28% | [81.48, 94.02] | |
| E13 V-JEPA ViT-H @ LikePhys L16 | 59.77% | [50.92, 69.71] | |
| E13 V-JEPA ViT-H @ LikePhys L17 | 59.73% | [50.73, 69.98] | |
| E14 HunyuanVideo @ IntPhys2 L16 | 79.61% | [67.39, 88.84] | |
| E14 HunyuanVideo @ IntPhys2 Lrel | 79.23% | [68.57, 88.06] | |
| E15 HunyuanVideo @ LikePhys L16 | 26.26% | [19.77, 34.63] | |
| E15 HunyuanVideo @ LikePhys Lrel | 29.26% | [22.69, 37.20] | |
| **E15 V-JEPA ViT-L @ LikePhys L16** | **52.14%** | **[41.79, 62.41]** | ❌ |
| **E15 V-JEPA ViT-L @ LikePhys L13** | **53.67%** | **[43.52, 64.29]** | ❌ |
| E17 Wan2.2-A14B low @ IntPhys2 L16 | 98.90% | [90.57, 108.62] | |
| E17 Wan2.2-A14B low @ IntPhys2 Lrel | 95.66% | [86.01, 106.36] | |
| E17 Wan2.2-A14B high @ IntPhys2 L16 | 98.83% | [90.18, 109.60] | |
| E17 Wan2.2-A14B high @ IntPhys2 Lrel | 95.63% | [85.84, 106.46] | |
| E18 A14B low@t300 L16 | 87.92% | [77.26, 98.89] | |
| E18 A14B low@t300 Lrel | 89.27% | [76.50, 105.63] | |
| E18 A14B high@t900 L16 | 97.37% | [79.13, 119.68] | |
| E18 A14B high@t900 Lrel | 96.56% | [81.02, 114.65] | |
| **E19 MiniMax-H3 L16** | **60.56%** | **[42.30, 76.34]** | ❌ |
| E19 MiniMax-H3 Lrel | 79.56% | [68.45, 88.13] | |

### 2.2 D2（25 格，本表新增）：**全部 25/25 稳健**

**全表最小下界 +0.0248**，来自 E13 ViT-H @ LikePhys L17（CI [+0.0248, +0.0753]）；
排在其后的两格是 E13 ViT-H @ LikePhys L16（[+0.0251, +0.0747]）与
E15 V-JEPA ViT-L @ LikePhys L16（[+0.0255, +0.0819]）。**三格都在 LikePhys 上** ——
这正是 D2 gap 最小的数据集（交付值 +0.0289 … +0.1476）。
IntPhys 2 侧的最小下界是 **+0.2471**（E13 V-JEPA ViT-L L16 flat），比 LikePhys 高一个数量级。

完整数值见 `E20ext_cells.json`。

## 3. 三个新增脆弱格的含义

| 格 | 点估计 | CI | 为什么重要 |
|---|---|---|---|
| **E19 MiniMax-H3 L16** | 60.56% | **[42.30, 76.34]** | **唯一一个"pass 但区间跨阈值"的高显著性格**。E19 交付值 63.87% 本就标注了 `comparable_across_cells: false`（real AUC 0.5739 < 0.6 弱信号）；E20-ext **独立地**得出同一判断：这一格的 pass **不能当确定结论引用** |
| **E15 V-JEPA ViT-L @ LikePhys L16** | 52.14% | [41.79, 62.41] | 交付值 49.41% 已在 `D1_TALLY` 标为 `at_threshold`。区间跨阈值**印证**该标注 |
| **E15 V-JEPA ViT-L @ LikePhys L13** | 53.67% | [43.52, 64.29] | 同上（交付值 47.97%） |

**这三格有一个共同点：全部是 V-JEPA 或 H3，全部贴在阈值附近。**
E15 的两格在 `E15_vjepa_probe_seeds.json` 里本就记录了"跨 probe seed 的判定会翻"；
E20-ext 从**另一条轴**（场景抽样）再次确认了同一不确定性。**两条独立证据指向同一结论。**

## 4. 与交付值的一致性（自检）

| cell | E20-ext | 交付值 | 差 |
|---|---|---|---|
| E13 V-JEPA L16 flat | 92.91% | 92.86% | +0.05 |
| E13 V-JEPA L13 flat | 88.11% | 88.26% | −0.15 |
| E13 V-JEPA L16 timeavg | 94.19% | 94.63% | −0.44 |
| E13 ViT-H L16 | 82.66% | 83.25% | −0.59 |
| E14 L16 | 79.61% | 77.09% | +2.52 |
| E17 a14b low L16 | 98.90% | 99.12% | −0.22 |
| E18 low@300 L16 | 87.92% | 89.40% | −1.48 |
| E19 H3 L16 | 60.56% | 63.87% | −3.31 |

偏差 **−3.31 … +2.52 pp**，方向不一，量级在 E16 的 2.51 pp probe 划分 sd 量级内。
**这不是逐位复现**，原因与 E20 报告 §2/§6 相同（折外池化 vs 逐折平均；且 E13/H3 只用了单次特征取数）。

## 5. 局限（与 E20 一致，不重复论证）

1. **区间不含重拟合变异**（见 `E20_REPORT.md` §2）。
2. **V-JEPA 2 只有 seed 0**（确定性编码器，无噪声抽样），所以 E13/E15 的 V-JEPA 格的区间
   只反映场景重采样，不含 feature-seed 变异 —— 这一点使它们的区间**相对偏窄**。
3. **E13 的 `t0`(clean) 格**读的是 `cache/cogvideox_feats_real_intphys2_T0`，与 E8 的 t=600
   不是同一批特征，不可与 E8 行互相比较。
4. 覆盖仍限于 L16 / Lrel 两个层；E8 之外的 22 个格已全部覆盖，**但 E8 的 Lrel 层仍未做**。

## 6. 复现

```bash
python3 scripts/e20_extend.py --workers 14 --threads 8   # 14 min
python3 scripts/audit_d1_count.py && python3 scripts/audit_d2_count.py
python3 scripts/attach_d1d2_ci.py                        # 顺序不可颠倒
```

`results/E20_bootstrap_ci/oof_cache/` 存每个 (cell, seed, arm) 的折外分数，
重跑时命中缓存，下游改动不再需要重新拟合（这是第一次跑崩后加的）。
