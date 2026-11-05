# E15 附录：LikePhys 侧补 D2（写于计算之前，2026-10-28）

`E15_REPORT.md` §4 把这件列为未完成项：两个新模型（V-JEPA 2 ViT-L、HunyuanVideo）
**LikePhys 侧的 D2 没有重做**，因为 E15 预注册只问了 D1 那一个问题。
本附录把它补上，**只用现有缓存**（`appearance` 组 50 clip 已在 920 个提取里），零 GPU。

## 判据（与 E8 完全一致，不改）

E8 在 LikePhys 上用的是**清单自带的** `color_change` 组，不是人造扰动：

- **正类**：`kind == "color_change"`（50 clip，5 个场景 × 10 subgroup）—— 外观改变，物理不变
- **负类**：`group == "physics" and label == 0`（120 clip，物理合法）—— 未编辑
- **判定**：`AUC(合法 vs 变色) ≥ AUC(合法 vs 物理违反)` → D2 fail
- **分组**：`(scenario, subgroup)`，与 E8 的 `subkey` 同口径
- **参考格**：V-JEPA 用 L16（clean，无 t）；Hunyuan 用 L16（t=600）
- **地板**：V-JEPA 用 permutation-within-scenes；Hunyuan 用 seed-spread（与各自 E13/E14 一致）

**为什么用清单自带的而不是自造扰动**：E8 §14.2 的 LikePhys 行（0.9641–1.0000）全部
基于这个组，自造扰动会让两边不可比。这一点在 `E8_REPORT.md` §14.2 与 `PREREG.md`
（`color_change` 含时序成分的声明）里都已记录。

## 已知限制（先写，避免事后合理化）

LikePhys 的 `color_change` **含时序成分**——LikePhys 原文三处写明变色发生在
"mid-flight" / "mid-simulation" / "mid-flow"，且被列为 invalid variant。
所以它不是一个"纯静态外观扰动"，与 IntPhys 2 的 `cj`（逐帧恒定 HSV）**不完全同质**。
→ 报告必须写明：**LikePhys 的 D2 与 IntPhys 2 的 D2 用的不是同一种扰动**，
   因此只能各自与本数据集的物理对照比，不能横向比 gap 的大小。

## 预期与判据

E8 的 6 个模型在 LikePhys 上 D2 **6/6 fail**（外观 AUC 0.9641–1.0000）。
本次补测两个模型，**判据不变**，结果照实报（含若出现 pass 的情形）。
