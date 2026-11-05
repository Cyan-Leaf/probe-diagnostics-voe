# E15 预注册：LikePhys 侧的跨范式检验

**写于任何 LikePhys 提取之前：2026-10-28**

本文件是 `E13_PREREG.md` §8 与 `E14_PREREG.md` §5 里那两条 "LikePhys 暂缓" 的**解冻附件**，
不是新方向。两份预注册都写明：暂缓的理由是"先用 IntPhys 2 判定，避免全做完才发现判据有问题"。
IntPhys 2 的判定现已产出，解冻条件成立。

---

## 1. 要补的缺口（这是唯一的动机）

E13 报告 §7 第 4 条写着：

> **Limitations 增一条**：V-JEPA 2 只有 IntPhys 2 的结果。LikePhys 侧未测，
> 所以"非生成式编码器上 D1 也低"这件事**尚未检验**。

这是现在整个项目最大的一处空白，因为：

- **D1 的判定是数据集依赖的**，这已是 E8 的结论（IntPhys 2 上 6/7 pass，LikePhys 上 7/7 fail）
- E13 补了"非生成式 vs 生成式"这条轴，但**只在 IntPhys 2 上**
- 于是有一个直接的、可以一句话说清的问题：

> **在 LikePhys 上，非生成式编码器的 D1 也 < 50% 吗？**

- 若 **是** → "D1 由数据集决定"在范式轴上也成立，D1 的结论可以写成跨范式
- 若 **否** → D1 低是"生成式训练 + LikePhys"的联合产物，那么 E8 §14.5 那句
  "真正决定 D1 判定的是数据集，不是 backbone" **必须补上"但要看训练目标"**

**两种结果都必须照实写，且第二种会改主结论的措辞。** 这是本次唯一要回答的问题。

## 2. 模型与口径（全部沿用，不新增选择）

| 项 | 值 |
|---|---|
| V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34`，几何 256/64 帧，8192 token |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo@e8c2aaa66fe3`，几何 512/49 帧，13312 token |
| 输入 | LikePhys 原视频，`C.read_spanning` 在整段上均匀取帧（与 E13/E14 完全同一函数） |
| 参考格 | **L16 / t=600**（Hunyuan）；**L16 / clean**（V-JEPA，无 t） |
| 相对深度格 | Hunyuan L32（单流，**不是** L16 的深度对照）；V-JEPA L13 |
| 随机初始化 | norm-aware，与 E13/E14 完全相同 |
| 地板 | V-JEPA 用 **permutation within scenes**（`(scenario, subgroup)` 分组）；Hunyuan 用 seed-spread |
| probe / folds / C 网格 | 冻结不变 |

**LikePhys 的 D2 不需要 `cj` 变体**：清单里自带 `appearance` 组（50 clip，`color_change`），
E7/E8 一直用它。→ 本次不做 `cj`，只做 `plain` 的 real + random。

### 2.1 必须声明的偏离：64 帧输入 vs 56–61 帧的 clip

V-JEPA 2 的 pos embed 写死 64 帧，而 LikePhys 的 clip 只有 **56/60/61** 帧
（实测 40 个 clip 全是 60 帧）。`C.read_spanning(path, 64, ...)` 用
`np.linspace(0, n-1, 64).round()` 取索引，**短 clip 会出现重复索引**：

| clip 帧数 | 64 个索引里互不相同的 | 重复 |
|---|---|---|
| 56 | 56 | 8 |
| 60 | **60** | **4** |
| 61 | 61 | 3 |

后果量化：重复的是相邻两帧（时间上相隔 1/59 的间隔），因此
**tubelet=2 的 32 个时间位置里，约 2 个位置是"冻结"的**（同一帧出现两次）。
这不会改变 token 数（仍 8192），也不会改变训练/随机的对比（两边同样处理），
但它让 LikePhys 上 V-JEPA 的时间轴比 IntPhys 2 上略不平滑。

→ **声明方式**：报告必须写"LikePhys 的 V-JEPA 输入含 4/64 重复帧（60 帧 clip）"。
→ **不做的事**：不改为"循环补帧"或"改 pos embed"，那会让 LikePhys 与 IntPhys 2
   的预处理不同，从而污染跨数据集比较——本实验恰好就是要做跨数据集比较。

对比之下 **HunyuanVideo 不受影响**：它取 49 帧 < 56，`linspace` 不会重复。

## 3. 判据（写死）

**Q：`D1_likephys < 50%` 是否成立？**

- **成立（`pass_dataset`）**：V-JEPA 2 在 LikePhys 上 D1 < 50%
  → "D1 由数据集决定"跨范式成立，E13 的 Limitations 那条可以删掉
- **不成立（`training_objective_matters`）**：V-JEPA 2 在 LikePhys 上 D1 ≥ 50%
  → D1 低需要"生成式训练 **且** LikePhys"两个条件同时成立；
    **E8 §14.5 与 E13 §7 的措辞都要改**
- **不可解读**：T10 门槛不过 → 如实报，不给方向

**D1 的方向判定与数值同时报**，因为 LikePhys 上 `real AUC` 很高（0.84–0.97），
分母 `real − 0.5` 不小，T10 门槛不构成问题（与 LTX 在 IntPhys 2 的情形相反）。

**不做的事**：
- 不用 IntPhys 2 的任何数字去推断 LikePhys（那是 E8 已经证明会翻的）
- 不改阈值、不改 probe、不改随机初始化规则
- 不同时补 D3–D6

## 4. 提取量

| 项 | 数量 |
|---|---|
| V-JEPA 2 ViT-L LikePhys plain real | 920 |
| V-JEPA 2 ViT-L LikePhys plain random | 920 |
| HunyuanVideo LikePhys plain real | 920 |
| HunyuanVideo LikePhys plain random | 920 |
| **合计** | **3680 次前向** |

按实测速度（V-JEPA 1.1 s/clip、Hunyuan 4.1 s/clip），**4 卡约 25 分钟**。

（ViT-H 的 LikePhys **不做**：它是 E13 的规模轴辅助格，主结果已表明规模不翻判定，
再补 LikePhys 只是重复。若主结果出现反常再补。）

## 5. 交付路径

```
results/E15_likephys/
    E15_PREREG.md      本文档
    E15_cells.json     两模型 × 两层的逐格值
    E15_verdict.json   Q 的判定
    E15_REPORT.md      结论 + 对 E8/E13 措辞的具体修改清单
    manifest.json
```

---

## 6. 为什么这次只做两件事

因为现在的瓶颈**不是模型数量**，而是一处具体的、写在自己报告 Limitations 里的空白。
再加模型只会让"覆盖 7 个族"变成"覆盖 8 个族"，而这一格补上会让主结论的措辞从
"数据集决定"变成"数据集 + 训练目标共同决定"，或者反过来把它钉死。

**这是本轮唯一还能改变主结论措辞的实验。**
