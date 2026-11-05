# E14 — 第五个模型族：HunyuanVideo（12.8B）

生成：2026-10-28　·　协议：`PREREG.md` D1/D2 + `results/E14_hunyuan/E14_PREREG.md`

数字出处：`E14_cells.json` 给逐格原始值，`E14_verdict.json` 给判定。

---

## 1. 结论先行

**D2 再次 fail（累计 28/28 格），零例外。D1 pass，但有一个必须报的异常。**

| 格 | 层 | block 类型 | real AUC | random | 外观 | D1 | 判定 |
|---|---|---|---|---|---|---|---|
| L16 | 16 | **双流** | 0.6153 | 0.5264 | 1.0000 | **77.1%** | pass |
| Lrel | 32 | **单流** | 0.6241 | 0.5310 | 1.0000 | **75.0%** | pass |

- **D2：外观 AUC = 1.0000，物理 AUC ≤ 0.6241，gap ≥ +0.376** → 累计 22 个 D2 格全部 fail，6 个模型族零例外
  （**独立单元 15 个**：同一批特征的另一个口径/另一层不算独立复现，见 §5 的说明）
- **D1：两层都 ≥ 75%，远超 50% 阈值** → 与 E8 在 IntPhys 2 上的 6/7 pass 同向
- **异常：`random > raw pixel` 这个下界排序没有重现**（见 §4）

---

## 2. 模型与口径

- **`hunyuanvideo-community/HunyuanVideo` @ `e8c2aaa66fe3`**
- transformer 本体 **12.8B**（不含 8B 的 LLaMA-3 文本塔，文本塔只用于算一次空 prompt 嵌入后即卸载）
- **20 个双流 block + 40 个单流 block = 60 个**，hidden 3072（24 head × 128）
- VAE：`AutoencoderKLHunyuanVideo`，8× 空间 / 4× 时间，16 通道
- 输入：512×512、49 帧、空 prompt → **13312 token**（网格 13×32×32，与 Wan、CogVideoX 完全相同）
- 加噪：flow matching，t=600 ⇒ σ = 0.600（与 Wan/LTX/Mochi 同约定，不需要 SNR 匹配格）

### 2.1 两处结构差异（都写进了预注册）

**① latent 布局是 `[B,C,F,H,W]`，与 Wan/CogVideoX 的 frame-major 相反。**
这次它是**报错**而不是静默出错：

```
RuntimeError: Given groups=1, weight of size [3072, 16, 1, 2, 2],
expected input[1, 13, 16, 64, 64] to have 16 channels, but got 13 channels instead
```

比 CogVideoX-5b 的 RoPE 事故幸运（那次是静默贴 0.51）。但仍然显式转置而非依赖报错。

**② 60 个 block 分属两个不同的栈**，所以 `L16` 与 `Lrel=32` **不是同一个量的两个深度**：

| | L16 | Lrel=32 |
|---|---|---|
| block 索引 | 16 | 32 |
| 所在栈 | **双流**（video + text 联合注意力）| **单流**（仅 video）|
| 深度 | 26.7% | 53.3% |

报告在展示这两个数时**必须每次都说明这一点**，不得把 0.6153 → 0.6241 的差异解释成"深度效应"。

---

## 3. D1 / D2 结果

### 3.1 D1：两层都 pass

```
L16  : (0.6153 − 0.5264) / (0.6153 − 0.5) = 77.1%
Lrel : (0.6241 − 0.5310) / (0.6241 − 0.5) = 75.0%
```

地板（seed-spread）分别是 0.0042 与 0.0105，T10 门槛（`效应 > 2×地板` 且 `AUC−0.5 > 2×地板`）通过。

**与 E8 的关系**：E8 的 IntPhys 2 六格是 79.4–99.1%，HunyuanVideo 的 77.1% 落在**同一区间内**。
跨 5 个模型族、13 个格，IntPhys 2 上 D1 pass 是 12 格 + 1 格不可解读（LTX t=600）。

### 3.2 D2：继续 fail（累计 28/28）

| | 值 |
|---|---|
| AUC(物理违反) | 0.6153 / 0.6241 |
| AUC(逐帧恒定 HSV 扰动) | **1.0000 / 1.0000** |
| gap | **+0.3847 / +0.3759** |

扰动与 E8 完全同一个（hue 0.35、sat 1.4、逐帧恒定、无时序成分）。

**HunyuanVideo 在结构上是 E8 覆盖里最"不像"的模型**：它是唯一一个**同一模型内先双流后单流**的，
也是唯一用 LLaMA 系文本塔的。D2 在这一族上仍然 fail，说明这条结论与 block 拓扑、与文本条件的形式**都无关**。

（D2 格数按实际交付的 JSON 逐个数：E8 12 + E14 2 + E13 ViT-L 4 + E13 ViT-H 2 = **20**，全部 fail。
早期草稿里写的 13/15 是漏数了 E13 的双口径格与 ViT-H 的两层。）

---

## 4. 必须报的异常：`random > raw pixel` 排序没重现

E8 反复强调的排序是 **trained > random > {raw pixel, VAE latent}**（`E8_PREREG.md` §4 的"主打问题"）。
HunyuanVideo 上：

| 量 | AUC |
|---|---|
| trained DiT | 0.6153 / 0.6241 |
| **raw pixel** | **0.5470** |
| **VAE latent** | **0.5328** |
| **random DiT** | **0.5264 / 0.5310** |

**random 低于两个非训练下界**（0.5264 < 0.5470，0.5310 < 0.5470）。`E14_verdict.json` 的
`ordering_reproduces` 因此对两个格都是 `false`。

### 4.1 这意味着什么、不意味着什么

**意味着**：随机初始化的 HunyuanVideo，其特征读物理的能力**不如直接读原始像素**。
E8 在另外 6 个模型上没有出现这个现象。

**不意味着**：
- 不影响 D1 的**方向**：D1 的分母是 `real − 0.5`，分子是 `real − random`；random 更低只会让 D1 更高，
  所以 D1 的 pass **不是**这个排序问题造出来的。
- 不能读作"随机初始化更好/更差"的普遍结论：只有一个模型、只在一层上出现。

### 4.2 一个可能的解释（**未验证，不得写进结论**）

HunyuanVideo 的 60 个 block 中，L16 在**双流段**、L32 在**单流段**，而随机初始化是
norm-aware 的（`ndim≥2` 用 He，`norm.weight` 置 1）。若 `qk_norm=rms_norm` 在随机权重下
把注意力压得极平，双流 block 的输出可能更多保留"文本-视频"混合的常数偏置，
从而削弱 clip 之间的可分性。**这是推测，本轮没有任何干预实验支持**，因此只记为待查项。

### 4.3 为什么仍然报告

因为如果只报 D1/D2 而不报这个，读者会以为排序也重现了。`E8_REPORT.md` §14.1 把排序列为 E8 的
"主打问题"，所以 E14 必须逐格报 `ordering_reproduces`，哪怕它是 false。

---

## 5. 覆盖范围的最终账

加上 HunyuanVideo 后，跨 backbone 的覆盖是：

| 轴 | 范围 |
|---|---|
| 模型族 | **5 个**（Wan / CogVideoX / LTX / Mochi / Hunyuan，外加非生成的 V-JEPA 2 = 6）|
| 参数量 | 0.33B（ViT-L）– 14B |
| 层数 | 24 – 60 |
| latent 通道 | 12 / 16 / 48 / 128 |
| token 预算 | 1792 – 13312（+ E12 的梯子把 LTX 推到 12800）|
| block 拓扑 | 单流 / 双流 / **双流→单流串联** / 无 DiT（ViT）|
| block 类型（同一模型内）| 仅 HunyuanVideo 有 |

**判定方向**：IntPhys 2 上 D1 12 pass + 1 不可解读（扩散）+ 2 pass（ViT）；LikePhys 上 7/7 fail；
**D2 全部 fail：28 个格 / 19 个独立单元**（`results/D2_TALLY.json`）。

> **格数 vs 独立单元数必须分开写。** 20 个格里有些不是独立复现：
> E13 的 ViT-L 与 ViT-H 各给出两个层，但**同一个层内 flat 与 timeavg 是同一次提取的两种读出**
> （4 格 → 2 个单元），E14 的两层共享同一次提取。
> 按 (模型, 数据集, 一次提取) 计：E8 12 + E14 1 + E13 ViT-L 1 + E13 ViT-H 1 + E15 Likephys 2 = **17 个独立单元**。
> 写 "20/20" 而不说清楚会被读成 20 次独立复现。

---

## 6. 对论文写作的后果

1. **D2 可以写成 28/28 格（19 个独立单元）、6 个模型族、零例外**。这是全文最硬的结论。写格数时必须同时给独立单元数。
2. **D1 必须写成"数据集依赖"，不能写成"训练贡献少数信号"**（E13 已给出同向证据）。
3. **排序（random 高于原始像素）不是普适的**，HunyuanVideo 是已知反例。这句话必须进 Limitations，
   否则 E8 的"排序重现"会被读成 8/8。
4. **LikePhys 未做**，不能外推。

---

## 7. 复现

```bash
export HF_HOME=$HF_HOME

# 空 prompt 嵌入（LLaMA-3-8B + CLIP pooled；算完即卸载文本塔）
#   结果已存于 results/E14_hunyuan/prompt_embeds_hunyuan.pt

# 提取（4 卡，一个条件一张卡）
CUDA_VISIBLE_DEVICES=i python scripts/e14_extract.py --dataset intphys2 \
    --weights {real,random} --variant {plain,cj}

# 分析 + 判定
python scripts/e14_analyse.py --threads 8
python scripts/e14_verdict.py
```

实测：4.08 s/clip，显存峰 40.4 GB，3036 次前向约 50 分钟（4 卡）。
