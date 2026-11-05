# E8 预注册补充（写于任何提取之前）

上游协议：`PREREG.md` D1 / D2，参考格 `layer 16 / t=600`。
本文档只记录**换 backbone 时必须先定死的三件事**，不重复 D1/D2 的定义。

## 1. 模型

`zai-org/CogVideoX-2b` @ `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`

选它的理由只有一条：**`num_layers = 30`，与 Wan2.2-TI2V-5B 相同**。
参考格是 layer 16，在两者上都等于第 16/30 层、相对深度 53.3%，
所以"固定 layer 16"在两个 backbone 上是同一件事，不需要重新定义参考格。
（LTX-Video 为 28 层，layer 16 会漂到 57%，所以没选。）

输入口径与 Wan 一致以便比对：512×512、49 帧、空 prompt。
- latent：[1, 16, 13, 64, 64]（8× 空间、4× 时间下采样）
- 视频 token：13 × 32 × 32 = **13312**，与 Wan TI2V-5B 完全相同
- 隐层维度：1920

## 2. t=600 不是同一个噪声水平（必须并列报两个 timestep）

两个 backbone 的加噪形式本来就不同，**不能把 t 当成可以直接对齐的量**：

| | 加噪形式 | t=600 处的有效噪声 |
|---|---|---|
| Wan2.2 | flow matching，`z_t=(1-σ)z₀+σε`，σ=t/1000 | σ=0.600，SNR=0.444 |
| CogVideoX | VP/DDIM，`z_t=√ᾱ·z₀+√(1-ᾱ)·ε` | ᾱ=0.0458，σ_eff=**0.9768**，SNR=0.048 |

CogVideoX 在 t=600 已经几乎是纯噪声（97.7%），
此时任何 probe 都接近随机，D1 的分母 `AUC_real−0.5` 会退化，
**T10 的可解释性门槛会直接把该格判为 NOT interpretable**。

因此预注册两个格，**都报，都不改口径**：

- **主格 `t=600`（字面）**：与 `PREREG.md` 的参考格逐字对齐，这是 E8 必须交的格。
- **辅格 `t=298`（SNR 匹配）**：取 SNR 与 Wan t=600 相等处（0.4449 vs 0.4444），
  用于回答"排序是否重现"这个真问题。**标 secondary，不替代主格。**

t=298 由 `Ṅ = argmin |SNR(t) − 0.4444|` 算出，在跑任何提取之前写死于此。

## 3. 随机初始化对照：不能照抄 Wan 的初始化

`wan_probe_lib._init_weights_` 把**所有一维参数置零**，其中包括 RMSNorm 的 weight。
实测 Wan 上这没问题（层内归一化的调制走 `scale_shift_table`，是二维参数）。

**CogVideoX 上照抄会让整个网络塌成常数。** 已实测：按同样规则置零后，
不同 clip 之间的特征 std = **0.000000**（完全相同），
因为 `CogVideoXLayerNormZero` 的 affine 归一化 weight 被清零后 `norm(x)=0`，
block 退化为 `x = shift`，与输入无关。

→ **CogVideoX 的随机对照采用 norm-aware 初始化**，规则写死如下：

- `ndim >= 2`：`N(0, 2/fan_in)`（与 Wan 相同的 He 规则）
- `ndim < 2` 且参数名以 `norm.weight` 结尾：**置 1.0**
- 其余一维参数（所有 bias、norm.bias）：置 0
- 浮点 buffer（`patch_embedding.pos_embedding`）：`N(0, 0.02)`（不置零）

实测该规则下不同 clip 的特征 std 非零（0.000158），且不退化。
**这是一次有意的偏离，必须在 `manifest.json` 的 `deviations` 里声明**，
并说明它使随机下界的绝对值不可与 Wan 直接比较（只比排序）。

## 4. 判据（与 PREREG.md 一致，不改）

- **D1**：`(AUC_real − AUC_random) / (AUC_real − 0.5) < 50%` → 主张不成立
  - 叠加 T10 可解释性门槛：仅当 `效应 > 2×地板` **且** `AUC_real − 0.5 > 2×地板` 时可解读
- **D2**：`AUC(valid vs 外观扰动) ≥ AUC(valid vs 物理违反)` → 无法区分，主张不成立
  - 地板 = seed-spread（定义 A），feature seed 0/1/2
- **分组**：IntPhys2 用 `SceneIndex`，LikePhys 用 `(scenario, subgroup)`
- **主打问题**：**排序**（训练 > 随机 > 像素）与 D1/D2 的**判定方向**是否重现，
  **不要求**数值复现

## 5. 数据与提取量（先算清再跑）

| 数据集 | real plain | random plain | real cj | random cj |
|---|---|---|---|---|
| IntPhys2 Main | 1012 | 1012 | 506 | 506 |
| LikePhys | 920 | 920 | — | — |

LikePhys 的 D2 直接用清单里已有的 `color_change` 组（50 clip）作外观对照，
**不需要额外跑 jitter**；IntPhys2 没有自带重着色对照，所以需要 `cj` 变体（只对 Possible 做）。

合计 4876 次单 clip 提取，每次 3 个 feature seed。
