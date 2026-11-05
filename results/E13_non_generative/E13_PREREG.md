# E13 预注册：非生成式编码器上的 D1/D2

**写于任何 V-JEPA 2 提取之前：2026-10-28**
上游协议：`PREREG.md` D1 / D2，参考格沿用 `layer 16 / t=600`

---

## 1. 为什么要做这个

E8 的最终矩阵覆盖 **7 个模型**，但它们有一个共同点：**全是 video diffusion transformer**，都靠去噪目标训练、都从同一个 benchmark 的"被编辑过的视频"里读特征。

E8 §1 原本的结论是「训练只贡献少数信号」，报告把它收窄为「**这些 benchmark + 线性读出**这个组合的特性」。但这个收窄**没有证据支持**——所有 7 个模型都是生成式的。若换一个**不生成视频**的编码器仍然看到同样模式，收窄成立；若模式消失，那说明这条结论依赖去噪训练本身，而这才是更精确的说法。

E13 就用一个非生成式编码器问这个问题。

## 2. 模型：V-JEPA 2

| | |
|---|---|
| 仓库 | `facebook/vjepa2-vitl-fpc64-256` @ `b3c1679b7c34` |
| 规模 | ViT-L，约 0.3B |
| 训练目标 | **联合嵌入预测（JEPA），不是去噪** |
| 输入 | 64 帧 @ 256×256（模型自带 pos embed 定死这个尺寸）|
| token | tubelet 2 → 32 个时间位置 × (16×16)=256 → **8192 token** |
| 层 | 24 |
| VAE | **无** |
| timestep | **无** |

第二个规模 `facebook/vjepa2-vith-fpc64-256`（ViT-H，约 0.6B，32 层）作规模轴，**只在主结果出来后**才跑，避免又一次"先跑一大堆再想统计"。

## 3. D1 / D2 在非生成式模型上怎么定义（**这是本文档最重要的部分**）

### 3.1 D1 照原样定义，但有一处必须改

原 D1：
`D1 = (AUC_real − AUC_random) / (AUC_real − 0.5)`，其中 random 是**同架构随机初始化**。

- **"random = 同架构随机初始化" 这一条不变**，V-JEPA 2 也做 `_init_norm_aware` 对照。
  所以 `D1` 的**字面定义可以直接用**，不需要改判据、不需要改阈值。
- **变的只有一件事**：原定义里 "real features" 是在**加噪 latent 上**取的（t=600）。
  V-JEPA 2 没有加噪步，只能从**干净视频**取特征。

→ **因此 E13 的 D1 与 E8 的 D1 不是同一个量，禁止直接比较数值。** 这是本预注册的第一条硬约束。

能比的是**判定方向**：训练过的编码器是否也把大部分可分性交给训练？
为让比较有意义，E13 额外报一个**同口径对照**：CogVideoX-2b 在 **t=0（干净 latent）** 上的 D1。
两者都是"干净输入 + 训练 vs 随机"，这才可比。这个 t=0 格**必须跑**，否则 E13 的判定无法与 E8 对接。

### 3.2 D2 照原样定义，且**完全可比**

D2 问的是"外观扰动是否比物理违反更容易读出"。它只依赖
（i）物理违反对照、（ii）外观扰动对照、（iii）线性读出，**与模型是否生成、是否加噪无关**。

→ **D2 的判定与 E8 的 12 个格完全可比**，这是 E13 最干净的一条。外观对照沿用 E8 的 `cj` 变体
（逐帧恒定 HSV 扰动，hue 0.35 / sat 1.4），不新造。

### 3.3 下界：只剩三个，**必须说明为什么**

E8 有四个下界（raw pixel / VAE latent / random DiT / trained DiT）。V-JEPA 2 **没有 VAE**，
所以 `VAE latent` 这一档不存在。

→ E13 报三个下界：`raw pixel` / `random-init ViT` / `trained ViT`。
→ 报告里必须写明"第四个下界缺失"及其原因，**不得为了凑齐四格而塞一个别的量进去**。

## 4. 输入口径的三处偏离（跑数前写死）

### 4.1 几何：256 / 64 帧（不是 512 / 49 帧）

V-JEPA 2 的位置编码写死在 256×256×64。强行改尺寸或帧数会让 pos embed 不匹配，
**要么报错、要么静默插值**（后者是这个项目已经踩过的坑，见 E8 §12 的 RoPE 事故）。

→ 采用模型原生几何 256×256、64 帧。由此产生两个后果，都要写进报告的 Limitations：

- **分辨率降到 256**，物理违反的像素证据变少（与 E12 的 §4.2/§4.3 是同一类问题）
- **帧数升到 64**，IntPhys 2 的 636 帧取 64 帧与 E8 取 49 帧的时间跨度相同（都覆盖全程）

### 4.2 池化：按 tubelet 时间位置，不按 latent 帧

E8 的板式是"每个 latent 帧一个 D 维向量"。V-JEPA 2 的 tubelet=2，64 帧 → 32 个时间位置。
→ 池化后得到 **32 × 1024** 维向量，与 E8 的板式同构（时间位置 × 通道），下游 probe 不用改。

### 4.3 特征取哪层：24 层，报 L16 与 Lrel（53.3% → L13）

与 E8 同规则：`L16` 绝对 + `Lrel = round(0.533×24) = 13` 相对深度，**两个都报**。
不因为哪个好看而选哪个。

## 5. 判据（**跑数前写死**）

### Q1：训练的作用是否同样小？

- **口径**：`D1_clean = (AUC_trained − AUC_random) / (AUC_trained − 0.5)`，干净输入
- **对照格**：CogVideoX-2b 的 `t=0` 干净格 D1（新增，必须跑）
- **判据**：
  - `V-JEPA D1 < 50%` **且** `CogVideoX t=0 D1 < 50%` → **训练的作用小，与是否生成无关**（支持收窄）
  - `V-JEPA D1 ≥ 50%` **且** `CogVideoX t=0 D1 ≥ 50%` → 同样是"与是否生成无关"，但方向相反（收窄不成立）
  - **两者不一致 → `inconclusive`，并且是 E13 最有信息量的结果**：说明"训练的作用大小"依赖训练目标本身
- T10 可解读性门槛照用：`效应 > 2×地板` 且 `AUC − 0.5 > 2×地板`。不满足就报 NOT interpretable，不给 D1 数

### Q2：D2 是否重现？

- **判据**：`AUC(外观扰动) ≥ AUC(物理违反)` → D2 fail（与 E8 同方向）
- 这一条**只有一个方向**，与 E8 一致，没有"哪种更好"的主张
- 若外观 AUC 也顶到 1.0（E8 里 IntPhys2 上全部 ≥ 0.9986），必须照 E8 §14.2 的写法处理：
  说明 gap 的方向在物理一侧大范围变化下仍成立，**不拿天花板当证据**

## 6. 交付路径（定死）

```
results/E13_non_generative/
    E13_PREREG.md          本文档
    E13_cells.json         逐格 AUC / d' / 每 seed 值 / 地板
    E13_verdict.json       Q1/Q2 判定 + 与 E8 的可比性声明
    E13_REPORT.md          模型、几何、下界、判定、三处偏离
    manifest.json          revision、几何、脚本 sha256、提取量
```

`E13_verdict.json` 顶层键（定死）：

```json
{
  "model": "facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34",
  "reference_cell": {"layer": 16, "t": null, "why_null": "non-generative: no noising step"},
  "geometry": {"height": 256, "width": 256, "frames": 64, "tokens": 8192},
  "lower_bounds": {"raw_pixel": 0.0, "random_vit": 0.0, "trained_vit": 0.0,
                   "missing": "vae_latent — model has no VAE"},
  "q1": {"d1_vjepa": 0.0, "d1_cogvideox_t0": 0.0, "verdict": "consistent|inconsistent|inconclusive"},
  "q2": {"auc_physics": 0.0, "auc_appearance": 0.0, "verdict": "fail|pass"},
  "comparability": "D1 not numerically comparable to E8; D2 is",
  "note": ""
}
```

## 7. 提取量

| 项 | 数量 |
|---|---|
| V-JEPA 2 干净特征 (real) | 1012 (IntPhys2 plain) + 506 (cj) |
| V-JEPA 2 干净特征 (random init) | 1518 |
| CogVideoX-2b t=0 对照 | 1518 × 2 |
| **合计** | **6072 次单 clip 前向** |

ViT-L 一次前向远小于 48 层 DiT，预计 **< 30 分钟（4 卡）**。

## 8. 不做

- 不做 LikePhys：先看 IntPhys 2 的结果，避免又一次"全做完才发现判据有问题"
- 不做 ViT-H 规模轴：等主结果
- 不重跑任何 E8 / E12 的格
- 不把 V-JEPA 2 的 D1 数值与 E8 的 D1 数值并列在同一张表里（只能并排列判定方向）
