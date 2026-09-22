# Protocol as executed — T3 (E0 null floor) and T4 (E1 CALIPER discriminability)

写作时间：**2026-09-16，在任何 T3/T4 数字产生之前**。
`PREREG.md` 仍未撰写（属负责人，见 `handoff/HANDOFF.md` 未决事项）。本文件不是预注册，
只是「执行者把自由度提前钉死」的记录，供负责人写 PREREG 时取用。

## 0. 为什么先写这个

因子实验的结论完全取决于协议是否在看到结果前固定。本轮的自由度（层集合、timestep 集合、
标签定义、划分方式、正则强度选择流程）全部在此固定，跑完不改。任何事后调整都会在报告里
显式标注 exploratory。

## 1. 硬件与数值前提（T2 已核实，见 `results/T2_env/`）

- **物理 GPU 0 有缺陷 SM，禁止使用**：`dmesg` 有 `NVRM: Xid 13 ... Illegal Instruction
  Encoding`（GPC 10 多个 TPC/SM），平凡 kernel `x*2+1` 在 1.7e7 个元素里错 8070 个，
  最大误差 2.4e3。DMA 正常，所以**不崩溃，只出错数**。
- 所有实验入口调用 `wan_probe_lib.preflight()`，自检不过直接抛异常，绝不落盘结果。
- 实验只用物理 GPU 1/2/3（`CUDA_VISIBLE_DEVICES` 显式指定），每条结果记录物理设备号。
- 模型：`Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff731`，transformer bf16、VAE fp32。
- prompt embeds：空 prompt，离线算好（`results/T2_env/prompt_embeds_empty.pt`），
  运行时不加载 T5。

## 2. 数据与标签（`results/T3_null_floor/manifest.json`，920 clips）

LikePhys-Benchmark @ `30fc7a19`，12 scenarios × 10 subgroups。

| group | 内容 | n | 用途 |
|---|---|---|---|
| `physics` label 0 | `valid_*` | 120 | 主 probe 正类（物理合法） |
| `physics` label 1 | 全部 violation kinds | 630 | 主 probe 负类（物理违反） |
| `temporal` | `temporal_disorder_*` | 120 | **不进主 probe**，帧序打乱对照（对齐 `2606.09646` 的 temporal control） |
| `appearance` | `color_change_*` | 50 | **不进主 probe**，纯外观扰动对照候选（SPEC E7） |

配对键：`(scenario, subgroup)`。同一 subgroup 内的 valid 与 violation 共享场景布置，
因此配对检验用这个键。

## 3. 状态提取协议（固定）

- 输入尺寸：**512×512，49 帧**（4k+1，H/W 为 32 倍数）。LikePhys 原生 512×512/30fps/~60 帧，
  只取前 49 帧，不重采样（ball_collision 原生 1024×1024，双线性降到 512×512）。
- latent：`(1, 48, 13, 32, 32)`，逐通道 `latents_mean/std` 归一化。
- 加噪：flow matching，`sigma = t/1000`，`z_t = (1-sigma) z_0 + sigma * eps`，
  `v_target = eps - z_0`。
- timestep：**per-token** `[B, seq_len]`（`expand_timesteps=true`），seq_len = 13×16×16 = 3328。
- **timestep 网格**：`t ∈ {200, 400, 600, 800, 950}`。
  含 t=950 是有意的：`2603.14294` 只做到 600，高噪声区无人测过（SPEC E5）。
- **层集合**：**全部 30 层**（`blocks[0..29]`），不做挑选。
- 特征池化：每层输出 `[B, 3328, 3072]` → 按 latent 帧分组，对空间 token 取均值
  → `[13, 3072]`，存 fp16。Wan 的 block hidden states 只含视频 token（文本走 cross-attn），
  因此不存在「去掉 text token」这一步。
- **seed 集合**：噪声 seed `{0,1,2,3,4}`。
  - denoising error：5 个 seed 全做（E0 主体）。
  - hidden states：t=600 存 seed 0/1/2（probe 级 null floor）；其余 timestep 只存 seed 0。

## 4. E0（null floor）报告口径

在报告任何 delta 之前先报噪声地板。三层：

1. **clip 级**：同一 (clip, t)，5 个 seed 的 denoising-error 标准差与极差；
   相对量 = 极差 / 该 clip 均值。
2. **效应级**：valid vs violation 的配对差（按 `(scenario, subgroup)` 配对），
   与 seed 噪声直接比。**判据：若 |效应| 不显著大于 seed 噪声，停下报告。**
3. **probe 级**：同一协议、只换噪声 seed 重训线性 probe，AUC/accuracy 的波动幅度。
   这一项就是因子实验误差棒的来源（SPEC v0.2 硬规则②）。

统计：配对 Wilcoxon 符号秩检验 + 配对差的 bootstrap 95% CI（10,000 次，seed 12345），
效应量报 Cliff's delta 与配对 Cohen's d。**per-scenario 分布优先于全局均值。**

## 5. E1（CALIPER 可区分性）报告口径

依据 `2609.08250`：干净固定机位场景下，线性 probe 可能无法区分真正编码物理的表征与下界。

**四个特征源，同一套 probe 协议**：

| 名称 | 特征 | 维度/帧 |
|---|---|---|
| `real_dit` | Wan2.2-TI2V-5B DiT 第 ℓ 层池化状态 | 3072 |
| `random_dit` | **同架构、随机初始化**（fan-in 缩放正态，seed 0）的 DiT，同层同 timestep | 3072 |
| `vae_latent` | 归一化 VAE latent 按空间池化 | 48 |
| `raw_pixel` | 帧降采样到 32×32 RGB 后展平 | 3072 |

probe 协议（对四个源完全相同）：

- **仅线性**：`sklearn.linear_model.LogisticRegression`，`max_iter=2000`，
  L2 正则，`C ∈ {0.001, 0.01, 0.1, 1, 10}` 在**训练折内**用 3-fold 内层 CV 选，
  不看测试折。输入按帧展平（13×D），先 `StandardScaler`（只 fit 训练折）。
- **划分**：`GroupKFold(5)`，**group = subgroup**，保证同一 subgroup 的 valid 与 violation
  不跨折泄漏。类别不平衡用 `class_weight="balanced"`。
- **指标**：AUC-ROC 与 accuracy 都报（文献里两种都有，SPEC §3.1）。
- **判据**：`real_dit` 与三个下界中**最强者**的差距，是否超过 E0 probe 级噪声地板的
  2 倍。若不超过 → **场景集不可区分，停下报告**（HANDOFF T4 第 3 条）。

## 6. 本轮不做

E2–E4 单轴对照（Milestone 内容）、E8 guidance 相关性（P2 且对方无代码）、
inversion 取状态（E3 的对象，本轮只做 forward-noising）。

## 7. 落盘约定

每个结果文件必带 `env`（torch/diffusers/GPU/物理设备号/模型 revision/代码状态）、
`preflight`、`args`、`seeds`。缺任一项视为结果不可信。
