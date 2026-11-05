# E14 预注册：第五个模型族（HunyuanVideo，12.8B）

**写于任何 HunyuanVideo 提取之前：2026-10-28**
上游协议：`PREREG.md` D1 / D2 + `results/E8_second_backbone/E8_PREREG.md`，参考格 `layer 16 / t=600`

---

## 1. 为什么还要加一个模型

E8 已成 7 个 backbone × 2 数据集，判定方向一次没翻。**再加一个同类模型，边际价值很低**，这一点必须先说清楚，否则这一轮就是在堆数量。

加 HunyuanVideo 的**唯一理由**：它在 E8 覆盖的四点上都与已有模型不同，而且是**同时不同**：

| 维度 | 已有覆盖 | HunyuanVideo |
|---|---|---|
| 参数量 | 1.3B–14B | **12.8B**（transformer 本体，不含文本塔）|
| block 结构 | 单流 DiT（Wan / LTX / Mochi）或 双流（CogVideoX）| **双流 20 + 单流 40，同一模型内串联** |
| 文本条件 | UMT5 / T5 | **LLaMA-3-8B（4096 维）+ CLIP pooled（768 维）双路** |
| 隐层宽度 | 1920–3072 | **3072**（24 head × 128）|
| token 预算 | 1792 / 9216 / 13312 | **13312**（与 Wan、CogVideoX 同）|

「同一模型内先双流后单流」这条**没有任何已有 backbone 有**。如果 D1/D2 在这一族上也一致，那么"这不是架构细节的产物"这句话的覆盖面就真的到头了 —— 再往上加模型只是重复。

## 2. 输入口径

与 E8 逐项对齐，唯一例外是 latent 布局：

- **512×512、49 帧、空 prompt**，与 E8 全部 backbone 相同
- VAE：`AutoencoderKLHunyuanVideo`，**8× 空间、4× 时间**，16 通道
- latent：`[1, 16, 13, 64, 64]`（**`[B,C,F,H,W]` 布局，与 Wan/CogVideoX 的 frame-major 相反**）
- patch：`patch_size=2, patch_size_t=1` → token 网格 `13 × 32 × 32 = 13312`

> ⚠️ **布局差异是本次唯一一处真实陷阱。** 实测直接把 frame-major 的 `[1,13,16,64,64]` 喂进
> `conv3d` 会报 `expected input ... 16 channels, but got 13 channels`。这**不是**静默错误
> （比 CogVideoX-5b 的 RoPE 事故幸运），但仍必须显式转换而非依赖报错。

## 3. 参考格

- `num_layers`：报告 **两层并报**，与 E8 §15 同口径
  - `L16` 绝对层（在 60 个 block 里的第 16 个双流块，深度 26.7%）
  - `Lrel = round(0.533 × 60) = 32`（53.3% 相对深度，与 Wan2.2-TI2V-5B 的 layer 16 同深度）
- **索引口径说明**：`transformer_blocks`（20）+ `single_transformer_blocks`（40）拼成 60 个，
  第 0–19 是双流、第 20–59 是单流。`L16` 落在**双流段**，`Lrel=32` 落在**单流段**。
  → 这两个格跨了两个不同的 block 类型，**所以它们不是同一个量的两个深度**。报告必须写明，
  不得把 `L16` 与 `Lrel` 的差异解释成"深度效应"。

- t=600：flow matching（`FlowMatchEulerDiscreteScheduler`），与 Wan/LTX/Mochi 同一约定，
  σ = 0.600。**不需要额外报 SNR 匹配格**（那是 CogVideoX 的 VP 加噪才需要的）。

## 4. 判据（与 E8 完全一致，不改）

- **D1**：`(AUC_real − AUC_random) / (AUC_real − 0.5) < 50%` → 主张不成立
  - T10 门槛照用：`效应 > 2×地板` 且 `AUC_real − 0.5 > 2×地板`
- **D2**：`AUC(外观扰动) ≥ AUC(物理违反)` → fail
  - IntPhys2 用 `cj` 变体（逐帧恒定 HSV，hue 0.35 / sat 1.4），与 E8 同一个扰动
- **地板**：seed-spread（定义 A），feature seed 0/1/2，与 E8 相同（HunyuanVideo 是扩散模型，有噪声抽样）
- **随机初始化**：沿用 norm-aware 规则（`ndim>=2` He，`norm.weight` 置 1，其余 0），
  **不照抄 Wan2.2 的"全部一维置零"**（会让 RMSNorm/LayerNorm 塌陷，见 `E8_PREREG.md` §3）

## 5. 提取量

| 项 | 数量 |
|---|---|
| IntPhys2 plain (real) | 1012 |
| IntPhys2 cj (real) | 506 |
| IntPhys2 plain (random) | 1012 |
| IntPhys2 cj (random) | 506 |
| **合计** | **3036 次单 clip 前向** |

实测单次前向 2.93 s（含 60 个 block 的 hook），显存峰 25.3 GB。
4 卡并行约 **40 分钟**。

**LikePhys 不做**：先用 IntPhys 2 判定，与 E13 的处理一致，避免又一次"全做完才发现判据有问题"。

## 6. 交付路径（定死）

```
results/E14_hunyuan/
    E14_PREREG.md          本文档
    prompt_embeds_hunyuan.pt
    E14_cells.json
    E14_verdict.json       顶层键与 E8_verdict_schema.json 一致，单行
    E14_REPORT.md
    manifest.json
```

`E14_verdict.json` 顶层键（与 `E8_verdict_schema.json` 的行结构一致）：

```json
{
  "model": "hunyuanvideo-community/HunyuanVideo@e8c2aaa66fe3",
  "reference_cell": {"layer": 16, "t": 600},
  "relative_depth_cell": {"layer": 32, "depth": 0.533, "note": "falls in the single-stream段；不是 L16 的深度对照"},
  "lower_bounds": {"raw_pixel": 0.0, "vae_latent": 0.0, "random_dit": 0.0, "trained_dit": 0.0},
  "d1": {"training_share": 0.0, "verdict": "pass|fail|NOT interpretable", "threshold": 0.5},
  "d2": {"auc_physics": 0.0, "auc_appearance": 0.0, "verdict": "fail|pass"},
  "note": ""
}
```

## 7. 不做

- 不做 D3–D6
- 不做 LikePhys
- 不把 L16 与 Lrel 当同一个量的两个深度
- 不因为某一层好看而改参考格
