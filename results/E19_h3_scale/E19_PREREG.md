# E19 预注册：MiniMax-H3（33B）—— 规模轴

**写于任何 E19 提取之前：2026-10-31**

---

## 1. 为什么做这个

现有覆盖的参数量上限是 **14B**（Wan2.1-14B / Wan2.2-A14B 各一）。
`NEXT_STEPS_AND_DIVISION.md` 判过「再加同类模型边际≈0」，但那个判断针对的是**同族、同规模、
同 VAE 布局**的模型。H3 一次换掉三样：

| 轴 | 现有上限 | **H3** |
|---|---|---|
| 参数量 | 14B | **33.1B** |
| 文本编码器 | T5 / LLaMA-3-8B / CLIP | **Qwen3-VL-32B**（取第 50 层）|
| VAE 布局 | 16 通道、f8 或 f16 | **24 通道、f16t4d24** |
| 序列组织 | 连续视频 latent | **text/audio/video 打包的单一序列** |

所以它检验的不是「更大一号的 Wan」，而是**这三件事一起换掉之后，D1/D2 的判定方向是否仍然不变**。

## 2. 设计（与 E8 对齐，只有一处不得不不同）

- **参考格**：**L16**（`PREREG.md` §1.0 定死的参考深度）+ **相对深度 16/30 对应的层**（50 层 → L27）
- **t = 600**（σ = 0.600，flow matching，与 Wan / LTX / Mochi / Hunyuan 同一约定）
- **帧数与尺寸**：49 帧、512×512（与其他 backbone 相同，保证可比）
- **数据集**：IntPhys 2 Main（1012 plain + 506 cj）
- **四组下界**：raw pixel / VAE latent / random-init / trained（`E8_lower_bounds` 口径）
- **三项判定**：D1（> 50%）、D2（物理 vs 外观）、下界排序（训练 > 随机 > 像素）

**不得不不同的一处**：H3 的 `forward` 要求一条 packed 序列里的 audio 行。
本实验**只读 video 行**；audio 行按正确长度与 tag 传入，不读回任何东西。
这一点必须写进报告，否则读者会以为测了音视频联合表征。

## 3. 判定（写死，不可事后改）

### Q1：D1 是否 > 50%？

- 统计量：`D1 = (AUC_real − AUC_random) / (AUC_real − 0.5)`
- 阈值：**50%**（`PREREG.md` D1，不可改）
- 门槛：必须先过 `e8_verdict.interpretable`（T10：效应与分母都 > 2× 地板），
  不过门槛的格**不报 D1**，只报 `NOT interpretable` + 原始 AUC

### Q2：D2 是否 fail？

- 统计量：`AUC_appearance`（逐帧恒定 HSV，hue 0.35 / sat 1.4，与 E8 同一变体）vs `AUC_real`
- 判据：`AUC_appearance ≥ AUC_real` → **fail**；否则 pass
- **预注册预期**：**fail**。理由是前 6 族 32 格全部 fail、零例外，
  如果 33B 上出现 pass，那是本项目**第一个反例**，必须立即上报而不是当作噪声

### Q3：下界排序是否重现「训练 > 随机 > 像素」？

- 只判**排序**，不追求数值复现（`NEXT_STEPS_AND_DIVISION.md` A 的明令）
- 数值差异写进报告的 `note`，不为对齐去调参

## 4. 已知的坑（写于提取之前）

1. **H3 的 VAE 是远程代码**（`trust_remote_code=True`），`encode` 返回原始 moments
   （2C 通道，前 C 是均值），**不是** `latent_dist`。归一化用 checkpoint 自己的
   `latents_mean` / `latents_std`，**不是** Wan 那样的标量 `scaling_factor`。
2. **packed 序列的 rotary 时钟从 text 行之后开始** —— text 长度会整体平移 media 位置。
   所以打包必须**复用 diffusers 的 `build_packed_sequence`**，不得自己重写（E11 的教训）。
3. **video 行在序列末尾**（text | keyframe conditions | audio | video）。hook 必须按
   `num_latent_frames × rows_per_frame` 从尾部切，不能硬编码 text+audio 的偏移。
4. **环境隔离**：H3 需要 `diffusers 0.42.1 + transformers ≥4.60 + hub ≥1.32`，
   而主环境的 `transformers 4.57` 要求 `hub <1.0`。**必须用 `.venv_h3/`**，
   不得升级主环境（会破坏 E8–E18 的可复现性）。
5. **threads = 4**（E16 规矩），分析阶段必须显式传。

## 5. 交付

```
results/E19_h3_scale/
    E19_PREREG.md      本文档
    E19_cells.json     参考格 + 相对深度格 × (real/random/appearance) + 四组下界
    E19_verdict.json   D1 / D2 / 下界排序
    E19_REPORT.md
    manifest.json      含 diffusers/transformers 版本（与其余任务不同！）
```

## 6. 不做

- **不做** LikePhys（先做 IntPhys 2，与 E17/E18 一致）
- **不做** D3–D6（`NEXT_STEPS_AND_DIVISION.md` 明令：六项全做会出现第三种判定）
- **不做** t=298（SNR 匹配格）—— 只有一个模型做它会破坏可比性
- **不读** audio 行的任何输出
- **不改** D1/D2 阈值、probe 实现、随机初始化规则、颜色抖动参数

## 7. 与 E18 教训的关系

E18 证明：**换 σ 就换噪声抽取，random 臂水平会动，D1 分母只有 0.1 左右时会把 0.01 放大成 10 pp。**

E19 的所有格**只用一个 t**（t=600），所以不涉及跨 σ 比较，
**E18 的混淆在这里不适用**。但报告必须显式写这一句，否则读者会以为没检查过。

**同样要报**：每格的 `AUC_real`。若某一格 `AUC_real < 0.6`（E8 §15.2 弱信号区），
该格的 D1 数值**不得**与其它格比大小，只能报判定。
