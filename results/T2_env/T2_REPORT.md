# T2 — 环境与最小链路 · 结果报告

执行日期：2026-09-16 · 对应 `handoff/HANDOFF.md` §T2（v0.2）
产出目录：`results/T2_env/`

---

## 0. 结论

最小链路已跑通：**加载模型 → VAE encode → 单次 forward → denoising error**，
四个高危风险（#1 归一化 / #3 expand_timesteps / #5 flow target / #8 T5 离线）**逐个实测验证**。

但 T2 期间发现一件比链路本身更重要的事：

> ### ⚠️ 物理 GPU 0 有缺陷 SM，会静默产出错数
>
> `dmesg`：`NVRM: Xid (PCI:0000:1b:00): 13 ... Graphics SM Warp Exception on (GPC 10, TPC 2..5,
> SM 0/1): Illegal Instruction Encoding`（多个 TPC/SM 反复报）。
> 平凡 kernel `y = x*2+1` 在 1.678e7 个元素中错 **8070** 个，最大偏差 **2.4e3**；
> host↔device DMA 正常，所以**程序不崩溃，只是数字错**。
> GPU 1/2/3 在同样测试下 **bit-exact**（0 处错误）。
>
> **实测污染证据**：在 GPU0 上算出的 UMT5 prompt embeds 与 GPU1 版本相比，
> 925,696 个元素中有 **2,995 个不同（0.32%）**，最大偏差 **0.00296**（数值本身上界 0.75）——
> 幅度小、看起来完全正常、肉眼与常规断言都抓不到。
> 该文件已改名为 `CORRUPTED_gpu0_prompt_embeds_empty.pt.bak` 隔离保留作为证据。

因此：**全部实验只用物理 GPU 1/2/3**，且 `wan_probe_lib.preflight()` 成为所有入口的强制前置自检
（不通过直接抛异常，绝不落盘）。这条如果没抓出来，后面每个数字都会「漂亮且错误」。

---

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | 4 × RTX 4090 24564 MiB，driver 535.54.03，CUDA 12.2；**GPU0 故障，禁用** |
| CPU / 内存 | 128 核 / 1007 GB |
| 磁盘 | `$WORKDIR` 41 T 可用（用户配额 8.2 T 可用） |
| 外网 | 无直连，全部经 `<http-proxy>`；`export.arxiv.org` 必须 https |
| uv | 0.12.15（`pip install uv` 到 `/opt/conda/bin/uv`） |
| venv | `.venv`（`uv venv --system-site-packages`，复用系统 torch），一律 `uv run --no-project` |
| HF 缓存 | `HF_HOME=$HF_HOME` |

依赖（完整清单 `results/T2_env/pip_freeze_venv.txt`，81 项）：

| 包 | 之前 | 现在 | 说明 |
|---|---|---|---|
| torch / torchvision | 2.5.1+cu124 / 0.20.1 | 不变 | 沿用系统版本，避免 2.5 GB 重下 |
| **diffusers** | 0.29.2 | **0.36.0** | 0.29 无 `AutoencoderKLWan` / `WanTransformer3DModel` |
| **transformers** | 4.43.1 | **4.57.6** | UMT5 加载 |
| **peft** | 0.15.2 | **0.21.0** | diffusers 0.36 硬性要求 ≥0.17，否则 import 即失败 |
| **datasets** | 缺→2.14.4 | **4.8.5** | 2.14 与 pyarrow 25 不兼容（`pa.PyExtensionType` 已移除） |
| 新增 | — | scikit-learn 1.9.1 / matplotlib 3.11.2 / imageio-ffmpeg 0.6.0 / joblib / threadpoolctl / pypdf | |
| ffmpeg | 系统无 | 走 imageio-ffmpeg 自带二进制 | |

**依赖坑（都会直接 import 失败或静默行为不一致，记录备查）**：
peft 版本门槛、datasets↔pyarrow、`imageio` 的 pyav 插件未装（视频解码统一走 **decord**）。

---

## 2. 下载与校验

| 资源 | 大小 | 状态 |
|---|---|---|
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff731` | **34.20 GB / 21 文件** | ✅ 逐文件字节数校验通过（`scripts/verify_snapshot.py`，0 missing / 0 mismatch） |
| `JianhaoDYDY/LikePhys-Benchmark` @ `30fc7a19` | 0.16 GB / **920 clips** | ✅ |
| `facebook/IntPhys2` @ `a077a2f9` | 1.7 GB | ✅（Debug + HeldOut；含 `Debug/metadata.csv`，字段 `condition/type(1_Possible|1_Impossible)/occluder/Difficulty/Camera`，`SceneIndex` 即配对键） |
| `PhysionLabs/Physion-Eval` @ `1c53fd7a` | 元数据 17 MB（`videos.zip` 25.6 GB 暂未拉） | ✅ 见 §3 |

**下载注意**：HF 对代理出口 IP 做限流（`429 ... We had to rate limit your IP` 于
`xet-read-token`）。解法：`HF_HUB_DISABLE_XET=1` + `max_workers=4` + 指数退避重试
（`scripts/t2_download_data.py`）。所有 revision **已 pin**，写进脚本常量。

---

## 3. Physion-Eval 核实（HANDOFF 新增高优先级项）

**结论：可获取、无 gating、格式清晰，但它不能直接给我们的 self-gen 侧打标签。**

- 仓库 `PhysionLabs/Physion-Eval`，license `other`，**未 gated**，downloads 1120 / likes 18
- 文件：`Physion_Eval_20260322.json`（4.77 MB）、`all_metadata_with_caption.json`（12.43 MB）、
  `videos.zip`（**25.6 GB**）
- **记录数 9,569**（论文摘要称 10,990 条 expert reasoning traces —— 数量口径不一致，
  可能是「轨迹 ≠ 记录」或版本差异，**引用时必须以实际文件为准**）
- 字段：`id / filename / has_glitches(0|1) / reasoning(自然语言) / glitch_severity(int) /
  model / Glitch_Category / caption`
- 生成模型是**外部生成器**（首条记录 `model = "Sora 2"`）
- README 明示：**为降低版权风险，每条生成视频删掉了前 5 帧**

对 SPEC v0.2 的两点直接影响（事实，非方案建议）：

1. **`human-score` 标签轴**：Physion-Eval 可直接用，省掉大量人评。
2. **`分布轴` 的 self-gen 侧仍缺标签**：Physion-Eval 标的是 Sora 2 等**别人模型**的生成结果，
   不是 Wan2.2-TI2V-5B 自己的生成结果。SPEC §3.1 要的 `self-gen` 是「**该模型**自己生成的视频」，
   这部分标签 Physion-Eval 顶不掉 —— 与 `handoff/HANDOFF.md` 未决事项
   「self-gen 侧的标签获取方案（当前最大瓶颈）」一致，此处只是把它确证了。
3. 「删掉前 5 帧」对 Wan 因果 VAE 的首帧单独处理有影响（首个 latent 帧只由第 1 帧决定），
   若用 Physion-Eval 做 probe，需在报告里声明这一裁剪。

---

## 4. 最小链路实测（`results/T2_env/min_chain.json`）

配置：LikePhys `ball_drop/subgroup_000/valid_00.mp4`，512×512×49 帧，
latent `(1,48,13,32,32)`，seq_len = 13×16×16 = **3328**，空 prompt embeds，物理 GPU1。

| 量 | 值 | 与 `MODELS.md` 的关系 |
|---|---|---|
| DiT 参数量 / bf16 权重 | 5.0 B / **9.31 GB** | ✅ 证实 §1.3 的推论「bf16 约 10 GB」（原文标注「未在真机验证」） |
| 峰值显存（VAE fp32 + DiT bf16 + 49 帧 activation） | **14.18 GB** | 24 GB 卡余量充足 |
| VAE encode（含两次 pass） | 3.2 s | 与 §4.2 估计同量级 |
| **单次 forward** | **0.34 s** | §4.2 估 480P 约 1 s，实测更快（因 512×512 只有 3328 token） |
| T5 峰值显存（离线算 embeds 时） | 10.64 GB | 风险 #8：运行时不加载，已落盘复用 |

### 四个高危风险的实测验证

| 风险 | 正确做法的数值 | 错误做法的数值 | 判定 |
|---|---|---|---|
| **#5 flow target = `ε − x₀`** | MSE 0.029（t=600） | 用 `ε` 当 target：**0.854**；用 `−x₀`：**1.001** | 差 **29 倍**，方向明确 |
| **#1 逐通道 latent 归一化** | MSE 0.029（t=600） | 不归一化：**0.107** | 差 **3.7 倍** |
| **#3 per-token timestep** | 0.02927 | 标量 timestep：0.02923 | MSE 几乎一样，但 `v_pred` 相对 L2 差 **1.06%**（t=200 时 1.83%）——**正是「不报错但语义不对」** |
| **#8 T5 不常驻** | 空 prompt embeds `[1,226,4096]` 落盘 | — | 运行时 0 GB text encoder |

denoising error 随 t 的形状（seed 0，单 clip）：

| t | 200 | 400 | 600 | 800 | 950 |
|---|---|---|---|---|---|
| MSE(v_pred, v_target) | 0.0639 | 0.0329 | **0.0293** | 0.0407 | 0.1408 |

U 形，最低点在 t≈600 —— 与 `2603.14294` 选 t∈{200,400,600} 的区间一致，
但它没测 t≥800（我们测了，误差在 t=950 陡增 4.8 倍，SPEC E5 关心的高噪声区确实是另一个 regime）。

**种子噪声预览**（t=600，同一 clip，seed 0/1/2）：MSE = 0.02927 / 0.03082 / 0.03099，
极差 0.00172 = 均值的 **5.9%**。→ 这个量级不可忽略，E0 必须做全量（T3）。

---

## 5. 对 `research/MODELS.md` 的两处修正（工程层面）

1. **§2.5 伪代码里的 `sch.scale_noise(z0, t, eps)` 在 diffusers 0.36 会崩**：
   `scale_noise` 只存在于 `FlowMatch*` 系列 scheduler，`UniPCMultistepScheduler` **没有这个方法**。
   本项目改为自己做插值：`sigma = t/1000`，`z_t = (1-sigma)·z0 + sigma·eps`。
   与 `FlowMatchEulerDiscreteScheduler.scale_noise` 的定义一致，且与
   `timesteps = sigmas * num_train_timesteps` 自洽（源码核对）。
2. **`next(transformer.parameters()).dtype` 会取到 fp32**：diffusers 把
   `_keep_in_fp32_modules`（`time_embedder` / `scale_shift_table` / `norm1..3`）保持在 fp32，
   即使模型以 bf16 加载。用它推断输入 dtype 会得到
   `Input type (float) and bias type (c10::BFloat16) should be the same`。
   应改用 `transformer.patch_embedding.weight.dtype`。

另核实：`model_index.json` 的 `expand_timesteps=true` / `boundary_ratio=null` /
`transformer_2=[null,null]`，`vae` 的 `z_dim=48`、`scale_factor_spatial=16`、`scale_factor_temporal=4`、
`latents_mean/std` 各 48 个值，`scheduler` 为 `flow_prediction` + `use_flow_sigmas` + `flow_shift=5.0`
—— **`MODELS.md` 的这些数字全部与本地权重一致**。

---

## 6. LikePhys 数据结构（对后续实验有用的两点）

12 scenarios × 10 subgroups，每 subgroup 恰好 1 个 `valid` + N 个违反类型 + 1 个 `temporal_disorder`。
配对键 `(scenario, subgroup)` 天然成立（同一场景布置）。

| 发现 | 数量 | 用处 |
|---|---|---|
| **`color_change_*`** 出现在 5 个场景（ball_drop / cloth_drape / faucet / flag / river） | **50 clips** | LikePhys 把它当违反项，但它本质是**纯外观改变**。→ SPEC E7「外观扰动对照」有现成数据，不必自己造 |
| **`temporal_disorder_*`** 全 12 场景 | **120 clips** | 帧序打乱对照，正好对齐 `2606.09646` 的 temporal control |

因此 T3/T4 的 manifest 把这两组**从主 probe 中剔除**、单独成组
（见 `results/T3_null_floor/protocol_as_executed.md` §2）：
physics 750（120 valid / 630 violation）+ temporal 120 + appearance 50。

---

## 7. 产出文件

```
results/T2_env/
├── T2_REPORT.md                                 ← 本文件
├── min_chain.json                               最小链路全部数字（含四个风险验证）
├── gpu_health.json                              4 卡逐项自检 + nvidia-smi + dmesg Xid
├── gpu0_corruption_prompt_embeds.json           GPU0 污染的量化证据
├── CORRUPTED_gpu0_prompt_embeds_empty.pt.bak    被污染的 embeds（保留作证）
├── prompt_embeds_empty.pt / .meta.json          正式使用（GPU1 计算）
├── data_inventory.json                          三个数据集的结构与标注格式
├── pip_freeze_venv.txt                          依赖锁定清单
├── hf_api_*.json                                四个仓库的文件清单与体积（校验用）
└── download_*.log                               下载日志

scripts/
├── wan_probe_lib.py        共享原语（含 preflight 硬自检）
├── t2_download_wan22.py / t2_download_data.py   pin 了 revision 的下载脚本
├── t2_prompt_embeds.py     离线 prompt embeds
├── t2_min_chain.py         最小链路 + 风险验证
├── t2_inspect_data.py      数据集结构核实
├── t2_diag_gpu.py / t2_diag_gpu0.py / t2_diag_vae.py   GPU 故障诊断三件套
├── verify_snapshot.py      权重完整性校验
└── hf_repo_summary.py / hf_dataset_summary.py   仓库清单汇总
```

## 8. 给负责人的提示

1. **GPU0 需要报修**；在修好之前，算力实际是 3×4090 而非 4×4090。
   SPEC §8.2 对老师说的「4×RTX 4090」需要按 3 卡口径重新表述，或注明其一故障。
2. Physion-Eval 顶不掉 self-gen 侧标签（§3.2）—— 这仍是最大瓶颈，与 HANDOFF 判断一致。
3. Physion-Eval 记录数 **9,569**，与论文摘要的 10,990 不一致，引用时以实际文件为准。
4. `PREREG.md` 仍缺。本轮执行者已把可提前钉死的自由度写进
   `results/T3_null_floor/protocol_as_executed.md`（在跑数之前写的），供负责人写 PREREG 时取用。
