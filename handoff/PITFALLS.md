# PITFALLS — 四轮踩过的坑

> 写于 2026-09-17，来自 T1–T4 的真实排障记录。**读者是三个月后的你自己或接手的人。**
> 组织方式：**症状 → 根因 → 处理 → 怎么防**。
> 排在前面的是「不会报错、只会让数字变错」的那一类 —— **这类最贵**。
>
> 配套：`handoff/MACHINE_HANDOFF.md`（怎么用）、`results/T2_env/T2_REPORT.md`（原始记录）。

## 索引

| # | 坑 | 类别 | 代价 |
|---|---|---|---|
| [1](#1-gpu-静默产出错数不崩溃) | GPU 缺陷 SM 静默错数 | **静默错误** | 会毁掉全部数字 |
| [2](#2-flow-matching-的-target-用成了-ε) | flow target 用错 | **静默错误** | MSE 差 29 倍 |
| [3](#3-latent-没做逐通道归一化) | latent 未归一化 | **静默错误** | MSE 差 3.7 倍 |
| [4](#4-per-token-timestep-传成标量不报错但语义不对) | per-token timestep | **静默错误** | v_pred 偏 1–1.8% |
| [5](#5-temporal_disordercolor_change-混进-physics-主组) | 对照组混入主组 | **静默错误** | 结论方向都会变 |
| [6](#6-t5-text-encoder-1064-gb-常驻显存) | T5 常驻显存 | 资源 | 24 GB 卡直接紧张 |
| [7](#7-unipcmultistepscheduler-没有-scale_noise) | `scale_noise` 不存在 | 崩溃 | 照抄伪代码即崩 |
| [8](#8-nexttransformerparametersdtype-取到-fp32) | dtype 推断错 | 崩溃 | conv 类型不匹配 |
| [9](#9-依赖版本门槛四个都是-import-就炸) | 依赖版本门槛 | 崩溃 | import 阶段失败 |
| [10](#10-hf-下载被限流429) | HF 限流 | 中断 | 下载半途死 |
| [11](#11-hf_hub_offline1-时-diffusers-仍然要连网) | 离线模式仍连网 | 崩溃 | 分片 checkpoint 加载失败 |
| [12](#12-probe-拟合慢到不可接受两次都是自己作的) | probe 性能 | 效率 | 4 小时 → 15 分钟 |
| [13](#13-joblib-worker-在父进程被杀后继续跑) | 孤儿进程 | 效率 | 占满 CPU 无产出 |
| [14](#14-视频解码imageio-的-pyav-插件没装) | 解码插件 | 崩溃 | 小坑，记一下 |
| [15](#15-set--euo-pipefail-下-grep-无匹配会杀掉脚本) | bash set -e | 崩溃 | 编排脚本秒退 |

---

## 1. GPU 静默产出错数（不崩溃）

**症状**
- `VAE encode` 抛 `RuntimeError: CUDA error: an illegal instruction was encountered`，
  但换个 backend / 换 dtype / 重跑，有时又不报错。
- 更可怕的是**不报错的那些次**：预处理出来的像素范围本该是 `[-1, 1]`，
  实际打印出 `[-1.73, 5.90]`；同样的代码在 CPU 上是 `[-1.0, 0.81]`。

**根因**
物理 GPU 0 有**缺陷 SM**。`dmesg` 证据：

```
NVRM: Xid (PCI:0000:1b:00): 13, Graphics SM Warp Exception on (GPC 10, TPC 2, SM 0):
      Illegal Instruction Encoding
（GPC 10 的 TPC 2/3/4/5、SM 0/1 反复出现）
```

**host↔device DMA 是好的，SM 计算是坏的** —— 所以数据搬进搬出没问题，
一旦在卡上做运算就可能出错，而且**大部分元素是对的**，错的是少数。

量化：
- 平凡 kernel `y = 2x+1`，1.678e7 个元素错 **8070 个**，最大偏差 **2.4e3**
- 在它上面算出的 UMT5 prompt embeds：925,696 个元素里 **2995 个不同（0.32%）**，
  最大偏差 **0.00296**，而数值本身上界只有 0.75 ——
  **这种幅度不会触发任何 NaN/Inf 检查，画出来的曲线也完全正常。**

**处理**
1. 立即停用该卡，`CUDA_VISIBLE_DEVICES` 只暴露健康卡。
2. **重算所有曾在坏卡上产出的中间件**。本项目重算了 prompt embeds；
   坏版本改名 `CORRUPTED_gpu0_prompt_embeds_empty.pt.bak` 保留作证。
3. 报修（Xid 13 + 具体 GPC/TPC/SM 编号是有效的报修材料）。

**怎么防**
- `wan_probe_lib.preflight(device)` 已内置：跑 `y=2x+1` 与 CPU 逐位比对，
  不过就**抛异常**，**绝不落盘**。所有入口脚本第一件事就是调它。
- 每个结果文件带 `preflight` 与 `env.cuda_visible_devices`，事后可追哪张卡算的。
- 新机器到手第一件事：`scripts/t2_diag_gpu.py`（逐卡逐算子）+ `t2_diag_gpu0.py`（含 dmesg）。
- **判读标准**：健康卡上除 `matmul_fp32`（允许 ~2e−4 浮点累加差）外，
  所有项的 `max|gpu−cpu|` 应当**逐位为 0**。

> **教训**：第一次看到 `illegal instruction` 时我以为是 torch cu124 与驱动 CUDA 12.2 不匹配，
> 去查 SDPA backend、试 MATH/EFFICIENT/FLASH、试 bf16/fp32、试 `enable_tiling`——
> **全是浪费**。真正定位靠的是「同一段计算 CPU 跑一遍对比」。
> **凡是怀疑数值不对，第一步永远是和 CPU 对答案，不是换配置。**

---

## 2. flow matching 的 target 用成了 `ε`

**症状**
没有症状。MSE 算出来是个正常的小数字，曲线也有形状。**这是最危险的一类。**

**根因**
Wan2.2 是 **rectified flow / flow matching**，模型预测的是速度场：

```
x_t      = (1 − σ) · x_0 + σ · ε          σ = t / 1000
v_target = ε − x_0          ← 不是 ε，也不是 x_0
```

按 ε-prediction 的习惯写 `target = eps` 不会报错，只是在算另一个量。

**处理 / 证据**
实测（`results/T2_env/min_chain.json`，t=600，同一 clip）：

| target | MSE |
|---|---|
| **`ε − x₀`（正确）** | **0.0293** |
| `ε`（错误） | 0.8537 → **差 29 倍** |
| `−x₀`（错误） | 1.0006 |

**怎么防**
- 用 `wan_probe_lib.flow_noise()`，它同时返回 `(z_t, eps, v_target)`，不给你写错的机会。
- 新模型接手时，**先做这个三选一的对照**：正确的 target 应该让 MSE 低一个量级。
  这是个零成本的自检。

---

## 3. latent 没做逐通道归一化

**症状**
同样没有症状。MSE 偏大，但你不知道「大」是多少才算不对。

**根因**
Wan 的 VAE 用**逐通道** `latents_mean` / `latents_std`（TI2V-5B 是 **48 个值**，
不是一个标量 `scaling_factor`）。漏做 → 送进 DiT 的输入分布错位。

**处理 / 证据**
`results/T2_env/min_chain.json` 的 `risk1_no_normalisation`（t=600）：

| | MSE |
|---|---|
| 归一化 | **0.0293** |
| 不归一化 | 0.1070 → **差 3.7 倍** |

**怎么防**
`encode_latents(vae, px, normalize=True)` 是默认值；
`normalize=False` 只在明确要做「未归一化」对照时用。
反归一化写法：`z / (1/std) + mean`（注意 pipeline 源码里 `latents_std` 变量存的是 `1/std`，
容易看串）。

---

## 4. per-token timestep 传成标量（不报错，但语义不对）

**症状**
不报错。MSE 几乎一样（0.02927 vs 0.02923），**看起来完全没差**。

**根因**
Wan2.2-TI2V-5B 的 `model_index.json` 里 `expand_timesteps = true`，
意味着 timestep 要展开成 **per-token** 的 `[B, seq_len]`：

```python
temp_ts  = (ones[f, h, w][:, ::2, ::2] * t).flatten()   # patch_size=(1,2,2)
timestep = temp_ts.unsqueeze(0).expand(B, -1)           # [B, seq_len]
```

传标量 `[B]` **会被广播，不会报错**，但走的是 `timestep.ndim == 1` 分支，
condition embedding 的形状语义完全不同。

**处理 / 证据**
两种写法的 `v_pred` **相对 L2 差 1.06%（t=600）到 1.83%（t=200）**——
比 MSE 上看到的差异大得多，而 MSE 把它平均掉了。
**用聚合指标（MSE）自检是查不出这个坑的，必须比张量本身。**

**怎么防**
- 用 `wan_probe_lib.per_token_timestep()`。
- 迁移到别的模型前，先读 `model_index.json` 的 `expand_timesteps`
  与 `boundary_ratio`（A14B 还要按 `boundary_ratio` 选 `transformer` / `transformer_2`，
  选错专家 = 在完全错误的噪声区间上问模型）。

---

## 5. `temporal_disorder`/`color_change` 混进 physics 主组

**症状**
probe AUC 异常高，且「物理违反」的结论变得很漂亮。

**根因**
LikePhys 的 920 个 clip 里：
- `temporal_disorder_*` **120 个**（全 12 场景）：把帧顺序打乱，**不是物理违反**，
  是时序破坏；
- `color_change_*` **50 个**（5 场景）：只改颜色，**物理完全合法**。

LikePhys 自己把它们列在违反项里。**如果照单全收当负类**，
probe 学到的就是「任何编辑痕迹」，AUC 会虚高，而且结论方向会被带偏。

**处理**
`scripts/t3_build_manifest.py` 把它们**单独成组**，`label=None`，不进主 probe：

```
physics    750 = 120 valid + 630 violation
temporal   120（帧序对照）
appearance  50（外观对照）
```

**这不是取巧，恰恰相反 —— 这两组后来成了 v0.3 最重要的对照组**：
`valid vs color_change` 的 real_dit AUC = **1.0000**，比 `valid vs violation` 的 0.9698 **还高**，
直接证明 probe 读的是编辑痕迹而非物理。

**怎么防**
- 用任何 benchmark 之前，**把它的类别清单逐条读一遍**，
  问「这一类真的属于我要测的属性吗」。本项目就是靠这一步拿到了免费的对照组。
- 分组固定在 `PREREG.md` §1.3，**不得混入**。

---

## 6. T5 text encoder 10.64 GB 常驻显存

**症状**
24 GB 卡上跑着跑着 OOM，或者 batch/分辨率上不去。

**根因**
Wan2.2 的 UMT5 text encoder 磁盘 11.36 GB，加载后实测峰值 **10.64 GB**。
probing 任务里 prompt 是固定的（我们用空 prompt），**没有任何理由让它常驻**。

**处理**
`scripts/t2_prompt_embeds.py` 离线算一次，存成 fp16 的 `[1, 226, 4096]`
（`results/T2_env/prompt_embeds_empty.pt`，1.8 MB），
运行时 `load_prompt_embeds()` 直接读盘，**全程不加载 text encoder**。

省下来的效果：最小链路峰值显存 **14.18 GB**（VAE fp32 + DiT bf16 + 512×512×49 帧 activation），
24 GB 卡余量充足。

**怎么防**
换 prompt 集合时重跑一次 `t2_prompt_embeds.py --prompts "..." "..."` 即可，
embeds 里同时存了 `prompts` 与 `seq_lens` 便于核对。
注意它**必须在健康卡上算**（见坑 #1）。

---

## 7. `UniPCMultistepScheduler` 没有 `scale_noise`

**症状**
```
AttributeError: 'UniPCMultistepScheduler' object has no attribute 'scale_noise'
```

**根因**
`research/MODELS.md` §2.5 的伪代码写的是 `sch.scale_noise(z0, t, eps)`，
但在 diffusers **0.36** 里 `scale_noise` 只定义在 `FlowMatch*` 系列
（`scheduling_flow_match_euler_discrete.py` 等），**UniPC 没有**。
而 Wan2.2 的 `scheduler_config.json` 用的正是 `UniPCMultistepScheduler`。

**处理**
自己做插值，不依赖 scheduler：

```python
sigma = t / 1000
z_t   = (1 - sigma) * z0 + sigma * eps
```

与 `FlowMatchEulerDiscreteScheduler.scale_noise` 的定义一致，
也与 `timesteps = sigmas * num_train_timesteps` 自洽（源码核对过）。
已封装在 `wan_probe_lib.flow_noise()`。**`MODELS.md` 已于 2026-09-16 实机修正。**

---

## 8. `next(transformer.parameters()).dtype` 取到 fp32

**症状**
```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```
—— 明明模型是用 `torch_dtype=torch.bfloat16` 加载的。

**根因**
diffusers 的 `WanTransformer3DModel._keep_in_fp32_modules` 包含
`time_embedder` / `scale_shift_table` / `norm1..3`，
这些模块**即使整体以 bf16 加载也会保持 fp32**。
`next(parameters())` 可能先迭代到其中之一 → 推断出 fp32 → 用它去 cast 输入 → 与 conv 的 bf16 权重不匹配。

**处理**
用真正参与第一层计算的权重来推断：

```python
dtype = transformer.patch_embedding.weight.dtype
```
已修进 `wan_probe_lib.forward_v()`。**`MODELS.md` 已于 2026-09-16 实机修正。**

**顺带**：`AutoencoderKLWan.from_pretrained(..., dtype=...)` 会报
`TypeError: __init__() got an unexpected keyword argument 'dtype'` ——
diffusers 的模型加载参数名是 **`torch_dtype`**（transformers 那边才提示 `dtype`，别混）。

---

## 9. 依赖版本门槛（四个，都是 import 就炸）

| 症状 | 根因 | 处理 |
|---|---|---|
| `ImportError: peft>=0.17.0 is required ... but found peft==0.15.2` | diffusers 0.36 在 `utils/constants.py` **import 阶段**做 `dep_version_check("peft")` | `uv pip install "peft>=0.17.0"` |
| `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` | datasets 2.14.4 用了 pyarrow 25 已删除的符号 | `uv pip install "datasets>=3.6,<5"` |
| `cannot import name 'AutoencoderKLWan'` | 系统自带 diffusers **0.29.2**，没有 Wan 类 | `uv pip install "diffusers==0.36.0"` |
| `ImportError: The 'pyav' plugin is not installed` | `imageio.v3.immeta(..., plugin="pyav")` 需要 pyav | **改用 decord**，全项目统一 |

**怎么防**：`results/T2_env/pip_freeze_venv.txt` 是锁定清单；
新机器重建后跑一次「import 冒烟」：
```bash
uv run --no-project python -c "import torch,diffusers,transformers,peft,datasets,sklearn,decord; \
from diffusers import AutoencoderKLWan, WanTransformer3DModel; print('ok')"
```

---

## 10. HF 下载被限流（429）

**症状**
```
huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests
for url: .../xet-read-token/...
We had to rate limit your IP (150.109.196.248).
```
下载跑到一半死掉。

**根因**
代理出口 IP 是**共享**的，HF 按 IP 限流；`xet` 传输路径的 token 端点最先被限。

**处理**
```bash
export HF_HUB_DISABLE_XET=1     # 绕开 xet，走普通 CDN
# 并发从 8 降到 4；失败后指数退避重试（已写进 t2_download_data.py）
```
IntPhys2 下载时还遇到过 `us.aws.cdn.hf.co Read timed out`，
`snapshot_download` 会自动 resume，重试即可（实测 523 秒完成）。

**怎么防**
下载与实验分开跑；实验阶段 `HF_HUB_OFFLINE=1`，别让长任务因为网络抖动中断。

---

## 11. `HF_HUB_OFFLINE=1` 时 diffusers 仍然要连网

**症状**
```
huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach
https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/revision/...
```
—— 文件明明已经全在本地缓存里。

**根因**
对**分片 checkpoint**，`diffusers.models.modeling_utils.from_pretrained` 会走
`_get_checkpoint_shard_files` → 调用 Hub 的 `model_info` API 去解析分片清单，
**即使所有分片都已缓存**。

**处理**
不用 repo id，**直接指向本地快照目录**：

```python
snap = Path(os.environ["HF_HOME"])/"hub"/f"models--{repo.replace('/','--')}"/"snapshots"/revision
WanTransformer3DModel.from_pretrained(str(snap/"transformer"), torch_dtype=torch.bfloat16)
```
已封装为 `wan_probe_lib.snapshot_dir()`，所有加载都走它。
好处不止是离线可用：**实验不再依赖网络状态，可复现性也更强**。

---

## 12. probe 拟合慢到不可接受（两次，都是自己作的）

**症状 A**：单个 probe 配置要 **17 分钟**，全量 118 配置估算 4 小时。

**根因 A**：我给 C 路径加了 `warm_start=True`，以为能加速。
在 **p ≫ n**（39,936 维 vs 600 样本）时，上一个更小 C 的解是**很差的初值**，
lbfgs 迭代次数暴涨。实测：同一配置 **warm start 1038 秒 vs 冷启动 ~100 秒，慢 10 倍**。

**处理 A**：去掉 `warm_start`，每个 C 冷启动。

---

**症状 B**：冷启动后单配置 2 分钟，但开 24 个 worker 并行后又劣化到 **18 分钟/配置**，
load average 飙到 136，总吞吐反而下降。

**根因 B**：每个 worker 都在 600×39,936 的稠密矩阵上跑 BLAS，
**内存带宽**是瓶颈，不是 CPU 核数。24 workers × 4 threads 把带宽打满，互相拖慢。

**处理 B**：两步。
1. 并行度降到 **12 workers × 8 threads**；
2. 真正的解法：**行空间精确换基**。
   L2 正则的线性模型最优解必然落在训练矩阵的行空间内
   （`A = U S Vᵀ`，`‖w‖² = ‖Vᵀw‖²`，`A w = (U S)(Vᵀw)`），
   所以在 `Z_tr = U S`（≤600 维）上拟合**与直接在 39,936 维上拟合等价**，
   测试点用同一个 `V` 映射。用 Gram 矩阵（n×n 特征分解）求 `V`，比对 n×p 做 SVD 便宜得多。

   **这是精确变换不是降维近似**，已验证：`results/T4_caliper/rowspace_equivalence_check.json`
   显示 C ≤ 1 时 AUC **逐位相同**，加速 7–11 倍。
   （C=10 时两者有差异，因为**直接拟合那一侧没收敛**，行空间版本反而更接近真解。）

最终：**118 配置 15 分钟**。

**怎么防**
- 先量一个配置的耗时再决定并行度，别直接开满。
- p ≫ n 的线性模型，**先想行空间/Gram**，再想加机器。
- 「优化」要有等价性验证脚本（`t4_verify_rowspace.py`），否则你不知道是加速了还是改了结果。

---

## 13. joblib worker 在父进程被杀后继续跑

**症状**
`pkill -f t4_caliper.py` 之后，`top` 里仍有 12 个 python 各占 **900–1000% CPU**，
每个已累计 170+ 分钟 CPU 时间，而收集结果的父进程早就没了 —— **算了半天全扔掉**。

**根因**
joblib 的 **loky** backend 用独立的 `popen_loky_posix` 子进程，
父进程被 SIGKILL 后它们不会自动退出。

**处理**
```bash
ps -eo pid,etimes,cmd | grep "[p]ython"      # 找出 LokyProcess-* 与 resource_tracker
kill -9 <每个 pid>
```
确认 `/proc/loadavg` 回落再启动新任务（load average 是滞后指标，
**以 `top` 里实际的 %CPU 为准**）。

**怎么防**
长任务用 `nohup ... &` + 记下 PID；
要中止时先 `kill` 父进程，**再检查一遍有没有 loky 孤儿**。

---

## 14. 视频解码：imageio 的 pyav 插件没装

**症状**：`ImportError: The 'pyav' plugin is not installed.`

**根因**：`imageio.v3.immeta(path, plugin="pyav")` 需要 pyav，环境里没有；
系统 PATH 里也没有 ffmpeg 可执行文件。

**处理**：统一用 **decord 0.6.0**（`VideoReader` / `get_batch`），
需要 ffmpeg 二进制时用 `imageio_ffmpeg.get_ffmpeg_exe()` 拿自带的那个。
`wan_probe_lib.read_video_frames()` 已固定走 decord。

---

## 15. `set -euo pipefail` 下 grep 无匹配会杀掉脚本

**症状**：编排脚本 `t3t4_pipeline.sh` 刚打印一行就 `Exit 1` 退出。

**根因**：
```bash
done_count=$(grep -l "pattern" logs/*.log 2>/dev/null | wc -l)
```
grep 无匹配时返回 1，`pipefail` 让整个管道返回 1，
命令替换赋值的退出码就是 1，`set -e` 直接终止脚本 ——
**而「无匹配」正是轮询等待时的正常状态。**

**处理**
```bash
done_count=$( { grep -l "pattern" logs/*.log 2>/dev/null || true; } | wc -l)
```

---

## 附：排障心法（这四轮总结出来的）

1. **怀疑数值不对 → 先和 CPU 对答案**，不要先换配置。坑 #1 就是这么定位的。
2. **不报错的坑比报错的贵**。#1–#5 全是「不报错」类，其中 #2 差 29 倍、#3 差 3.7 倍。
   每接一个新模型，**先做正确/错误写法的对照**，把「正确时该长什么样」量出来。
3. **聚合指标会掩盖错误**。#4 在 MSE 上看只差 0.01%，在张量上差 1.8%。
   **自检要比张量，不要只比标量。**
4. **先量一个单元的耗时，再决定并行度**（#12）。
5. **优化必须配等价性验证**（#12 的 `t4_verify_rowspace.py`）。
6. **benchmark 的类别清单要逐条读**（#5）——
   本项目最有价值的两个对照组就是这么白捡的。
7. **每个结果文件都带 `env` + `preflight`**。
   否则出了问题你分不清是哪张卡、哪个版本、哪次跑的。
