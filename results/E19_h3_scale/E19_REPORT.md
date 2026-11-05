# E19 — MiniMax-H3（33.1B）：规模轴

生成：2026-10-31　·　协议：`PREREG.md` D1/D2 + `results/E19_h3_scale/E19_PREREG.md`

数据：`E19_cells.json`，判定：`E19_verdict.json`，环境：`manifest.json`

运行环境：4×NVIDIA B200（183 GB／卡），解释器 **`.venv_h3/bin/python`**（diffusers 0.42.1 /
transformers 5.22.1 / hub 2.4.0）—— **不是** `python3`，见 §6。

---

## 1. 结论先行

**三个判定方向全部与既有 6 族一致，没有出现反例。**

| 判定 | 结果 | 与 E8 六族的一致性 |
|---|---|---|
| **D1 @ L16** | **63.87%**（pass），但 real AUC 0.5739 < 0.6 → **弱信号，不可跨格比较** | 方向一致 |
| **D1 @ L27**（相对深度） | **80.97%**（pass），real AUC 0.6414 → 可比较 | 方向一致 |
| **D2 ×2 格** | **两格全 fail**（外观 1.0000 vs 物理 0.5739 / 0.6414） | **第 33–34 格，仍零例外** |
| **下界排序** | `trained(0.5739) > pixel(0.5454) > vae(0.5334) > random(0.5267)` | **与 E8 六族同序** |

**但有三件事必须同时说，否则这个"一致"会被读得过强**：

1. **参数量涨了 2.4 倍，real AUC 反而比 14B 更低**（0.5739 vs Wan2.1-14B 的 0.6667）。
   **规模没有单调改善物理可读性。**
2. **每帧 token 数只有 Wan 的 1/4**，所以这不是一个干净的"只更大"对比（见 §3.1）。
3. **L16 与 L27 差 17 个百分点**，而 E8 在 Wan 上 L16 与 L21 只差 4 个百分点。
   **深度在 H3 上比在 Wan 上重要得多**，这本身是新信息（见 §3.2）。

---

## 2. 结果表

`threads = 4`（E16 规矩）。参考格 L16，t=600，49 帧请求 → **39 帧实际编码 → 12 个 latent 帧**。

| 层 | 深度 | real AUC | random | 外观 | **D1** | 地板 | 判定 | 弱信号 |
|---|---|---|---|---|---|---|---|---|
| **L16** | 16/50 | 0.5739 | 0.5267 | 1.0000 | **63.87%** | 0.0112 | pass | **是**（< 0.6）|
| **L27** | 27/50（=16/30 相对深度）| 0.6414 | 0.5269 | 1.0000 | **80.97%** | 0.0111 | pass | 否 |

逐 feature seed（3 个）：

| 层 | real（3 seed） | random（3 seed） | 外观 |
|---|---|---|---|
| L16 | 0.5718 / 0.5763 / 0.5735 | 0.5255 / 0.5385 / 0.5161 | 1.0000 / 1.0000 / 1.0000 |
| L27 | 0.6369 / 0.6376 / 0.6496 | 0.5252 / 0.5388 / 0.5168 | 1.0000 / 1.0000 / 1.0000 |

**real 臂的 seed 间波动 ≤ 0.013，random 臂 ≤ 0.023** —— 判定不依赖挑 seed。

### 2.1 D2：第 33–34 格

| 层 | 物理 AUC | 外观 AUC | gap |
|---|---|---|---|
| L16 | 0.5739 | **1.0000** | **+0.4261** |
| L27 | 0.6414 | **1.0000** | **+0.3586** |

**L16 的 +0.4261 是本项目 32 格里的最大 gap**（此前最大是 HunyuanVideo 的 +0.35 量级）。
外观 3 个 seed 全部恰好 1.0000。

→ **D2 累计 34 格 / 22 个独立单元 / 7 个模型族，全部 fail，零例外。**
（`scripts/audit_d2_count.py` 数出，权威 `results/D2_TALLY.json`。）

### 2.2 下界：与 E8 同序，但**先要纠正一张任务卡上的说法**

任务卡（`NEXT_STEPS_AND_DIVISION.md` A）写预期排序是「训练 > 随机 > 像素」。
**E8 实际测出来的不是这个。** 从 `E8_cells_intphys2.json` 逐模型读：

| 模型 | trained | random | vae | pixel | 实测排序 |
|---|---|---|---|---|---|
| cogvideox2b | 0.6443 | 0.5213 | 0.5381 | 0.5471 | trained > pixel > vae > random |
| cogvideox5b | 0.6030 | 0.5043 | 0.5387 | 0.5471 | trained > pixel > vae > random |
| ltx | 0.5506 | 0.5261 | 0.5181 | 0.5490 | trained > pixel > random > vae |
| mochi | 0.6157 | 0.5239 | 0.5218 | 0.5464 | trained > pixel > random > vae |
| wan21_14b | 0.6667 | 0.5015 | 0.5288 | 0.5471 | trained > pixel > vae > random |
| wan21_1b3 | 0.6502 | 0.5132 | 0.5288 | 0.5471 | trained > pixel > vae > random |

**`random` 在六个模型里一次都没有超过 `pixel`。** 而 **`trained > pixel` 是 6/6、`trained > vae` 也是 6/6**；
`vae` 与 `random` 的相对次序按模型翻转（3:3）。

→ 所以 E19 应该判的是 E8 **实际建立**的两条，不是任务卡上那张三元的：

| 断言 | E8 六族 | **H3** |
|---|---|---|
| `trained > raw_pixel` | 6/6 | ✅ 0.5739 > 0.5454 |
| `trained > vae_latent` | 6/6 | ✅ 0.5739 > 0.5334 |
| `trained > random` | 6/6 | ✅ 0.5739 > 0.5267 |
| 完整四元次序 | 不稳定 | `trained > pixel > vae > random`（与 4/6 个模型同序）|

**边距很薄**：`trained − pixel = 0.0285`，而地板是 0.0112。**只相当于 2.5 倍地板**，
刚过 T10 的 2× 门槛。这条不能当强证据用。

---

## 3. 三个必须写进 Limitations 的发现

### 3.1 H3 每帧 token 数只有 Wan 的 1/4 —— 这不是干净的"规模"对比

| | latent 帧 | 每帧 token | **总 token** |
|---|---|---|---|
| Wan2.1/Wan2.2（E8/E17）| 13 | 32×32 = 1024 | **13312** |
| **H3（E19）** | 12 | 16×16 = 256 | **3072** |

H3 的 `patch_size` 是 `(1,2,2)`，把 2×2 个 latent 像素并成一个 token（32×32 像素一块），
而 Wan 系是 1×1。所以 H3 在**参数量 2.4 倍**的同时，**token 数只有 0.23 倍**。

**后果**：`real AUC 更低`（0.5739 vs 0.6667）这个观察**不能归因于规模**，
因为 token 预算同时也变了，而 E12 已经在 LTX 上证明 token 预算会显著改变 real AUC
（6 档之间从 0.5304 摆到 0.5519）。

→ 正确的表述：**"33B 且低 token 预算"这个组合下 D1/D2 方向不变**，
不能写成"33B 的 D1 更高/更低"。

### 3.2 深度在 H3 上比在 Wan 上重要得多 —— 一个新发现

| 模型 | 参考层 D1 | 深一层 D1 | Δ |
|---|---|---|---|
| Wan2.2 A14B（E17）| L16 **99.12%** | L21 **95.33%** | −3.8 pp |
| **H3（E19）** | L16 **63.87%** | L27 **80.97%** | **+17.1 pp**（方向相反）|

**H3 的 D1 随深度上升，Wan 的随深度下降。** 这是本项目第一次观察到方向相反的深度趋势。

两种可能的解释，**本实验无法区分**：
- H3 是 50 层、单流、packed 序列，L16 只到 32% 深度；Wan 的 L16 在 40 层里是 40% 深度
- H3 的 L16 real AUC 落在弱信号区（0.5739），D1 比值本身在那里不可靠

→ **不能写"更深的层物理信息更多"**。只能写：**这个格对层敏感，且与 Wan 的方向相反**。

### 3.3 L16 落在 E8 §15.2 的弱信号区

`real AUC = 0.5739 < 0.6`。按 E8 §15.2 与 E12 的 6 档复现，**该格的 D1 比值不可与其它格比大小**，
只能报判定。引用时必须带 `real AUC`。

这也是为什么本报告的"主格"用 L16（`PREREG.md` 定死的参考深度），
但**结论强度依赖 L27**（0.6414，在门槛之上）。

---

## 4. 覆盖范围的最终账（E19 之后）

| 轴 | 范围 |
|---|---|
| 模型族 | **7 个**（Wan / CogVideoX / LTX / Mochi / HunyuanVideo / V-JEPA 2 / **MiniMax-H3**）|
| 模型变体 | **15 个** |
| 参数量 | 0.33B – **33.1B** |
| block 拓扑 | 单流 / 双流 / 双流→单流串联 / 双专家 / 无 DiT / **packed 多模态单流** |
| 训练目标 | 去噪 + JEPA 联合嵌入预测 |
| 加噪形式 | flow matching / VP-DDIM |
| token 预算 | **3072** – 13312（E12 的梯子 + H3）|
| 文本编码器 | T5 / LLaMA-3-8B / CLIP / **Qwen3-VL-32B** |
| VAE | 16 通道 f8 / f16 / **24 通道 f16t4d24** |

**判定**：D2 **34 格 / 22 单元 / 7 族，全部 fail**；D1 @ IntPhys 2 **24 pass + 1 不可解读**（共 25 格）;
D1 @ LikePhys **8 fail + 2 落在阈值上**（未变，E19 没做 LikePhys）。

---

## 5. 复现

```bash
# 权重（代理单流只有 65 KB/s，必须 96 路并行）
python3 scripts/piq_par_download.py --repo MiniMaxAI/MiniMax-H3 \
    --dest $HF_HOME/minimax_h3 \
    --include transformer vae text_encoder tokenizer processor scheduler \
    --workers 96 --chunk-mb 8
# 实测 6.3 MB/s；transformer 62 GB + vae 9.8 GB + text_encoder 62 GB

# 提取（4 卡，一个进程一卡；峰值显存 real 119.8 GB / random 133.4 GB）
bash scripts/e19_queue.sh                        # 实测 19:29:55 -> 20:05:14，35 分钟
# 分析
.venv_h3/bin/python scripts/e19_analyse.py --threads 4    # 854 秒
.venv_h3/bin/python scripts/e19_verdict.py
.venv_h3/bin/python scripts/e19_manifest.py
```

**自检**：`pixel_features()` 的 12 个锚点是 `0,4,8,12,16 | 17,21,25,29,33 | 34,38`，
由 `17→5`、`5→2` 的 `(n-1)//4+1` 关系推出，与 latent 臂的 12 帧对齐；
不用 `linspace` 是因为它落不到这些位置，两臂会错位。

---

## 6. 环境：为什么必须用另一个解释器

| | 主环境（E8–E18）| **`.venv_h3`（E19）** |
|---|---|---|
| python | 系统 `python3` 3.12 | venv（`--system-site-packages`）|
| torch | 2.9.0a0+145a3a7bda.nv25.10 | 同左（共享）|
| diffusers | **0.36.0** | **0.42.1** |
| transformers | **4.57.0** | **5.22.1** |
| huggingface_hub | 0.36.0 | **2.4.0** |

**两套互斥**：`transformers 4.57` 要求 `hub < 1.0`，而 H3 的 `diffusers 0.42.1` 要求 `hub ≥ 1.32`。
升级主环境会让 E8–E18 无法复现，所以 **E19 的一切都用 `.venv_h3/bin/python`**。

`.venv`（旧的）是死链，指向不存在的 `/opt/conda/bin/python3.11` —— 不要用。

---

## 7. 不做

- **不做** LikePhys（与 E17/E18 一致，IntPhys 2 优先）
- **不做** D3–D6（`NEXT_STEPS_AND_DIVISION.md` 明令）
- **不做** t=298（SNR 匹配格）—— 只有一个模型做它会破坏可比性
- **不读** audio 行的任何输出（packed 序列里 audio 存在、按正确长度与 tag 传入，
  但探针只切 video 行）
- **不改** D1/D2 阈值、probe 实现、随机初始化规则、颜色抖动参数

---

## 8. 实现坑（新会话务必先读）

| 坑 | 症状 | 修法 |
|---|---|---|
| **FL2VA/transformer 不能直接加载** | 参数名是 `blocks.*` / `final_layer.*`，diffusers 找不到 `diffusion_pytorch_model.safetensors` | 用仓库**顶层** `transformer/`（diffusers 格式，14 分片）；`FL2VA/` 是原始 checkpoint |
| **H3 VAE 要 ImageNet 归一化** | 传 Wan 的 [-1,1] 不报错，只是特征全错 | `(u/255 − mean)/std`，mean/std 为 ImageNet 常量 |
| **VAE 要采样 posterior + fp16 舍入** | 取 mode 会与官方配方不同 | 照抄 `encoders.py::encode_vae_condition` |
| **`latents_mean` 在外层 config** | `vae.config['latents_mean']` KeyError | 读 `vae/config.json`，不在内部 config |
| **帧数必须 `17n+5`** | 49 帧会被静默对齐成 56（17 latent）| 用 **39** 帧 → 12 latent，与其它 backbone 的 13 接近 |
| **`(N-5)//4+1` 是错的** | 39 帧算出 9，正确是 12 | 用 `video_latent_num_frames(N,17,5)` |
| **特征张量第一维是"槽位"不是层号** | `IndexError: index 16 is out of bounds for dimension 0 with size 2` | 用 `layer_slot()` 映射 |
| **`raw_pixel_32` 是 Wan 的 4 帧分组** | 39 帧给 10 个 bin，与 latent 的 12 个错位 | 分析阶段按 H3 的 17/17/5 重算 |
| **一个进程一卡** | random 臂峰值 133.4 GB，两进程会 OOM | 4 卡 4 任务，没有第二波 |
| **不能用 `pkill` 之后立刻重启** | `pkill -9` 会连带杀掉刚启动的队列（rc=137）| 用 `setsid` 启动，或先确认旧进程已退出 |

---

## 9. 一句话

**33B 上 D1 通过、D2 仍 fail、下界排序与 E8 六族同序 —— 三个方向都没翻。**
但 real AUC 反而比 14B 低，因为 token 预算同时降到了 1/4，所以这不能读成"规模无效"；
而 L16→L27 的 +17 pp（Wan 是 −3.8 pp）是本项目第一次看到反向的深度趋势，未解释。
