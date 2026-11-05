# E15 — LikePhys 侧的跨范式检验

生成：2026-10-28　·　协议：`PREREG.md` D1/D2 + `results/E15_likephys/E15_PREREG.md`

本文是 `E13_PREREG.md` §8 与 `E14_PREREG.md` §5 里两条「LikePhys 暂缓」的解冻执行。
数字出处：`E15_cells_vjepa.json`、`E15_cells_hunyuan.json`、`E15_vjepa_probe_seeds.json`、
`E15_verdict.json`。

---

## 1. 结论先行

**这轮补上了 E13 自己写在 Limitations 里的最大空白，答案是「测出来了，但刚好压在阈值上」。**

| 模型 | 范式 | LikePhys D1 | 判定 |
|---|---|---|---|
| Wan2.1-1.3B | 生成 | 32.9% | 低于阈值 |
| Wan2.1-14B | 生成 | 32.1% | 低于阈值 |
| Wan2.2-TI2V-5B | 生成 | 28.5% | 低于阈值 |
| CogVideoX-2b | 生成 | 42.6% | 低于阈值 |
| CogVideoX-5b | 生成 | 43.5% | 低于阈值 |
| LTX-Video | 生成 | 32.5% | 低于阈值 |
| Mochi-1 | 生成 | 34.4% | 低于阈值 |
| **HunyuanVideo** | 生成 | **27.0%** | **低于阈值**（新增）|
| **V-JEPA 2 ViT-L** | **非生成** | **49.4% ± 2.3** | **`at_threshold`**（新增）|

（上表 9 行；生成式 8 行全部低于 50%，非生成式 1 行压在阈值上。）

两件事：

1. **HunyuanVideo 是一次干净的复现**：27.0%（L16）/ 28.1%（L32），落在 E8 的 28.5–43.5% 区间内
   （略低于下沿）。→「D1 在 LikePhys 上低」对**第五个生成式族**成立。
2. **V-JEPA 2 是 49.4%，六个 probe seed 里 5 个低于 50%、1 个 53.8%**。
   **它比 8 个生成式模型中最高的一个（CogVideoX-5b，43.5%）还高 5.91 个百分点，但没有干净地越过阈值。**

→ 所以 E13 那条 Limitations **不能删掉，只能改写**：
   原句「非生成式编码器上 D1 也低**尚未检验**」→ 现在应改为
   **「非生成式编码器的 D1（49.4%）恰好落在判定阈值上，与生成式模型的 27–44% 有明显差距，
   但不足以判为『不低于阈值』」**。
   （精确差距：49.41% − 43.49% = **5.91 个百分点**；生成式一侧 8/8 全部落在 27.0–43.5% 区间内。）

---

## 2. 这不是「没结果」，而是一个预注册没覆盖的结果

`E15_PREREG.md` §3 只定了三种结局：`pass_dataset` / `training_objective_matters` / `不可解读`，
并且**默认每个格只用一个 probe seed**。实测落在两者之间，而且**换 seed 就换结论**：

### 2.1 L16 的六个 probe seed

| probe seed | real AUC | random AUC | **D1** |
|---|---|---|---|
| 1000 | 0.9686 | 0.7167 | **53.76%** |
| 2000 | 0.9692 | 0.7348 | 49.96% |
| 3000 | 0.9808 | 0.7466 | 48.70% |
| 4242 | 0.9807 | 0.7522 | 47.52% |
| 5000 | 0.9792 | 0.7498 | 47.88% |
| 6000 | 0.9712 | 0.7421 | 48.62% |
| **均值 ± sd** | | | **49.41% ± 2.29%** |

L13 同向：均值 47.97% ± 3.54%，5/6 低于 50%，最高 54.03%。

### 2.2 为什么必须报 sd，而不能只报 49.41%

**如果只报 seed 1000，结论是 53.8% →「非生成式编码器上 D1 不低于阈值」。**
**如果只报 seed 4242，结论是 47.5% →「不低于阈值」被推翻。**

这正是 `E8_REPORT.md` §15.2 禁止的那件事：**在多个格里挑一个好看的**。
所以判定标签用第四种状态 `at_threshold`（±1sd 带 [0.471, 0.517] 跨过 0.5），
并把六个 seed 全列出来。

**预注册漏了这一层，这是本轮的方法论错误，责任在我**：E13 给 V-JEPA 的地板用的是
**label-permutation**（模型确定性前向，无 feature seed），但**没规定 probe 自身的
group-fold 划分 seed**。生成式模型那边这个 seed 是固定的（1000+feature_seed），
到了 V-JEPA 就只剩 1000 这一个值，于是它的不确定性被系统性低估。

→ **补救措施（本轮已做）**：把 probe seed 扫描写成一个独立的敏感性文件
`E15_vjepa_probe_seeds.json`，并在判定里用它。
→ **应该做的（未做）**：把同样的 seed 扫描施加到 E8 的 12 个格上，
   看它们的 D1 是否也在 ±2–3 个百分点内摆动。**如果也在摆，那么 E8 那些 28.5% vs 50%
   的差距是安全的（差 6 个百分点以上），但 E8 内部 42.6% vs 43.5% 这种
   「谁更接近阈值」的排序则不可靠。** 这一条已写进待办。

---

## 3. HunyuanVideo：干净的第五个生成式族

| 格 | 层 | block 类型 | real AUC | random | D1 | 三个 feature seed 的 D1 |
|---|---|---|---|---|---|---|
| L16 | 16 | 双流 | 0.9422 | 0.8228 | **27.0%** | 28.6 / 26.9 / 25.5 |
| Lrel | 32 | 单流 | 0.9508 | 0.8239 | **28.1%** | 27.9 / 29.4 / 27.1 |

- 与 E8 的 6 个模型（28.5–43.5%）**同区间**，且在低端
- **两层一致**（27.0 vs 28.1），不像 LTX 在 IntPhys 2 上那样层间翻转
- 地板 0.0145 / 0.0165，T10 门槛通过（`real − 0.5` = 0.44–0.45，远大于 2×地板）

**注意 L32 是单流块，不是 L16 的深度对照**（`E14_PREREG.md` §3），这里并列只为显示
「换到另一个栈，判定不变」，不能读成深度效应。

---

## 4. LikePhys 的 D2：两个新模型已补测（都是 fail）

此项原列为未完成项，现已用现有缓存补上（零 GPU）。预注册写在
`E15_D2_ADDENDUM.md`（**写于计算之前**），数据在 `E15_D2_likephys.json`。

判据与 E8 完全一致：**物理合法片段 vs 清单自带的 `color_change` 片段**（120 vs 50 clip，
分组 `(scenario, subgroup)`）。

| 模型 | 参考格 | 外观 AUC | 物理 AUC | gap | 判定 |
|---|---|---|---|---|---|
| V-JEPA 2 ViT-L | L16 / clean | **0.9975** | 0.9686 | +0.0289 | **fail** |
| HunyuanVideo | L16 / t=600 | **0.9970** | 0.9422 | +0.0547 | **fail** |

- 加上 E8 的 6 个模型（0.9641–1.0000，6/6 fail），**LikePhys 侧的 D2 现在是 8/8 全部 fail**
- 加上 IntPhys 2 侧的 14 格，**D2 总计 28 个格全部 fail（19 个独立单元、6 个模型族）**
  —— 由 `scripts/audit_d2_count.py` 数出，见 `results/D2_TALLY.json`

### 4.1 一处必须声明的不可比性

LikePhys 的 `color_change` **含时序成分**——LikePhys 原文三处写明变色发生在
"mid-flight" / "mid-simulation" / "mid-flow"，且被列为 invalid variant。
而 IntPhys 2 的 `cj` 是**逐帧恒定 HSV**、无时序成分。

→ 所以 **LikePhys 与 IntPhys 2 的 D2 gap 不能横向比大小**（+0.029 与 +0.276 不是同一件事）。
→ 能横向比的是**方向**，而方向在 28 个格上一致。

→ 这也意味着 V-JEPA 2 在 LikePhys 上 gap 只有 +0.0289，**远小于**它在 IntPhys 2 上的 +0.2759，
   但这**不是**"外观扰动变得更容易/更难"的证据——两个数据集的扰动本来就不同质。

---

## 5. 对主结论的具体修改清单

| 文档 | 现在写的 | 应改为 |
|---|---|---|
| `E13_REPORT.md` §7 第 4 条 | 「非生成式编码器上 D1 也低**尚未检验**」| 「已检验：49.4% ± 2.3%，压在阈值上，见 E15」|
| `E8_REPORT.md` §14.5 | 「真正决定 D1 判定的是**数据集**，不是 backbone」| **保留**，但要加一句：*训练目标的影响未被排除——非生成式编码器落在阈值上（E15，49.4% ± 2.3），比生成式一侧最高的 43.5% 高 5.9 个百分点* |
| `E8_HANDOFF.md` §5.4 | §14.3 的 token 预算解释 | 已被 E12 否证（另见 E12_REPORT §8）|

**第 2 行是唯一一处会改主结论措辞的地方**，且改法是「加限定」而不是「改结论」：
数据集仍然是主因（8 个生成式模型 **8/8** 在 27.0–43.5%），但「不是 backbone」这句
不能再说成「不是任何模型属性」。

---

## 6. 已知限制

1. **V-JEPA 2 的 64 帧输入 vs 60 帧 clip**：`linspace` 会重复 4 帧（已写进预注册 §2.1）。
   这不改变训练/随机的对比（两边同样处理），但让时间轴略不平滑。
2. **probe seed 只对 V-JEPA 做了扫描**，8 个生成式模型没有。它们的 feature seed 三个值是
   固定的 0/1/2，probe seed 固定为 1000+seed，所以**没有**同等意义上的不确定性估计。
3. ~~**ViT-H 的 LikePhys 未做**~~：**已补做**。结果是本轮最出乎意料的一格——
   ViT-H 在 LikePhys 上 D1 = **54.1% ± 3.40**（1/6 划分低于 50%），
   比 ViT-L 的 **49.4% ± 2.29**（5/6 低于）**更高**，
   与 IntPhys 2 上"更大更弱"的方向**相反**。差 +4.68 个百分点 = 1.64 个合并 sd，
   处于"可能真实、也可能不是"的地带。详见 `E13_REPORT.md` §6.1.1 与
   `E13_cells_vith_likephys.json`。
4. **D2 未在 LikePhys 上补做**（§4）。

---

## 7. 复现

```bash
export HF_HOME=$HF_HOME

# 提取（4 卡，一条件一卡）
CUDA_VISIBLE_DEVICES=0 python scripts/e13_extract.py --model vitl --dataset likephys --weights real   --variant plain
CUDA_VISIBLE_DEVICES=1 python scripts/e13_extract.py --model vitl --dataset likephys --weights random --variant plain
CUDA_VISIBLE_DEVICES=2 python scripts/e14_extract.py --dataset likephys --weights real   --variant plain
CUDA_VISIBLE_DEVICES=3 python scripts/e14_extract.py --dataset likephys --weights random --variant plain

# 分析 + 判定（分析脚本本轮为一次性脚本，见 logs/gpu/e15_analyse_*.log 的 heredoc）
python scripts/e15_verdict.py
```

实测：V-JEPA 0.24 s/clip（920 clip 用 219 s）、Hunyuan 2.98–3.11 s/clip（约 46 分钟），
4 卡并行总耗时约 50 分钟。
