# E8 交接文档（Handoff）

**写给：接手的下一个会话**
**写于：2026-10-26**（E8 原始会话已于 2026-10-24 03:01 结束，机器已关）

本文档的唯一目的：让一个**没有读过原始对话、没有 GPU、没有模型权重**的新会话，
能在十分钟内判断出「E8 做到哪了、哪些数可信、哪些坑已经踩过、下一步能做什么」。

如果你只读一份文件，读 `results/E8_second_backbone/E8_REPORT.md` 的 **§14**（最终矩阵）。
本文档不重复结论，只补报告里没有的东西。

---

## 0. 一句话状态

**E8 已完成并超出原定范围**（原计划 1 个 backbone，实际 7 个模型 × 2 数据集）。
所有交付物齐全、JSON 可解析、报告数字基本可追溯。
**未完成的是"过程记录"**：日志无索引、10 条问题只写了 1 条、代码版本没钉死。本文档补这些。

---

## 1. 先建立这个认知：哪些东西可以放心用

| 资产 | 位置 | 可信度 | 说明 |
|---|---|---|---|
| 最终判定矩阵 | `E8_REPORT.md` §14 | **高** | 本文档核过，数字与 `E8_verdict.json` 一致 |
| 逐格原始值 | `E8_cells_{intphys2,likephys}.json` | **高** | 带 `per_seed_auc`、`n`、`started/finished` |
| 四组下界 | `E8_lower_bounds.json` | **高** | |
| 模型清单/偏离声明 | `manifest.json` | **高** | 5 条偏离都写了 `why` / `consequence` |
| §11 / §13 的旧表 | `E8_REPORT.md` | **低，不要引用** | 已被 §14 取代，且 LTX 数字与 §14 不一致（见 §5.2）|
| 特征缓存 | `cache/` (464G) + `_intermediates/` (46G) | **高** | 可直接重算，无需 GPU |

**关键：`cache/` 里 6 个模型 × real/random × intphys2/intphys2_cj/likephys 全部齐全**
（已逐个核对，`_cj` 各 506 个文件）。所以**所有分析都能在无 GPU 机器上重跑**。

---

## 2. 复现命令（无 GPU 可跑）

只有 `e8_analyse.py` / `e8_verdict.py` 这两步是纯 CPU + 缓存：

```bash
cd $REPO

# 分析（读 cache/，产出 cells）
python3 scripts/e8_analyse.py --backbones cogvideox2b --dataset intphys2

# 判定（读 cells，产出 verdict）
python3 scripts/e8_verdict.py --backbones cogvideox2b --dataset intphys2
```

**需要 GPU 的只有 `e8_extract*.py`（重新提取特征）。除非要加新模型，否则用不到。**

⚠️ `cache/` 里已有 `_cj` 变体，但若你要跑 `--layers` 指定新层，仍要先确认对应层的 `.npy` 存在，
否则会撞上本文档 §4.1 记录的那个 AssertionError。

---

## 3. 日志索引（107 个日志，哪个是终态）

`logs/e8/` 下 107 个文件，混杂着成功、失败、版本迭代。**按"看最后一个日志"是错的**，
因为多个模型的早期日志以报错结尾，而后期日志才是终态。

### 3.1 分析类日志（`analyse_*.log`）

| 终态 | 日志 | 说明 |
|---|---|---|
| ✅ 终态 | `analyse_grid2.log` | CogVideoX-2b 的 6×6 全网格；`analyse_grid.log` 是同一任务的无收尾早期版本 |
| ✅ 终态 | `analyse_w14b_ip2.log` | Wan 14B / IntPhys2；`analyse_w14b_ip.log` 中止于缺 `_cj` 缓存 |
| ✅ 终态 | `analyse_ltx_ip4.log` | LTX / IntPhys2；`ip`(缺缓存) / `ip2`(IndexError) / `ip3`(IndexError) 均失败 |
| ✅ 终态 | `analyse_cog5b_rope_ip2.log` | CogVideoX-5b / IntPhys2；带 `rope_` 前缀的是修复后重跑 |
| ✅ 终态 | `analyse_mochi_ip.log` / `analyse_mochi_lp.log` | Mochi 两数据集 |
| ✅ 终态 | `analyse_lp_both.log` / `analyse_w1b3_*.log` / `analyse_cog5b_lp.log` | LikePhys 侧 |
| ❌ 失败 | `analyse_ltx_scan.log` | **末尾是 AssertionError**，但 §15 引用的 9 个逐层值在报错之前已打印（已抄进本文档 §5.3）|
| ❌ 失败 | `analyse_cog5b_layers.log` | 缺 `cogvideox5b_feats_real_intphys2_cj`（当时该缓存尚未生成）|
| ❌ 失败 | `analyse_w14b_ip.log` | 缺 `wan21_14b_feats_real_intphys2_cj` |
| ❌ 失败 | `analyse_ltx_ip2.log` / `ip3.log` | `IndexError: index 834 is out of bounds for axis 0 with size 834` |

### 3.2 提取类日志（`{model}_{real,rand}*.log`）

大量 `shard N done in Xs` 结尾的是**成功的提取分片**，`?` 是因为结尾没有 `wrote` 字样。
**关键区分：带 `rope_` 前缀的是 CogVideoX-5b 修复 RoPE 之后的重跑，不带的是首轮作废数据。**

| 作废 | 有效 |
|---|---|
| `cog5b_real_s0/s1.log`、`cog5b_rand_s0/s1.log`、`cog5b_realcj.log`、`cog5b_randomcj.log`（10-23 22:04–22:34）| `cog5b_rope_*.log`（10-23 23:26 起）|

分界点是 **`scripts/cogvideox_probe_lib.py` 的修改时间 10-23 22:52**。早于此时间的 5b 提取全部无效。

### 3.3 下载类日志

| 日志 | 结果 |
|---|---|
| `dl_cog5b.log` | ❌ `RuntimeError: Data processing error: CAS service error` |
| `dl_cog5b2.log` / `dl_wan14b2.log` | ✅ 重试成功（`DONE`）|
| `dl_ltx.log` / `dl_mochi.log` / `dl_wan21.log` | ✅ |
| `w14b_real.log` | ❌ `OSError: no file named diffusion_pytorch_model.bin`（权重下载不完整，重下后由 `w14b_real_s0/s1.log` 成功）|

---

## 4. 已知坑（重跑前必读）

### 4.1 分析脚本硬依赖 `_cj` 缓存

`e8_analyse.py` 的 `load_X()` 会 `assert files`，缺 `_cj` 缓存直接 AssertionError。
**这个错在本次会话里犯了 3 次**（w14b、cog5b、ltx 各一次），因为当时提取还没跑完就开始分析。
→ 重跑前先确认目标模型的 `cache/{model}_feats_{real,random}_intphys2_cj` 有 506 个文件。

### 4.2 `_appearance_cj` 的 n_pos 不匹配（未修复）

`analyse_ltx_ip2.log` / `ip3.log` 报 `IndexError: index 834 out of bounds for axis 0 with size 834`，
位置在 `e8_analyse.py:220` 的 `if len(np.unique(y[te])) < 2`。
现象是分组标签长度（840）与实际样本数（834）不一致——LTX 有若干 clip 提取失败导致数量对不齐。
**`ip4` 绕过了这个问题但没有修根因**。若将来对 LTX 做新的分组分析，**这个 bug 会再次出现**。
→ 排查方向：先核对 `cache/ltx_feats_*_intphys2_cj` 的文件数与 `E8_cells` 里的 `n` 是否一致。

### 4.3 随机初始化：不能照抄 Wan 的规则

`wan_probe_lib._init_weights_` 把所有一维参数置零。Wan 上无害（调制走二维 `scale_shift_table`），
但在 CogVideoX / LTX 上会让 affine norm weight 归零、整个 block 塌成常数（实测 clip 间 std = 0.000000）。
→ 已改为 norm-aware 初始化，记录在 `manifest.json` 的 `deviations[0]`。
→ **后果：随机下界的绝对值不可与 Wan E7 直接比较，只能比排序。**

### 4.4 CogVideoX-5b 的 RoPE（已修复，但要知道影响面）

5b 的 `use_rotary_positional_embeddings=True`，位置编码要由调用方传入 `image_rotary_emb`；
不传不会报错，整个前向位置盲。**症状极隐蔽：8 层物理 AUC 全贴 0.51，而外观对照仍 0.9975。**
→ 修复在 `cogvideox_probe_lib._rotary_embeds()`（10-23 22:52）。
→ 2b 用可学习位置表、Wan 内部自建 RoPE，均不受影响。

---

## 5. 报告里需要修正的地方（**已知缺陷，别照抄**）

### 5.1 `E8_verdict.json` 不符合定死的 schema

`NEXT_STEPS_AND_DIVISION.md` 要求顶层 `model / reference_cell / lower_bounds / d1 / d2 / note`。
实际顶层是 `reference_cell / d1_threshold / prereg / thresholds_from / datasets`，
**五个必需键全缺**，信息被拆进 `datasets.{intphys2,likephys}.backbones.*`。
内容没丢，但按"JSON 键齐不齐"的验收标准**字面判 fail**。

### 5.2 §13 与 §14 的 LTX 数字互相矛盾

| | §13 | §14（最终）|
|---|---|---|
| IntPhys2 外观 AUC | 0.9990 | **0.9986** |
| IntPhys2 gap | +0.4483 | **+0.4461** |
| LikePhys gap | +0.1233 | **+0.1215** |

§14 与 `E8_verdict.json` 一致，**以 §14 为准**。§13 的三个数在任何交付 JSON 里都找不到。

### 5.3 §15 的逐层扫描值不在任何交付 JSON 里

报告原文写"（9 层，`E8_cells_intphys2*.json`）"——**指错了文件**。那批值只在
`logs/e8/analyse_ltx_scan.log` 第 3100–3116 行（且该日志末尾报错）。已抄录如下：

LTX-Video-0.9.5 / IntPhys2 / t=600 / real / physics（n=1012，9 层连续扫描）：

| 层 | real AUC | ±seed |
|---|---|---|
| L0 | 0.5305 | 0.0028 |
| L4 | 0.5357 | 0.0063 |
| L8 | 0.5419 | 0.0027 |
| **L12** | **0.5548** | 0.0041 |
| L15 | 0.5524 | 0.0094 |
| L18 | 0.5478 | 0.0146 |
| L21 | 0.5382 | 0.0188 |
| L24 | 0.5288 | 0.0315 |
| L27 | 0.5171 | 0.0179 |

峰值 0.5548 在 **L12**，参考格 L16（0.5506）并非峰值。这与报告 §10 那条
"参考格不是信息量最大的格"是同一个现象。

### 5.4 过时表述

- 报告标题仍是 `E8 — Second backbone: CogVideoX-2b`，实际覆盖 7 个模型。
- **§8「未完成与不做」整段过时**：写着"只跑了 CogVideoX-2b、规模轴尚未开始"，但后面全做完了。
  读 §8 会严重误判进度。
- `RUNLOG.md` 声称"`audit` CLEAN、`crosscheck` PASS"——**这两个脚本不存在**，无法复核。
- `RUNLOG.md` 声称"主动记录的问题共 10 条，详见 RUNLOG 历史"——**RUNLOG.md 无历史章节**，
  §12 只写了 RoPE 一条。本文档 §4 是按日志反推的 4 条，**其余 6 条已丢失**。
- `START_HERE.md`（9-21）与 `NEXT_STEPS_AND_DIVISION.md`（10-21）均未更新，
  还写着 E8 待认领。**新会话别从这里判断进度。**

---

## 6. 代码版本（本目录不是 git repo，只能靠指纹）

`manifest.json` 的 `code_rev` 是空串。以下为 2026-10-26 时的脚本 sha256 前 16 位，
**用于判断"是否还是当时那版代码"**：

| 脚本 | sha256[:16] | 职责 |
|---|---|---|
| `cogvideox_probe_lib.py` | `c97f0ab6641b7b84` | CogVideoX 前向 + RoPE 修复 |
| `e8_backbones.py` | `647d86330f023bc8` | 模型注册表 |
| `e8_extract.py` | `18b5b8cdb09bd2fb` | CogVideoX 专用提取 |
| `e8_extract_generic.py` | `f1cfdd32bece3932` | 通用提取 |
| `e8_extract_ltx.py` | `625198204a7fa4d1` | LTX 专用提取 |
| `e8_extract_mochi.py` | `d5b352e54ac4b1e4` | Mochi 专用提取 |
| `e8_analyse.py` | `fd16dd08752af7ac` | probe + cells |
| `e8_verdict.py` | `030b0770caf92d4b` | D1/D2 判定 |

若重算结果与 `E8_cells_*.json` 不一致，先比这些指纹。

---

## 7. 时间线（仅关键节点）

```
10-23 17:31  E8_PREREG.md 定稿（跑任何提取之前）
10-23 18:49  e8_verdict.py
10-23 19:58  6×6 全网格跑完（CogVideoX-2b / IntPhys2）
10-23 22:04  CogVideoX-5b 首轮提取 → 22:52 发现 RoPE bug，作废
10-23 22:52  cogvideox_probe_lib.py 修复
10-23 23:26  5b 重跑开始（rope_ 前缀日志）
10-24 00:20  LTX 提取 / 00:47 LTX 逐层扫描
10-24 01:52  e8_analyse.py 最后修改
10-24 02:45  分析中断一次 → 02:47 重新拉起并合并（cells 里 merged_from_previous: true）
10-24 02:58  E8_REPORT.md
10-24 03:00  E8_lower_bounds.json + E8_verdict.json
10-24 03:01  RUNLOG.md  ← 会话正常收尾
```

**"02:45 中断"不是崩溃，是分析进程重开**，`E8_cells_intphys2.json` 的
`previous_started: 2026-10-24 02:45:15` 记录在案，数据已合并，无需重跑。

---

## 8. 下一步还能做什么（按是否需要 GPU 分类）

**不需要 GPU（只依赖现有 cache / JSON）：**

| 任务 | 说明 |
|---|---|
| 修 §5 的四处报告缺陷 | 纯文本/JSON 编辑 |
| `E8_verdict.json` 补齐 schema 五键 | 从现有结构搬迁，不用重算 |
| LTX 扫描值落成 JSON | 值已在 §5.3 |
| 修 `_appearance_cj` 的 n_pos bug（§4.2）| 会影响将来所有 LTX 分析 |
| **E9**（剔除饱和场景重算）| `NEXT_STEPS_AND_DIVISION.md` B 项，零 GPU |
| **E11**（非线性读出补强）| 同上，零 GPU |
| D7 起草、T16 文献重扫 | 零 GPU |

**需要 GPU：**

- 新增 backbone 的特征提取
- T15 干净机器复现（需 `nvidia-smi` 环境快照 + 冒烟路径）

---

## 9. 交接自检清单

新会话接手后，用这五步确认自己理解对了：

1. `python3 -c "import json;print(list(json.load(open('results/E8_second_backbone/E8_verdict.json'))))"`
   → 应看到顶层是 `datasets`，**不是** `d1`/`d2`（即 §5.1 的缺陷仍在，除非已修）
2. 读 `E8_REPORT.md` **§14**，确认 7 模型 × 2 数据集矩阵
3. 读本文档 §5.3 的 LTX 9 层表，确认峰值是 L12 的 0.5548
4. 确认 `cache/` 里 6 个模型的 `_cj` 缓存各 506 文件（无 GPU 重算的前提）
5. 若打算改分析脚本，先记下 §6 的指纹，改完比对结果是否变化

---

**最后一条提醒**：本仓库的四条红线（`NEXT_STEPS_AND_DIVISION.md` §3）依然有效，
其中最容易被违反的一条是"不手改 `results/`"——上面 §5 的修正
**应当通过脚本重新生成 JSON，而不是手工编辑**。手工改会让 §6 的指纹校验失去意义。
