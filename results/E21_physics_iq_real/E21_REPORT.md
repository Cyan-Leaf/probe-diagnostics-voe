# E21 — 真实世界视频（Physics-IQ）：probe 能读出「哪一次拍摄」吗

> 执行于 2026-11-05，4×B200。**判据写定于提取之前**：`E21_PREREG.md`（D-take-1..3 与阈值）。
> 原始数据：`manifest.json`（396 clip，含逐文件 sha256）/ `E21_cells.json` / `E21_verdict.json`。
> 脚本：`e21_extract.py` / `e21_analyse.py`。

## 0. 一句话结论

**在真实拍摄的视频上，probe 读不出「同一场景的哪一次拍摄」——AUC 0.4891，落在随机水平。**

而正对照成立：同一批特征区分**机位方向**达 **0.9717**。所以这不是管子坏了，是 take 轴确实不可读。

这条结果把 D2 的失效范围**收窄**了：probe 读的是**编辑过程引入的痕迹**（重上色），
不是「真实世界里任何与物理无关的变化」。

## 1. 这个实验能问什么、不能问什么

**不能问**：它**不是** possible/impossible 判据。核实过官方仓库与磁盘上的 396 个文件：
*"Each scenario is recorded twice to capture natural physical variations"* ——
`take-1` / `take-2` 是**同一场景的两次自然拍摄，两次物理上都合法**。
所以 E21 **不能**回答「真实视频上能否检测物理违反」，也**不得**写成「真实视频上的物理理解」。
`NEXT_STEPS_AND_DIVISION.md` 那条「需补 possible/impossible 配对标签」是对的：本数据里没有这个任务。

**能问**：take 轴是一个**我们没有构造**、且**物理合法**的变异。
D2 的对照（重上色）是我们自己施加的，因此一直有一个反驳：
「probe 读的可能只是你们施加的那个变换」。take 轴让这个反驳可以直接检验。

| 任务 | 变的是什么 | 物理 | 谁造的 |
|---|---|---|---|
| D1（IntPhys 2 / LikePhys）| possible vs impossible | **变了** | benchmark 作者 |
| D2 对照（重上色）| 仅色相/饱和度 | 合法 | **我们** |
| **E21 take** | **同一场景的两次真实表演** | **合法，两次都真实** | **世界** |

## 2. 数据与协议

| 项 | 值 |
|---|---|
| clip 数 | **396**（`take-1` 198 + `take-2` 198）|
| 场景 | **66**，每场景 3 个机位、2 次拍摄 |
| 配对完整性 | **198/198 个 (scene, perspective) 键两次拍摄齐全**（`manifest.json` 逐条核） |
| 规格 | 30 FPS、240 帧（8 s）、原始 3840×2160 |
| 读帧 | **49 帧全 clip 均匀采样**（`linspace`），512×512 — 与 E7 读 IntPhys 2 完全同一路径 |
| backbone | **Wan2.1-T2V-1.3B**，L16 / t=600，feature seeds 0/1/2 |
| 读出 | `e8_analyse.probe` **原样导入**（E11 实测重写版偏 2.3 pp）|
| 分组 | **scene**（66 组）。两次拍摄、三个机位必须同折 —— 否则 probe 可以记住场景，take 任务会因错误的原因变得可学 |
| 线程 | 8（`--threads 8`），全表一致 |

提取：4 卡各 1 shard，real 臂 322–326 s，random 臂 113–116 s，单卡峰值 14.4 GB。

## 3. 结果

### 3.1 三个臂（real backbone，L16/t=600）

| 任务 | 特征来源 | n | 组数 | AUC | seed sd | **分离度** |
|---|---|---|---|---|---|---|
| **take** | **DiT L16** | 396 | 66 | **0.4891** | 0.0309 | **0.5201** |
| take | raw pixel 32 | 396 | 66 | 0.5050 | 0.0086 | 0.5082 |
| take | VAE latent | 396 | 66 | 0.5306 | 0.0076 | 0.5306 |
| take | **random-init DiT** | 396 | 66 | 0.5070 | 0.0054 | 0.5070 |
| **perspective** | **DiT L16** | 264 | 66 | **0.9717** | 0.0126 | 0.9717 |
| perspective | raw pixel 32 | 264 | 66 | 0.8751 | 0.0268 | 0.8751 |
| perspective | VAE latent | 264 | 66 | 0.7557 | 0.0212 | 0.7557 |
| perspective | random-init DiT | 264 | 66 | 0.9221 | 0.0127 | 0.9221 |

逐 seed（take, DiT）：`0.5087 / 0.5052 / 0.4535`。三个 seed 都在 0.5 附近，没有一个越过 0.6。

### 3.2 判定

| id | 规则 | 值 | 阈值 | 判定 |
|---|---|---|---|---|
| **D-take-1** | 分离度(take) ≥ 0.60 | **0.5201** | 0.60 | **refuted** |
| **D-take-2** | 分离度(take) ≥ AUC_physics(E8) − 0.05 | **0.5201** | 0.5958 | **refuted** |
| **D-take-3** | AUC(perspective) ≥ 0.95（正对照）| **0.9717** | 0.95 | **hold** |

参照值 `AUC_physics(E8)` = **0.6458**（`E8_cells_intphys2.json` 的 `wan21_1b3|intphys2|t600|L16|real|physics`
seed-0，**只读引用，E21 不重算**）。

### 3.3 三条必须一起读

**(a) take 的分离度低于每一个平凡下界。**
DiT 0.5201 < VAE latent 0.5306。也就是说，**训练过的 transformer 在这条轴上的可读性还不如
VAE 的均值池化特征**。这与物理任务上的排序（trained > pixel，E8 6/6）方向相反。

**(b) random-init 与 trained 几乎相同（0.5070 vs 0.5201，差 1.3 pp），在 seed sd（3.09 pp）之内。**
按项目自己的规矩（E17：差 0.06 pp ≪ 噪声 2.51 pp 时不主张差异），**不能说训练引入了 take 信息**。

**(c) 正对照很强，且 raw pixel 也能做到 0.8751。**
机位方向本质上是**像素级**属性 —— random-init DiT（0.9221）甚至高于 raw pixel（0.8751）。
所以 perspective 成立只证明「管子能学到一个明显的真实差异」，
**不能**当成「DiT 表示里有机位信息」的证据。

## 4. 这条结果改了什么

**D2 的失效范围被收窄，而不是扩大。**

D2 在 34/34 格上失败，我们一直把它的含义写作「probe 读的是编辑痕迹」。
E21 给出一个关键限定：在**未经编辑的真实素材**上，一个同样自然的、与物理无关的变异
（哪一次拍摄）**完全读不出来**。所以

> probe 读的不是「任何与物理无关的变化」，而是**编辑过程引入的那类痕迹**。

这把「线性 probe 什么都读」这个最伤的反驳挡掉了：不是的，它对真实世界的 nuisance 是盲的，
它只对**合成编辑留下的统计痕迹**敏感。这是 D2 结论更强、也更准确的版本。

**同时它也不支持过度解读。** take 与 physics 不是同一个任务：
- take 的两次拍摄共享**同一场景布局、同一批物体、同一机位**，差异可能小到本来就不可线性读出；
- 而 physics 任务是 possible vs impossible，差异是**被作者刻意放大**的。

所以正确的表述是「真实世界里一个自然的 nuisance 读不出来」，
**不是**「真实视频上 probe 没有 nuisance 敏感性」。

## 5. 预注册修正（如实登记）

**修正 1：判定改用无方向分离度 `max(AUC, 1−AUC)`。**
`E21_PREREG.md` §4 把规则写成 `AUC_take ≥ 0.60`，**这隐含假设 take-1→take-2 有固定方向**。
没有任何东西固定这个方向：AUC 0.40 与 0.60 一样说明可区分。
按原规则，一个落在下尾的结果会被判成 "refuted"，而数据其实显示**更强**的 take 信号。
实测 0.4891（近随机）**在两种判法下结论相同**，但这个规则会误判，所以必须改。
修正内容与时间写在 `E21_verdict.json.rule_amendment`。

**修正 2（观察，不改数）：** 首次分析时 `random|take` 因 random 臂尚未提取完而记为 NaN，
补跑后得到 0.5070。该臂的提取与 real 臂同脚本同参数，仅 `--backbone random`。

## 6. 局限（不待追问，主动写）

1. **66 个场景是有效 n，不是 198。** 三个机位不独立（同一场布景），
   任何场景级断言的有效样本量都只有 66。
2. **take-1/take-2 的差异不止一处**：光照漂移、机位微动、物体摆放都可能不同。
   因此即便 AUC 高也**无法指出** probe 具体读的是哪一个 nuisance；
   本实验只能说「读不出这个整体差异」，不能定位。
3. **没有违反标签**，所以不能与 Physics-IQ 排行榜分数对比
   （那是「生成 + 打分」流水线，不是对真实素材做 probe）。
4. **单一 backbone**。只测了 Wan2.1-1.3B；不过 E21 的目的是取证「真实素材上是否可读」，
   不是再做一条模型规模轴 —— 按 `NEXT_STEPS_AND_DIVISION.md`「模型数量已饱和」的口径。
5. **随机水平的下界**：raw pixel 0.5082、random-init 0.5070。
   take 的 0.5201 距这两者只有约 1.2–1.3 pp，与 seed sd（3.09 pp）同量级。

## 7. 复现

```bash
export HF_HOME=/path/to/hf_cache
python3 scripts/e21_extract.py --stage manifest
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/e21_extract.py --stage extract \
      --backbone real --shard $i --num-shards 4
  CUDA_VISIBLE_DEVICES=$i python3 scripts/e21_extract.py --stage extract \
      --backbone random --shard $i --num-shards 4
done
python3 scripts/e21_analyse.py --threads 8
```

数据：Physics-IQ `full-videos/take-{1,2}/30FPS/`，共 396 条，合计约 3.3 GB（已下到 `data/physics_iq/`）。
`manifest.json` 逐文件记录 sha256 与字节数，可用于校验重新下载的副本。
