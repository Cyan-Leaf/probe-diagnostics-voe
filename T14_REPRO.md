# T14_REPRO — L0 冒烟复现路径

> 2026-09-22。对应 `PROPOSAL_v1.0_*.md` §7。
> 实测数据：`results/T14_release/L0_tolerance.json`（5 次独立场景抽样）。
> 脚本：`repro/l0_smoke.py`。

---

## 0. 结论先说

**L0 能稳定复现的只有「次序关系」，不是数字。**

| 量 | 24 场景子集（5 次抽样） | 全集 | 可否给容差 |
|---|---|---|---|
| `appearance_trained` | **1.0000，sd = 0.0000**，5/5 全中 | 1.0000 | ✅ **精确复现** |
| `physics_trained` | 0.5598 ± 0.0339，范围 [0.5240, 0.6094] | 0.6458 | ⚠️ 只能给区间 |
| `physics_random` | 0.5103 ± 0.0326，范围 [0.4746, 0.5527] | 0.5168 | ⚠️ 只能给区间 |
| **`d1_share`** | **0.788 ± 0.866，范围 [−0.636, +1.552]** | 0.885 | ❌ **不可用** |

**`d1_share` 在这个子集规模上没有意义，我们不为它给容差。**

---

## 1. ⚠️ `d1_share` 为什么不可用（如实报告，不挑 seed）

五次抽样的逐次结果：

| seed | physics_trained | physics_random | d1_share |
|---|---|---|---|
| 0 | 0.6094 | 0.5343 | 0.687 |
| 1 | 0.5665 | 0.4746 | 1.382 |
| 2 | 0.5323 | **0.5527** | **−0.636** |
| 3 | 0.5240 | 0.4867 | 1.552 |
| 4 | 0.5667 | 0.5031 | 0.953 |

两个问题：

1. **seed 2 上随机初始化反而高于训练模型**（0.5527 > 0.5323），
   于是 `d1_share` 变成负数。
2. `d1_share = (real − random)/(real − 0.5)`，当 `real` 接近 0.5 时分母趋零，
   比值被放大到无意义。

**这恰好是我们自己在 §5.2 构造效度里发现的 D1 退化**
（A2 案例：real AUC 0.5326 时 D1 报出 1.113），
现在在真实的小子集上再次出现。**协议里的可解释性门槛正是为此而加。**

> **我们没有挑 seed。** 若只报 seed 0（share 0.687，看起来最接近全集的 0.885），
> 会给出「L0 能复现 D1」的错误印象。

### 后果：L0 的判据是三条定性检查，不是数值

```
appearance_above_physics   1.0000 > 0.5598   PASS
physics_above_chance       0.5598 > 0.5      PASS
random_below_trained       0.5103 < 0.5598   PASS
```

**5 次抽样全部通过这三条。** 这是 L0 承诺可复现的内容。

> 若要复现 `d1_share` 的数值，需要**全部 253 个场景**，
> 不是 30 分钟的冒烟路径能覆盖的。

---

## 2. L0 做什么

复现 proposal 开篇那个例子的**次序关系**：

- 同一批特征、同一套折，只换标签定义
- **外观任务（纯调色）≫ 物理任务**
- 训练模型仅略高于随机初始化

流程：下数据 → 抽 24 个场景（96 clip）→ 提特征（real / random / real+调色）→
训 probe → 出三个 AUC。**不写任何缓存**，
这样第二次运行不会因为复用第一次的中间产物而假性通过。

---

## 3. 实测成本（本机）

| 项 | 实测 |
|---|---|
| 单次运行 | **约 7–10 分钟** |
| 提特征速度 | 1.73–1.85 s/clip |
| DiT forward 次数 | 约 240 次 / 轮 |
| 模型加载 | 3 次（real、random、real 复用） |
| 显存峰值 | < 24 GB（单卡 4090 实测通过） |
| 磁盘写入 | 仅一个 JSON（KB 级） |

5 次抽样共约 45 分钟。**单次运行满足「≤30 分钟」的要求。**

---

## 4. ⚠️ L0 对外部环境的全部假设（未在干净机器上验证）

**我们无法用自己的机器验证 L0 在别处能跑。**
本机已有数据、已装依赖、已设环境变量，任何一条都会造成假性通过。
以下逐条列出 L0 依赖什么，**需要一次独立运行来确认**。

### 4.1 软件

| 项 | 本机版本 | 假设 |
|---|---|---|
| Python | 3.11.10 | ≥ 3.10（用了 `X \| None` 语法） |
| PyTorch | 2.5.1+cu124 | 需 CUDA 版；CPU 版会因 `torch.cuda` 调用失败 |
| CUDA | 12.4 | 未测其他版本 |
| diffusers | **0.36.0** | **≥ 0.36**：`WanTransformer3DModel` 在早期版本不存在（`PITFALLS` 记录过 0.29.2 会失败） |
| transformers | 4.57.6 | 未测边界 |
| decord | 0.6.0 | **manylinux wheel，macOS / Windows 需自行编译** |
| opencv-python | 4.11.0.86 | 仅 colour jitter 验证用 |
| scikit-learn | 1.9.1 | `_rowspace` 依赖其 `PCA` / `LogisticRegression` 行为 |

完整清单见 `release/requirements.txt`。

### 4.2 硬件

- **单张 ≥ 24 GB 显存的 NVIDIA 卡**。峰值受 49 帧 × 512×512 的 VAE 编码支配。
  更小的卡**未测**，可能需要减少 `--scenes` 或改分辨率（后者会改变数字）。
- **未在多卡、CPU-only、非 NVIDIA 上测试。**

### 4.3 数据

| 项 | 说明 |
|---|---|
| IntPhys 2 | **约 12 GB**，需从 HuggingFace 或 `dl.fbaipublicfiles.com` 下载 |
| 定位方式 | 依次尝试 `INTPHYS2_ROOT` → `HF_HOME`/`~/.cache/huggingface` → `data/IntPhys2` |
| 需要的部分 | **仅 `Main` split**，但下载是整包 |
| 模型权重 | Wan2.2-TI2V-5B，**约 10 GB**，首次运行自动下载 |

### 4.4 网络

- 首次运行需访问 **HuggingFace**。内网环境可能需要代理或镜像。
- 本机是在 `HF_HUB_OFFLINE=1` + 已缓存权重的条件下跑的，
  **「首次从零下载」这条路径我们没有走过**。

### 4.5 一个已知的未解耦依赖

`repro/l0_smoke.py` 通过 `sys.path` 引入 `scripts/` 下的
`wan_probe_lib` / `e7_extract` / `e10_s1s3` / `t4_caliper`。
**这些模块假设仓库布局不变。** 单独拷贝 `l0_smoke.py` 出去会失败。

另外它读 `results/T2_env/prompt_embeds_empty.pt`（空 prompt 的文本嵌入，约 KB 级），
该文件**在仓库内**，不需另外下载。

---

## 5. 需要独立验证的清单

请在**一台没碰过这个项目的机器**上执行，并如实回报：

```bash
git clone <repo> && cd <repo>
pip install -r requirements.txt

huggingface-cli download facebook/IntPhys2 --repo-type dataset --local-dir data/IntPhys2

python repro/l0_smoke.py --dry-run      # 应打印计划，不占 GPU
python repro/l0_smoke.py --scenes 24    # 应在 30 分钟内出三个 AUC
```

需要确认的：

- [ ] `pip install -r requirements.txt` 在干净环境能否装上（尤其 **decord**）
- [ ] 模型权重**首次下载**是否顺利（我们只跑过已缓存的情形）
- [ ] 显存峰值是否 < 24 GB
- [ ] 三条定性检查是否全部 PASS
- [ ] 总耗时

> **在拿到独立运行结果之前，不得在 proposal 或 README 中宣称 L0「已验证可复现」。**
> 当前表述一律为「未在干净机器上验证」。

---

## 6. 与全集数字的关系

| 量 | 全集 | L0 子集 |
|---|---|---|
| physics_trained | 0.6458 | 0.5598（低约 0.09） |
| physics_random | 0.5168 | 0.5103 |
| appearance_trained | 1.0000 | **1.0000** |

子集的 `physics_trained` 系统性偏低是预期的 ——
253 场景降到 24 后每折的训练样本大幅减少，probe 欠拟合。
**`appearance` 不受影响**，因为调色的可分性极强，小样本也足够。

**这本身就是一条证据**：
物理任务的信号弱到需要全部数据才测得出，
而外观任务的信号强到 96 个 clip 就能满分。
