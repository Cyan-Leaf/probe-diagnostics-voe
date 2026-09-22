# E6 — 把 D1–D6 代入文献

> 执行于 2026-09-18。**只依据论文原文**（硬规则⑤）。
> 原文来源：`results/E6_literature/raw/<id>.{abs.html,html,pdf,txt}`，
> 抓取状态见 `fetch_status.json`（四篇全部拿到 HTML 全文，63–94 K 字符）。
> 证据抽取：`scripts/e6_evidence.py`、定位：`scripts/e6_section.py`。
>
> **取不到原文的格子写「无法核实」；本轮四篇全部可核实，无此情况。**

## 0. 一句话结论

**四篇核心论文没有一篇报告 D1（随机初始化下界）。**
D2 只有 Punzo 报告了近似物（时序对照，非属性无关编辑对照）。
最完整的是 Punzo（D4 ✅、D5 ✅、D2 部分），最薄的是 LikePhys（D1–D4 全缺）。

## 1. 四篇的设定（原文核实）

| 论文 | 数据 | 取状态方式 | 标签 | 读出 | 报告结果 |
|---|---|---|---|---|---|
| **Invisible Hand** `2606.05328` v1 2026-06-03 | IntPhys + InfLevel | **显式反向采样**恢复轨迹 | VoE 配对 | linear probe | **81.27%** per-video acc（WAN-1.3B）vs V-JEPA **71.36%** |
| **Seeking Physics** `2603.14294` v3 2026-08-05 | 生成视频 | 前向加噪 | 人工/自动评分 | probe → verifier → 轨迹选择 | 见 §4 |
| **Punzo 等** `2606.09646` v2 2026-09-13 | IntPhys2 + MVP | frozen features | VoE 配对 | linear / MLP / temporal attentive | 扩散弱于 V-JEPA |
| **LikePhys** `2510.11512` v3 2026-03-06 | 自建配对 | — | VoE 配对 | denoising likelihood | PPE |

## 2. D1–D6 × 论文 核对表

图例：✅ 报告了 · ⚠️ 部分/近似 · ❌ 未报告 · ➖ 不适用

| | Invisible Hand | Seeking Physics | Punzo 等 | LikePhys |
|---|---|---|---|---|
| **D1** 随机初始化下界 | ❌ | ❌ | ❌ | ❌ |
| **D2** 属性无关扰动对照 | ❌ | ❌ | ⚠️ | ❌ |
| **D3** 天花板检查 | ❌ | ❌ | ⚠️ | ⚠️ |
| **D4** 噪声地板 | ⚠️ | ❌ | ✅ | ❌ |
| **D5** 跨读出方式一致性 | ❌ | ⚠️ | ✅ | ❌ |
| **D6** 有界量用于饱和区 | ➖ | ➖ | ➖ | ➖ |

### D1 随机初始化下界 —— **四篇全部未报告**

这是本次核对最强的一条。逐篇机械检索 `random(ly) initiali[sz]ed` / `untrained` /
`from scratch` / `random weights`：

| 论文 | `untrained` | `from scratch` | `random` 的实际用法 |
|---|---|---|---|
| Invisible Hand | 0 | 0 | 仅 1 次，指 `fixed random seed of 42` |
| Seeking Physics | 0 | 0 | 8 次，主要是 **`Random Sel.`** 基线 |
| Punzo 等 | 0 | 0 | 16 次，全部是随机**种子**或随机**标签** |
| LikePhys | 0 | 0 | 3 次，指 `50% random-guess threshold` 与随机数种子 |

**必须分清的两处近似物（都不是 D1）**：

1. `2603.14294` 的 **`Random Sel.`** 是「给 verifier 随机**分数**」——
   原文：*"Random Sel. : identical drop schedule with random verifier scores given"*。
   它是**选择流程**的基线，不是**表征**的下界：网络仍是训练好的。
2. `2606.09646` 的 **Random Labels Control**（附录 A.7）是打乱**标签**——
   原文：*"train the temporal attentive probe with randomized labels to determine whether
   high performance can arise without a meaningful relationship between the
   representations and target labels"*。
   这是 **D4 的地板**，回答「probe 会不会凭空拟合」，
   **不回答**「这个表征相对随机权重贡献了多少」。

> **为什么这条重要**：我们在 LikePhys 上测得随机初始化 DiT 的 probe AUC **0.8353**，
> 而训练好的是 0.9686 —— 训练只贡献 **28.5%**。
> 在 IntPhys 2 上是 0.5168 vs 0.6458（贡献 88.5%）。
> **同一个 D1 在两个数据集上给出完全相反的判断**，
> 说明缺了它就无法知道「probe 的成绩有多少来自训练」。

### D2 属性无关扰动对照

- **Punzo ⚠️（最接近但不等价）**：附录 A.6 有两个输入级对照 ——
  *Frame-shuffled*（打乱帧序，"disrupts motion trajectories ... while preserving the set of
  frames and their individual appearance"）与 *Single-frame*（抽一帧重复成整段）。
  这两个测的是「**是否依赖时序**」；D2 要测的是「**能否被物理合法的编辑骗到**」。
  Single-frame 对照最接近（若单帧就够说明靠外观），**但方向仍不同**：
  它移除信息，不是**加入**一个物理合法的扰动。
- **Invisible Hand ❌ / Seeking Physics ❌**：无匹配段落。
- **LikePhys ❌ 且有一处必须点明**：LikePhys **有**颜色变化的素材，
  但原文把它列在 **invalid variants（违反项）** 里，不是对照组：
  > *"In invalid variants, the sphere's **color changes mid-flight**"*（Ball Drop）
  > *"the cloth's **color changes mid-simulation**"*（Cloth Drape）
  > *"fluid **color shifts mid-flow**"*（Faucet Flow）

> #### ⚠️ 这条反过来影响了我们自己的结论，必须记录
>
> 我们在 E1/E3 里把 LikePhys 的 `color_change` 当作**外观对照**，得到 AUC **1.0000**。
> 但原文三处都写明变色发生在 **mid-flight / mid-simulation / mid-flow** ——
> **它含一个时序突变**，所以 probe 可能读的是那个突变而不是颜色本身。
> 这是我们此前没意识到的混淆。
>
> **E7 的自造对照恰好排除了它**：IntPhys 2 上我们施加的是**逐帧完全相同**的色相/饱和度
> 变换（实测帧间差分相关 **0.9989**、V 通道差 **0.000000**），**没有任何时序成分**，
> 而 probe 仍达 AUC **0.9980**（弱档，RGB 变化仅 0.043）。
> **因此「probe 对物理合法的外观编辑比对物理违反更敏感」这一结论在去掉时序混淆后依然成立**，
> 而且 IntPhys 2 的版本是更干净的证据。

### D3 天花板检查

- **Punzo ⚠️**：正文多处讨论模型处于 chance 水平（IntPhys2 本身难），
  属于**下边界**的讨论，不是上边界饱和的系统检查。
- **LikePhys ⚠️**：原文提到 *"only a handful of models significantly achieve PPE lower than
  the 50% random-guess threshold in more than a few scenarios"* ——
  同样是在讨论接近随机，不是饱和剔除。
- **Invisible Hand ❌ / Seeking Physics ❌**：未检索到饱和/天花板的系统检查。

> 我们的 LikePhys 数据里 **12 个场景有 5 个 real−random < 2× 地板**（饱和），
> 若不剔除会让全局均值失真。

### D4 噪声地板

- **Punzo ✅**：三随机种子（42/101/102）+ 报 sd，**且**有 Random Labels Control（A.7）。
  这是四篇里唯一同时给了**种子地板**与**置换地板**的。
- **Invisible Hand ⚠️**：*"Error bars demonstrates standard error of the mean across 5 seeds"*
  —— 有种子地板，但未见标签置换地板。
- **Seeking Physics ❌ / LikePhys ❌**：未检索到。

### D5 跨读出方式一致性

- **Punzo ✅**：linear / MLP / temporal attentive **三种** probe 并报，且比较其差异。
- **Seeking Physics ⚠️**：probe → verifier → 轨迹选择是**一条链**上的不同阶段，
  不是对同一表征的两种独立读出。
- **Invisible Hand ❌**：仅 linear probe。
- **LikePhys ❌**：仅 denoising likelihood。

> 我们的 D5 结果：同一表征上 probe 与 denoising error **在外观对照上结论相反**
> （全 5 个 t 重现），且 E4 进一步给出**各自在测什么**
> （probe 读编辑痕迹、denoising error 56% 方差由复杂度解释）。

### D6 有界量用于饱和区

**四篇均标 ➖ 不适用**：D6 是 2026-09-17 由本项目 S4 的判据设计错误升级而来的通则
（`PREREG.md` §2.1），**晚于全部四篇**。列出它是为了说明该通则的来源，
不构成对这些论文的批评。

## 3. 「若补做缺失的检查，结论最可能如何变化」

> **以下全部是推断，不是结论。** 依据是我们自己在两个数据集上的实测方向，
> 但这些论文的模型、数据与读出都与我们不同，**无法据此断言其结论会改变**。

| 论文 | 缺失项 | 推断（**非结论**） |
|---|---|---|
| **Invisible Hand** | D1、D2 | 其核心主张是「物理信号在 DiT 内部、**不在** VAE latent」。**我们在 IntPhys 2 上的实测支持该主张**（见 §4），因此补 D1 大概率**不会**推翻它；但 D1 会给出「81.27% 里有多少来自训练」的量，这是目前未知的。补 D2 风险更高：若外观对照也达到高分，「linearly decodable 的是物理」这一步会需要限定。 |
| **Seeking Physics** | D1、D2、D4、D5 | 其 verifier 的 AUC 0.638 已接近随机，补 D4（地板）后**可能有部分格子落进地板**。它已有 `Random Sel.` 对照说明「多轨迹采样本身贡献了大部分提升」（原文自陈），方向上与「训练增量有限」一致。 |
| **Punzo 等** | D1 | 已有 D4、D5、部分 D2，是四篇里最稳的。补 D1 会直接回答其核心问题「不同预训练范式编码了什么」—— **没有随机初始化基线时，「扩散弱于 V-JEPA」无法区分「扩散训练没学到」与「两者的架构先验不同」**。 |
| **LikePhys** | D1–D4 | 其 PPE 建立在 denoising likelihood 上，而我们 E4 测得该读出量 **56% 的方差由复杂度解释**、场景内匹配后主效应**符号反转**。**推断**：PPE 的部分排序可能反映 clip 复杂度而非物理。这需要在 LikePhys 自己的设定下重做才能确认。 |

## 4. 我们的数据能直接说明的地方

### 4.1 ✅ Invisible Hand 的「信号不在 VAE latent」—— 在更接近的设定下**成立**

原文（摘要）：
> *"this signal is **absent from the VAE latent input** and emerges inside the denoising
> transformer itself"*

| 设定 | VAE latent probe AUC | DiT probe AUC |
|---|---|---|
| 我们 · **LikePhys** | **0.8200** | 0.9686 |
| 我们 · **IntPhys 2**（本轮新测） | **0.5358** | 0.6458 |

`PREREG.md` §5 原本要求「必须报告该不一致并说明设定不同，不可直接对比」。
**本轮把设定往他们那边挪了一步**（IntPhys 2 是他们所用 IntPhys 的后继、同为 VoE 构造），
结果是：**VAE latent 掉到 0.5358，几乎随机，与他们的主张一致。**

→ **结论更新**：那条不一致是 **LikePhys 特有**，不是普遍性质。
（仍非严格复现：IntPhys 2 ≠ IntPhys，且我们报 AUC 他们报 per-video accuracy。）

### 4.2 这同时解释了 LikePhys 为什么「什么都能测出来」

| 下界 | LikePhys | IntPhys 2 |
|---|---|---|
| raw pixel | 0.7886 | **0.5476** |
| VAE latent | 0.8200 | **0.5358** |
| 随机初始化 DiT | 0.8353 | **0.5168** |
| 训练好的 DiT | 0.9686 | 0.6458 |

LikePhys 的违反是**在合法视频上做编辑**产生的，所以连**原始像素**都能拿到 0.79；
IntPhys 2 的违反是**引擎渲染出的真实事件**，像素与 VAE latent 都拿不到东西。
**这是「基于编辑的 benchmark 上，probe 读的是编辑痕迹」这一论点最直接的证据**，
而且它不是推断，是两个数据集上同一套下界的实测对比。

## 5. 方法学说明

- 检索用**方法词**而非概念词（`RELATED_WORK.md` 硬规则 1 的同一逻辑）：
  例如 D1 查 `random(ly) initiali[sz]ed` / `untrained` / `from scratch` / `random weights`，
  而不是查「lower bound」。
- 每个「❌」都经过**两轮**确认：先用 `e6_evidence.py` 的方法词组，
  再对 `random|untrained|scratch|initial|baseline|chance` 做全文词频与上下文人工过目。
- 版本时间线已核（`fetch_status.json`）：
  Invisible Hand 仅 v1；Seeking Physics v1/v2/v3（撞我们的内容在 v3）；
  Punzo v1 2026-06-08 / **v2 2026-09-13**（本轮读的是 v2）；LikePhys v1/v2/v3。
