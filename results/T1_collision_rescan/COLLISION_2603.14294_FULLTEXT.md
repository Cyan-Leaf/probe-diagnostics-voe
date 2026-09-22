# 全文逐条对照：arXiv:2603.14294 "Seeking Physics in Diffusion Noise"

执行日期：2026-09-16 · 依据：v3 全文 PDF（15 页，含附录，实抓）
原始文件：`raw/2603.14294v3.pdf` / `raw/2603.14294v3.txt`
项目页：https://thegreatestcj.github.io/Physics-in-Noise/ （未访问；论文未给出 code 链接）
作者：Chujun Tang（HKUST-GZ 实习期间完成 / Brown）、Lei Zhong（Edinburgh）、Fangqiang Ding（HKUST-GZ，通讯）
版本：v1 2026-03-15 → v3 2026-08-05 · cs.CV · 未见会议/期刊标注（体例、`Table S*` 编号与 DPO/VideoREPA 对照看像 AAAI 投稿）

---

## 0. 四个核心问题的直接回答

| 问题 | 答案 | 证据 |
|---|---|---|
| **probe 是否线性？** | **探测分析是线性的**（logistic regression）；**但真正部署的 verifier 不是**——它有 1 层 causal self-attention + 2 层 MLP，0.55M–1.37M 参数 | "fit a **logistic regression** probe for PC, and report mean AUC-ROC under 5-fold CV"（Probing Analysis §Setup）；verifier 结构见 Method §Physics Verifier + Table S1 |
| **有没有 layer × timestep heatmap？** | **有网格，但很粗，且不是 heatmap**：ℓ∈{5,10,15,20,25} × t∈{200,400,600} = **5×3 = 15 格**，只在 **CogVideoX-2B（30 blocks）** 上做，以表格（Table 2）呈现 | Table 2，含 VAE latent 基线列 |
| **有没有量化 probe 精度与生成质量的落差？** | **没有。一处都没有。** probe/verifier 的 AUC 与生成侧 PhyGenBench 分数分别报告，**从未配对比较、从未定义落差量、从未按场景分解** | Table 2/5（AUC）与 Table 3/4（生成分数）之间无任何桥接分析 |
| **guidance 的 λ / timestep 扫描做到什么程度？** | **完全没扫**：`w_phy = 0.5` 单点固定，注入窗口 `t∈[400,600]` 单点固定（就是两个 selection checkpoint 之间），per-frame 梯度范数 clip 1.0，反传只过前 `ℓ_best=10` 个 block | 附录："injected with strength **w_phy = 0.5** inside the window **t∈[400,600]**, i.e. only between the two selection checkpoints" |

---

## 1. 它实际做了什么（精确参数，供复现/对照）

### 1.1 取状态的方式 —— 与我们的「降级路径」相同，不是 inversion

```
video → VAE latents → forward diffusion 加噪至 t∈{200,400,600} → 冻结 DiT 单次 forward
      → 取 block ℓ 后的 hidden states → 丢掉 text token → 视频 token 空间 mean-pool
      → f^(ℓ)_t ∈ R^{F×D}（F=13 latent frames）→ 时间维 flatten → logistic regression
```

**它用的是 forward noising，不是 inversion。** 即 SPEC §3.1 的「*降级路径*」正是这篇的主路径；
Invisible Hand 的 inversion 路径**没有人做过跨方法比较**。

### 1.2 数据 —— 用的是「生成视频 + 人工标注」，不是真实 valid/invalid 配对

- 探测分析：**VideoPhy** ~4,500 条视频，来自 **7 个 T2V 生成器**（LaVie / VideoCrafter2 / ZeroScope / Gen-2 / OpenSora / SVD / Pika），人工标注 physical commonsense (PC) 与 semantic accuracy (SA)，**PC ≥ 3 记为正类**
- 部署 verifier 的训练集（backbone-matched）：CogVideoX-2B **343** 条 / CogVideoX-5B **1,736** 条 / Wan2.1-14B **591** 条，85/15 划分（seed 42）
- **没有用 LikePhys 的 valid/invalid 配对数据**、没有用 IntPhys / IntPhys2 / InfLevel

### 1.3 探测结果 —— 数字比想象的低得多

Table 2（CogVideoX-2B，线性 probe，5-fold CV AUC）：

| timestep | VAE latent | ℓ=5 | **ℓ=10** | ℓ=15 | ℓ=20 | ℓ=25 |
|---|---|---|---|---|---|---|
| t=200 | 0.539 | 0.569 | **0.606** | 0.604 | 0.574 | 0.579 |
| t=400 | 0.534 | 0.545 | 0.610 | 0.608 | **0.624** | 0.594 |
| t=600 | 0.537 | 0.565 | **0.638** | 0.595 | 0.583 | 0.585 |

- 最好一格 **AUC 0.638**（ℓ=10, t=600），比 VAE 基线最多 +0.101
- within-source 线性 probe：0.534（Pika）– 0.712（LaVie），overall **0.652**（Table 1）
- cross-source 迁移几乎不成立：对角均值 0.611 vs 非对角均值 0.538（Fig. 3b）
- 部署 verifier（非线性）held-out AUC：**0.684 / 0.660 / 0.657**（CogX-2B / CogX-5B / Wan2.1）
- **LikePhys 作为 training-free 基线：0.554 / 0.488 / 0.627** —— 在 CogVideoX-5B 上 **低于随机**（Table 5）

### 1.4 生成侧机制

**(A) Progressive Trajectory Selection**（selection，非我们的路线）
N=4 条轨迹并行，在 t∈{600,400} 两个 checkpoint 用 verifier 打分、保留 top ρ=0.5，池子 4→2→1。
省钟表时间 37%（CogVideoX-2B：490s vs Best-of-4 的 778s，A5000）。

**(B) CFG-Style Reward Gradient Guidance**（= SPEC §3.4 的机制）

```
ε̃_t = ε̂_t − w_phy · σ_t · ∇_{z_t} log r_t(z_t)          (Eq. 6)
r_t(z) = f_φ(PoolFeat(h_t(z); ℓ_best)) ∈ [0,1]
```
- 因为 `r = Sigmoid(u)`，`∇log r = (1−r)∇u`，verifier 自信时梯度自动衰减
- `w_phy = 0.5`，窗口 `t∈[400,600]`，per-frame grad-norm clip 1.0，反传只过 blocks 1..10
- 明确论证只在「moderate noise」注入，否则 reward hacking / off-manifold drift

### 1.5 生成侧结果（PhyGenBench 160 prompts，PhyGenEval：S1 VQAScore+CLIP-FlanT5-XXL / S2 GPT-4o 事件排序 / S3 GPT-4o 自然度 0–3）

| backbone | Base | Random Sel. | Best-of-4 | +Selection | +Reward Gradient | DPO | VideoREPA |
|---|---|---|---|---|---|---|---|
| CogVideoX-2B | 0.370 | **0.490** | 0.515 | 0.515 | — | — | — |
| CogVideoX-5B | 0.363 | — | — | 0.365 | **0.496** | 0.475 | 0.492 |
| Wan 2.1-14B | 0.569 | — | — | **0.612** | 0.606 | 0.558 | — |

Physics-IQ（CogVideoX-5B-I2V，42/66 场景）：0.242 → **0.262**，spatial IoU 0.292 → 0.317，MSE 0.0064 → 0.0063。

---

## 2. 它没做的 / 做得薄的（逐条，全部有出处）

| # | 缺口 | 具体情况 | 严重度 |
|---|---|---|---|
| 1 | **没有 null floor / 种子方差** | 全文 0 次 `variance`、0 次 `error bar`、0 次 `p-value`、0 次 `bootstrap`、0 次 multi-seed。baseline 是 **single-seed**（seed 42+i），主表全部是单点数字 | **高** |
| 2 | **没有随机方向 guidance 对照** | 只有 **Random *Selection*** 对照（随机打分代替 verifier 打分），**且只在 CogVideoX-2B 上做**。头条结果 CogVideoX-5B 的 0.363→0.496 **没有任何等强度随机扰动对照** | **高**（正是 SPEC §4.2 对照组 1） |
| 3 | 自己承认「大部分提升来自多轨迹采样本身」 | 原文：Random Selection 0.490 vs Base 0.370 —— "**much of the headline improvement comes from multi-trajectory sampling itself**"。physics 信号只解释 0.490→0.515 这 0.025 | **高** |
| 4 | **没有人评** | 评判全靠 GPT-4o（PhyGenEval S2/S3 + pairwise）。Wan 上 pairwise **157/160 是平局**，作者自己标注 "should not be interpreted as conventional win rates" | **高** |
| 5 | **没有 λ / 注入窗口扫描** | `w_phy=0.5`、`t∈[400,600]` 均为单点，无消融 | 中 |
| 6 | **落差从未被量化** | AUC 与生成分数分别报告，两者之间没有任何配对分析、没有 per-scenario 落差分布 | 中（这是 SPEC §3.3 的位置） |
| 7 | 外观混淆控制较弱 | 只做了两件事：within-source 重训 + 用 **VQAScore 单一代理**做线性残差化。**没有** matched-rendering 的纯外观扰动组，**没有** CALIPER 式的相机/光照重采样 | 中 |
| 8 | probe 与 verifier 的口径不一 | 结论句写 "physical plausibility is **linearly** decodable"，但下游全部用非线性 verifier；线性 probe 的最好成绩只有 AUC 0.638 | 中 |
| 9 | 层扫描很粗 | 30 层里只测 5 层（步长 5），噪声只测 3 档，只测一个 backbone | 中 |
| 10 | guidance 在强 backbone 上无效 | Wan 2.1-14B：0.612→0.606（不升反降），GPT-4o 判平 157/160。作者归因于 verifier 弱（Wan 只有 591 条标注） | 中（对 RQ3 是重要负面证据） |
| 11 | 未做 timestep 800+ | 附录说明不在 t=800 放 checkpoint（因该处不可靠）——**高噪声区是空的** | 低 |
| 12 | Wan **2.1** 不是 2.2 | 且 28GB bf16 需单张 80GB H200；作者提到多卡切分 Wan 会引入 tiling/blurring 伪影 | 低（工程信息，对我们有用） |

作者自陈的 limitation 原文四条：probing signal **moderate**；verifier 需按 backbone 分别训练、不能 plug-and-play；标注数据少、覆盖窄；评测依赖 GPT-4o 继承其噪声与偏差。

---

## 3. 又发现两篇（一篇是 v3 引文里挖出来的，先前扫描未命中）

### 3.1 `arXiv:2606.09646` —— 第三篇 probing 论文

**"How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms"**
（Punzo, Caselli, Pantelidis, Massafra, Lo Sardo, Salehi；v1 **2026-06-08** / v2 **2026-09-13**，cs.CV）

> frozen-feature probing on **IntPhys2 and Minimal Video Pairs (MVP)**, compare **V-JEPA / VideoMAE / LTX-Video** ... **Layerwise analyses** show physics-relevant information is weakest in early layers and **most accessible at intermediate-to-late depth** ... **temporal controls** show that disrupting frame order substantially reduces performance ... accessibility depends on pretraining paradigm, representational depth, and **readout mechanism**（线性 vs 时序 probe）

- 撞的是 E1 的「层扫描 + probe 形式对比 + 时间打乱对照」，数据集是 IntPhys2（我们的第二数据源）
- **它的结论与 Invisible Hand 相反**：V-JEPA 最强，扩散生成器（LTX-Video）"weaker but non-trivial"；而 Invisible Hand 声称 DiT 探针**优于** V-JEPA/VideoMAE

### 3.2 文献内部存在一处未被任何人调和的三方矛盾（事实陈述）

| 工作 | 被探测对象 | 数据 | 取状态方式 | 数字 |
|---|---|---|---|---|
| Invisible Hand `2606.05328` | video DiT | IntPhys + InfLevel（**真实/高保真 VoE**） | **inversion** | **81.27% accuracy**，优于 V-JEPA/VideoMAE |
| Seeking Physics `2603.14294` | CogVideoX-2B DiT | VideoPhy（**生成视频** + 人工 PC 标注） | **forward noising** | 线性 probe **AUC 0.638 / 0.652**；非线性 verifier 0.66–0.68 |
| Punzo et al. `2606.09646` | V-JEPA / VideoMAE / LTX-Video | IntPhys2 + MVP | frozen-feature | 扩散模型**弱于** V-JEPA |

**同一个问题（「视频扩散模型内部是否编码物理」），三篇结论从 81.27% 到接近随机，跨越极大，且没有任何一篇解释这个差异。**
可能的混杂因素（真实 vs 生成分布 / inversion vs forward noising / VoE 配对 vs 人工评分 / accuracy vs AUC / backbone）**全部纠缠在一起，无人拆解**。

---

## 4. 最终对照：SPEC 的哪些格子被占、哪些还空

| SPEC 条目 | 状态 | 说明 |
|---|---|---|
| §3.1 DiT 中间层 probe 判别物理 | **占**（三篇） | 2603.14294 + 2606.05328 + 2606.09646 |
| §3.1 layer × timestep 二维扫描 | **半占** | 2603.14294 有 5×3 粗网格（单 backbone）；Punzo 有 layerwise 无 timestep；**细网格 + 多 backbone + 高噪声区仍空** |
| §3.1 「仅用线性 probe」的方法论洁癖 | **空** | 2603.14294 线性只用于分析、部署用非线性；`2602.12218`(PhyIP) 从另一侧论证了低容量 probe 才准 |
| §3.1 inversion 取状态 | **半空** | 只有 Invisible Hand 用 inversion；**inversion vs forward-noising 的对照没人做** |
| §3.2 生成侧评分 + **人评校准** | **空** | 2603.14294 用 GPT-4o 代替人评并自陈其偏差；`2603.19607` Physion-Eval 提供了现成的人类专家推理标注 |
| §3.3 **ShowGap 配对量化 + per-scenario 分布** | **空** | 无人做 |
| §3.4 内部 probe 梯度注入采样 | **占** | Eq.6 与我们设计的形式同族（CFG 加性注入） |
| §3.4 λ × 注入窗口扫描 | **空** | 单点 0.5 / [400,600] |
| §4.2 对照组 1（随机方向 guidance） | **空** | 只有 random *selection*，且只在 2B |
| §4.2 对照组 2（纯外观扰动，matched rendering） | **半空** | 只有 VQAScore 残差化 + within-source；CALIPER `2609.08250` 表明这远不够 |
| §6.1 **null floor / 种子地板** | **空** | 该子领域仅 `2608.00617` 做过（另一指标族） |
| §6.2 预注册 | **空** | 无人做 |
| §6.3 配对检验 / p 值 / 效应量 | **空** | 2603.14294 全无 |
| RQ2 H2a vs H2b 归因 | **空** | 见 §3.2 的三方矛盾——这正是 H2a/H2b 的实证入口 |
| E7 跨模型 | **占** | CogVideoX-2B/5B + Wan2.1-14B（+ Punzo 的 LTX-Video / V-JEPA / VideoMAE） |
| 主力模型 Wan **2.2** | **空** | 三篇都没用 Wan 2.2；Wan2.2-TI2V-5B 是唯一能在 24GB 上完整跑起来的一档 |
| 认知科学框架（VoE looking vs reaching / A-not-B） | **空**（视频域） | LRM 域已被 `2608.07077` 占 |

---

## 5. 交给负责人的事实清单（本 agent 不做方案选择）

1. **§3.4 作为「Novelty 支柱」已不成立**：机制（内部 verifier 梯度按 CFG 形式注入 latent、只反传前若干 block、不动权重）已被 2603.14294 完整实现并在三个 backbone 上验证。
2. **但该论文的严谨性缺口极大且可点名**：单种子、无方差/CI/p 值、无随机方向对照、无人评、λ 与窗口未扫、自己承认头条提升的大部分来自多轨迹采样。SPEC §6 的护栏正好逐条对上它的缺口。
3. **它自己在 Wan 2.1-14B 上给出了负面结果**（guidance 0.612→0.606，GPT-4o 判平 157/160）。SPEC §2 的「可证伪性声明」（否定结果亦是强结论）因此已有先例数据点。
4. **文献层面出现一处显式矛盾**：同一命题的三篇结果从 81.27% 到接近随机（§3.2 表），无人调和。这是纯事实，不含叙事建议。
5. **T3（null floor）与 T4（probe 扫描）在技术上仍然可执行且不与任何已发表工作重复其数字**——它们复现/加严的对象从「Invisible Hand 一篇」变成「三篇互相矛盾的结果」。是否值得做、以什么叙事做，属于方案层面。
6. 需要人做的事：`research/RELATED_WORK.md` §1 需增列 2603.14294、2606.09646 两篇为「已被占据的位置」，§3 的三条 novelty 声明需重写（第 2 条已失效）。**本 agent 未改动任何一阶段文档。**
7. 建议在和老师谈之前先看一眼项目页 https://thegreatestcj.github.io/Physics-in-Noise/ ，确认是否已放出代码（论文正文未给 code 链接；若有代码，复现成本大幅下降，但撞车程度也更实）。

---

## 附：证据定位（便于人工复核）

| 主张 | 位置 |
|---|---|
| logistic regression 线性 probe | 正文 Probing Analysis → Setup 段 |
| ℓ∈{5,10,15,20,25}, t∈{200,400,600} | 同上 + Table 2 |
| within-source AUC 表 | Table 1 |
| UMAP + cross-source AUC 矩阵 | Figure 3(a)(b) |
| VQAScore 残差化控制 | 正文 Control Analyses 段 |
| verifier 结构（causal attn + MLP，0.55–1.37M） | Method → Physics Verifier + Table S1/S2/S3 |
| Eq. 6 guidance 形式 | Method → CFG-Style Reward Gradient Guidance |
| `w_phy=0.5`、`t∈[400,600]`、clip 1.0 | 附录 implementation 段 |
| 主结果表 | Table 3 |
| Physics-IQ 结果 | Table 4 |
| verifier vs LikePhys AUC | Table 5 |
| VideoREPA 对照 | Table S6 |
| 单种子 / seed 42+i 说明 | 附录 Reproducibility 段 |
| 四条 limitation | Discussion and Conclusion 末段 |
| Punzo et al. 引文 | References，`arXiv:2606.09646` |
