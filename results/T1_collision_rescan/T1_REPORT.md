# T1 — 重扫撞车 · 结果报告

执行日期：2026-09-16 · 执行者：第二阶段编码 agent（开发机 4×RTX4090）
对应任务：`handoff/HANDOFF.md` §T1 · 触发规则：`START_HERE.md` 硬规则 ①

---

## 0. 结论（TL;DR）

**撞车。且是硬撞车。按硬规则 ① 停在 T1，不进入 T2。**

| 级别 | 工作 | 撞掉了什么 |
|---|---|---|
| **P0 硬撞车** | **`arXiv:2603.14294` Seeking Physics in Diffusion Noise**（v1 2026-03-15，**v3 2026-08-05 才加入 guidance**） | SPEC §3.1 probe + §3.4 self-guidance + §4.2 外观对照，**三项同时**。**已读全文，逐条对照见 `COLLISION_2603.14294_FULLTEXT.md`** |
| **P0 硬撞车** | **`arXiv:2606.09646` How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms**（v1 2026-06-08 / v2 2026-09-13） | E1 的 layerwise probing + probe 形式对比 + 时间打乱对照，数据集用 IntPhys2。**从 2603.14294 的参考文献里挖出，前述关键词检索未命中** |
| P1 部分撞车 | `arXiv:2609.06207` PhysWeep（2026-09-05） | RQ1「没人直接测这个落差」+ RQ2「两种失效机制」的表述 |
| P1 部分撞车 | `arXiv:2608.29904` Off-Manifold Refinement（BMVC 2026） | §3.4 的注入形式（中段 ODE 步给 velocity 加梯度） |
| P1 部分撞车 | `arXiv:2609.04649` ReaDiT Guidance（2026-09-04） | §3.4 的「读出 DiT 内部特征→引导生成」框架（非物理语义） |
| P2 遗漏项 | `arXiv:2601.18577` Self-Refining Video Sampling（2026-01，一阶段漏掉） | 「不需要外部 verifier 也能改善物理」的口号位 |
| P2 遗漏项 | `arXiv:2608.07077` Transformers Struggle to Use Their Emergent World Models | knowledge-action dissociation 的框架叙事（LRM 域，非视频） |

**尚未被占的位置（事实陈述，不含方案建议）**：见 §5。

**这是方案层面的判断，属于项目负责人。本报告只给证据，不改方向。**

---

## 1. 扫描范围与可复现性

| 通道 | 覆盖 | 结果 |
|---|---|---|
| Awesome-Physics-Cognition-based-Video-Generation | README 全量 152 条论文行 | 最新条目 = 2026-06（`2606.00499`），**该清单滞后约 3 个月，且不含 Invisible Hand `2606.05328`** —— 不能单独作为防线 |
| arXiv API 定向查询 | 15 条 query（`scripts/t1_arxiv_scan.py`） | 594 篇去重 |
| arXiv API 窗口全量 | 4 条宽查询 × 分页（`scripts/t1_arxiv_followup.py`），窗口 2026-06-16 ~ 2026-09-16 | 723 篇去重，域内热点 208 篇逐条过题目 |
| arXiv API 补充定向 | 6 条（linear probe / show gap / self-guidance / probe+video diffusion …） | 命中 P0 撞车 |
| 摘要精读 | 36 篇（`arxiv_candidates.json` / `arxiv_batch2.json` / `arxiv_batch3.json`） | — |

原始数据全部落盘：`results/T1_collision_rescan/`（`raw/*.atom`、`raw/*.html`、`arxiv_*.json`、`arxiv_query_log.json`、`triage.md`）。
网络：开发机**无直连外网**，全部经海外代理 `<http-proxy>`；`export.arxiv.org` 必须走 **https**（http 返回 421）。

---

## 2. P0 硬撞车：Seeking Physics in Diffusion Noise

- arXiv: [2603.14294](https://arxiv.org/abs/2603.14294) · cs.CV (+cs.AI/cs.LG/cs.RO) · 15 pages · 无 code 链接 · 未见期刊/会议标注
- 作者：Chujun Tang, Lei Zhong, Fangqiang Ding
- 版本：v1 **2026-03-15** → v3 **2026-08-05**

### 原文摘要（实抓全文）

> Do video diffusion models encode signals predictive of physical plausibility? We **probe intermediate denoising representations of pretrained Diffusion Transformers (DiTs)** and find that physically plausible and implausible videos are **partially separable in mid-layer feature space, even at high noise levels**. **Within-source and perceptual-quality controls** suggest that this signal is not fully explained by generator identity or generic visual quality. We distill the signal into a **lightweight, backbone-specific physics verifier trained on frozen features** and use it in two complementary inference-time mechanisms under a fixed multi-trajectory budget: **progressive trajectory selection**, which scores trajectories at intermediate checkpoints and prunes weak candidates early, and **reward-gradient guidance, which steers surviving trajectories by backpropagating through only the first few DiT blocks**. Experiments on **PhyGenBench and Physics-IQ** across **CogVideoX-2B/5B and Wan 2.1-14B** show that progressive selection matches verifier-based Best-of-4 on CogVideoX-2B while reducing wall-clock inference time by 37%, whereas reward-gradient guidance substantially improves physical consistency on CogVideoX-5B, **all without fine-tuning the video generator**.

### 与 SPEC 的逐条对照

| SPEC 条目 | 我们计划做的 | 2603.14294 已做的 | 判定 |
|---|---|---|---|
| §3.1 判别侧 Internal Physics Probe | DiT 中间层状态上训 probe 二分类物理合理性 | 完全相同（mid-layer feature space，frozen features） | **撞** |
| §3.1 layer × timestep 扫描 | 二维网格找信号最强位置 | 报告了「mid-layer」「even at high noise levels」，说明扫过层与噪声水平 | **大概率撞**（需读正文确认是否出 heatmap） |
| §3.4 Self-Guidance from Internal Probe（**Novelty 支柱**） | 把 probe 梯度注入去噪，`x_t ← x_t + λ∇ log p_probe` | **reward-gradient guidance**，反传过前几个 DiT block 来 steer 轨迹 | **撞，且是同一机制** |
| §1.3 与 WMReward 的区分（「不需要外部模型」） | 用模型自身内部 probe，无外部 reward model | verifier 就训在**自己的 frozen features** 上，同样不需要外部世界模型 | **撞，区分点失效** |
| §3.4「不改权重，全程 training-free」 | 与 PISA 类后训练区分 | "all without fine-tuning the video generator" | **撞** |
| §4.2 对照组 2（纯外观扰动，"本项目最大的科学风险"） | 控制外观/渲染统计混淆 | within-source + perceptual-quality controls | **撞** |
| E7 跨模型 | ≥2 个模型验证普遍性 | CogVideoX-2B/5B + **Wan 2.1-14B** | **撞**（且覆盖我们的主力模型家族） |
| E2/E3 生成侧评分 | PhysicsIQ 那套指标 | PhyGenBench + **Physics-IQ** | **撞** |

### 它没做的（据摘要，未读正文）

1. **没有量化「知—行落差」本身**：它把 probe 当工具（verifier），不把 probe accuracy 与 generation plausibility 做配对比较、不报 per-scenario 落差分布。SPEC §3.3 的 ShowGap 指标本身仍未被定义/测量。
2. **没有 H2a vs H2b 归因**：没有「probe 精度在生成分布上是否崩塌」这条曲线（SPEC RQ2 / E4）。
3. **没有认知科学框架**：无 A-not-B / looking-vs-reaching / VoE 叙事。
4. **没有 null floor**：摘要未提种子噪声地板；主张是效果导向（-37% wall-clock、"substantially improves"）。
5. **probe 形式可能不是线性**："lightweight, backbone-specific physics verifier" 未说线性；SPEC §3.1「仅用线性 probe」的方法论洁癖它未必遵守（需读正文）。
6. **未用 LikePhys 的 valid/invalid 配对数据**，用的是 PhyGenBench + Physics-IQ。

> ⚠️ 以上 6 条基于**摘要**。全文（15 页，无 code）需要人工读一遍再定性。本 agent 未下载 PDF 正文做进一步核实——是否值得投入取决于负责人对方向的决定。

### 2.1 全文已读（负责人指示后补做）

全文逐条对照报告：**`COLLISION_2603.14294_FULLTEXT.md`**。四个关键问题的答案：

| 问题 | 答案 |
|---|---|
| probe 是否线性 | 探测分析线性（logistic regression），**部署的 verifier 非线性**（causal attn + 2 层 MLP，0.55–1.37M 参数） |
| layer×timestep heatmap | 有 5×3 粗网格（ℓ∈{5,10,15,20,25} × t∈{200,400,600}，仅 CogVideoX-2B），表格非 heatmap |
| 落差量化 | **完全没有**。AUC 与生成分数分别报告，从未配对 |
| λ / timestep 扫描 | **完全没有**。`w_phy=0.5` 单点，窗口 `t∈[400,600]` 单点 |

**关键数字**：线性 probe 最好 **AUC 0.638**（ℓ=10,t=600），within-source overall **0.652**，
非线性 verifier held-out **0.657–0.684**；LikePhys 作 baseline 只有 **0.488–0.627**（CogX-5B 上低于随机）。
→ 与 Invisible Hand 的 **81.27% accuracy** 差距极大，两篇用的数据/取状态方式完全不同（真实 VoE + inversion vs 生成视频 + forward noising），**无人调和**。

**版本时间线（重要）**：v1/v2 摘要**只有 progressive trajectory selection**，
**reward-gradient guidance（撞 §3.4 的那部分）是 v3（2026-08-05）才加入的**，
且项目页 https://thegreatestcj.github.io/Physics-in-Noise/ 至今仍是模板占位状态
（Code 按钮指向 `github.com/YOUR REPO HERE`，PDF 链接是 `<ARXIV PAPER ID>` 占位）——**未放出代码**。

**它的严谨性缺口（可逐条点名）**：单种子基线、全文 0 处方差/CI/p 值/bootstrap、
**无随机方向 guidance 对照**（只有随机 *selection* 对照且仅在 2B 上做）、
无人评（GPT-4o 判，Wan 上 157/160 平局）、λ 与注入窗口未扫、
自陈「头条提升的大部分来自多轨迹采样本身」（Random Sel. 0.490 vs Base 0.370，Selection 0.515）、
在 Wan 2.1-14B 上 guidance **无效**（0.612→0.606）。

---

## 3. P1 部分撞车（三篇，均在本次扫描窗口内）

### 3.1 PhysWeep（`2609.06207`，2026-09-05，Rasul Khanbayov, Hasan Kurban）

> "Image-to-video generators are often credited with absorbing physical dynamics as implicit world models, a claim the community currently checks with plausibility scores... **Plausibility is the wrong test on its own**, because a clip can look natural while encoding the wrong value of the governing physical parameter, and **no existing benchmark measures this gap directly**. PhysWeep closes it with a fixed, label-free audit, treating a frozen generator as a black box, recovering the realized parameter from generated pixels, and reporting how often generation is trackable at all, how far the realized value sits from the requested one, and **which, if either, of the literature's two proposed failure mechanisms the data support**."

撞击点：
- 「**没有基准直接测这个落差**」这句话被它抢先占用，虽然它测的落差是 **requested vs realized 物理参数**（黑箱、像素侧），不是 **probe accuracy vs generation plausibility**（白箱、表征侧）。
- 「**两种失效机制择一**」的论证结构与我们 RQ2 的 H2a/H2b 高度形似（需读正文确认它指的哪两种）。
- 它是纯评测，不做 guidance；不碰内部状态。

### 3.2 Off-Manifold Refinement / OMR（`2608.29904`，**BMVC 2026 已接收**）

> "...an inference-time method that injects world-model feedback **directly into a single sampling trajectory**. During **scheduled middle ODE steps**, we **augment the generator velocity with the gradient of an adapter-space V-JEPA 2.1 su[rrogate]**..."

撞击点：SPEC §3.4 的注入形式（在选定 timestep 区间给 velocity 加梯度）已被实现并发表。
保留点：它的 reward 来自**外部** V-JEPA 2.1（WMReward 家族），我们「内部 probe」的对偶身份在与它对比时仍成立——但在与 §2 的 2603.14294 对比时不成立。
附带信息：其 related work 已把「candidate selection / gradient-based world-model guidance / generator-internal refinement / post-training」列成四条既有路线，说明这个空间在 2026 下半年已被系统性占领。

### 3.3 ReaDiT Guidance（`2609.04649`，v1 2026-09-04 / v2 2026-09-13）

> "DiT Readout (ReaDiT) Guidance, a lightweight framework for **controlling generation with DiT models via their internal feature representations**. ReaDiT Guidance uses **features from a single DiT block** to steer the generative process according to spatial targets - like depth, pose, or edge maps... naturally extends to video generation, enabling camera and motion control."

撞击点：「从 DiT 内部特征读出 → 引导自己的生成」这个**框架**已成型且通用化。我们的差异只剩语义层面（物理合理性 vs 几何/空间控制）。
`RELATED_WORK.md` §3 的第 2 条 novelty 声明（"没有人用模型自身的内部物理 probe 去引导自己的采样"，n4 轮结论「self-guidance 系列全是图像质量/可控性」）**已过期**。

---

## 4. 一阶段漏掉的两篇（不在本次窗口，但确属漏检）

| 工作 | 内容 | 影响 |
|---|---|---|
| `2601.18577` **Self-Refining Video Sampling**（v1 2026-01-26 / v2 2026-05-20） | 把预训练生成器当自己的 refiner，推理时 inner-loop 迭代精修，**无外部 verifier、无额外训练**，motion coherence 与 **physics alignment** 提升，>70% 人评偏好，明确对比了 "guidance-based sampler" | 「不需要外挂，模型自己就能修物理」的口号位已被占。机制不同（denoising autoencoder 内环 + 不确定性选区，非 probe 梯度）。**此篇就在 Awesome 清单里（Jan., 2026 行），一阶段过清单时漏读** |
| `2608.07077` **Transformers Struggle to Use Their Emergent World Models**（2026-08-07，cs.AI） | 小 Transformer 与前沿 LRM 都**线性可解码**出忠实的世界模型（Sierpinski 三角）、且**因果相关**，但仍在任务上失败 | 「可线性解码 ≠ 会用」这个论点已在 LRM/规划域被完整论证。我们的视频扩散实例仍空，但「首次提出这个 dissociation 框架」这一说法不再成立 |

### 4.1 第三、第四篇 probing 论文（关键词检索打不到，靠引文网络挖出）

| 工作 | 内容 | 影响 |
|---|---|---|
| `2606.09646` **How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms**（Punzo 等，v1 2026-06-08 / v2 2026-09-13） | 在 **IntPhys2 + Minimal Video Pairs** 上做 frozen-feature probing，比较 **V-JEPA / VideoMAE / LTX-Video**；**layerwise 分析**（早层最弱、中后层最强）；**probe 形式对比**（线性 vs 时序）；**时间打乱对照** | 撞 E1 的层扫描 + probe 形式 + 时序对照。**且结论与 Invisible Hand 相反**：V-JEPA 最强、扩散生成器较弱 |

**由此形成一处文献内部的三方矛盾（纯事实）**：

| 工作 | 数据 | 取状态 | 数字 |
|---|---|---|---|
| Invisible Hand `2606.05328` | IntPhys + InfLevel（真实 VoE） | **inversion** | **81.27% acc**，优于 V-JEPA/VideoMAE |
| Seeking Physics `2603.14294` | VideoPhy（**生成视频**+人工标注） | **forward noising** | 线性 probe **AUC 0.638/0.652** |
| Punzo 等 `2606.09646` | IntPhys2 + MVP | frozen-feature | 扩散模型**弱于** V-JEPA |

同一命题的三篇结果跨度从 81.27% 到接近随机，**混杂因素（真实 vs 生成分布、inversion vs forward noising、VoE 配对 vs 人工评分、accuracy vs AUC、backbone）全部纠缠，无人拆解**。

---

## 5. 仍然空着的位置（只列事实）

1. **ShowGap 指标本身**：同一模型上把「probe 判别精度」与「生成物理合理性」做**配对、per-scenario** 测量并报告落差分布 —— 未见任何工作做过。PhysWeep 测的是 requested-vs-realized 参数（黑箱），2603.14294 只把 probe 当工具。
2. **H2a vs H2b 归因**：probe 精度在**生成分布**上 vs 真实分布上的对比曲线（SPEC E4）—— 未见。**§4.1 的三方矛盾正是这条轴的实证入口**（81.27% 在真实 VoE、0.65 在生成视频，无人拆解）。
3. **认知科学框架**：VoE looking-time / A-not-B / knowledge-action dissociation 作为诊断视频生成模型的框架 —— 视频域未见（LRM 域见 `2608.07077`）。
4. **严谨性叙事**：null floor 先行 + 预注册 + 人评校准，在这个子领域刚刚开始出现（见 §6 的 `2608.00617`、`2609.13257`），仍有空间，但**已经不是无人区**。2603.14294 在这一项上是全空的（单种子、无方差、无随机方向对照、无人评）。
5. LikePhys 的 valid/invalid 配对数据 + Wan **2.2** 组合尚未被覆盖（2603.14294 用 PhyGenBench/Physics-IQ + Wan **2.1**；Punzo 用 IntPhys2/MVP + LTX-Video；Invisible Hand 用 IntPhys/InfLevel）。
6. **λ × 注入窗口的扫描**、**inversion vs forward-noising 的对照**、**细粒度 layer×timestep 网格（含高噪声区 t≥800）** —— 三者均无人做。

---

## 6. 不撞车但必须引用的新发现（对后续实验有直接用处）

| 工作 | 为什么重要 |
|---|---|
| `2602.12218` The Observer Effect in World Models (PhyIP) | 实验证明**高容量 probe / 微调式评测会破坏被测表征**，低容量线性 probe 才是准确评测。**直接支持 SPEC §3.1「仅用线性 probe」的方法论选择**，是我们这条洁癖的外部背书 |
| `2609.08250` **CALIPER**（2026-09-08，cs.RO） | 「clean、固定机位场景 + linear probe **无法区分**真会推理物理的 encoder 与不会的」：干净场景下所有表征（含随机初始化 ViT、raw pixels）都逼近 oracle 上界（±0.02 R²），重采样相机/光照/外观后差异才显现。**这是对 E1 结论解释力的直接威胁，也是 §4.2 对照组 2 必要性的最强外部论据** |
| `2608.00617` Diagnosing Under-Development of Irreversible Processes | 同一子领域已有工作按「指标 null-degenerate 检验」（纯噪声上得 0.50）+ 9 人人评来立论。**我们 E0 null floor 的做法已被同行采用**——说明这条护栏有效，但也不再是差异化优势 |
| `2609.13257` Sampling headroom is not selection gain (CVA) | 在 192 个 Physics-IQ 场景上：候选池 4→16 使 oracle 质量 +9.23 IQ，但 Flow/Cycle/VideoReward 都无法可靠捞出来；12 个自适应策略无一胜过均匀分配。**对 RQ3「test-time 改物理」的先验判断有直接约束** |
| `2603.19607` Physion-Eval | 5 个模型生成视频上的 **10,990 条专家推理轨迹**、22 类细粒度物理失效、时间定位标注。**可能直接省掉 SPEC §3.2 人评的大部分工作量**（C 角色） |
| `2609.04200` Principia | calibration-independent 的关系式物理一致性分数（8 类现象，真实录制）。生成侧自动指标候选，规避帧率/尺度/标定歧义 |
| `2608.05948` GAUGE / `2608.02150` PhyCheck / `2608.19583` VGI-Bench / `2608.27345` PAWBench | 2026 下半年新增的物理保真度评测基准，E2 指标选择时应对比 |
| `2606.27371` Feature Self-Guidance（flow 模型，多样性） / `2607.29122` Frozen pixel diffusion guides itself / `2603.17825` Steering Video DiT with Massive Activations / `2605.00874` Latent Space Probing（CogVideoX 潜空间探针做内容审核） | 「内部特征自引导」「潜空间探针」在 2026 年已是成熟机制族，Related Work 需要整段处理，不能只引 2412.05827 |
| `2512.13290` LINA | Physical Alignment Probe 数据集 + prompt/visual latent 空间的学习式干预 + causality-aware 去噪调度 |
| `2512.05513` Know-Show（VLM benchmark） | **命名冲突**：标题里已有 "Know-Show" 概念，若沿用 "Know ... Show" 系列标题会撞名 |

---

## 7. 一阶段为什么漏了 P0（复盘，供改进检索规程）

`2603.14294` 的标题与摘要**完全不含**以下词：`gap`、`show`、`knowledge`、`action`、`self-guidance`、`linear probe`、`plausibility preference`。
它叫 **"Seeking Physics in Diffusion **Noise**"**，方法词是 `verifier`、`reward-gradient guidance`、`trajectory selection`。

- `RELATED_WORK.md` §4 的 n2（「模型识别得出但生成不出」）、n4（「自引导/内部表征引导采样」）、n5（「probe 精度与生成质量落差」）三轮都是**概念词检索**，打不到这篇的用词。
- Awesome 清单**没有收录它**（该清单同样漏了 `2606.05328`）。
- 本次能命中，靠的是 `abs:"video diffusion" AND (abs:probe OR abs:probing OR abs:"linearly decodable")` 这条**方法词 + 域词的机械组合**查询（`z5`）。

**规程建议（工程层面，非方案层面）**：novelty 检索必须包含「域词 × 方法词」的机械笛卡尔积查询，不能只做概念词检索；且必须对**目标窗口做全量分页拉取**后人工过题目，不能只看 top-k 相关度排序。本次 T1 的三个脚本可直接复用做月度重扫（`SPEC.md` §7 要求每月一次）。

**第二条规程**：`2606.09646`（第三篇 probing 论文）连本次的机械组合查询都没打到——它的标题用 "video foundation models" 而非 "video diffusion"，摘要不含 `plausibility`。
它是从 P0 撞车论文的**参考文献**里挖出来的。
→ **凡确认了一篇最接近的工作，必须把它的 Related Work 与参考文献整段读完**，引文网络遍历是关键词检索之外不可省的一步。

---

## 8. 待负责人决策的事项（agent 不做选择）

1. ~~`2603.14294` 需要人读全文~~ → **已按指示读完，见 `COLLISION_2603.14294_FULLTEXT.md`**。结论：probe 分析线性但部署 verifier 非线性；layer×timestep 只有 5×3 粗网格；落差**完全未量化**；λ/窗口**完全未扫**。
2. `research/RELATED_WORK.md` §1 需增列 `2603.14294`、`2606.09646` 两篇为「已被占据的位置」，§3 的三条 novelty 声明中第 2 条（内部 probe 自引导）**已失效**，第 1、3 条需重新表述。该文件是「novelty 声明的唯一依据」，改动属于方案层面，**本 agent 未改动任何一阶段文档**。
3. `SPEC.md` §3.4 / E5 / E6 的定位需要重新判断。
4. T2–T4 是否推进（**当前指示：全停**）。三项与 novelty 归属的关系：
   - T2（环境 + 最小链路）：与 novelty 无关，任何重定位下都要用；
   - T3（E0 null floor）：与 novelty 无关，且恰好是 `2603.14294` 的最大缺口；
   - T4（E1 probe 扫描）：从「复现 Invisible Hand 一篇」变成「面对三篇互相矛盾的结果」——是否值得做、以什么叙事做，属于负责人。
5. 是否把 `2609.08250` CALIPER 的「clean scene 不可区分」检验并入 E1 的必做项（这会实质提高 E1 的工作量与说服力）。
6. `2603.14294` **未放出代码**（项目页 Code 按钮至今是 `github.com/YOUR REPO HERE` 占位）→ 若要做对照复现，需要自己实现其 verifier 与 Eq.6 注入。
7. 需要人判断的一个时间线细节：撞 §3.4 的 reward-gradient guidance 是 **v3（2026-08-05）** 才加进去的，v1/v2 只有 trajectory selection。该部分在原文中是最薄的一块（单 λ、单窗口、无随机方向对照、在 Wan 上无效）。

---

## 附：本次扫描产物清单

```
results/T1_collision_rescan/
├── T1_REPORT.md                      ← 本文件
├── COLLISION_2603.14294_FULLTEXT.md  ← P0 撞车论文全文逐条对照
├── arxiv_hits.json           594 篇（15 条定向 query 去重）
├── arxiv_window_broad.json   723 篇（窗口全量分页）
├── arxiv_candidates.json     16 篇精读（第一批）
├── arxiv_batch2.json         14 篇精读（第二批）
├── arxiv_batch3.json         6 篇精读（第三批）
├── arxiv_batch4.json         1 篇精读（Punzo 等 2606.09646）
├── arxiv_query_log.json      每条 query 的原文与命中数
├── triage.md                 自动打分 triage
└── raw/                      *.atom 原始响应、awesome_main.md、abs_*.html、
                              2603.14294v3.pdf/.txt、projpage_physics_in_noise.html
scripts/
├── t1_arxiv_scan.py          定向 query
├── t1_arxiv_followup.py      窗口分页 + 指定 id 批量
├── t1_fetch_abstracts.py     按 id 取摘要
├── t1_triage.py              自动打分
├── t1_parse_abs.py           abs 页元信息（venue/comments/code）
├── pdf_to_text.py            PDF → txt
└── html_to_text.py           HTML → txt + 外链
```

## 附二：环境事实（T2 未开工，但已确认）

| 项 | 状态 |
|---|---|
| GPU | 4 × RTX 4090（24564 MiB each），driver 535.54.03，CUDA 12.2 |
| CPU / 内存 / 磁盘 | 128 核 / 1007 GB / `$WORKDIR` 剩余 41 T |
| 外网 | **无直连**（github/arxiv/huggingface 全部 timeout），必须走 `<http-proxy>`；`export.arxiv.org` 只接受 **https**（http → 421） |
| `huggingface.co` DNS | 解析到 `2a03:2880:f10d:183:face:b00c:0:25de`（Meta 段，非内网），直连仍不通；**经代理返回 200** |
| uv | 未预装，已用 `pip install uv` 装到 `/opt/conda/bin/uv`（0.12.15） |
| venv | `.venv`（`uv venv --system-site-packages`，复用系统 torch 2.5.1+cu124，4 卡可见），运行方式 `uv run --no-project python ...` |
| 已装 | torch 2.5.1+cu124 / torchvision 0.20.1 / transformers 4.43.1 / **diffusers 0.29.2（过旧，跑 Wan 需升级）** / decord / opencv / pypdf |
| 缺 | ffmpeg、scikit-learn、matplotlib、imageio-ffmpeg、datasets、新版 diffusers |
