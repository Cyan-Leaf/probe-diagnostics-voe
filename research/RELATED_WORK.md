# Related Work & Novelty Audit

调研日期：2026-09-15，**2026-09-16 经 T1 重扫大幅修订**。
**本文件是 novelty 声明的唯一依据，写 report 的 Related Work 章节直接从这里取。**

> ## 撞车史（四次）
>
> - **第一轮（09-15）**：LikePhys / WMReward / Invisible Hand 三篇，撞掉最初的三个切口。
>   原因：只查了 V-JEPA / VideoPhy / Physics-IQ 三条线就下「真空」结论。
> - **第二轮（09-16，T1 重扫）**：`2603.14294` / `2606.09646` 两篇 P0 硬撞车，
>   撞掉「Show Gap」方案的核心 novelty（内部 probe 自引导）。
>   原因：第一轮的「穷尽检索」仍是**概念词检索**（gap/show/self-guidance/linear probe），
>   而 `2603.14294` 标题叫 *Seeking Physics in Diffusion **Noise***，方法词是
>   `verifier`/`reward-gradient guidance`/`trajectory selection`，**概念词一个都不沾**。
>
> ## 检索硬规则（违反即视为未完成检索）
>
> 1. **必须做「域词 × 方法词」的机械笛卡尔积查询**，不能只做概念词检索。
>    命中 `2603.14294` 靠的是 `abs:"video diffusion" AND (abs:probe OR abs:probing OR abs:"linearly decodable")`。
> 2. **必须对目标窗口做全量分页拉取后人工过题目**，不能只看 top-k 相关度排序。
> 3. **凡确认一篇最接近的工作，必须把它的 Related Work 与参考文献整段读完。**
>    `2606.09646` 连机械组合查询都没打到（它说 "video foundation models" 不说 "video diffusion"，
>    摘要不含 `plausibility`），是从 `2603.14294` 的参考文献里挖出来的。
> 4. **必须核对 arXiv 版本时间线**。`2603.14294` 撞我们的那部分是 **v3（2026-08-05）** 才加入的，
>    v1/v2 摘要里没有。只看最新版会误判撞车范围与对方的成熟度。
> 5. Awesome 清单**不能单独作为防线**：滞后约 3 个月，且漏收 `2606.05328`、`2603.14294`。
> 6. 月度重扫复用 T1 的三个脚本（`scripts/t1_arxiv_*.py`）。

---

## 0.5 Probing 方法论谱系（2026-09-19 新增，**09-19 晚经 T10.4 原文核实后修订**）

**D1–D4 不是我们发明的判据，它们各自对应 probing 方法论中已确立的做法。**
这是协议合理性的**文献侧依据**（实验侧依据见 `PROTOCOL_VALIDITY.md` 与 `results/T10_validity/`）。

> ⚠️ **本节初版有三处错误，已按原文修正。错误内容保留在此，供报告写作时引以为戒。**

| 文献 | 核心贡献 | 对应 |
|---|---|---|
| **Hewitt & Liang**, EMNLP 2019, *Designing and Interpreting Probes with Control Tasks*<br>`10.18653/v1/D19-1275` · arXiv `1909.03368` (v1, 仅一版) · cited 847 | **control tasks**：构造只能由 probe 自身拟合的任务，要求 probe 具备 **selectivity** | **D2 同源** ✅ 核实无误 |
| **Zhang & Bowman**, 2018 | **首先观察到** probe 在随机初始化表征上表现意外地强 | **D1 的原始出处** |
| **Voita & Titov**, EMNLP 2020, *Information-Theoretic Probing with MDL*<br>arXiv `2003.12298` (v1, 仅一版) · cited 335 | 复述上述观察并指出 accuracy 的**差异**不足以反映表征差异，**进而提出以 MDL 码长替代 accuracy** | **D1 的方法论依据（半条，见下）** |
| **Elazar et al.**, TACL 2020, *Amnesic Probing*<br>`10.1162/tacl_a_00359` · arXiv `2006.00995` (v3 2021-02-19) · cited 330 | 无法从 probing 结果推出行为结论；主张关注信息**如何被使用** | **D5 的思想来源（非方法来源）** |

### 修正一：引文必须逐字

❌ 我们初版写的：probe accuracy「does not substantially favour pretrained representations over randomly initialized ones」

✅ **原文逐字**（摘要）：
> "Despite widespread adoption of probes, **differences in their accuracy** fail to adequately
> reflect differences in representations. For example, **they do not substantially favour**
> pretrained representations over randomly initialized ones."

主语是 **differences in their accuracy**（accuracy 的**差异**），不是 `probe accuracy`。
机械检索 `does not substantially favou?r` → **0 命中**。
实质意思未被曲解，但**直接引语必须逐字改对**。

### 修正二：归属

「随机初始化上 probe 依然很强」这一观察的**原始出处是 Zhang & Bowman (2018)**，
Voita & Titov 原文两处明确标注：
> "This is clearly seen when using them to compare pretrained representations with randomly
> initialized ones **Zhang and Bowman (2018)**."

→ **正确表述**：Zhang & Bowman 首先观察，Voita & Titov 复述并量化，并提出 MDL 替代方案。

### ⚠️ 修正三（实质）：他们的药方是 MDL，而我们的 D1 是 accuracy 系的量

Voita & Titov **不是**一篇「指出 accuracy 有问题」的批评文章，而是**给出替代方案**的方法论文：
> "This again confirms that accuracy alone does not reflect what a representation encodes.
> **With MDL probes, we will see that codelength shows large difference between trained and
> randomly initialized representations.**"

| | accuracy | MDL 码长 |
|---|---|---|
| 能否区分 trained vs random | **不能**（他们的批评） | **能**（他们的结论） |

而我们的 D1 = `(AUC_real − AUC_random) / (AUC_real − 0.5)`，**正是 accuracy 系的量**。

**→ 因此「D1 有文献依据」这句必须限定**：
- ✅ 有依据的是「**必须设随机初始化对照**」这件事；
- ❌ **没有**依据的是「用 AUC 差比值来做」这个具体形式——那恰是他们建议替换掉的。

### ✅ 已补测：MDL 码长下 D1 结论存活

按 Voita & Titov §2.2.2 的 online (prequential) 码在我们的缓存上重算
（C 扫描后取 C=1e-3，判据为**使两者中较弱一方最优**，避免比较偏向训练模型）：

| | 压缩率 | 码长 (kbits) |
|---|---|---|
| 训练 DiT | **2.162** | 0.347 ± 0.006 |
| 随机初始化 DiT | **1.579** | 0.475 ± 0.011 |

| 统计量 | 训练贡献份额 |
|---|---|
| AUC（D1 主数值） | **24.6%** |
| **MDL 码长（稳健性检验）** | **31.8%** |

**两个统计量一致：训练只贡献少数。**
→ **我们的结论不是「用错统计量」的产物，它在他们自己推荐的度量下同样成立。**
报告中 D1 主数值仍为 AUC 系，**码长版本并列作为稳健性检验**。

### 另两篇的注意事项

- **Hewitt & Liang** ✅ 无问题（`selectivity` 65 次、`control task` 50 次，用法一致）。
  但不宜把 control tasks 说成已解决一切 —— Voita & Titov 指出其调参方式在真实任务上不总有效。
- **Elazar et al.** ⚠️ 该文明确把自己的控制与 Hewitt & Liang 的 selectivity **区分开**
  （原文 "Control over Selectivity — **Not to be confused with Hewitt and Liang (2019)**"）。
  → 引用时**不可混为一谈**。我们的 D2 对应前者；D5 借的是 Elazar 的**思想**
  （同一信息在不同用法下结论不同），**不是**其方法（INLP 消除 + 行为影响）。

### 我们的位置

**视频物理 probing 这条线上尚无工作报告随机初始化下界**
（E6 核实：四篇论文 `untrained` / `from scratch` 命中均为 0）。
**D5 与 D6 为本项目新增**，其正当性不诉诸文献，而由 T10 的消融与构造效度实验支撑。

---

## 1. 已被占据的位置（不可再做）## 1. 已被占据的位置（不可再做）

### 1.1 LikePhys (ICLR 2026)

- arXiv: [2510.11512](https://arxiv.org/abs/2510.11512) · [code](https://github.com/YuanJianhao508/LikePhys) · [data](https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark)
- 作者: Jianhao Yuan et al. · v1 2025-10-13 / v3 2026-03-06 · 23 pages
- **做了什么**：training-free，用 denoising objective 作为 ELBO-based likelihood surrogate，
  在 curated valid/invalid 配对视频上区分物理合法与非法。指标 **PPE (Plausibility Preference Error)**，
  声称与 human preference 强对齐，优于 SOTA evaluator baseline。
- **覆盖范围**：12 scenarios × 4 physics domains；支持 animatediff / cogvideox / hunyuan_t2v / ltx / mochi。
- **额外结论**：系统性 benchmark 了 model design 与 inference settings 的影响；
  发现**随 model capacity 与 inference scaling，物理理解呈改善趋势**。
- **对我们的影响**：封死「用 likelihood 探针测物理」与「scaling 趋势分析」两个切口。
- **可复用资产**：benchmark 数据集（valid/invalid 配对，高质量）、evaluator 代码、PPE 指标实现。

### 1.2 WMReward — Inference-time Physics Alignment (2026-01)

- arXiv: [2601.10553](https://arxiv.org/abs/2601.10553) · v1 2026-01-15 / v2 2026-02-27 · 22 pages
- **做了什么**：把提升生成物理合理性当作 **inference-time alignment** 问题。
  用 latent world model（**V-JEPA-2**）的物理先验作为 reward，搜索并引导多条候选去噪轨迹，
  实现 test-time compute scaling。
- **战绩**：ICCV 2025 Perception Test **PhysicsIQ Challenge 第一名**，62.64%，超过前 SOTA 7.42%。
- **覆盖范围**：image-conditioned / multiframe-conditioned / text-conditioned 三种生成设定，含 human preference 验证。
- **对我们的影响**：封死「用外部 reward 在推理时改进物理生成」这个切口。
- **关键区分点**：**它用的是外部世界模型（V-JEPA-2）作为 reward。我们用模型自身的内部 probe。**
  这是对偶解法，不是重复。见 §3。

### 1.3 The Invisible Hand of Physics (2026-06)

- arXiv: [2606.05328](https://arxiv.org/abs/2606.05328) · Parsa Esmati et al. · v1 2026-06-03
- **做了什么**：沿真实视频对应的 latent trajectory 探测视频扩散模型。
  通过把确定性采样过程近似求逆（从 clean video latent 反向积分学到的 velocity field 回到噪声），
  拿到模型中间状态与 attention map。
- **核心结论（对我们极其重要）**：
  1. physical plausibility **可从 diffusion transformer 状态线性解码**，
     在 IntPhys 与 InfLevel 上平均准确率 **81.27%**；
  2. **优于** V-JEPA 与 VideoMAE 等专门的表征学习 baseline；
  3. 该信号 **在 VAE latent 中不存在**，是在 denoising transformer 内部涌现的；
  4. 尽管模型没有用自监督预测目标训练。
- **对我们的影响**：**不是坏消息。** 它替我们证明了本提案最难自证的前提——「模型内部确实知道物理」，
  并给出了可引用的具体数字（81.27%）与可复用的 inversion 方法。
- **它没做的**：标题里的 "Know More Than They **Show**"，**Show 那一侧全文未做**。
  纯 probing 论文，在 real video 的 inverted trajectory 上做线性解码，**没有生成任何一帧视频**，
  也**没有量化 probe 精度与生成质量之间的落差**，更没有把 probe 回接到采样过程。

### 1.4 Seeking Physics in Diffusion Noise（P0，2026-09-16 T1 发现）

- arXiv: [2603.14294](https://arxiv.org/abs/2603.14294) · Chujun Tang, Lei Zhong, Fangqiang Ding · 15 pages
- 版本：v1 **2026-03-15** → v3 **2026-08-05**。**撞我们的 guidance 部分是 v3 才加入的。**
- 项目页 https://thegreatestcj.github.io/Physics-in-Noise/ 至今模板占位，**未放出代码**。
- **做了什么**：probe 预训练 DiT 的中间去噪表征，发现物理合理/不合理视频在
  mid-layer feature space 部分可分、即使在高噪声水平下亦然；有 within-source 与
  perceptual-quality 对照；把信号蒸馏成 frozen-feature 上训练的轻量 physics verifier；
  在固定多轨迹预算下用两种推理时机制——**progressive trajectory selection** 与
  **reward-gradient guidance（反传过前几个 DiT block 来 steer 轨迹）**。
  实验在 PhyGenBench + Physics-IQ 上，跨 CogVideoX-2B/5B 与 **Wan 2.1-14B**，全程不微调生成器。
- **撞掉了什么**：旧方案的 §3.1 probe + §3.4 self-guidance + §4.2 外观对照 + E7 跨模型 + E2/E3 生成侧评分，
  以及「不需要外部世界模型」这个与 WMReward 的区分点（它的 verifier 也训在自己的 frozen features 上）。
- **关键数字**：线性 probe 最好 **AUC 0.638**（ℓ=10, t=600），within-source overall **0.652**；
  非线性 verifier held-out **0.657–0.684**；LikePhys 作 baseline 仅 **0.488–0.627**（CogX-5B 上低于随机）。
- **它的缺口（可逐条点名，全部经全文核实）**：
  - 探测分析用线性 logistic regression，但**部署的 verifier 非线性**（causal attn + 2 层 MLP，0.55–1.37M 参数）
  - layer×timestep 只有 **5×3 粗网格**（ℓ∈{5,10,15,20,25} × t∈{200,400,600}，仅 CogVideoX-2B），且**未覆盖高噪声区 t≥800**
  - **完全没有量化落差**：AUC 与生成分数分别报告，从未配对
  - **λ 与注入窗口完全未扫**：`w_phy=0.5` 单点，窗口 `t∈[400,600]` 单点
  - **单种子基线，全文 0 处方差/CI/p 值/bootstrap**
  - **无随机方向 guidance 对照**（只有随机 *selection* 对照，且仅在 2B 上做）
  - 无人评（GPT-4o 判，Wan 上 157/160 平局）
  - 自陈「头条提升的大部分来自多轨迹采样本身」（Random Sel. 0.490 vs Base 0.370, Selection 0.515）
  - **在 Wan 2.1-14B 上 guidance 无效（0.612→0.606），原文未解释** ← 见 §3 RQ3

### 1.5 How Do Video Foundation Models Encode Intuitive Physics?（P0，同上）

- arXiv: [2606.09646](https://arxiv.org/abs/2606.09646) · Punzo 等 · v1 2026-06-08 / v2 2026-09-13
- **做了什么**：在 IntPhys2 + Minimal Video Pairs 上做 frozen-feature probing，
  比较 V-JEPA / VideoMAE / LTX-Video；**layerwise 分析**（早层最弱、中后层最强）；
  **probe 形式对比**（线性 vs 时序）；**时间打乱对照**。
- **撞掉了什么**：旧 E1 的层扫描 + probe 形式对比 + 时序对照。
- **结论与 Invisible Hand 相反**：V-JEPA 最强、扩散生成器较弱。

### 1.6 P1 部分撞车

| 工作 | 撞点 | 仍可区分之处 |
|---|---|---|
| **PhysWeep** [2609.06207](https://arxiv.org/abs/2609.06207)（09-05） | 「没有基准直接测这个落差」这句话被抢占；「两种失效机制择一」的论证结构 | 它测的是 **requested vs realized 物理参数**（黑箱、像素侧），不是表征侧；纯评测，不碰内部状态 |
| **Off-Manifold Refinement** [2608.29904](https://arxiv.org/abs/2608.29904)（BMVC 2026） | 注入形式：中段 ODE 步给 velocity 加梯度 | reward 来自**外部** V-JEPA 2.1 |
| **ReaDiT Guidance** [2609.04649](https://arxiv.org/abs/2609.04649)（09-04） | 「读 DiT 内部特征 → 引导生成」已成通用框架 | 语义是几何/空间控制（depth/pose/edge），非物理 |
| **Self-Refining Video Sampling** [2601.18577](https://arxiv.org/abs/2601.18577) | 「不需要外挂，模型自己就能修物理」的口号位 | 机制是 denoising autoencoder 内环 + 不确定性选区，非 probe 梯度。**此篇就在 Awesome 清单里，第一轮漏读** |
| **Transformers Struggle to Use Their Emergent World Models** [2608.07077](https://arxiv.org/abs/2608.07077) | knowledge-action dissociation 的框架叙事 | LRM/规划域，非视频 |

### 1.7 命名冲突

[2512.05513](https://arxiv.org/abs/2512.05513) 已有名为 **Know-Show** 的 VLM benchmark。
**禁止使用 "Know ... Show" 系列标题。**


| 工作 | 做了什么 | 与我们的关系 |
|---|---|---|
| X-VoE (ICCV 2023) | VoE 范式测直觉物理，含解释模块 | **讲师 Chi Zhang 为合著者**。范式来源、必引。数据 128×128×15帧，仅作低分辨率对照 |
| IntPhys 2 (2025, FAIR) | UE5.4 高真实感 VoE benchmark，4 原则 | 候选主数据集之一；Invisible Hand 也用它 |
| IntPhys (2018) | 原始 VoE benchmark，15k train / 100 帧 | 历史脉络 |
| Physics-IQ (WACV 2026) | 3840×2160 / 30fps 真实录制物理 benchmark | 生成侧评分候选；WMReward 的竞赛场地 |
| VideoPhy / VideoPhy-2 | 文本到视频的物理常识评测 | 生成侧评分参考 |
| V-JEPA intuitive physics (2025) | 预测式表征的 VoE surprise | WMReward 的 reward 来源；表征侧对照 |
| PISA (ICML 2025) | 物理后训练（watching stuff drop） | 对照：他们改权重，我们不改 |
| DiffPhy / Think Before You Diffuse | LLM 引导的物理感知生成 | 对照：他们加外部 LLM，我们不加 |
| CausalVQA (Meta) / HVCR | 视频因果推理 benchmark | **均为判别式 VQA，未触及生成模型**。C′ 方向仍空，列为 future work |
| Self-Guidance (NeurIPS 2023 / 2412.05827) | 用模型自身注意力/表征引导生成 | **方法论上最接近的先例，但全是图像质量/可控性，无物理、无视频** |

---

## 3. 我们的位置（**v0.3 · 2026-09-17，E0/E1 结果回来后**）

> **重要**：v0.2 的定位（因子拆解三条轴、赌分布轴主导）已被自己的实验数据改写。
> 详见 `SPEC.md` §1.0。以下是 v0.3 定位。

### 3.0 我们手上已有的、别人没有的数字

| 发现 | 数字 | 文献中是否有人报过 |
|---|---|---|
| 随机初始化同架构下界 | **AUC 0.8576**，占总信号 **75.4%**；全网格 t=200/400/600 训练增量仅 **20.8%/22.5%/28.5%** | **无人报过**（`2609.04264` 在 JEPA 上有方向一致的独立观察，见 §3.35） |
| 纯外观编辑（物理合法）的可分性 | **5/5 个 timestep 上外观 > 物理**，差值 +0.031 ~ +0.102 | **无人报过** |
| **特异性对照** | **时序**扰动上两读出**都**反应 → 排除「probe 只是更灵敏」 | **无人报过** |
| `shadow_camera` probe 塌到随机水平 | **0.4927**，CI [0.467, 0.520]；同设计下其余 11 场景 **0/11** 次 < 0.60 | **无人报过** |
| **denoising error 主要是复杂度计** | `temporal_grad` 单变量解释配对差 **56%** 方差（r=+0.750）；复杂度匹配后主效应**符号反转**（−0.01137 → +0.00112） | **无人报过** |
| **饱和区读出量陷阱（D6）** | 同一实验内：real AUC 降幅 **0.0000**（5/7 场景）而 d′ **7/7 下降**、保留 0.754 | **无人报过** |
| **四篇核心论文无一报告 D1** | `untrained` / `from scratch` 命中数**全为 0**；两处近似物已辨明不是 D1 | **本项目 E6 首次系统核对** |
| **两数据集下界对照** | raw pixel 0.7886 (LikePhys) vs **0.5476** (IntPhys 2)；VAE latent 0.8200 vs **0.5358** | **无人报过** |
| **协议可区分 benchmark** | 同一 D1：LikePhys 28.5% ❌ / IntPhys 2 **88.5% ✅**；同一 D2：**两边都失败** | **无人报过** |
| 天花板效应 | 12 场景中 5 个 real−random = 0.00 | **无人报过** |
| **读出方式的混淆敏感性相反** | denoising error 对外观免疫（p=0.13）/ probe 完全不免疫（1.000） | **无人报过** |
| **读出方式的有效噪声区间不同** | t=950：denoising error p=0.88 失效 / probe 仍 0.9443 | **无人报过** |
| 高噪声区 null floor | t≥800 效应落入噪声（p=0.127 / 0.883） | `2603.14294` 网格只到 600 |
| 三层噪声地板 | clip 级 / 效应级 / probe 级全部量化 | `2603.14294` **全文 0 处方差/CI/p 值** |
| `vae_latent` AUC 0.8200 | 与 Invisible Hand「信号不在 VAE latent」结论**不一致** | 需在报告中讨论 |

### 3.1 三方矛盾（原始动机）+ 被忽略的第四列

| 工作 | 数据 | 取状态 | 标签 | **读出方式** | 结果 |
|---|---|---|---|---|---|
| Invisible Hand `2606.05328` | 真实 VoE | inversion | VoE 配对 | probe | **81.27% acc** |
| Seeking Physics `2603.14294` | 生成视频 | fwd noising | 人工评分 | probe→verifier | AUC **0.638** |
| Punzo 等 `2606.09646` | IntPhys2+MVP | frozen feat | VoE 配对 | probe | 扩散**弱于** V-JEPA |
| LikePhys `2510.11512` | 自建配对 | — | VoE 配对 | **denoising likelihood** | PPE |

**v0.2 只列了前三列。§3.0 的数据表明第四列（读出方式）可能是主导差异来源** ——
这是 v0.2 的设计缺陷，被自己的实验补上。

### 3.2 尚未被占的位置（v0.3）

1. **诊断协议 D1–D6（minimal necessary，边界已实测）**：把「随机初始化下界 + 属性无关扰动对照 +
   天花板检查 + seed 地板 + **跨读出方式一致性** + **饱和区读出量规则**」
   作为表征级主张的必要条件。
   **D6 尤其反直觉**：`sd = 0.0000` 常被读作"测量极稳定"，
   实际往往意味着该统计量已无分辨率 —— 本项目有同一实验内的直接对照。
   D1–D4 的思想散见于个别工作（CALIPER 有 D1、`2608.00617` 有 D4），
   **但从未被系统化为协议；D5 无人提出。**
2. **读出方式作为独立混淆轴**：无人做过 denoising-error 与 probe 在
   同一模型、同一 clip、同一对照上的并行比较。
3. **相机运动是 probing 有效性的前提**（H4 / E10）：CALIPER 在机器人域提出
   「重采样相机后差异才显现」，**但无人在视频扩散 + 物理 probing 上验证**，
   也无人指出现有 benchmark 的固定机位设计使结论失效。
4. **高噪声区（t≥800）的读出方式分化**：两种读出在此区间分道扬镳，无人报告。

### 3.3 三句话定位

- 现有「视频扩散模型编码物理」的 probing 结论，**在标准场景集上主要由低层编辑痕迹驱动**：
  随机初始化网络占 75.4% 信号，纯外观改变比物理违反更好分。
- **读出方式（denoising error vs linear probe）是被忽略的第四条混淆轴**，
  且我们能说出各自在测什么：**probe 读编辑痕迹、denoising error 读画面复杂度（56% 方差）**。
  **两者都不是物理** —— 这机制性地解释了文献中 81% → 随机的跨度。
- 我们给出**诊断协议 D1–D6**（D5 跨读出一致性、D6 饱和区读出量规则为新提出；**已验证为 minimal necessary，并给出不充分性的反例**），
  并检验现有工作有多少能通过。

### 3.35 独立佐证：`2609.04264`（2026-09-17 重扫发现，**不撞车**）

**Spectral-Target Physical Latent Structuring for JEPA-Style World Models**（v3）
报告 JEPA 式潜空间世界模型存在 **"physical representation laziness"**：
潜状态不塌缩，但**不编码关键物理属性**，导致下游规划失败。

| | `2609.04264` | 本项目 |
|---|---|---|
| 模型类 | JEPA 潜空间世界模型 | 视频扩散（rectified flow） |
| 动作 | **训练期修复**（加 Fourier 辅助头） | **事后审计**（probe 到底测到了什么） |
| 指标 | planning success rate | probe AUC / denoising MSE + D1–D6 |

**不构成 P0**（不同模型类、训练期修复 vs 事后审计）。
但它的「physical representation laziness」与本项目 D1 的
「训练相对随机初始化只贡献 20.8–28.5%」是**同一现象的两种表述** ——
**必须写进 Related Work 作为独立佐证**：不同模型类、不同方法，得到方向一致的观察。
**这对 Soundness 是加分项**（外部独立验证），不是威胁。

### 3.4 抗撞车性质（v0.3 进一步增强）

v0.2 的价值来自「测量做得干净」；**v0.3 的价值来自「数据已经在手」**。

即使他人发表相关工作，§3.0 那八行数字仍是我们自己跑出来的、可复现的、
带三层误差棒的结果。**撞车风险从「方案失效」降级为「需要补引用」。**

### 3.5 认知科学框架：v0.3 不写

v0.2 曾设为条件性写入（条件：分布轴主导）。**该条件随分布轴废弃一并取消。**
v0.3 的论证是纯测量方法论的，与发展心理学无结构对应，**因此不写**。
依据：授课教师 2026-09-16「没有必要为了课程强行凑配」。

## 4. 检索证据索引

| 轮次 | 查询意图 | 结论 |
|---|---|---|
| 第一轮（09-15，5 组概念词查询） | VoE / likelihood / self-guidance / 知行差距 | 命中 LikePhys、WMReward、Invisible Hand；**漏掉 P0** |
| T1（09-16） | 15 条定向 + 4 条窗口全量分页 + 6 条机械组合 | 594 + 723 篇去重，208 篇逐条过题目，36 篇精读 |
| T1 关键命中 | `abs:"video diffusion" AND (abs:probe OR abs:probing OR abs:"linearly decodable")` | 命中 `2603.14294` |
| T1 引文遍历 | 读 `2603.14294` 参考文献 | 挖出 `2606.09646`（关键词检索完全打不到） |

| CP-1 重扫（09-17） | arXiv API **406 不可达**（代理拒绝，逐项排除非限流/非语法/非 UA） | 改走 OpenAlex（72 机械组合 + cursor 分页）+ arXiv RSS（702 条全量）|

原始数据落盘于 `results/T1_collision_rescan/`（09-16）与 `results/T1_rescan_20260917/`（09-17）。
脚本：`t1_arxiv_scan.py`（已加 `--out`/`--window`，**月度重扫必须传新目录，否则覆盖已交付产出**）、
`t1_openalex_scan.py`、`t1_arxiv_rss.py`。

> **⚠️ 两条必须遵守的重扫局限（2026-09-17 实测）**：
> 1. **OpenAlex 的 `publication_date` 是索引日期，不是 arXiv 投稿日** ——
>    窗口内返回的全是 `2404.*`/`2507.*` 这类旧编号，**一条 `2609.*` 都没有**。
>    **OpenAlex 不能替代 arXiv 的 `submittedDate` 窗口。**
> 2. **arXiv RSS 只覆盖最近一个公告周期。**
>    09-17 那次窗口恰为 1 天，吻合；
>    **下次重扫若间隔 > 7 天，不得用 RSS 冒充完整重扫** ——
>    必须先解决 arXiv API 或换出口，否则上报。
>
> **每次重扫第一步：确认 `export.arxiv.org` 是否恢复。**

## 5. 对后续实验有直接用处的工作（T1 §6 发现，非撞车）

| 工作 | 用处 |
|---|---|
| **PhyIP / The Observer Effect in World Models** [2602.12218](https://arxiv.org/abs/2602.12218) | 证明**高容量 probe / 微调式评测会破坏被测表征**，低容量线性 probe 才准确。**「仅用线性 probe」这条方法论洁癖的外部背书** |
| **CALIPER** [2609.08250](https://arxiv.org/abs/2609.08250) | 干净固定机位场景下线性 probe 不可区分（±0.02 R²）。**升级为 E1 必做检验**，也是外观对照组必要性的最强论据 |
| **Physion-Eval** [2603.19607](https://arxiv.org/abs/2603.19607) | **实测 9,569 条记录**（摘要称 10,990，口径不一致，引用以实际文件为准）、22 类失效标注。标的是 **Sora 2 等外部生成器**输出，**无法为 self-gen 侧打标签**（T2 §3 核实）→ v0.3 降级为 Future Work 引用。另注：每条视频**删掉前 5 帧**降版权风险 |
| **CVA / Sampling headroom is not selection gain** [2609.13257](https://arxiv.org/abs/2609.13257) | 192 个 Physics-IQ 场景：候选池 4→16 使 oracle +9.23 IQ，但 Flow/Cycle/VideoReward 都捞不出来，12 个自适应策略无一胜过均匀分配。**对「test-time 改物理」的先验判断有直接约束** |
| **Diagnosing Under-Development of Irreversible Processes** [2608.00617](https://arxiv.org/abs/2608.00617) | 同子领域已采用「指标 null-degenerate 检验（纯噪声上得 0.50）+ 9 人人评」。**说明 E0 有效，但不再是差异化优势** |
| **Principia** [2609.04200](https://arxiv.org/abs/2609.04200) | calibration-independent 的关系式物理一致性分数，规避帧率/尺度/标定歧义。生成侧自动指标候选 |
| GAUGE `2608.05948` / PhyCheck `2608.02150` / VGI-Bench `2608.19583` / PAWBench `2608.27345` | 2026 下半年新增物理保真度基准，指标选择时应对比 |
| Feature Self-Guidance `2606.27371` / Frozen pixel diffusion guides itself `2607.29122` / Steering Video DiT with Massive Activations `2603.17825` / Latent Space Probing `2605.00874` | 「内部特征自引导」「潜空间探针」在 2026 年已是成熟机制族，**Related Work 需整段处理，不能只引 2412.05827** |
| LINA `2512.13290` | Physical Alignment Probe 数据集 + prompt/visual latent 干预 + causality-aware 去噪调度 |


| 轮次 | 查询意图 | 结论 |
|---|---|---|
| Q1 | inference-time physics guidance / best-of-N | 命中 WMReward（2601.10553） |
| s2 | 生成世界模型的因果推理 | CausalVQA / HVCR / CounterScene，均判别式 |
| s3 | ACRE 后续工作 | 无生成模型方向的后继 |
| s4 | 知行差距 / 自我验证 | 全部 LLM 领域（Self-[In]Correct 等） |
| s5 | 发展心理学 knowledge-action dissociation | Baillargeon VoE 范式文献，确认 A-not-B 对应 |
| n1 | 判别+生成双侧配对评测 | 无直接命中 |
| n2 | 模型识别得出但生成不出 | 命中 Invisible Hand（2606.05328） |
| n3 | knowledge-action 迁移到神经网络 | 仅 LLM 的 language-vs-thought（Mahowald TiCS 2024） |
| n4 | 自引导 / 内部表征引导采样 | 仅图像方向，物理+视频为空 |
| n5 | probe 精度与生成质量落差 | 仅 LLM generation-verification gap |

补充资源：[Awesome-Physics-Cognition-based-Video-Generation](https://github.com/minnie-lin/Awesome-Physics-Cognition-based-Video-Generation)
—— 该领域的论文清单，**开工前应再过一遍，作为撞车的最后一道防线。**
