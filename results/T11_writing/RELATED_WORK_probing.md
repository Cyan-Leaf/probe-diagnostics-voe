# Related Work — Probing Methodology

> 草稿，2026-09-20。T11.5 产出。
> 引文已按硬规则④逐字核过原文（`results/T10_validity/raw/*.txt`），
> 核对记录与三处修正见 `results/T10_validity/T10_4_methodology_audit.md`。

---

## 2.x Probing methodology

用线性 probe 读取冻结表征已是标准做法，但该领域早已确立了若干必要的对照，
而视频物理这条线上基本没有采用。

**Control tasks.** Hewitt and Liang (2019) 提出 **control tasks**：
构造一个只能由 probe 自身拟合的任务，要求 probe 在目标任务与对照任务之间具备
**selectivity**（高目标任务分 + 低对照任务分）。
我们的属性无关扰动对照（D2）与之同源：
若一个物理合法的外观编辑比目标违反更容易被分辨，
则该 probe 无法区分目标属性与一般编辑痕迹。

> 需注意其局限：Voita and Titov (2020) 指出，
> 按 control task 差值来调 probe 超参在真实任务上**不总有效**——
> 原文举例 *"in the tuned setting, scores for these layers remain very close
> (e.g., 97.3 and 97.0)"*。我们未使用那种调参方式，故不受此影响，
> 但引用 control tasks 时不宜将其视为已解决一切。

**随机初始化基线.** Zhang and Bowman (2018) 首先观察到 probe 在**随机初始化**表征上
表现意外地强。Voita and Titov (2020) 复述该观察并将其一般化，
其摘要原文为：

> "Despite widespread adoption of probes, **differences in their accuracy** fail to
> adequately reflect differences in representations. For example, **they do not
> substantially favour** pretrained representations over randomly initialized ones."

**注意主语是 accuracy 的差异，且该观察的原始出处是 Zhang and Bowman (2018)** ——
Voita and Titov 在引入此句时明确标注了来源
（*"This is clearly seen when using them to compare pretrained representations with
randomly initialized ones Zhang and Bowman (2018)"*）。

我们的随机初始化下界（D1）沿用其前半部分（**必须设此对照**）。
但必须说明：Voita and Titov 的论文并非仅为批评 accuracy，
而是提出**替代度量** —— MDL 码长，并报告

> "With MDL probes, we will see that codelength shows large difference between trained and
> randomly initialized representations."

即**在他们的实验中，accuracy 区分不了而码长能区分**。
我们的 D1 主数值是 AUC 系的量，正属其批评对象，
因此按其建议**同时报告 online prequential 码长**：
训练 DiT 压缩率 2.162、随机初始化 1.579，训练贡献 31.8%，
与 AUC 系的 24.6% **给出同一判断**。
**故 D1 的结论不依赖于统计量的选择。**

**从 probing 到行为.** Elazar et al. (2020) 指出**无法从 probing 结果直接推出行为结论**，
主张关注信息**如何被使用**而非是否被编码，并提出 amnesic probing
（用 INLP 移除信息后观察行为变化）。
这是我们跨读出方式一致性（D5）的**思想来源**，但方法不同：
我们比较两种**既有**读出方式（linear probe 与 denoising error）
在同一表征上的结论差异，不做信息移除。

> 该文明确将自己的控制与 Hewitt and Liang 的 selectivity **区分开**：
> *"Control over Selectivity — **Not to be confused with Hewitt and Liang (2019)**
> Selectivity. Although recommended to use when performing standard probing, we argue it
> does not fit as a control for amnesic probing."*
> 二者针对不同 probing 范式，**引用时不可混为一谈**。

**本文的位置.** 据我们所知，视频物理 probing 这条线上**尚无工作报告随机初始化下界**：
对四篇核心论文做机械检索，`untrained` 与 `from scratch` 的命中数**均为 0**
（§5 的 E6 核对，每个「未报告」经两轮确认）。
两处近似物都不是 D1 ——
`2603.14294` 的 `Random Sel.` 是随机 verifier **分数**（网络仍是训练好的）；
`2606.09646` 的 Random Labels Control 是**噪声地板**（D4），回答的是另一个问题。

因此本文的贡献不在于**提出**这些检查，而在于
**把领域已公认的判据系统地代入这条线，并补上其中缺失的部分**。
**D1–D2 的正当性来自上述文献；D3–D6 为本文新增，其正当性来自 §5.10 的消融与构造效度实验。**

---

## 引用条目

| 文献 | arXiv | 版本（已核） | 对应 |
|---|---|---|---|
| Hewitt & Liang, EMNLP 2019, *Designing and Interpreting Probes with Control Tasks* | `1909.03368` | v1 2019-09-08（仅一版） | D2 同源 |
| Voita & Titov, EMNLP 2020, *Information-Theoretic Probing with MDL* | `2003.12298` | v1 2020-03-27（仅一版） | D1 文献依据 + MDL 补测 |
| Elazar et al., TACL 2020, *Amnesic Probing* | `2006.00995` | v1 2020-06-01 / v2 2020-12-05 / **v3 2021-02-19**（读的是 v3） | D5 思想来源 |
| Zhang & Bowman, 2018 | — | 经 Voita & Titov 转引，**未取到原文** | D1 观察的原始出处 |

> ⚠️ Zhang & Bowman (2018) 本轮**未取到原文**，
> 其内容依据 Voita & Titov (2020) 的转引。
> 按硬规则④，正式投稿前须补核原文，否则应标注为转引。
