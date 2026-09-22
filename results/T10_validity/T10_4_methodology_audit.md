# T10.4 — 三篇奠基文献的引用核实

> 执行于 2026-09-19。**只依据原文**（硬规则④）。
> 原文：`results/T10_validity/raw/{1909.03368,2003.12298,2006.00995}.{abs.html,html,pdf,txt}`
> 抓取状态与版本时间线：`methodology_fetch_status.json`（三篇全部拿到 HTML 全文，53–64 K 字符）

## 0. 结论

**谱系主张方向正确，但有三处必须修正**，其中第 3 条涉及对我们自己 D1 的一个实质批评。

| # | 问题 | 严重度 |
|---|---|---|
| 1 | 引文措辞与原文不符（主语错） | 低，但要改 |
| 2 | **观察的原始出处不是 Voita & Titov，是 Zhang & Bowman (2018)** | 中，影响归属 |
| 3 | **他们开出的药方是 MDL，而我们的 D1 恰恰用的是他们批评的 accuracy** | **高** |

---

## 1. 修正一：引文措辞

我们写的（`PROTOCOL_VALIDITY.md` §6、`RELATED_WORK.md` §0.5）：

> 原文：probe accuracy「does not substantially favour pretrained representations over
> randomly initialized ones」

**原文实际是**（摘要，逐字）：

> "Despite widespread adoption of probes, **differences in their accuracy** fail to
> adequately reflect differences in representations. For example, **they do not
> substantially favour** pretrained representations over randomly initialized ones."

差别：主语是 **differences in their accuracy**（accuracy 的**差异**），不是 `probe accuracy`；
动词是 `they do not`，不是 `it does not`。
机械检索 `does not substantially favou?r` → **0 命中**，`substantially favou?r` → 1 命中。

**实质意思没有被曲解**，但既然是直接引语，**必须逐字改对**。

## 2. 修正二：归属

我们写「他们 2020 年就明确指出」。原文在引入这句时**明确标注了出处**：

> "This is clearly seen when using them to compare pretrained representations with
> randomly initialized ones **Zhang and Bowman (2018)**."

而在 §4 开头又一次归属：

> "Probes using these representations show surprisingly strong performance for both token
> **Zhang and Bowman (2018)** and sentence **Wieting and Kiela (2019)** representations."

→ **「随机初始化基线上 probe 依然很强」这个观察的原始出处是 Zhang & Bowman (2018)**，
Voita & Titov 是**复述并量化**它的人。

**修正后的表述**：

> D1 的文献依据：**Zhang & Bowman (2018)** 首先观察到 probe 在随机初始化表征上表现出
> 意外强的性能；**Voita & Titov (EMNLP 2020)** 复述该观察并指出 accuracy 的差异
> 不足以反映表征差异，进而提出以 MDL 码长替代 accuracy。

## 3. ⚠️ 修正三：他们的药方是 MDL，而我们的 D1 用的是 accuracy

**这一条最重要，它是对我们自己协议的一个实质批评。**

Voita & Titov 的论文**不是**一篇「指出 accuracy 有问题」的批评文章，
而是一篇**给出替代方案**的方法论文。原文（§4 开头）：

> "This again confirms that accuracy alone does not reflect what a representation encodes.
> **With MDL probes, we will see that codelength shows large difference between trained and
> randomly initialized representations.**"

实验结论（§4.2 "Trained vs random"）：

> "As expected, codelengths for the randomly initialized model are larger than for the
> trained one... **gain from using context for the randomly initialized model is at least
> twice smaller than for the trained model.**"

**也就是说**：

| | accuracy | MDL 码长 |
|---|---|---|
| 能否区分 trained vs random | **不能**（他们的批评） | **能**（他们的结论） |

**而我们的 D1 是**：`(AUC_real − AUC_random) / (AUC_real − 0.5)` —— **一个 accuracy 系的量**。

→ **按 Voita & Titov 的立场，我们的 D1 用的正是他们判定为不充分的那类统计量。**
他们会说：正确的做法是改用码长。

### 3.1 这对我们意味着什么（三点，都要写进报告）

1. **我们的实测结论仍然成立且与他们一致**：
   我们测到随机初始化 DiT 达到训练模型的 86%（0.8353 / 0.9686），
   这正是他们所说「accuracy 不能实质区分两者」在新模态上的表现。
   **我们观察到的现象，就是他们预言的现象。**

2. **但「D1 有六年前的文献依据」这句要限定**：
   有依据的是「**必须做随机初始化对照**」这件事；
   **没有依据的是「用 AUC 差比值来做」这个具体形式** —— 那恰是他们建议替换掉的。

3. **可行的加强**：把 MDL 码长作为 D1 的第二读出量补测。
   若码长也显示 trained ≈ random，则我们的结论在他们推荐的统计量下同样成立，
   批评失效；若码长显示大差距，则我们的 24.6% 低估了训练的贡献，**结论需要修正**。
   → 已实现为 `scripts/t10_mdl.py`，结果见 §5。

## 4. 另两篇的核实结果

### Hewitt & Liang 2019（D2 同源）—— ✅ 无问题

`selectivity` 全文 65 次、`control task` 50 次，核心概念与我们的用法一致：
构造一个只能被 probe 本身学到的对照任务，要求 probe 具备高目标分 + 低对照分。

**我们的 D2 与之同源无误。** 一处细节值得记：
Voita & Titov 指出 Hewitt & Liang 的调参方式在真实任务上**不总有效**
（"tuning a probe setting by maximizing difference with the synthetic control task
does not help: in the tuned setting, scores for these layers remain very close
(e.g., 97.3 and 97.0)"）——
我们没有做那种调参，因此不受此影响，但引用时不宜把 control tasks 说成已解决一切。

### Elazar et al. 2020（D5 思想来源）—— ⚠️ 一处需注意

核心主张（无法从 probing 结果推出行为结论、应看信息如何被使用）与我们的 D5 方向一致。

**但必须注意**：该文明确把自己的控制与 Hewitt & Liang 的 selectivity **区分开**：

> "Control over Selectivity — **Not to be confused with Hewitt and Liang (2019)**
> Selectivity. Although recommended to use when performing standard probing, we argue it
> does not fit as a control for amnesic probing..."

→ 引用时**不可**把 Elazar 的控制与 Hewitt & Liang 的 selectivity 混为一谈，
它们是针对不同 probing 范式的不同对照。我们的 D2（外观对照）对应前者，
D5（跨读出方式）借的是 Elazar 的**思想**（同一信息在不同用法下结论不同），
**不是**他的具体方法（INLP 消除 + 行为影响）。**这个区别要在 Related Work 里写明。**

## 5. 补测：用 MDL 码长重做 D1

见 `T10_5_mdl.json` 与 `T10_REPORT.md` §5。

## 6. 版本时间线（硬规则：必须核）

| 论文 | 版本 | 本轮读的 |
|---|---|---|
| Hewitt & Liang `1909.03368` | v1 2019-09-08（仅一版） | v1 |
| Voita & Titov `2003.12298` | v1 2020-03-27（仅一版） | v1 |
| Elazar et al. `2006.00995` | v1 2020-06-01 / v2 2020-12-05 / **v3 2021-02-19** | v3 |

## 7. 建议的 Related Work — *Probing Methodology* 一节（草稿）

> **Probing methodology.** 用线性 probe 读取冻结表征已是标准做法，
> 但该领域早已确立了若干必要的对照。
> Hewitt and Liang (2019) 提出 **control tasks**：
> 构造一个只能由 probe 自身拟合的任务，要求 probe 在目标任务与对照任务之间具备
> **selectivity**；我们的外观对照（D2）与之同源。
> Zhang and Bowman (2018) 观察到 probe 在**随机初始化**表征上表现意外地强，
> Voita and Titov (2020) 复述该观察并指出 probe accuracy 的差异不足以反映表征差异，
> 转而提出以 **MDL 码长**作为替代度量；
> 我们的随机初始化下界（D1）沿用前一半（必须设此对照），
> 并按其建议**同时报告码长**（§5）。
> Elazar et al. (2020) 进一步指出无法从 probing 结果直接推出行为结论，
> 主张关注信息**如何被使用**；这是我们跨读出方式一致性（D5）的思想来源，
> 但方法不同——他们用 INLP 消除信息后观察行为变化，
> 我们比较两种**既有**读出方式在同一表征上的结论差异。
> 据我们所知，**视频物理 probing 这条线上尚无工作报告随机初始化下界**
> （E6 核实：四篇论文中 `untrained` / `from scratch` 命中均为 0）。
> **D5 与 D6 为本文新增**，其正当性不诉诸文献，而由 §(validity) 的消融与构造效度实验支撑。
