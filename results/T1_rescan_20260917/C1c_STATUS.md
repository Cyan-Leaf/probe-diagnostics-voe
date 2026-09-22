# C1c 撞车重扫 — 2026-09-17

> CP-1 的第三项检查（`SPEC.md` §7.0：「是否又撞车 / 重扫无 P0」）。
> 上次完整重扫：2026-09-16，产出在 `results/T1_collision_rescan/`（只读）。
>
> **本文件分两部分**：§1–2 记录 arXiv API 受阻的排查（保留作证），
> §7 起是**换数据源后实际完成的重扫**与结论。

## 0. 结论

**未发现新的 P0 撞车。**
arXiv API 在本机被代理拒绝，改用 **OpenAlex（机械组合 + 全量分页）**
与 **arXiv listing RSS（最新公告周期全量）** 两条独立路径完成检索，
硬规则 1、2 已复刻，规则 3 本轮不适用（没有出现新的「最接近那篇」），
规则 4 部分满足（RSS 带版本号与 replace 标记）。

## 2. 受阻原因：arXiv API 在本机不可达

| 现象 | 证据 |
|---|---|
| 所有新查询 **HTTP 406 Not Acceptable** | 15 条 query + 3 条 known-id 全部 406 |
| **响应体为空**，0.4 s 秒回 | 未到 arXiv 上游，代理侧直接拒 |
| **不是限流** | 间隔 60 s 重试 4 次，仍然 406；间隔无影响 |
| **不是 UA / Accept 头** | 5 种 UA（含浏览器、curl）结果一致 |
| **不是查询语法** | `all:electron` 200、`all:physics` 406；同一语法两种结果 |
| 早期少数 200 是**代理缓存命中** | 重复过的 URL 返回 200，全新 URL 一律 406 |

排查路径（逐项排除，不是猜）：

1. header → 5 种 UA 全部一致，排除
2. `sortBy` 参数 → 去掉后仍 406，排除
3. 字段前缀（`abs:` / `all:` / `ti:` / `au:`）→ 同一前缀时通时不通，排除语法
4. 限流 → 60 s 间隔 ×4 无改善，且 0.4 s 秒回，排除
5. **结论：代理 `<http-proxy>` 对 `export.arxiv.org` 的新请求做了拒绝**

其他出口也不可用：

- `Fetch` 工具 → `403 Unable To Access On The IDC Network Segment`
- `http://export.arxiv.org` → 已知 421（`PITFALLS.md` 记载）

## 3. 实际做到的：google_search 降级检索

用 `google_search` 做了域词 × 方法词的组合检索（4 组关键词 × 3 查询）。
**局限**：搜索引擎按相关度返回，**无法按 `submittedDate` 窗口过滤，也无法分页拉全量**，
因此不满足硬规则 2。

### 3.1 命中并核对的论文

| arXiv id | 标题 | 日期 | 是否已在既有记录 | 撞车判定 |
|---|---|---|---|---|
| `2609.15562` | PIVOT: Physics-Grounded Verification for AI-Generated **Audio-Video Detection** | 2026-09-14 | ✅ 在 `arxiv_hits.json`（q13 命中） | **不撞车** |
| `2609.06207` | Does a Video Generator Realize the Physics You Ask For? | 2026-09-05 | ✅ 已triage，见 `T1_REPORT.md` | 既有结论不变 |
| `2609.08215` | PhysFlow: Physics-Aware Optical Flow for Motion Controllable Video | 2026-09-08 | ✅ 已覆盖 | 不撞车 |
| `2609.00656` | Physically Plausible Video Generation via Visual-Semantic … | 2026-09-01 | ✅ 已覆盖 | 不撞车 |
| `2607.23472` | Visual In-Context Physics Reasoning for Physically Plausible Video | 2026-07 | ✅ 已覆盖 | 不撞车 |
| `2606.18943` | Physics-IQ Verified | 2026-06 | ✅ 已覆盖 | 不撞车 |
| `2606.04811` | （视频扩散内嵌动作生成，跨形态零样本策略迁移） | 2026-06 | ❌ **未出现过** | 见 §3.3 |

### 3.2 `2609.15562` PIVOT 的判定依据

摘要关键句：
> "We therefore explore detecting AIGC by assessing whether the depicted event
> satisfies measurable **physical** constraints…"

它做的是**用物理一致性检测视频是否为 AI 生成**（deepfake / AIGC detection），
**不是探测生成模型内部表征是否编码物理**。问题设定、输入、输出都不同：

| | PIVOT | 本项目 |
|---|---|---|
| 输入 | 待检测的视频（来源未知） | 已知 valid/violation 配对视频 |
| 被测对象 | **视频本身**是否物理自洽 | **模型内部状态**是否可线性解码出物理合法性 |
| 输出 | 真 / AI 生成 | probe AUC、denoising error |

上次重扫抓到它但未升级到 triage，**这个处理是正确的**。

### 3.3 `2606.04811` — 唯一新出现的 id

google 摘要片段：「embeds action generation into the video diffusion process,
achieving zero-shot policy transfer across embodiments」——
属于 **robot policy / embodied control**，与「物理合法性探针」不同赛道。
**初判不撞车**，但**因 API 不可达，未能读到完整摘要与参考文献**，
故标记为 **待核（unverified）**，不作为「已排除」计入。

## 4. 硬规则执行情况（逐条）

| # | 规则 | 本轮 | 说明 |
|---|---|---|---|
| 1 | 必须做域词 × 方法词的**机械笛卡尔积查询** | ❌ | API 不可达；google_search 只能做少量组合，且返回受相关度排序污染（4 组里 2 组返回了完全无关结果） |
| 2 | 必须对目标窗口做**全量分页拉取**后人工过题目 | ❌ | 搜索引擎无 `submittedDate` 窗口过滤，无法分页 |
| 3 | 必须把最接近那篇的 **Related Work 与参考文献整段读完** | ❌ | 无法取全文（Fetch 被 IDC 段限制，API 不可达） |
| 4 | 必须核对 arXiv **版本时间线** | ❌ | 同上 |
| 5 | Awesome 清单不能单独作防线 | — | 本轮未依赖 |
| 6 | 复用 `t1_arxiv_*.py` | ⚠️ | 脚本已跑但全部 406；已为其加 `--out` / `--window` 参数（避免覆盖只读目录），网络恢复后可直接重跑 |

## 5. 本轮对脚本的改动

`scripts/t1_arxiv_scan.py` 新增两个参数，**默认行为不变**：

- `--out DIR`：输出目录。**月度重扫必须传新目录** ——
  原脚本把 `results/T1_collision_rescan/` 写死，直接跑会覆盖已交付的 Preliminary results
  （硬规则②）。
- `--window YYYYMMDD-YYYYMMDD`：替换 `WINDOW` 常量，免去改源码。

本轮的空产出（15 条 query 全 406）存于 `results/T1_rescan_20260917/`，
**保留作为 API 不可达的证据**，其 `arxiv_query_log.json` 里每条都带 `ERROR HTTP 406`。

## 6. 给负责人的建议

C1c 要真正完成，需要下列之一：

1. **换出口**：确认还有哪个代理可达 `export.arxiv.org`（本轮只验证了 squid1 不行）；
2. **换数据源**：Semantic Scholar / OpenAlex API 支持按日期窗口 + 全量分页，
   可复刻硬规则 1、2，且通常不在同一封禁名单上；
3. **人工**：在能上网的机器上跑 `t1_arxiv_scan.py --out <新目录> --window 20260916-20260918`，
   把产出拷回来。

在此之前，**CP-1 不应判为通过** —— C1c 是它的三项之一。

---

# 第二部分：换数据源后实际完成的重扫

> 负责人裁定（2026-09-17）：换数据源（OpenAlex / S2 API）。以下为执行结果。

## 7. 数据源连通性实测

| 源 | 结果 |
|---|---|
| `export.arxiv.org/api/query` | ❌ 406（见 §2） |
| **`api.openalex.org`** | ✅ 200，支持 `from/to_publication_date` + cursor 分页 |
| `api.semanticscholar.org` graph search | ❌ 429 |
| `api.semanticscholar.org` **bulk** | ✅ 200 |
| **`rss.arxiv.org/rss/<cat>`** | ✅ 200 |
| `api.crossref.org` | ❌ 429 |

## 8. 路径 A：OpenAlex 机械组合 + 全量分页

`scripts/t1_openalex_scan.py`，窗口 2026-09-16 → 2026-09-18。

- **硬规则 1**：6 个域词 × 12 个方法词 = **72 条机械组合查询**
  （域词：video diffusion / video generation / world model / video foundation model /
  text-to-video / diffusion transformer；
  方法词：probe / probing / linearly decodable / linear probe / internal representation /
  hidden states / verifier / reward guidance / denoising error / readout /
  plausibility / intuitive physics）
- **硬规则 2**：cursor 全量分页，另加 3 条无方法词的窗口 sweep

结果：72 组合中仅 **2 组**有命中，且都指向同一篇
`2608.20983`（灾害谣言严重度评估）—— OpenAlex 的 `search` 是模糊匹配，**属误命中**。
窗口内共 101 条唯一作品，标题同时含 `video` 与
`diffus|physic|probe|world model|plausib|generat` 的：**0 条**。

### 8.1 ⚠️ OpenAlex 的关键局限（必须记录）

**OpenAlex 的 `publication_date` 是索引/版本日期，不是 arXiv 投稿日。**
窗口内返回的 arXiv id 是 `2404.16944` / `2507.04157` / `2412.03259` 这类旧编号，
**一条 `2609.*` 都没有**。
因此 OpenAlex **不能替代 arXiv 的 `submittedDate` 窗口**，必须配路径 B。

## 9. 路径 B：arXiv listing RSS（这才是真正覆盖新窗口的那条）

`scripts/t1_arxiv_rss.py`，抓 `cs.CV` / `cs.LG` / `cs.AI` 的最新公告。

| 类目 | 条目 |
|---|---|
| cs.CV | 177 |
| cs.LG | 325 |
| cs.AI | 304 |
| **去重后** | **702** |

对全部 702 条做**域词 × 方法词的机械正则筛**（规则 1 的等价实现），
命中 `domain AND (method OR physics)` 的 **5 条**，逐条判定：

| arXiv id | 标题 | 既有记录 | 判定 |
|---|---|---|---|
| `2609.17521` | PhysStream: Streaming Physics-Grounded Video Generation… | 已知 | **不撞车**：可控视频**生成**（结构化场景记忆 + 速度增量控制），不探测内部表征 |
| `2609.16074` | World-Action Models for Robot Learning and Control: A Survey | 已知 | 不撞车：机器人综述 |
| `2609.16864` | TEMPO: Learning Temporal Context for Dynamic Robot Manipulation | 新 | 不撞车：VLA 操作策略 |
| `2607.19190` v4 | Agentic Real2Sim: Physics-based World Modeling with VLM | 新 | 不撞车：real-to-sim 重建 |
| **`2609.04264` v3** | **Spectral-Target Physical Latent Structuring for JEPA-Style World Models** | 新 | **不撞车，但概念相邻** — 见 §9.1 |

### 9.1 `2609.04264` 的判定依据（本轮唯一需要细读的）

摘要要点：JEPA 式**潜空间世界模型**中发现新的失效模式
「physical representation laziness」——潜状态不塌缩但**不编码关键物理属性**，
导致下游规划失败；解法是训练期加一个轻量 Fourier 辅助头做物理结构化监督。

| | `2609.04264` | 本项目 |
|---|---|---|
| 模型类 | JEPA 潜空间世界模型 | 视频扩散（rectified flow）|
| 动作 | **训练期修复**（加辅助损失） | **事后审计**（probe 到底测到了什么）|
| 指标 | planning success rate | probe AUC / denoising MSE + D1–D5 诊断 |
| 核心主张 | 加了辅助头能改善规划 | 随机初始化已达 real 的大部分，且 probe 读的是编辑痕迹 |

**不构成 P0。** 但它的「physical representation laziness」与本项目 D1 的
「训练相对随机初始化只贡献 20.8–28.5%」是**同一现象的两种表述**，
**应写进 Related Work 作为独立佐证**（不同模型类、不同方法，得到方向一致的观察）。

> 规则 4（版本时间线）：RSS 给出它是 **v3 / replace**，`2607.19190` 是 **v4 / replace-cross**。
> 完整版本历史仍需 abs 页（API 不可达），故规则 4 标为**部分满足**。

## 10. 硬规则执行情况（最终）

| # | 规则 | 状态 | 说明 |
|---|---|---|---|
| 1 | 域词 × 方法词机械笛卡尔积 | ✅ | OpenAlex 72 组合 + 对 702 条 RSS 做同构正则筛 |
| 2 | 窗口全量分页后人工过题目 | ✅ | OpenAlex cursor 分页；RSS 是公告周期全量，702 条标题全过 |
| 3 | 读最接近那篇的 Related Work 与参考文献 | ➖ | **本轮不适用**：没有出现新的「最接近」论文，最接近仍是 `2603.14294`，其引文上轮已遍历 |
| 4 | 核对 arXiv 版本时间线 | ⚠️ | RSS 提供版本号与 replace 标记；完整历史需 abs 页，不可达 |
| 5 | Awesome 清单不单独作防线 | ✅ | 未依赖 |
| 6 | 复用 `t1_arxiv_*.py` | ✅ | 已跑（全 406，留证），并新增两个替代脚本 |

## 11. 本轮新增/改动的脚本

| 脚本 | 用途 |
|---|---|
| `t1_arxiv_scan.py` | **新增** `--out` / `--window`，默认行为不变。原脚本把输出目录写死在只读的 `T1_collision_rescan/`，直接跑会覆盖已交付产出 |
| `t1_openalex_scan.py` | **新建**：OpenAlex 机械组合 + cursor 全量分页，自动与既有 T1 记录比对 |
| `t1_arxiv_rss.py` | **新建**：arXiv listing RSS 全量 + 域词×方法词正则筛 |

## 12. C1c 最终判定

**通过（无新 P0），但附两条局限**：

1. arXiv API 不可达，规则 3/4 未能完整执行（本轮无新「最接近」论文，影响可控）；
2. RSS 只覆盖最近一个公告周期。本轮窗口本就只有 1 天（上次重扫 09-16 → 今天 09-17），
   两者吻合；**但下次重扫若间隔超过一周，RSS 不够，必须先解决 arXiv API 或代理问题。**

> **给下一轮的提醒**：把「确认 `export.arxiv.org` 是否恢复」放在重扫的第一步。
> 若仍不可达且间隔 > 7 天，**不要用 RSS 冒充完整重扫**，应上报。
