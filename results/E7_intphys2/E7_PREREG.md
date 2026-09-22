# E7 预注册 — IntPhys 2 上的 D1–D4 迁移性检验

> **写定于 2026-09-18，在任何 IntPhys 2 特征被提取之前。**
> 依据 `handoff/HANDOFF.md` T8.2（「判据方向先写死再跑」）、
> `PREREG.md` §2.1（D6 通则）、§3（D1–D5）、D4（地板选择规则）。
> 本文件冻结后如有偏离，按硬规则④标注 exploratory。

## 1. 数据

| 项 | 值 | 来源 |
|---|---|---|
| 数据集 | IntPhys 2 **Main** split | `facebook/IntPhys2` @ `a077a2f9` |
| clip 数 | **1012** | `Main/metadata.csv` |
| 场景数 | **253**，每场景恰好 4 个 clip | 同上，`SceneIndex` 计数验证 |
| 类型构成 | `1_Possible` / `1_Impossible` / `2_Possible` / `2_Impossible` **各 253** | 同上 |
| 物理条件 | solidity 292 / permanence 240 / immutability 240 / continuity 240 | 同上 |
| **相机** | **Moving 596 / Fixed 416** | 同上 |
| 难度 | Medium 400 / Hard 336 / Unknown 172 / Easy 104 | 同上 |
| 分辨率 | **512 × 512**（与 LikePhys 相同，无需 resize） | decord 实测 |
| 时长 | **636 帧 @ 60 fps ≈ 10.6 s**（抽样 min 635 / max 800） | decord 实测 |

`HeldOut` split（344 clip）**不用**：官方不发布其 metadata。
`Debug` split（60 clip）**不用**：官方定位为「模型标定 / 压缩伪影敏感性」，不是评测集。

## 2. 标签与分组（冻结）

- **标签**：`Impossible → 1`，`Possible → 0`。
  与 LikePhys 的约定一致（那里 `label=1` 是 violation，已在 T7 核对过符号）。
- **分组**：`GroupKFold(5)`，group = **`SceneIndex`**。
  这是官方的配对键，同一场景的 4 个 clip 共享布景与相机轨迹，**必须不跨折**。
- 池化任务 n = 1012（506 正 / 506 负），逐条件 n = 240–292。

## 3. 时序采样（**与 LikePhys 的唯一实质差异，必须声明**）

IntPhys 2 的 clip 是 **636 帧 @ 60 fps**，而 LikePhys 是约 60 帧 @ 30 fps。
Wan TI2V-5B 要求 `num_frames % 4 == 1`，本项目固定用 **49 帧**。

**决定：49 帧在全 clip 上均匀采样**（`stride = n_frames // 49`，约 12–16）。

理由：

1. **违反事件的发生时刻没有标注**（`Main/metadata.csv` 无时间列）。
   取原生步长的前 49 帧只覆盖 clip 的 **7.7%**，极可能整段错过违反事件 ——
   那测出来的低 AUC 会是采样错误而非模型局限。
2. IntPhys 2 的四类违反（permanence / immutability / continuity / solidity）
   都是**持续多帧**的状态变化（物体消失、穿透、突变），
   粗时间采样仍可见，不像瞬时碰撞那样会被跳过。
3. 参照：Punzo 等（`2606.09646`，Table 4）在 IntPhys2 上**全程用 16 帧/clip**，
   同样远低于原生帧率。

**代价（须在报告中声明）**：等效帧率降到约 4–5 fps，
逐帧表观位移比原生大一个量级，可能使输入偏离 Wan 的训练分布。
**这是迁移到不同时长数据集时无法回避的取舍，不是可调参数。**

## 4. 主读出量（**按 D6 在跑数前写死**）

**主读出量 = `d′`（无界）与 `AUC`（有界）并列，两者都是主读出量。**

`PREREG.md` §2.1 D6 规则 1–3 要求：baseline 落在有界统计量边界时，
主读出量须为无界量，**且必须在预注册阶段写死**。

**为什么现在就能判定有此风险**（不是看了结果才说）：
IntPhys 2 官方 README 明写

> "most models performing at chance levels (50%), in stark contrast to human performance"

即**预期 AUC 落在 0.5 附近**，正是 D6 点名的 `AUC ≤ 0.51` 下边界。
S4 那次是上边界（1.0）饱和，这次风险在下边界。
因此**不等结果，直接把 `d′` 与 `AUC` 同列为主读出量**，
避免重复 S4「事后换读出量只能算 exploratory」的教训。

- `d′ = (mean(s|y=1) − mean(s|y=0)) / pooled_sd(s)`，留出折上计算，与 S4 同一实现
- `AUC` 同时报，用于与文献（Punzo 用 VoE accuracy、Invisible Hand 用 accuracy）对齐

## 5. 噪声地板（**按 D4 在跑数前声明**）

**采用定义 A：seed-spread（固定 clip，只换 feature seed）。**

依据 `PREREG.md` D4 的选择规则：
「问『这个测量稳不稳』用 A；问『这个效应能不能推广到别的 clip』用 B」，
且 A 适用于「样本量不是瓶颈时（如 pooled n=750）」。

E7 的问题是**「D1–D4 这套诊断协议在另一个数据集上还成立吗」** ——
问的是测量本身，且 pooled n=1012、逐条件 n≥240，样本量不是瓶颈。**故用 A。**

**附加声明**：若任何分析单元的 n 降到 100 以下（例如逐条件 × 逐相机的交叉格），
该单元**额外**报定义 B 的置换地板，并明确标注是哪一个。
**不得**在看到结果后改换地板定义（硬规则③）。

## 6. 诊断逐条（判据在跑数前写死）

固定 layer 16 / t=600（与 LikePhys 的参考格一致，保证跨数据集可比）；
feature seed 0/1/2。

### D1 随机初始化下界

- 报 `(AUC_real − AUC_random) / (AUC_real − 0.5)`，以及 `d′` 版本
  `(d′_real − d′_random) / d′_real`
- **判定**：该比例 < 50% 时，「表征编码了该属性」**不成立**，只能说「部分成立」
- LikePhys 参照值：t=600 为 **28.5%**，全 t 范围 20.8–49.1%（均 < 50%，不成立）

### D2 属性无关扰动对照（**IntPhys 2 无自带外观对照，自造**）

- IntPhys 2 没有 LikePhys 的 `color_change` 那种物理合法的外观编辑，因此**自造**：
  对 **Possible** clip 施加**逐 clip 恒定的色相/饱和度变换**
  （HSV 空间，色相偏移 +0.35 turn、饱和度 ×1.4，**全帧完全相同**）。
  - **物理完全不变**：所有帧同一变换，不引入任何时序不一致
  - 语义对齐 LikePhys 的 `color_change`（只改颜色、物理合法）
  - 变换参数固定，不随 clip 变化，避免把「参数随机性」混进对照
- 任务：`possible vs colour_jittered_possible`
- **判定**：若 `AUC(possible vs 外观编辑) ≥ AUC(possible vs impossible)`，
  则该 probe **无法区分目标属性与一般编辑痕迹**，主张不成立
- LikePhys 参照值：外观 **1.0000** > 物理 0.9686 → **不成立**（全 5 个 t 均如此）

### D3 天花板检查

- **判定**：`real − random < 2 × 地板` 的单元视为**饱和**，须剔除后重算
- 分析单元：4 个物理条件 × {全体, Fixed, Moving}
- LikePhys 参照值：12 场景中 **5 个**饱和

### D4 噪声地板

- 用 §5 声明的定义 A，报 `auc_seed_spread` 与 `dprime_seed_spread`
- **判定**：效应必须 > **2 ×** 地板才可报

## 7. 预先声明的次要分析：Camera 因子（**不是事后想到的**）

IntPhys 2 **自带 `Camera ∈ {Fixed, Moving}`**（416 / 596）。

这是对 H4（「固定机位设计使 probing 失去意义」）的**天然检验**：

- S4 只能施加**合成**相机运动，且受 rail 2 限制只到真实量级的 **27%**
- IntPhys 2 的 Moving 是**真实、全量级**的相机运动，且与 Fixed 在同一数据集内配平

**预先写死的判据**：

| 结果 | 含义 |
|---|---|
| `Fixed` 上 probe 明显强于 `Moving` | 与 H4 方向一致，且是 S4 拿不到的全量级证据 |
| 两者相当 | H4 的相机运动机制在真实运动下**不成立** |
| `Moving` 反而更强 | 与 H4 相反 |

**注意**：Fixed/Moving 非随机分配（与 game_name、Difficulty 相关），
故此项为**相关性证据，不是因果**，报告须写明。

## 8. 结果方向与含义（HANDOFF 已写死，此处照抄）

| 结果 | 含义 |
|---|---|
| IntPhys 2 上 D1/D2/D3 **同样失败** | **协议可迁移，问题是系统性的** —— 更强的结论 |
| IntPhys 2 上 D1/D2/D3 **通过** | **协议能区分好坏 benchmark** —— 同样是好结果，并给出「什么样的数据集才能用」 |

**两个方向都可发表，不存在「失败」分支。不得因为哪个方向好看而调整判据。**

## 9. 缓存与产出

- 新缓存目录：`cache/feats_{real,random}_intphys2/`、`cache/feats_{real,random}_intphys2_cj/`
  （cj = colour jitter），**不覆盖任何现有缓存**（硬规则②）
- 结果目录：`results/E7_intphys2/`
- 每个结果文件带 `env` + `preflight` + `args` + seed
