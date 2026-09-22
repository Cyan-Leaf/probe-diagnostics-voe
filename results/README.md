# results/ 索引

《认知推理》大作业 · Disentangling Physics Probes (SPEC v0.2) · 第二阶段执行产出

> **2026-09-17 补注**：方案已升级到 **v0.3「What Do Physics Probes Actually Measure」**，
> 本目录的数字未变、结论未变，但其在方案中的角色变了（成为 v0.3 的两根支柱）。
> **换机器/换会话的接手者请先读 `handoff/MACHINE_HANDOFF.md` 与 `handoff/PITFALLS.md`。**
> 注意 `SPEC.md` §1.0 引用的 `results/T4_probe/` 实际目录名是 **`results/T4_caliper/`**。

| 目录 | 任务 | 状态 | 入口文档 |
|---|---|---|---|
| `T1_collision_rescan/` | T1 重扫撞车（上一轮） | 完成，触发方案重定位 | `T1_REPORT.md`、`COLLISION_2603.14294_FULLTEXT.md` |
| `T2_env/` | T2 环境与最小链路 | 完成 | `T2_REPORT.md` |
| `T3_null_floor/` | T3 / E0 null floor | 完成，**t ≤ 600 通过，t ≥ 800 不通过** | `T3_REPORT.md`、`protocol_as_executed.md` |
| `T4_caliper/` | T4 / E1 CALIPER 可区分性 | 完成，**预注册判据通过但控制实验推翻字面读法 → 已停** | `T4_REPORT.md` |

## 一句话结论

1. **T2**：链路跑通；**物理 GPU 0 有缺陷 SM，会静默产出错数**，已隔离并加了强制自检。
2. **T3**：噪声地板量化到三层；valid vs violation 的效应在 t ≤ 600 是地板的 46–116 倍（p ≤ 1.7e−4），
   **t ≥ 800 落进噪声**。效应方向为负且由 4 个流体/柔体场景主导。
3. **T4**：线性 probe 在 LikePhys 上 AUC 0.974，但**随机初始化网络 0.858、raw pixel 0.789**，
   且**纯外观扰动（color_change）被分到 AUC 1.000，比物理违反还高**
   → 这套场景集上的 probe 数字不能读作「模型编码了物理」。**按硬规则①停在 T4。**

## 复现

```bash
export HF_HOME=$HF_HOME HF_HUB_OFFLINE=1
# T2
CUDA_VISIBLE_DEVICES=1 uv run --no-project python scripts/t2_min_chain.py
# T3/T4（3 卡并行提取 → 分析）
bash scripts/t3t4_pipeline.sh
uv run --no-project python scripts/t4_per_scenario.py --layer 16 --t 600
uv run --no-project python scripts/t4_controls.py --layer 16 --t 600
uv run --no-project python scripts/make_plots.py --which e0 e1
```

**禁止使用物理 GPU 0**（`CUDA_VISIBLE_DEVICES` 只填 1/2/3）；所有入口已内置
`wan_probe_lib.preflight()` 自检，不通过会直接抛异常。

## 缓存（不在 results/ 内，体积大）

| 路径 | 内容 | 大小 |
|---|---|---|
| `cache/feats_real/` | 920 clips × 5 timesteps × 30 层池化状态 + 5 seed 的 denoising error | 15 GB |
| `cache/feats_random/` | 随机初始化 backbone，同协议（t=600，3 seeds） | 6.3 GB |
| `cache/consolidated/` | 上面两者的 `[n_clips, 30, 13, 3072]` fp16 memmap + index | 21 GB |
| `$HF_HOME` | Wan2.2-TI2V-5B (34.2 GB) + LikePhys + IntPhys2 + Physion-Eval 元数据 | 36 GB |

换标签定义重训 probe 是秒级；换场景集需重跑提取（3 卡约 1 小时）。
