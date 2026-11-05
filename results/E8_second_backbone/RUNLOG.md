# E8 运行日志（无人值守）— 已完成

最后更新：2026-10-24 01:xx　·　状态：**全部跑完，交付物已校验**

## 最终状态

| 模型 | 家族/加噪 | 参数 | 层数 | token | IntPhys2 | LikePhys |
|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B | Wan/flow | 5.0B | 30 | 13312 | ✅ E7 | ✅ E7 |
| Wan2.1-T2V-1.3B | Wan/flow | 1.3B | 30 | 13312 | ✅ | ✅ |
| CogVideoX-2b | CogVideoX/VP | 2.0B | 30 | 13312 | ✅ | ✅ |
| CogVideoX-5b | CogVideoX/VP | 5.0B | 42 | 13312 | ✅ | ✅ |
| Wan2.1-T2V-14B | Wan/flow | 14.0B | 40 | 13312 | ✅ | ✅ |
| LTX-Video-0.9.5 | LTX/flow | 2.0B | 28 | 1792 | ✅ | ✅ |
| Mochi-1-preview | Mochi/flow | 10.0B | 48 | 9216 | ✅ | ✅ |

## 最终结论

- **LikePhys：7/7 全部 fail**（28.5 / 32.1 / 32.5 / 32.9 / 34.4 / 42.6 / 43.5）
- **IntPhys 2：6/7 pass**（79.4–99.1），LTX 48.5% 判为 **NOT interpretable**
- **D2：12/12 全部 fail**，gap +0.0346 ~ +0.4461，零例外
- 覆盖 4 个模型家族、2 种加噪、1.3B–14B、28–48 层、12/16/48/128 通道、1792–13312 token
- **判定方向一次没有翻转过**

## 校验结果

- `audit`：D1 键齐全、阈值 0.5、判定与比值一致、D2 全覆盖 → **CLEAN**
- `crosscheck`：报告 12 个 model-dataset 数字全部可追溯到 JSON → **PASS**

## 主动记录的问题（共 10 条，详见报告 §12 与 RUNLOG 历史）

其中最严重的一条：**CogVideoX-5b 缺 RoPE**，导致首轮全套 5b 数据作废重跑。
报告 §12 完整记录了现象、根因、影响范围与为什么 T10 门槛没拦住它。

## 层选择敏感性（报告 §15）

LTX 在 L16（48.5%，不可解读）与 L15（60.7%，pass）之间翻转，
原因是对比值在 `real AUC ≈ 0.55` 时分母太小、不稳定。
**未采用更好看的 L15 作为主结果**，维持"不可解读"的诚实分类。

## 文件说明

| 文件 | 内容 |
|---|---|
| `E8_REPORT.md` | 主报告。**§14 是最终矩阵**，§11/§13 为历史版本（已标注取代） |
| `E8_PREREG.md` | 预注册（跑数前写死的时间步、初始化、判据） |
| `E8_lower_bounds.json` | 四组下界（trained / random / VAE latent / raw pixel），六模型 × 两数据集 |
| `E8_verdict.json` | D1/D2 判定，含 `training_share` / `verdict` / 阈值 / 可解读性 |
| `E8_verdict_{intphys2,likephys}.json` | 逐数据集版本（含更多中间层） |
| `E8_cells_{intphys2,likephys}.json` | 逐格原始值（AUC / d′ / 每 seed 值 / 地板） |
| `E8_cells_intphys2_grid.json` | CogVideoX-2b 的 layer × timestep 全网格（6×6） |
| `manifest.json` | 模型清单、几何、sha256、**全部偏离声明** |
| `_intermediates/` | 可重算的 X_*.npy 特征矩阵缓存（不属于交付物） |
