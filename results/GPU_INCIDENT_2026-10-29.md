# ⚠️ GPU 故障记录：2026-10-29 19:16 全部 4×B200 失效

**状态：不可用，需要管理员重启节点。用户态无法恢复。**

---

## 现象

```
$ python3 -c "import torch; print(torch.cuda.is_available())"
CUDA initialization: CUDA unknown error
False

$ python3 -c "import torch; torch.randn(4,4,device='cuda:0')"
RuntimeError: CUDA unknown error
```

`nvidia-smi` **仍然正常**：4 张 B200 全部可见、0% 利用率、0 MiB 显存、ECC 无错、NVLink 8 条链路全通
（53.125 GB/s）。**只有 CUDA runtime 层挂了。**

## 根因（来自内核日志）

```
NVRM: nvCheckOkFailedNoLog: Check failed: GPU lost from the bus
      [NV_ERR_GPU_IS_LOST] (0x0000000F)
NVRM: Xid (PCI:0001:c1:00): 154, GPU recovery action changed from
      0x0 (None) to 0x2 (Node Reboot Required)
NVRM: nvGpuOpsReportFatalError: uvm encountered global fatal error 0x60,
      requiring os reboot to recover.
```

**Xid 154 = GPU lost from the bus**，涉及 **8 个 PCI 地址**（`0000:18/6d/97/c0`、`0001:19/6c/96/c1`），
即本机可见的全部 GPU。恢复动作被内核明确设为 **`0x2 = Node Reboot Required`**。

## 时间

- 首次 Xid 154：内核时间戳 `4938563.88`
- 当前 uptime：`4940643.53` 秒
- → **故障发生在约 2080 秒（34.7 分钟）前**，即 **19:16 前后**
- 节点已连续运行 **57 天 4 小时**

## 已排除

| 检查 | 结果 |
|---|---|
| `nvidia-smi` 可见性 | ✅ 4 张卡全可见 |
| 驱动 / CUDA 版本 | 580.82.07 / CUDA 13.0，未变 |
| `/dev/nvidia*` 设备节点 | ✅ 齐全（nvidia0-3、nvidiactl、nvidia-uvm） |
| 残留 Python 进程占卡 | ✅ 无（`ps aux` 计数 0） |
| ECC 可纠正/不可纠正错误 | ✅ 全 0 |
| Remapped rows | ✅ 0 |
| NVLink 链路 | ✅ 8/8 通，53.125 GB/s |
| 尝试环境变量绕过（multicast 相关 3 种） | ❌ 均无效 |
| `cudaGetDeviceCount` 直接调用 | rc = **999**（`CUDA_ERROR_UNKNOWN`），n=0 |
| 多次重试（间隔 2 秒） | ❌ 全部 False |

## 没有做的事（**故意**）

**没有执行 `nvidia-smi -r`（GPU 复位）。** 理由：这是共享多用户机器，
复位会杀掉其他租户的进程，而且内核已经判定需要 **节点级重启**，单卡复位不解决问题。
这属于管理员操作，不应由我发起。

## 需要谁做什么

**需要管理员重启节点。** 在重启之前：

1. **所有 GPU 任务都无法进行**（E18 刚写好预注册、还没开始提取）
2. **已完成的所有结果不受影响** —— 特征已全部落盘在 `cache/`（883 GB），分析是 CPU 侧的
3. 重启后建议先跑一次健康检查：`scripts/wan_probe_lib.py::preflight()`
   （E8 时代的强制自检，源自旧机器一次静默错算 8070 个数的教训）

## 对项目进度的影响

**零损失。** 因为：

- E12–E17 的**全部提取都已完成**，报告与判定都基于已落盘的特征
- 唯一被打断的是 **E18**（自己噪声区间的两个格），它**还没开始提取**，
  预注册已写好、脚本已改好（`e17_extract.py --t 300/900`），重启后可直接跑
- E9 / E11（纯 CPU）**不受影响**，本轮继续做

---

## 附：为什么 `nvidia-smi` 活着而 CUDA 死了

`nvidia-smi` 走 NVML，只需和内核驱动通信读状态；CUDA runtime 需要 **UVM**
（统一虚拟内存）初始化上下文，而 UVM 已经进入 `global fatal error 0x60` 的终态，
只能靠重启清掉。所以"nvidia-smi 能看到卡"**不代表卡可用** —— 这一点值得写进
`handoff/PITFALLS.md`，避免下次误判为"驱动没问题，是代码问题"。
