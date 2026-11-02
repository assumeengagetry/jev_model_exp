# 备份执行状态

## 当前结论

**首轮现有稳定资产已在 ModelScope 国内站完成备份：1,372 个文件，21,372,577,529 bytes（约 21.37GB）。** 已逐目录对照远端文件树和大小，两个仓库均为私有。完成时间为 2026-10-31 09:56 UTC；10:04 UTC 独立复核回执见 [FINAL-RECEIPT.json](FINAL-RECEIPT.json)。

**这不是“所有未来实验产物均已生成”。** 增量备份进程 PID67837 已进入 watch 模式，最长继续 72 小时，在源机模型／评测完成后补传，不使用 GPU，不停止训练。

初始遇到的两项阻断已处理：

1. 初始凭据实际属于国际站 `.ai`，旧 SDK 送到国内站而被拒绝。用户后来指定有效 CN 凭据，现正式备份只走 `.cn`。
2. SSH 曾短暂被网关关闭，后续连接已恢复，已获取 1,438 个文件、约 21.72GB 的正式资产元数据清单；不含环境下载缓存或凭据。

- 模型仓库：[assumeengage/qwen3-06b-experiment-backup-20261031](https://modelscope.cn/models/assumeengage/qwen3-06b-experiment-backup-20261031)。
- 数据仓库：[assumeengage/jev-distill-data-20261031](https://modelscope.cn/datasets/assumeengage/jev-distill-data-20261031)。
- 端点固定为 `https://www.modelscope.cn`，两者可见性均为 private。
- 第一次 worker 在私有权限检查处因 SDK 字典返回值处理错误提前停止；修正后首轮上传已完整通过，不重跑训练或修改源文件。

## 已定位的首批迁移资产

| 内容 | 源机位置 | 状态 |
|---|---|---|
| Qwen 基座依赖 | `models/Qwen3-0.6B/` | 已看到实际文件 |
| 共同初始化及 tokenizer | `models/Qwen3-0.6B-common-pair-init-r32/` | 已存在；不是训练完成的 Student |
| 正式训练池 | `cache/pair-training/global-pool/` | 就绪记录为 **3,278.70 万 pair、约 65.28 亿有效 tokens** |
| 共享原文与候选 | `cache/pair-training/normalized/typed-full/` | 已看到清单与文件 |
| 两套编码缓存 | `cache/pair-training/tokenized/global-new/`、`typed-full/` | complete 标记已看到 |
| 冻结评测 suite | `datasets/evaluation-only/decision-index-suite-0.3/` | 已看到目录 |
| Knowledge、Tools 最终权重 | 对应 `runs/five-specialists-20261030/.../final/` | 两份完成回执均为 true |
| 其他最终 Student／融合权重 | 需随后续实验完成继续补入 | 目前未确定最终蒸馏 parent，不能用初始化代替 |

本次包含两个已完成专科 final、已训练分类头、共同初始化、Qwen 基座、正式训练池与两套编码缓存、完整来源文件，以及可稳定取得的中间 checkpoint／训练状态。两个已完成权重的 safetensors 头部均已确认包含 `[2,1024]` 的 score 权重；没有加载模型或运行 GPU 测试。

对首轮盘点的 1,438 个文件做交叉核对：云端已有 1,368 个原清单文件且大小一致，另外加入了上传期间新落盘的稳定文件。未进云端的 70 项为 **64 个未完成下载的 `.part` 文件**和 **6 个小元数据文件**；这 6 项已由 SCP 保存，逐项覆盖见 [coverage-exceptions.json](coverage-exceptions.json)。未完成下载片段不会被称为完整数据。

## 小文件分工

已向 `ses_e78d20bb6ffeeIq6l616oJr4f8` 发送明确任务：SCP 源码、配置、环境清单、训练／评测 manifest、映射、报告和结果小文件；该会话自己的 [交接记录](../scp-backup-20261031/HANDOFF.md)持续更新。

已独立核对该会话台账中的 **467 个小文件**均在本地存在，大小均与台账一致，约 38.19MB，覆盖源码、配置、环境相关资料、报告和训练／数据清单。还附有两个环境观察文件。没有把仅盘点但未落盘的条目计入完成数量。

为便于新机只从 ModelScope 拉取，这批小文件和环境资料另打包成 **469 文件、16.36MB 的归档**，已上传数据仓库 `support/scp-small-assets-20261031.tar.gz`。文件大小和远端路径已核对，见 [SMALL-ASSET-UPLOAD-RECEIPT.json](SMALL-ASSET-UPLOAD-RECEIPT.json)。归档已排除凭据，未新增 Git 提交或推送。

A800 执行规划、两个 requirements 文件和 PNG/SVG 卡分配图，也已上传到数据仓库 `support/a800/`。两仓库根目录的 `RESTORE.md` 已更新，回执见 [GUIDE-UPLOAD-RECEIPT.json](GUIDE-UPLOAD-RECEIPT.json)。

大文件 worker 的远端状态位置为 `/home/dev/assumeengage/backup/modelscope-20261031/upload-state.json`；增量记录在同目录 `watch-upload.log`。

已为增量进程配置完成通知，见 [ONGOING-BACKUP.json](ONGOING-BACKUP.json)。源机完成后可继续通知指定 SCP 会话补拉最终小报告和元数据；72 小时内源端没有完成的文件不会虚构为已备份。

## 尚未就绪或未在本台源机定位的资产

- 截至独立复核，Language、Retrieval、Arts、MIXED_ALL 及五个融合模型尚未全部进入最终权重备份；watch 会在源端产生完成回执后补传。
- 最终蒸馏 parent 仍需等上一阶段选定，不以初始化或任意一个专家代替。
- 本台源机所选目录未定位到 KnowLine 31B Teacher 权重和可复用的本轮 Teacher 概率缓存；本次不能声称已包含它们。国内站同名 `PelaAI/KnowLine-31B-Gen1` 公开仓库查询返回 404，Teacher 仍需按 A800 规划单独准备，或后续另做公共权重桥接。

相关文件：[新机器拉取说明](NEW-MACHINE-RESTORE.md)、[大文件回执](FINAL-RECEIPT.json)、[小文件云端归档回执](SMALL-ASSET-UPLOAD-RECEIPT.json)、[完整元数据盘点](remote-inventory.json)、[认证问题记录](AUTH-BLOCKED.md)、[后续待办](PLAN.md)。
