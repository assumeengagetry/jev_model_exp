# SCP 备份索引

最新实际落盘复核时间见 `SCP-COMPLETED.json`，主任务元数据交叉检查见 `MAIN-INVENTORY-CROSSCHECK.json`。

- 当前 root：`/home/dev/assumeengage`。
- 累计实际落盘 467 个源文件，约 38.19 MB；433 个稳定文件、34 个非最终快照。这是已存在文件的核验结果，不是本轮新增传输数量。
- 已有 24 个阶段资产在原位置重验，不重复下载。
- **`SCP-COMPLETED.json` / `SCP-COMPLETED.csv`：仅包含本目录内真实存在、大小/hash 核验通过的 467 个小文件。**
- `COPIED-FILES.csv`：含上述小文件及此前已有的 24 个阶段资产，不是新增小文件清单。
- `LEDGER.json`、`LEDGER.csv`：本执行方受限范围内的盘点/排除/传输台账，不是旧机完整盘点；其行数不代表已备份文件数。
- `SNAPSHOTS.csv`：非最终快照。
- `LARGE-ASSET-PATHS.csv`、`KEY-ASSETS.json`：交给 ModelScope 执行方的大资产准确路径。
- `HANDOFF.md`：首轮实际落盘与主任务验收结果、恢复依赖、历史缺口和后续通知边界。
- `MIGRATION-CODE-INDEX.md`：编码/输入格式、加载器、id2label、原 NCCL 手工同步与 CPU-Gloo 路径、prepare 和 eval 索引；新机 NCCL/DDP 是后续迁移目标，未改源训练。
- `checks/PRIORITY-CODE-RECHECK.json`：本轮 201 个源码/配置与源机 hash 一致，新增复制 0。
- `MAIN-INVENTORY-CROSSCHECK.json`：主任务指定资产范围的真实逐文件元数据清单及验收来源；盘点文件不作为上传/落盘证明，历史更正单独保留。
- `VERIFICATION.json`：传输与静态格式核验范围；不表示已运行模型或恢复实验。

| 来源类别 | 安放文件数 | 大小 MB |
|---|---:|---:|
| `AGENTS.md` | 1 | 0.00 |
| `README.md` | 1 | 0.00 |
| `activate.sh` | 1 | 0.00 |
| `cache` | 106 | 0.49 |
| `config` | 72 | 4.10 |
| `datasets` | 41 | 1.41 |
| `logs` | 24 | 0.40 |
| `reports` | 67 | 25.34 |
| `runs` | 25 | 0.04 |
| `scripts` | 36 | 0.24 |
| `src` | 93 | 6.16 |

## 使用边界

`files/` 镜像旧机 root 下的原始相对路径；远端 AGENTS 文档仅作为参考保存在 `reference/REMOTE-AGENTS.md`，不作为当前项目的新执行指令。源码/配置保留原字节与绝对路径，未安装、导入执行或改写。

本备份不包含权重、原始训练数据、大缓存或大预测；小 manifest 与源码齐全，不代表对应大 payload 已上传或可以独立重现。请按 HANDOFF 把共享 token cache、NPY 映射和冻结评测数据交由大资产方补齐。

主任务最新独立核对：两个 CN 私有仓库可写，已有 1372 个稳定大资产、约 21.37 GB；467 个小源文件＋2 个环境观察文件已打成 469 文件/约 16.36 MB 的 support 归档。首轮已知稳定资产可收尾；实验后续结果、历史缺失资产和新机恢复验证不冒充已完成。旧认证/SSH 阻断及早先不可靠的完成量只保留为历史来源，不是当前状态。

后续手动增量入口：`python3 reports/scp-backup-20261031/tools/incremental_scp.py --source <明确的小文件相对路径>`。收到主任务新结果通知后单次触发；入口不监视、不无限等待、不覆盖已存版本，实际复制后更新完成清单与台账。本轮已实际复用入口检查 201 文件，全部跳过重复传输；不另起 72h watcher。
