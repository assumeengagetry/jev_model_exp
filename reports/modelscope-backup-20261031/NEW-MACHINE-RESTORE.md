# 新机器从 ModelScope 国内站恢复

## 仓库

- 模型与训练状态：`assumeengage/qwen3-06b-experiment-backup-20261031`
- 训练池、编码缓存和来源数据：`assumeengage/jev-distill-data-20261031`
- 端点：**`https://www.modelscope.cn`**。
- 两个仓库均为私有，新机器需要这个账号或具有相应访问权限的 **CN 令牌**。本文不包含令牌。

大文件上传进行期间，仓库里可能只有部分文件。以 `BACKUP_STATUS.json` 和本任务上传回执为准，不把仓库创建成功当成全部资产已备份。

## 1. 安装轻量下载工具并登录

优先恢复到源机相同根目录，避免旧 manifest 中的绝对路径变化：

```bash
export ROOT=/home/dev/assumeengage
export MODELSCOPE_ENDPOINT=https://www.modelscope.cn
export MODELSCOPE_HOME="$ROOT/.secrets/modelscope-cn"
mkdir -p "$ROOT/envs" "$ROOT/_restore"

python3.11 -m venv "$ROOT/envs/backup-hub"
source "$ROOT/envs/backup-hub/bin/activate"
python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple \
  'modelscope-hub==0.6.1'

python -c 'import getpass; from modelscope_hub.api import HubApi; HubApi(endpoint="https://www.modelscope.cn").login(getpass.getpass("ModelScope CN SDK令牌: "))'
```

该环境不安装 PyTorch，不改 Student 或 Teacher 的运行环境。输入通过终端隐藏读取，不将令牌写进命令或本文件。

## 2. 先拉蒸馏迁移必需部分

上传保留源机相对目录，统一放在仓库的 `workspace/` 下。

```bash
modelscope-hub download \
  assumeengage/qwen3-06b-experiment-backup-20261031 \
  --repo-type model \
  --local-dir "$ROOT/_restore/models-repo" --max-workers 4 \
  --include 'workspace/models/**' 'workspace/runs/**' 'BACKUP_STATUS.json'

modelscope-hub download \
  assumeengage/jev-distill-data-20261031 \
  --repo-type dataset \
  --local-dir "$ROOT/_restore/data-repo" --max-workers 4 \
  --include 'workspace/cache/pair-training/**' \
            'workspace/datasets/evaluation-only/**' 'BACKUP_STATUS.json'
```

把两个下载目录的 `workspace/` 内容放回根目录。新机器的恢复目录应是独立的空目录，不要覆盖另一场实验已有的同名文件：

```bash
cp -a "$ROOT/_restore/models-repo/workspace/." "$ROOT/"
cp -a "$ROOT/_restore/data-repo/workspace/." "$ROOT/"
```

如果不能使用相同根目录，可保留自己的 `ROOT`，但需在副本中调整加载器的 base 路径及训练 manifest 的 `typed_normalized`、`typed_tokens`、`new_tokens` 等路径；编码二进制本体不因目录变化重新生成。

## 3. 需要全量来源备份时再拉原始数据

```bash
modelscope-hub download \
  assumeengage/jev-distill-data-20261031 \
  --repo-type dataset \
  --local-dir "$ROOT/_restore/data-repo" --max-workers 4
```

这部分用于保留原始来源和重建过程。直接开始已有训练池的蒸馏，优先恢复第 2 节，避免重复下载和 tokenization。

## 4. Student 和代码分别从哪里取

- `workspace/models/Qwen3-0.6B/` 是 adapter 依赖的基座。
- `workspace/models/Qwen3-0.6B-common-pair-init-r32/` 是共同初始化，**不是完成训练的蒸馏 Student**。
- 真正的已训练权重位于 `workspace/runs/five-specialists-20261030/<模型名>/final/`。
- 选择上一阶段最终确定的混合或融合模型作为 parent；其尚未完成或尚未备份时，不能用初始化冒充。
- 源码、配置、环境清单和报告等小资产，已由用户指定会话 SCP 回来，见 `reports/scp-backup-20261031/` 的台账；另提供一个同来源的小资产归档，放在数据仓库 `support/` 下，方便新机器直接拉取代码。
- Teacher 原权重未在源机 `models/` 中定位到；本备份不能声称已包含 31B Teacher。Teacher 的准备仍按 A800 蒸馏规划执行。

训练设置与 8 卡安排见 [A800 蒸馏执行版](../a800-8gpu-distillation-20261031/REPORT.md)。配套报告、依赖文件和卡分配图也已放进数据仓库 `support/a800/`，新机可一起拉取：

```bash
modelscope-hub download \
  assumeengage/jev-distill-data-20261031 \
  --repo-type dataset --local-dir "$ROOT/_restore/support-repo" \
  --include 'support/a800/**'

mkdir -p "$ROOT/code"
cp "$ROOT/_restore/support-repo/support/a800/requirements-student.txt" "$ROOT/code/"
cp "$ROOT/_restore/support-repo/support/a800/requirements-teacher-hf.txt" "$ROOT/code/"
```

## 5. 新机器直接拉代码和小文件归档

```bash
modelscope-hub download \
  assumeengage/jev-distill-data-20261031 \
  support/scp-small-assets-20261031.tar.gz \
  support/SCP-SMALL-ASSETS.json \
  --repo-type dataset --local-dir "$ROOT/_restore/support-repo"

mkdir -p "$ROOT/_restore/small-assets"
tar -xzf "$ROOT/_restore/support-repo/support/scp-small-assets-20261031.tar.gz" \
  -C "$ROOT/_restore/small-assets"

for section in scripts src config datasets cache runs reports; do
  [ -d "$ROOT/_restore/small-assets/workspace/$section" ] || continue
  mkdir -p "$ROOT/$section"
  cp -an "$ROOT/_restore/small-assets/workspace/$section/." "$ROOT/$section/"
done
```

这里仅补齐缺失文件，不用较早的 SCP 快照覆盖刚下载的大仓库 manifest、评测数据或训练状态。根层的旧 `AGENTS.md`、`activate.sh` 和 `README.md` 保留在归档中供参考，不自动套到新阶段。

该包来自已经 SCP 落盘的小文件副本，不包含认证目录、浏览器资料或 SSH 密钥；环境观察文件在包内 `backup-environment/`。这是旧工程的原样备份，不表示三臂蒸馏入口已经完整实现。环境按 A800 规划配置，也不要因恢复了旧控制脚本就重新启动五专科／融合实验。

首轮稳定资产上传完成之后，源机还会在最长 72 小时内增量备份新完成的训练、融合和评测产物。尚未生成的 MIXED_ALL 或融合 parent 不会被初始化权重替代。
