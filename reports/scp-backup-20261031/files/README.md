# AssumeEngage 训练工作区

所有工作区文件均位于 **`/home/dev/assumeengage`**（NFS 数据卷）。原 `/dev/assumeengage`、`/opt/assumeengage` 已移除。

## 进入环境

```bash
source /home/dev/assumeengage/activate.sh
# 或在已初始化 conda 的终端：
conda activate jev
```

新开的交互式 Bash / fish 会自动进入此目录及环境。运行 `fish` 即可使用已配置的 fish。

## 已安装

- Python 3.11、uv、fish、tmux、Git LFS、编译工具。
- Unsloth 2026.10.10、PyTorch 2.9.1 / CUDA 12.8、Triton 3.5.1、xFormers、bitsandbytes。
- Transformers 4.57.6、TRL 0.24.0、PEFT 0.18.1、Accelerate、datasets、HF CLI、ModelScope。
- TensorBoard、W&B（默认离线）、Jupyter kernel `jev` 及常用数据处理包。

安装记录：`config/installed-packages.txt`。未运行 GPU/模型测试、benchmark、训练或额外哈希校验。

## 路径

| 内容 | 路径 |
|---|---|
| Conda 环境 | `envs/jev` |
| Qwen3-0.6B 原始权重及 tokenizer | `models/Qwen3-0.6B` |
| Tasksource 文本决策语料 | `datasets/tasksource-jev-typed-decisions` |
| 按领域整理的 benchmark train | `datasets/benchmarks-train` |
| 上游原始压缩包 / 混合 split 来源 | `datasets/benchmark-sources` |
| Decision Index 官方复现代码 | `src/decision-index` |
| 缓存、运行输出、日志 | `cache/`、`runs/`、`logs/` |

具体下载状态见各数据目录的 `DOWNLOAD.json`、`HF_DOWNLOADS.json`、`EXTRACTED_TRAIN.json`，失败文件不会被标为完成。

## 数据使用

Tasksource 下载的是 `tasksource/tasksource-jev-typed-decisions` 的 **default、full、filtered-full 文本配置**。三者是不同训练混合版本，存在交叉，不应直接拼在一起。

```python
from glob import glob
from datasets import load_dataset

train = load_dataset(
    "parquet",
    data_files=glob("/home/dev/assumeengage/datasets/tasksource-jev-typed-decisions/data/train-*.parquet"),
    split="train",
)
```

`full/train-*.parquet` 和 `filtered-full/train-*.parquet` 是另外两个版本。约 140GB 的 vision 配置不适用于纯文本 Qwen3-0.6B，未下载；独立的 span-extraction 配置也未加入决策训练语料。

榜单仍展示六类：Knowledge、Language、Retrieval、Tools、Arts、Games；当前公开指数只对前五类计分，Games 中的 ChessBench 已并入 Knowledge。对应训练集的来源和限制见 `datasets/README.md`。

## 下载续传

```bash
python scripts/download-tasksource.py
python scripts/download-benchmarks-hf.py
python scripts/download-benchmark-extras.py
python scripts/prepare-benchmark-train.py
```

这些脚本只下载和整理数据，不运行训练、测试或哈希校验。网络适配保留 TLS 证书链和真实域名验证，并使用可续传的 HTTP Range。

## GPU 配置

`config/accelerate-single.yaml` 为 GPU 0 / bf16 默认配置；`config/accelerate-4gpu.yaml` 为四卡通用 Accelerate 配置。具体 Unsloth 训练入口是否支持该多卡方式，需按训练方案选择，未实际试跑。

## 国内源

- Python：阿里 PyPI；Conda：中科大 conda-forge。
- 模型：ModelScope；HF 默认镜像：`hf-cdn.sufy.com`。
- 大文件下载脚本包含此 pod 的 DNS/TLS/分段传输适配，训练时优先使用上述本地路径。
- Shell / pip / uv / conda 的配置正文保存在 `config/`，用户目录中仅放必要的配置入口；fish 等系统工具按系统包安装。
