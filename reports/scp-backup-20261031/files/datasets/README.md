# 数据来源与训练 split

## 已安排下载的官方训练来源

| 领域 | 来源 |
|---|---|
| Knowledge | MMLU auxiliary_train；ARC Easy/Challenge train；WinoGrande XL train；GSM8K main train |
| Language | HellaSwag train；ANLI 三轮 train；ContractNLI train；ACOS train；iSarcasmEval train；VAST train；NLI4CT train + 临床试验文档；RAGTruth train responses + 对应上下文 |
| Retrieval | BANKING77 train；CLINC150+OOS plus/train；Amazon ESCI 官方 train 的 Tasksource 已合并商品文本镜像；HoVer train；SGD train |
| Tools | When2Call SFT / preference train；API-Bank 作者发布的 level 1/2/3 training-data；ToolRet 官方 `ToolRet-Training-20w`，不是 evaluation Queries |
| Arts | Humicroedit 两个 subtask 的 train / train_funlines；New Yorker matching train；cfcolor 原始 train_vec；Habermas 原始训练标记记录 |
| Games | Boxoban unfiltered/medium 的官方 train levels；ChessBench 首个官方 action-value train 分片 |

这是一份来源清单，不代表所有下载已完成；以 `benchmarks-train/HF_DOWNLOADS.json`、`benchmark-sources/DOWNLOADS.json` 和各文件实际状态为准。

## 不伪造的训练集

- GPQA Diamond、HLE、MMLU-Pro、BBH、MuSR、SATA-Bench、SimpleBench、CRUXEval、CLadder 等在该评测中没有可直接作为训练语料的独立官方训练 split；即使某个镜像把评测题命名为 `train`，也不将其当作训练集。
- BFCL、RouterBench、BRIGHT、BPoMP、FinEntity、POP909、PhishNChips 等的榜单评测来源不能仅改名为 train；未确认独立训练划分前不放入训练目录。
- Home appliance simulator 是生成器，MiniWoB++、ScienceWorld、RTFM、Hanabi、Codenames 是交互环境，不是可统一下载的静态监督 train 表；未运行这些环境或生成实验数据。
- ForecastBench 已从当前公开指数退役。

## 大小与适用范围

- ChessBench action-value 完整 train 约 **1.1TB / 2148 个分片**；本卷初始仅约 104GB 可用，所以只安排官方首分片，绝不标记为完整训练集。
- Tasksource vision 约 140GB，且 Qwen3-0.6B 是纯文本模型，因此未下载。
- 原始压缩包可能含有多个 split，保留在 `benchmark-sources/`；`prepare-benchmark-train.py` 只抽取官方 train，不能直接把整个原始压缩包作为训练集。
- 没有运行交叉污染扫描或额外哈希校验；训练/评测隔离目前依据上游公开 split，不应宣称已经完成去污染验证。
- 上游数据有混合许可证，下载到本地不代表获得统一的再分发或商业授权。
