# 统计口径与一手来源

日期：2026-10-31。

## 本轮实测数据

- 服务器：`sh-09-gpu`。
- 请求：`/home/dev/assumeengage/datasets/evaluation-only/decision-index-suite-0.3/candidate-cache/normalized/rows-for-evaluation.jsonl.gz`。
- 候选映射：同目录 `field-options.parquet`。
- 真实分类输入 token 缓存：相邻 `tokenized/` 中的 prefix/suffix 长度偏移；按现有分类器的 `prefix + suffix + EOS` 计算，不重新运行模型。
- tokenizer：`/home/dev/assumeengage/models/Qwen3-0.6B/tokenizer.json`。
- 完整请求长度：同一 tokenizer 编码紧凑、排序键的 `{"state": ..., "questions": ...}`；不含答案、评分信息或来源元数据。它是统一内容口径，不是所有引擎都会使用的聊天模板。
- 模板：`/home/dev/assumeengage/scripts/pair_data_common.py` 的 `render_pair` 与现有分离 prefix/suffix 编码格式。
- 当前服务器 suite 由已发布的基础请求镜像、新增任务及官方 GSM8K 构造语义整理；GSM8K 使用本地重建候选，不宣称与维护者文件逐字节相同。

## 官方定义

| 内容 | 来源 |
|---|---|
| 榜单 | https://huggingface.co/spaces/multimodalart/jev-decision-index |
| 当前榜单 JSON | https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json |
| 榜单 methodology | https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json |
| 套件范围 | https://github.com/apolinario/decision-index/blob/main/docs/suite.md |
| 五维 37 项、权重、基线 | https://github.com/apolinario/decision-index/blob/main/decision_index/data/index-0.3.json |
| 版本、替换与子集 | https://github.com/apolinario/decision-index/blob/main/decision_index/editions.py |
| 实际行过滤 | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/io.py |
| 数据格式 | https://github.com/apolinario/decision-index/blob/main/docs/format.md |
| 基础评分与关联组 | https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/report.py |
| 分轨道评分 | https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/index.py |
| 五维加权 | https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/index02.py |
| 新增任务主字段 | https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/added.py |
| 并列正确答案 | https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/metrics.py |
| 参考引擎模板 | https://github.com/apolinario/decision-index/blob/main/decision_index/engines/transformers_engine.py |

## 数据构造与任务含义

| 任务 | 官方构造器/来源 |
|---|---|
| ANLI、NLI4CT、ESCI、iSarcasm | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/normalize_text.py |
| 原始多选题 | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/normalize_direct.py |
| GSM8K | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/gsm8k_v3.py |
| CRUXEval、BPoMP、SATA、API-Bank | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/adapters_selection.py |
| BFCL、ACOS、FinEntity | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/adapters_mechanical.py |
| ToolRet、BRIGHT | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/adapters_retrieval.py |
| POP909、cfcolor | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/adapters_creative.py |
| ChessBench、Habermas | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/adapters_scored.py |
| MMLU-Pro、BBH、RAGTruth、HoVer、When2Call、New Yorker、PhishNChips | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/adapters_added.py |
| Home appliances | https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/home_appliance.py |
| When2Call 与 BFCL 关系 | https://github.com/NVIDIA/When2Call#test |

## 容易混淆的版本数字

- 公开面板是 37 项；完整归档可评分请求覆盖 43 项。MMLU、ARC-Easy、ARC-Challenge、SimpleBench、RouterBench、SGD 仅展示，不计入当前公开指数。
- 归档中当前可评分请求是 140,178 条；本轮按活跃 37 项筛选得到 110,087 条，题数和候选数均逐行实际统计，不沿用旧页面总数。
- 37 项在统一排除前为 110,520 条，移除其中 433 条；全套归档的统一排除数 442 还包含非主榜项，不能机械地再从 110,087 减一次。
- ACOS 当前是 400 条源评论、1,565 请求；ContractNLI 是 123 份合同、2,091 判断；iSarcasm 的全部输入范围不等于其 headline 的 A-English 范围。
- 当前公开题型只有 choice/noul；报告中的性能 score 不是第三种输入题型。
- 当前 Full 是私有同技能与新领域结果，不能由这份公开数据画像推算私有题面长度或标签比例。
