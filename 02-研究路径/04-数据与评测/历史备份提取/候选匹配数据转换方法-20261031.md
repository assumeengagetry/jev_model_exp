# 并行补充训练数据

监控/协调会话 `ses_e7444cec5ffeRHet9AGZDT1Ckz` 在用户休息期间并行下载，执行会话继续负责正式训练、融合和评测。

- 下载入口：`scripts/overnight-download-supplements.py`。
- 文件位于 `datasets/overnight-supplement/train/<专科>/<来源>/`，独立于原数据目录，不覆盖原文件。
- 完成情况、来源、分片与镜像 revision 见 `DOWNLOAD.json`；`downloaded`/`present` 之外的条目不能假定已齐备。
- 只选择明确的 train 文件或上游完整训练专用源；不选择 test/validation 文件，不做模型测试或哈希校验。
- 这些是补充来源，不擅自替代计划指定的 `tasksource/zero-shot-label-nli` default 主底池。
- 正式数据转换时与已有训练来源统一归类、去重并记录实际 pair/token 数，然后纳入专科与公平曝光的 MIXED_ALL。

## 可复用的候选匹配转换缓存

`scripts/prepare-overnight-pairs.py` 已作为实际 CPU 数据准备任务启动，输出 `pairs/<专科>/<来源>.parquet` 和 `PAIRS.json`。

- v2 字段为 `state`、`question`、`criteria`、`candidate`、二分类 `label`、`domain`、`source`、`source_repo`、`group_id`；无相对选项上下文时 `criteria` 为空。
- NLI 保留三类关系，各类分别构造匹配/不匹配候选；ESCI保留四类相关性；xLAM按工具候选匹配，不转换为聊天生成式SFT。
- HH-RLHF/UltraFeedback使用真实偏好对；HelpSteer的 helpfulness/correctness/coherence 使用原始0–4人类评分候选。
- 相对偏好对保留双方响应作为 `criteria`，按原始行号确定性轮换选项顺序，避免缺少比较对象或“总选第一项”的位置捷径；与主执行会话的 `pair_data_common.criteria_context` 和 `render_pair` 共用格式。
- `PAIRS.json` 记录每来源实际 raw/used/pair 数、未转换条目及错误；尚不支持的来源明确标记 `raw_ready_adapter_pending`，不冒充已转换。
- 这些是中间缓存，不代表全局去重、共同输入格式的tokenization或正式训练已经完成。请主训练会话复用缓存，合并主底池与其他来源后做全局去重与统一tokenization。
