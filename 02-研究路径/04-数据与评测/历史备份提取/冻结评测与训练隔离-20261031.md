# 仅用于计划内正式评测

此目录严格排除在所有训练池、去重合并后的训练输入和 MIXED_ALL 之外。

`scripts/prepare-official-evaluation-data.py` 仅下载官方冻结评测输入，不运行模型测试、不做文件哈希校验，也不提前跑模型评测。

- 本地官方代码默认冻结输入版本：Decision Index suite `0.3`。
- 对应 HF 数据集、固定 revision 和完整状态见子目录 `DOWNLOAD.json`。
- 当前榜单展示版本为 `0.4.1`；正式报告应明确输入冻结版本和采用的五科/总分规则，不能冒充私有校准分数。
- 官方 `Suite(...).verify(strict=False)` 仍然计算哈希；本任务没有调用它。后续需要读取正式评测输入时直接复用本地文件和官方行筛选/评分逻辑，不附加用户未要求的校验。
- `excluded-questions.json`、官方 scoring subsets、退休/替换条目和修正后的 GSM8K 必须在正式评分时统一处理；所有11个主要条件使用完全相同的公共评测输入。
