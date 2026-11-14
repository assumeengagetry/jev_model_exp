# Base state＋一个已有LF v2：实际输入产品验收

- 实际CPU投影：102.2980095581秒，峰值RSS201121792bytes。
- 全24600train／375select，41674／615known field均有精确切点；5753／90unknown保留不tokenize，未打开confirm。
- 独立primary实际48.7907044608秒，于2026-11-12T14:19:32UTC确认原prefix IDs／全部row重拼／候选数／长度／EOS／来源与输出pins一致。
- `COMPLETE.json`和`FINAL_PRIMARY_VERIFICATION.json`为实际聚合回执；`PREPARATION.view.json`明确省略重复source bytes及input pins，原始完整产物和private sidecar留远端，未下载case/token/target。
- 严格state最后字符v1的ready=false结果保留。本版是独立`state-plus-one-existing-newline-v2`，不是round、改输入或完整Base复现。
- **仅输入边界ready；新训练、fresh张量对账、正式macro0、新mask数值/梯度/更新及质量尚未通过，不能据此宣称模型训练完成。**
