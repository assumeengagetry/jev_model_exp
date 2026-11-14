# Base-inspired 严格state末字符切点：实际不准入

**结果：完整CPU扫描／回读完成，但`ready=false`、`allow_new_state_boundary_training=false`。** 不改变原DATA_READY或正在运行的四臂；本版本没有模型／GPU／梯度／optimizer／D_confirm／public工作。

## 真实结果

原prefix独立分词不变，在Python字符串`固定系统串 + helper.text(state)`的最后字符后精确切开；跨切点BPE不得取整、重分词或删case。

| 原roster | 完整case | 全部field | Known field | 可精确切分 | 跨切点BPE，不能切 | Unknown保留不tokenize |
|---|---:|---:|---:|---:|---:|---:|
| train | 24600 | 47427 | 41674 | 12998 | 28676 | 5753 |
| D_select | 375 | 705 | 615 | 188 | 427 | 90 |

全部失败原因为`crossing_bpe`，不是offset缺口或Unicode重复offset本身被误拒。train保留全部332663候选／99607907原pair-token库存，select5190候选／1554484token；这里是原库存一次遍历，**不是训练曝光或新模型已消费token**。

- 准备器实际100.6339s，峰值RSS201367552bytes，14:03UTC完成；完整case／field／known mask和原候选长度／真实EOS没有删除或截断。
- 主代理随后用**独立的相邻offset端点枚举算法**重算全部known原prefix IDs／切点，再逐行核对sidecar／私有失败清单／计数及source/config/data/output pins；实际47.5031s，14:04:14UTC完成，仍明确ready=false。
- `complete=true`只表示本次扫描与回读完整，不是科学准入。Task2不得把它当训练ready，不得仅用“可精确”的case子集继续并复用原causal控制。

## 实施前review处置

独立只读review无确定P0/P1，三个P2在首次执行前修正：生成后立即pin sidecar并全程守护、全量回读重新推导offset切点／状态／失败计数；config先pin再读并统一使用同pin；先验证安全独占输出，再使主要preflight异常留下BLOCKED。没有为此新增unit/smoke/hash或放宽offset谓词。

## 结论与边界

这证明**该字符位置无法在保持原token序列时普遍作为attention切点**，不是Base的学习质量失败，不否定原干净数据准入，不把Candidate-P与state-P当同方法。该版本停止；若另定义固定分隔符所有权或重分词/新模板，需要显式独立设计和新namespace，不能默默round或改本版失败。

归档三份聚合证据：`COMPLETE.json`、标注的`PREPARATION.view.json`、`FINAL_PRIMARY_VERIFICATION.json`，及对应实际producer／primary源码。原sidecar、失败case身份／coordinate、原案例文本、target／token／权重均未下载到Git。
