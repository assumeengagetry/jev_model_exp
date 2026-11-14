# R2数据隔离前发现：旧group不是原生case group

这是本轮自主探索中对远端**既有准备源码**的只读核查，没有修改旧数据／代码或启动模型。

## 实际编码证据

`/home/dev/assumeengage/scripts/prepare-typed-pairs.py`：

- 33～45行：遍历原typed train文件，排除native dev／test及NO_TRAIN。
- 61～83行：以`(state,question,criteria,candidate)`精确去重，重复项累积label均值；首次保留的出处为`text(row.get('id'))`。
- 91～100行：输出规范化pair的字段名为`row_id`；它不是原始完整题的`group_id`，重复的其它出处已未保留。

`/home/dev/assumeengage/scripts/build-global-pair-pool.py`：

- 73～80行：按state／candidate再次跨来源去重，重复项只合并label并直接return，保留首次来源／group。
- 81～85行：global group key为`(source_id,str(group))`。
- 115～117行：typed来源命名为`typed_full/<task>`，传入的group为规范化`row_id`。
- 156～157行：这些值被写成`groups.parquet.original_group_id`与`SOURCES.json`。

因此，`original_group_id`这个名字**不表示它就是原native完整场景的group_id**。此前侦察将其与validation原生group直接比较，并未正确处理`typed_full/`前缀，所得“0匹配”不能证明case隔离。

## 对当前证据的影响

- 已发现的107条选中训练state重叠、150条public state重叠仍是实际精确文本结果。
- validation内state／group闭包排除257后得到的15293条仍只是候选库存，**从未准入为独立dev**，不因此升级为已证实干净。
- R0使用的旧readout dev一直标注历史trunk暴露未排除，只作机制诊断；上述发现不改变其用途，也不将其结果转成独立泛化结论。
- R1复用原冻结targets／顺序作单变量因果桥，不利用该validation选模型，因此不修改R1数据或目标来“顺带修复”。

## R2准入必须落实

1. 回原完整typed train／validation恢复真实`source + native group_id + id`，不能只读dedup后的global group数字。
2. 对所有真实已选训练state与原始记录建立关联闭包，传播多来源、兄弟field及重复state的历史暴露；旧dedup丢弃的来源必须从原始数据补回。
3. 对无法重建的来源／group保守隔离或排除，不能用“pair未重合”替代case未暴露。
4. 同一源的不同question/field变体若属于同case须同组；不同ANLI轮次不能只凭本地id碰撞合并成同一个实际样本，别名规则需来源证据。
5. Public仍只用于明确排除重叠与冻结后的正式评测；public答案不得成为训练、原始候选选择或engine路由的输入。

本发现是验证隔离的来源语义修正，不是“现有24.91存在public训练污染”的证明。历史训练／public是否有更深层关联仍须完成具体数据闭包审计，不能凭文件名或这个发现断言有／无。
