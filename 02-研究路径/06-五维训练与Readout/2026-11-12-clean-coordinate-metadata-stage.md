# Clean-origin数据准备：新坐标／来源许可metadata阶段

**状态：独立CPU metadata阶段已实际完成并直接复核；不是clean数据准入，也不是重新跑已完成的case图审计。** 用户已授权夜间自主推进attention方法实验，此阶段为后续fresh-origin训练提供完整case坐标和实际许可组合库存。

## 为什么需要这一步

已完成的audit-v2有最终root/flags与全部validation roster，但SQLite只存`keys(kind,value,node)`，没有全train逐行坐标。为了恢复完整case的全部字段／兄弟view，必须新增完整原native坐标投影，不能用抽样、旧dedup行或只含dev的roster自证完整。

本阶段直接复用封存图，**不重建keys、不union、不改parent、不修改旧flags**。只扫描source／group_id／kind／split／license／license_use，不读target、不tokenize、不加载模型。原任务输入semantic coverage及许可审批是后续独立门禁。

## 实际合同

- 严格绑定原15train（2529342行）＋2validation（15550行），最终存储2544892个唯一`(file_id,row_index)`坐标。
- 有合法native_group的行，按原family／group键查旧图，再逐级找最终root后读取flags；缺group行不删除，保存root=NULL、incomplete16，不据此放行。
- 保存来源／domain／许可组合全库存、实际kind／split数量及每source首坐标，不以首8条schema检查代替全库许可证明。
- 旧库`mode=ro`且`query_only`，mmap只读。source binding遇重复路径必须完全相同，不能用当前stat静默覆盖旧封存身份。
- 新namespace全程非阻塞flock；只有取得所有权的进程可写失败回执，重复启动不得污染另一进程产物。
- CPU2核、nice19、idle IO；45分钟／峰值RSS6GiB／至少保留20GiB scratch，失败保留partial，不抢占GPU或重启其他任务。
- 输出`complete_coordinate_inventory`仅证明本阶段；`DATA_READY=false`、`semantic_input_filter_complete=false`、`license_review_complete=false`、`clean_training_admitted=false`必须保留。

## 部署与验证状态

- 源码：`/tmp/opencode/jev-clean-census-20261112-local/code/metadata_inventory.py`。
- 已有独立只读review，修正继承stat覆盖与重复启动所有权／失败回执问题；未用synthetic/unit/smoke替代真实数据。
- exclusive远端源传输／逐字节回读成功：`/tmp/opencode/jev-clean-census-20261112-v1/`。
- 2026-11-12T09:17:24UTC实际启动CPU-only进程，owner PID687736/starttime140498690/boot绑定保存在远端`OWNER.json`；不是GPU训练。
- 采用只读pidfd阻塞观察器等exact进程退出，不轮询完成；最终计数／索引内容／来源stat待实际完成后由主代理直接核验。

### 实际完成与主代理回读

- 2026-11-12T09:18:23UTC完成，59.011954秒；SQLite165883904bytes，667source／667个source-license组合。
- 实际train2529342＋validation15550＝2544892个唯一原坐标；17文件逐个检查count／min／max等于原完整范围。
- 直接遍历全部1868994种实际root/flag组合，确认parent[root]==root及flags[root]与新表记录一致；332个继承source文件stat核对通过。没有重新构建旧图。
- 缺native group的坐标实际为0。按flags的潜在dev比旧eligible多149行；逐个对照原validation-roster后，全部额外行原因为`missing_original_identity`。因此**不能用flags-only候选替代旧dev身份门禁**，后续仍须非空／来源核实的完整provided input。
- fullsource-license库存已取得，但`unspecified`／`other`／Custom／Request Form等不因库存完整而获得许可；标准明确许可和需要来源例外的组合分别审查。
- 原始`COMPLETE.json`、`FINAL_VERIFICATION.json`、`INVENTORY.json`、`SOURCE_BINDING.json`已回读归档至`03-证据与图表/05-Attention与效率/attention-queue-20261112/results/clean-census/`，源码也有仓库逐字节快照。

本阶段仍明确`DATA_READY=false`、semantic filter／许可review未完成、clean training未准入，未生成任何新模型或public分数。

## 没有通过或尚未做的事

本阶段不批准任何许可，不证明来源无泄漏，不构造D_select/D_confirm，不签发100M训练曝光，不证明tokenizer或模型可训练，也不产生public分数。另一个clean-origin草案收到后仍为blocked／未运行；其source profiles为空、合成测试未运行，不能因子代理交付就直接启动训练。
