# 新state sidecar reader：全量真实产品验收

独立新namespace中的CPU/JSON reader真实执行16.838594秒；不是重新投影、tokenizer、模型运行或synthetic/smoke。

- 原24600train＋375select完整索引，所有case/ordinal/coordinate/native split及field顺序对齐。
- 41674＋615known field、332663＋5190candidate的cut／原prefix长度／完整pair长度／EOS／无损row重拼通过。
- 5753＋90unknown保留，四项边界信息null，不解释raw options/targets。
- 真实正式曝光macro0的16个train case各查询两次，值相同但cut列表独立；全部375select通过dev映射查询；持续stat/sourcebytes/inventory守护执行。
- 未导入Torch或Transformers，0模型forward，未打开confirm/public。未注入人为缺行、损坏或变更pin，不能称这些失败分支均已实测。

聚合`COMPLETE.view.json`明确省略sourcebytes；4份实际源码与远端执行版逐字节回读后归档。**本结果只签reader输入产品，不签新trainer集成、fresh、正式macro0、梯度/更新、质量或serve。**
