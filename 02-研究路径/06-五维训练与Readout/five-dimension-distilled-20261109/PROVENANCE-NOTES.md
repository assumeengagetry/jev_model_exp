# 发布前来源与复现边界

## 已核实

- 实际Student基座来自本地 `models/Qwen3-0.6B`；其原下载记录声明 repo为 `Qwen/Qwen3-0.6B`、source为 **ModelScope**、revision为 **master**。这不是已固定的HF commit，不能假称已核对两个站点的权重逐文件一致。
- 基座本地许可证为Apache-2.0。
- 实际Teacher为本地 `PelaAI/KnowLine-31B-Gen1`。其模型卡和本地LICENSE声明Apache-2.0，基座为 `google/gemma-4-31B-it`；Google官方Gemma4许可证页面同为Apache-2.0：<https://ai.google.dev/gemma/docs/gemma_4_license>。
- 共同未训练初始化为rank32／alpha64，分类头 `score` 在PEFT的 `modules_to_save` 内；不是只导出LoRA而丢掉分类头。

## 发布前必须处理

- 正式公共运行使用**实际训练所用的同一本地基座**，通过Engine的显式 `base_model` 参数传入。发布时应随提交包提供该实际基座，或先获得可核实的不可变源revision；不能未经核对就用任意HF `main` 权重替换。用户禁止额外权重hash，本次不以新增全量hash解决该问题。
- 公开／私有发布均需保留基座、Teacher和代码适用的许可证与署名，并披露训练来源。
- 训练数据包含ANLI等各自带用途限制的数据源，**不把模型组件的Apache-2.0等同于所有训练来源均无商业限制**。本轮是研究／benchmark用途；最终模型卡和使用范围需在发布前逐源说明，不提供未经确认的“可自由商用”承诺。
- 不上传公共评测全文、不上传私有训练缓存，不将重建的官方suite放入公开Hub dataset。仅结果目录按官方允许格式，在用户批准发布范围后上传。

本说明是技术来源记录，不替代法律意见。当前没有上传、公开权重或发起PR。
