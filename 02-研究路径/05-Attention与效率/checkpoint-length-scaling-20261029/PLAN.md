# 输入长度与gradient checkpointing

- 恢复旧medium和long_real_pairs的逐条长度、每个microbatch padded shape、有效tokens、padding占比。
- 固定原Qwen3-0.6B/R0、LoRA r16/alpha32、BF16、causal SDPA，物理microbatch16，累计2次。
- 固定256/512/1024/2048/4096/8192，使用同32对真实长文本裁出精确长度，零padding；逐档checkpoint On/Off。
- 每个组合独立进程重复两次，每次预热1个effective batch、测3个effective batch；记录前向、反向、总体耗时以及allocated/reserved。
- OOM单元保存错误及发生阶段后正常结束；128-token步长查找On/Off的首次OOM位置。
- 旧profile各重新测On/Off一次；另测物理microbatch8/32、长度1024/2048，补上batch迁移数据。
- 使用GPU0，不占其余卡，不中断已有工作；全部实验以110分钟为总时间预算。无optimizer step，不保存或修改任何模型权重。
- 报告直接给出具体输入长度、显存、速度、OOM位置和配置选择，不加入长篇方法限制或套话。
