# GPU协作交付：checkpoint固定长度资源测试已完成并释放

**当前状态：已收到本项新的完成/释放通知，GPU0预约解除。** 通知报告释放时四卡均0MiB；正式启动其他任务前仍须重新检查实时占用与其他预约。
49个测量单元与28层SDPA追踪已交付，主报告为`checkpoint-length-scaling-20261029/RESEARCH-REPORT.md`。
本会话未重复测量。比例训练仍因完整采样流与自然dev协议未冻结而不启动；资源释放不是训练就绪证明。

## 保留的预约历史

监视会话已通知将使用GPU0执行checkpoint On/Off固定长度与OOM边界测试，产物位于：
`reports/checkpoint-length-scaling-20261029/`。

本会话继续只做pair比例CPU准备：

- 不在GPU0启动任务，也不启动会占GPU0的四卡实验。
- 不重复该长度/显存/速度/OOM测试，不停止、重启或覆盖协作任务及其产物。
- GPU0在子进程切换/OOM处理间显示空闲，不代表预约解除。
- 先前推理长度诊断07:46:30释放GPU0的通知，不覆盖这次新预约。
- 新的完成/释放通知现已收到；继续检查完整比例采样表、自然dev/test和其他执行条件，按PLAN接续有界比例研究。

## 旧profile的复原信息

以下只读引用协作会话`checkpoint-length-scaling-20261029/original-profiles.json`，不是本会话重跑测量。

| profile | 实际长度范围 | 两个micro16的padded shape | 非padding / padded tokens，整profile | 整profile padding |
|---|---|---|---:|---:|
| medium | 192—205 | [16,208]、[16,208] | 6334 / 6656 | 4.84% |
| long_real_pairs | 2527—2807 | [16,2576]、[16,2808] | 83466 / 86144 | 3.11% |

4.84%和3.11%分别是两个profile的总padding比例，不是long两个microbatch各自的比例。
原long profile不是固定4096输入，不能将其OOM直接写成固定4096的实测结论。

新的256/512/1024/2048/4096/8192固定长度测试及128-token步长OOM搜索已由协作会话完成。
其计划使用R0、0.6B、r16/alpha32、BF16、物理micro16累积2；前后向探针不做optimizer step、不保存或修改模型权重。
具体资源数值与mask路径见`reports/language-understanding-study-20261029/CHECKPOINT-RESOURCE-ADDENDUM.md`；不将技术资源探针当作Language能力/训练长度最优性实验。
