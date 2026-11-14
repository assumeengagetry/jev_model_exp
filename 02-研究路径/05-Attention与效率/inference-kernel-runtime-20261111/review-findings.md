# 内核v2独立静态审查与处置

审查session：`ses_e34cff7f1ffedSXvbFNb38rQM8`。完整读取6个库外源文件及计划/账本，无模型执行、测试、编译、hash、文件修改或二次委派。

## 结论与修正

未静态确认Critical计算缺陷。RMS/QKnorm+RoPE、SiLU/up、LoRA scale/add舍入边界、16/8GQA寻址、prefix只读与suffix隔离、online softmax掩码、跨field索引恢复、因果右padding、pinned/图地址生命周期未发现确定性错误；不是运行验证。

- **Important I1已修正源码：** 完成回执新增本轮results/status/run-binding元数据，重用前验证它们、score与两份comparison；重用时重新只读计算两份完整结果对比并核对既有摘要，前后再复核绑定。不加载模型、不追加推理或hash。首次完成也验证这些输出后再写receipt。
- **表述问题重级为Important并修正：** 初始化provenance不再写“已测数值等价”，改为“待完整实际public对比”。因为新内核尚未运行，不能把验收意图冒充结果。

修正属于一轮源码fix，受no-extra-tests约束，没有RED/GREEN测试声明；实际JIT/capture/replay和完整public仍待执行。

## 保留的Minor/条件项

1. 协调器如果被手工用其它Python启动，采集的包版本可能不是子进程解释器。当前受控launcher明确使用`/home/dev/assumeengage/envs/jev/bin/python`启动协调器及其子进程，实际该环境版本已读取；不声称任意shebang启动方式都被支持。
2. request中途发生非致命异常时，runtime保留active及pinned源，后续请求可能连锁报错；本轮控制器遇非ok即终止生产，不尝试复用故障实例。没有为清理状态而不安全地提前释放异步源，也未进行故障注入。

## 证据边界

审查使用的旧源码来自本机辅助归档；主代理另从远端冻结目录只读取回了控制器/比较器并部署核对本轮字节。静态审查不证明JIT、H800实际执行、所有长尾、数值漂移或30分钟目标达成。
