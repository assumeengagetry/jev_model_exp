# 工作区约定

- 主目录是 `/home/dev/assumeengage`。环境、模型、数据、缓存、输出均放在此目录；不要再使用 `/dev/assumeengage` 或 `/opt/assumeengage`。
- 用户是 pod 内的 root，实际继承的 `HOME` 可能是 `/home/jovyan`。
- 集群仅用户一人使用。已授权实验范围内可直接使用现有 GPU、CPU、磁盘和 root 权限，不反复索取资源使用确认；不要安排与实验无关的安全加固等旁支工作。
- Conda 环境名为 `jev`；Bash 使用 `source /home/dev/assumeengage/activate.sh`。
- 优先国内源：PyPI 阿里、中科大 Conda、ModelScope 模型；HF 下载脚本在 `scripts/`，包含此 pod 所需的网络适配。
- 用户优先要求尽快跑实验、得到结论；不额外运行测试、烟测或文件哈希校验。复用现有环境、模型、数据和预处理缓存，独立下载/预处理及不冲突的训练尽量并行，关注有效 tokens/s 与端到端耗时。
- 正式参数和训练授权已在 `config/training-plan-five-specialists-2026-10-30.md` 明确：五专科加一个混合模型、五种融合、统一五维 benchmark 评测。前置数据与共同初始化就绪后直接执行，不再等待第二次开训确认；计划要求的正式评测不是额外测试，不可省略。
- 提速不等于缩成小池短训：保留充分数据、完整一轮训练和 MIXED_ALL 公平曝光，不为速度擅自缩减正式实验；遇到实际阻断再说明。
- 不把评测集当训练集。没有公开 train split 的 benchmark 必须明确标注。
- 不在文档、日志或回复中写入账号 token。
