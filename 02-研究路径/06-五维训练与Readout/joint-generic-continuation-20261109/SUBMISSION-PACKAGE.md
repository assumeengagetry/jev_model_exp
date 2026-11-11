# 联合续训候选：本地提交包边界

> **历史提交包准备快照，非当前训练状态。两段训练及原public24.97／加速版24.99已完成，见[当前方案](../current-scheme-20261110/README.md)；模型权重、原始语料、维护者访问和参榜PR仍未因本次报告Git推送而自动获准发布。**

此目录目前仅是待完成候选的材料说明，**不代表权重训练完成、正式成绩产出或已提交榜单**。

已授权执行链：当前五维KD完整final → 保留LoRA／分类头的“通用原标签＋五维原KD”联合续训 → 冻结联合final、实际训练基座、tokenizer和portable Engine → 唯一一次完整官方public0.3。执行状态见 `reports/joint-generic-continuation-20261109/README.md`。

实际基座是此前训练使用的ModelScope `Qwen/Qwen3-0.6B`、revision `master`；没有核实与任意HF不可变commit等同。正式可复现材料必须沿用实际基座快照或后来核实的固定来源，不能猜测HF main相同。PEFT权重与二分类score必须从真正完成的新final保留，不得以旧checkpoint或旧分数替换。

生产代码与私有训练数据／来源在独立scratch，公开包不得夹带原始训练样本、token流、鉴权资料或可识别用户信息。模型名、模型／成绩可见性、维护者访问、上传和参榜PR均仍待用户最终批准。当前没有上传、发布、commit、push或创建PR。
