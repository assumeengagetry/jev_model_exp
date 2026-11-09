# Qwen3-0.6B 参榜候选模型

用户目标：使用当前最佳训练方法，准备可提交 Decision Index 私有评测的模型。

## 当前选定模型

- 训练运行：`E17_E06_REPLAY`，从原始 Qwen3-0.6B 和共同未训练分类头重新训练，不是继续训练旧最佳权重。
- 训练已完成：100 万组文本对、7,813 步、约 3.14 亿有效 tokens；四个训练方向各 25 万，一遍。
- LoRA rank32／alpha64，学习率 1e-4，有效 batch128，seed42。
- 教师标签比例：Knowledge／Tools 50%，Retrieval 25%，Arts 0%；其余使用原始标签。
- Tools 教师 logit 校正量：−1.2826099976040561。
- 训练实际执行 FlashAttention-3，无 SDPA 回退。
- 本地重建公共评测记录：37 项、110,087 请求均成功，覆盖率 100%，Index **23.19**。原 E06 同方法为 23.16。
- 源权重：`/home/dev/assumeengage/runs/distillation-sharpness-20261106/E17_E06_REPLAY/KD_HYBRID/final/`。

另一个执行 session 已完成这次同方法重训，本 session 不重复启动第三次相同训练，不抢占仍在进行的其他评测。

## 文件角色

- `model/`：原样保存的 LoRA／分类头及模型配置，不是独立完整基座；重载仍需对应 Qwen3-0.6B 权重。
- `evidence/`：训练配置、完成回执、本地公开评分及逐请求预测；只保存原件，不补造官方完成字段。
- `source/`：本次模型实际依赖的推理、输入渲染代码快照；保留原路径设置，尚不是可移植的正式提交 Engine。

## 正式提交尚需完成

维护者要求完整 public 0.3 运行、可重新评分的结果及可自行启动的模型／推理代码。现有记录采用本地重建输入，manifest 未声明官方冻结输入的一致性；现有 `scores.json` 也不是官方 pipeline 的完整提交文件。因此 **23.19 只能按本地公开分报告，不能直接冒充官方接受的 public 或 Private／Full 成绩**。

提交前需要将推理代码整理为官方 Engine，并生成符合官方格式的 public 运行材料；权重和结果的 HF 仓库、访问范围及 GitHub PR 发布尚未执行。

官方流程：<https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard>。
