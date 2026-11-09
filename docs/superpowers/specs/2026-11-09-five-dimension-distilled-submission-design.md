# 最高分蒸馏配方的五维补全与参榜交付设计

## 用户目标与授权

用户要求暂停正在运行的五维原始标签扩大续训，优先按此前最高分的蒸馏配方补入语言维度，续训或重训由实施者选择，并整理成准备提交 Decision Index 的形式。用户已通过交互确认本会话直接实施本方案，并允许一次官方完整 public 0.3 运行及官方强制套件校验；不加消融、测试、烟测或额外哈希。发布／上传尚未授权。

资源暂停已单独获明确授权并完成。扩大训练根目录 `/tmp/opencode/jev-all-dimension-expanded-20261108` 的 `PIPELINE.json` 为 `paused_by_user`、`complete=false`、`automatic_resume_authorized=false`；2026-11-09T06:43:39Z 回执记录最后观察 68,180 步、1,117,589,917 token，可恢复 checkpoint 为 `checkpoints/stage2/step-000068000`，实际保存 68,000 步、8,704,000 pair、1,114,544,353 token 与 optimizer／scheduler／RNG；之后180步未保存。本会话已读取 `PAUSED.json` 原样副本并独立查询 `nvidia-smi`，无计算进程。不恢复该任务，不删除原件。

## 已知配方与证据边界

- 最高单次是 E24_GROUP_MIXED_RANK，本地 public 0.3 总分 23.96；相同配方重跑 20.51，不能把 23.96 视为稳定保证。
- 同数值控制重复的混合关系版本为 22.49；新任务保留这一数值控制，不刻意复刻旧的非确定性行为。
- E24 从原始 Qwen3-0.6B 和共同未训练 rank32 LoRA／分类头起步；四科各 250,000 pair，共 1,000,000 pair、一遍、314,094,515 有效 token。
- 点监督为原始标签与 KnowLine-31B 概率的 BCE 混合：knowledge 0.5、retrieval 0.25、tools 0.5、arts 0。Tools Teacher logit 加偏置 -1.2826099976040561。
- 另有 3,647 条已审计、未截断、互不重叠的 K/R 正负关系，关系目标为 `0.5 + 0.5 * teacher_relative_probability`，关系辅助权重 0.05。
- 旧语言专科约 35 分不是新五维模型的实测结果，28～32 只是目标，不能预填成绩。

证据：`reports/distillation-exploration-20261104/seventh-wave-results/E24_GROUP_MIXED_RANK/training-complete.json`、第九波报告、`reports/language-understanding-study-20261029.md`。

## 建议的单模型训练

### 选择重训，而非单独续训语言

从同一个未训练初始化重新联合训练五维，保留原四科的训练曝光，不采用只对 E24 做语言专科续训。这样不需要另猜一套续训学习率，也不假定语言训练不会遗忘四科。只训练一个候选，不开消融、系数网格或重复实验。

### 数据与 Teacher

- 原四科 1,000,000 pair、原始标签、原始 Teacher 概率和原关系成员原样复用，顺序身份为 K/R/Tools/Arts，语言追加在末尾，避免旧概率／关系索引错位。
- 语言目标约 250,000 pair，使用可追溯的原生训练切分：ANLI、HellaSwag、ACOS 两领域、RAGTruth、原生英语 iSarcasm、ContractNLI、NLI4CT、VAST。不使用公共评测行；不把广泛的 `typed_full` 语言池等同于原生目标数据。
- 按原题分组抽样，不拆候选集合。配额单位固定为9个任务：ANLI（三轮合并）、HellaSwag、ACOS laptop、ACOS restaurant、RAGTruth 回答级幻觉二分类、英语 iSarcasm 核心 sarcastic 二分类、ContractNLI、NLI4CT、VAST。ACOS 每条 review 的全体 aspect／sentiment 字段视为一个原题组。
- 非讽刺8任务按原题数平方根分配其余配额；不放回抽完整组，任务库存不足即封顶，剩余 pair 配额按相同平方根权重重新分配给尚有库存的任务。若总库存不足，不静默重复或扩大来源，而是 blocked 报告缺口。英语讽刺预留语言曝光的5%，按原生 sarcastic／non-sarcastic 类平衡；仅该平衡所需允许显式重复完整组。配额按 pair 曝光计，完整组允许边界少量溢出；记录独立原题数、pair 数与重复量。
- 英语 iSarcasm 只用核心 `sarcastic` 字段，不混入 Arabic，也不混入 irony／satire 等属性任务。RAGTruth 固定使用现有原生 converter 的回答级“是否含任何幻觉”二分类，监督来自 `bool(labels)`；不把它冒称旧语言专科全部任务视图。
- 原生 split、英语语言标记、来源标签与分组若不能从现存元数据可靠恢复，停止该源准入并报告缺口，不依靠名字猜测，不静默退回大池随机抽样。
- 只新增语言 Teacher 标注；复用同一 KnowLine-31B、原生 true/false 概率与 read-twice 设置。语言 Teacher 混合权重建议 0.5，原始软标签保留。这是新维度默认值，不是已验证最优值。
- 不为语言新增未经审计的排序关系；旧 K/R 3,647 条关系保持原语义。重建五维原子组批计划；`N`、实际尾批、曝光量均由冻结后的真实池计算。

这些是尽量保持 E24 四科蒸馏定义的五维扩展，不是旧35分语言专科的原样迁移：旧专科从已训练 R0 开始，新增50M token，D0:任务20:80及讽刺5%均按 token 计。本方案从 E24 未训练初始化开始，语言约250k pair，不宣称继承专科35分或其全部采样结论；新曝光定义需随整包设计获用户批准。

### 优化与存储

rank32／alpha64、LR 1e-4、有效 batch128、weight decay0.01、warmup0.03、seed42、一遍五维冻结池；保留显式 FA3 和严格 Torch/cuBLAS/FA3 数值控制，不回退 SDPA／FA2。实际 token 数由冻结数据的缓存长度统计，不按 pair 数虚构训练量。

Teacher 标注沿用已验证单卡 worker 分片机制；Student 沿用原子关系训练的单 GPU 路径，避免为四卡并行同时改变归约与原子关系语义。初稿CPU准备在独立 `/tmp/opencode/jev-five-dimension-distilled-20261109` 保留，正式采用 `/tmp/opencode/jev-five-dimension-distilled-20261109-v2` 命名空间：独立审查发现旧 ANLI 跨轮group_id碰撞，v2按原生 train 文件＋行号和完整标签候选恢复原题身份；未修改共享池。正式产物本地归档到 `artifacts/five-dimension-distilled-20261109/`，容量不足即阻塞，不清理旧模型。

## 可提交形式

交付一个固定 checkpoint 的 adapter、训练分类头、tokenizer／base revision 说明、可移植推理代码、依赖版本、精确训练与推理配置及数据来源说明；Teacher 不参与推理。实际发布前核对基座、Teacher 蒸馏条款和训练来源许可证，模型卡说明用途限制；不将“可以本地训练”自动理解成“可以任意许可证公开发布”。

推理必须复用真实的 pair 渲染、分类 margin 和 choice／noul 概率转换，不用 stock causal-LM 字母读出冒充本模型。所有候选返回概率，不能截断完整评测请求；容量不支持必须明示并计入错误覆盖。只用一个固定渲染，不按 benchmark 调 prompt。

Decision Index 官方要求完整 public 0.3 的未修改结果和 `scores.json complete=true`，并由维护者另算 Full 私有分；现有旧 wrapper 结果不能替代。建议用户本次批准一次参榜必需的完整公共运行，不进行额外评测或小样本烟测。官方套件导入自带输入一致性校验与用户之前不做 hash 的要求存在需说明的边界：不做额外模型／数据全量哈希，但正式工具强制的套件校验不能绕过。

准备阶段不上传、不创建公开仓库、不发 PR。模型与结果可先私有；上传、维护者访问方式和最终 PR 发布另经用户确认。不启用付费 HF Jobs，不发起与本任务无关的下载、服务重启或账户更改。全程不自动 commit／push。

## 验收与用户需要做的事

1. 用户审阅并批准上述单模型训练范围，明确允许本会话直接实施。
2. 明确是否同时批准一次官方 public 0.3 完整运行（包括官方强制套件校验，不包含额外 hash／烟测）。若不批准，训练和本地提交包仍可做，但不得声称已可提交。
3. 实际发布前选择权重／结果的可见性与维护者访问方式、模型名称；现阶段不需要用户提供密码或 Token。

训练通过依据为真实数据准入记录、Teacher 完整标注计数、优化步／五维曝光／FA3 前后向／final 导出记录；正式公共分只能在获准运行实际完成后报告。独立只读审查不替代这些执行证据。
