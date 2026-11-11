# 没有会话、没有AI时，如何人工重建与核对

[返回总说明](README.md)。这是依赖和操作顺序说明，**不是本次重新运行训练／评测的指令**。归档脚本是历史真实实现，有固定路径和所有权保护；不能在陌生目录盲目执行。

## 1. 先明确可复现到哪个层次

| 层次 | 本仓库本次提供 | 尚缺什么 |
|---|---|---|
| 理解算法／逐处审计 | 中文详解、公式、配置摘录、聚合结果和源码文件清单 | 代码本体另存库外，读者需另行取得；理解正文不依赖对话 |
| 对现成结果重算报告 | scores、status、readout聚合JSON和CSV | 原始逐请求文件未公开，不能独立重算所有原生metric |
| 重跑正式推理 | 两版Engine算法说明、固定kit commit、参数和命令形状 | 需要另取库外Engine、实际冻结基座、joint adapter/head、tokenizer和原suite资产 |
| 精确重训 | 两段训练算法、目标/组批定义、真实曝光与配置 | 需要另取库外trainer、共同初始化、冻结选池/索引/token streams/Teacher q/关系数组及完整构池资产 |
| 从公开来源另构一套近似实验 | 来源说明和历史converter参考 | 不可变revision/全部许可证与精确选池未完整闭包；结果不保证相同 |

**因此，本次是“可读、可审计的完整方案说明”，不是谎称已经公开了可一键复现的全资产release。** 权重、原始语料和逐请求预测没有因为用户要求推送报告就被自动发布。

## 2. 目录和角色

```text
02-研究路径/06-五维训练与Readout/current-scheme-20261110/
  README.md                   总说明和当前版本关系
  DATA-AND-TRAINING.md         数据/Teacher/损失/两段训练
  INFERENCE-AND-EVALUATION.md  两版正式执行和计分
  REPRODUCTION.md              本文
  SOURCE-ARCHIVE.md            库外代码定位、二级依赖和公开边界
01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md
01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md
03-证据与图表/06-五维训练与Readout/current-scheme-20261110/
  CURRENT_STATE.json           实际配置/完成计数摘录，路径脱敏
  SOURCE_MANIFEST.json         库外实际源码清单与字节对照记录
03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/
  RESULTS.json
  summary.csv
03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/
  scores.json / status.json / COMPARISON.json / RUNTIME.json 等
库外：jev-辅助归档/2026-11-10方案说明与近期报告/
  draft-before-pull/           拉取前97份材料的保全副本（含执行源码）
  extra-shared/               原生FA3加载/注册模块
```

`CURRENT_STATE.json` 对配置做了**字段摘录和路径脱敏**，不是逐字运行config。源码本身保持历史原件，不为把它伪装成“可移植版本”而改绝对路径。

## 3. 必须准备的原始资产清单

### A. 模型

1. 实际ModelScope基座快照；不能用同名但未核实等同的HF main替代。
2. `Qwen3-0.6B-common-pair-init-r32`：原未训练LoRA＋随机分类头＋config/tokenizer。仅重设seed不能证明重建了同一初始化张量。
3. KnowLine Teacher本地权重、native `knowline_layout.py/knowline_server.py`及发布config。
4. 若仅推理：joint final及实际base快照，不需要Teacher。
5. 若重做readout：另一条**原标签stage2 final**，不是joint final。

### B. 数据

1. 原typed/native/global池的来源清单、train原件、inputs/candidates字典、token streams及其offset。
2. 原四域selected各250k的六类数组：`state/candidate/target/source/group/length`。
3. 原四域Teacher q和**已经混合／校准的effective targets**；不能只拿一份q当所有目标。
4. 原K/R `relations.npy`与`teacher_relation_target.npy`。
5. 原生Language九任务train及group来源；特别是ANLI三轮、英语iSarcasm核心原标签、ACOS完整review字段。
6. canonical zero-shot-label NLI五个原始train分片。
7. 固定官方public0.3套件及upstream来源／强制校验信息。

每个来源还需要其许可证和用途检查。源码MIT/Apache等代码许可不能替代训练数据或Teacher权重条款；本次没有授权重发这些资产。

## 4. 环境要分开，不要装成一个大杂烩

### Student／正式推理

- Linux＋CUDA可用GPU；训练FA3路径依赖相符Hopper/CUDA kernel构建。
- Torch2.9.1+cu128、Transformers4.57.6、PEFT0.18.1、NumPy2.4.6及实际safetensors。
- 训练使用自定义注册 `flash_attn3_hopper`；正式推理使用SDPA。
- `student_flash_attention3_repro.py`（库外源码） 只注册／检查已安装FA3，不会让任意机器自动拥有同一kernel。
- 该确定性包装层还需要能以 `student_flash_attention3` 名称导入的原FA3加载／注册模块；它与实际kernel构建是三项不同依赖。原模块与包装层一并留在本次库外源码归档，不能只复制 `repro.py` 就认为依赖完整。

### Teacher

- 单独Python环境，Transformers5.12.1、Torch2.9.1+cu128。
- KnowLine发布helpers和权重；SDPA、BF16、原生read-twice。
- 不把Teacher的Transformers版本覆盖进Student环境。

### Readout

用Student同代库及实际FA3，代码显式禁TF32并采用FP32 readout；与正式Engine的autocast/TF32设置不是完全相同。

本轮加速运行设备、驱动、Python和包版本已从其原binding摘录到[RUNTIME.json](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/RUNTIME.json)，不含GPU UUID。这不是用旧环境快照替代本轮条件。本次没有为文档重装依赖，也没有声称在当前本地CPU环境完成模型验收。源码需按[库外归档说明](SOURCE-ARCHIVE.md)另行取得，不会因克隆研究库自动出现。

## 5. 人工重建第一段的顺序

这些步骤描述历史依赖，**不应直接对已有完成目录重新执行**：

1. 完整确认共同初始化、原四维selected和Teacher缓存身份；原件只读。
2. 在新的独立namespace布置第一段 `code/` 和config路径；所归档脚本由 `__file__` 父目录决定root，不能只改一个命令参数就假定其余路径都跟着变。
3. 准备九任务原生group，运行 `prepare_five_dimension.py` 的冻结逻辑。
4. 检查DATA_READY的逐域数量、Language完整group、ANLI身份恢复、unique及重复数量。
5. 按language-config启动三个单卡 `label_language.py --rank R --workers 3` 生产worker；该步骤会真实执行Teacher，需另有运行授权和资源。
6. 全部worker完成后merge，读取TEACHER_LABELS_READY；不能将partial NaN用hard label填掉。
7. 以 `prepare_five_dimension.py --targets` 固定effective targets和原子plan。
8. 训练 `train_five_dimension.py --arm KD_HYBRID`；设置必要数值环境、CUDA映射、共享scripts导入路径。
9. 完成后核对1,250,308 pair／366,985,783 token／9,769步／3,647关系及final导出；只有日志说“last step”不够。

原协调器 `run_pipeline.py`（库外源码） 自动处理资源和依赖，但含历史目录／暂停条件，不是对所有机器通用的安全启动器。

## 6. 人工重建第二段的顺序

1. 唯一父模型必须是本次五维KD真正完成final，而非旧原标签stage2或共同初始化。
2. 准备canonical原始train五分片，按原映射与原筛选生成generic。
3. 固定原五维全量回放和已烘焙effective targets；不重新做Tools bias。
4. 生成六stream数据、独立generic token streams、原子plan和DATA_READY。
5. 加载父参数前upcast分类头；真实比较393个张量，dtype和值必须相等。
6. 新建AdamW／scheduler，以2e-5训 `JOINT_HYBRID`；不恢复上一段optimizer作为“同一连续优化器”。
7. 核对1,667,242 pair／458,731,913 token／13,026步／尾42／3,647关系，final及完成回执一致。
8. 冻结本次final、实际base和Engine，再决定是否另有正式运行授权。

源码中的检查点保存还包括optimizer/scheduler、RNG、实际采样身份、attention计数、完整epoch游标。恢复时必须是同一run/data/code；不能把不同partial认领到新实验中。

## 7. 正式评测命令形状

下方是将历史路径替换成显式资产变量后的**手工模板**，本次没有执行：

```bash
# 先按原版本取得并核对 KIT；MODEL/BASE/SUITE 为已获授权的本地冻结资产。
# PYTHONPATH 包含原版 Engine 目录和 KIT 源目录。
python -B -u -m decision_index pipeline \
  --engine decision_engine:PairClassifierEngine \
  --model "$MODEL" \
  --option "base_model=$BASE" \
  --option max_microbatch_pairs=32 \
  --option max_microbatch_tokens=32768 \
  --edition 0.3 \
  --suite-dir "$SUITE" \
  --out "$NEW_OUTPUT"
```

新加速版将Engine改为 `fast_pair_engine:FastPairClassifierEngine`，并确保 `prefix_graph_runtime.py` 可导入；其运行协调器/实际命令来源见 run_public.py（库外源码）。

**不可把模板当作已复测证据。** 评测会产生GPU消耗和新完整结果，不能因为文档里写了命令就自动获得再次运行授权。

完成核对至少包括：

- runner140,620全ok、scores140,178全ok且 `complete=true`；
- 同一checkpoint/base/Engine/suite/runtime绑定，运行前后没有变；
- 没有截断／候选过滤／旧预测混用／正确性重试；
- 原生scores与completion receipt一致；
- public和private Full、请求延迟和整套墙钟分别报告。

## 8. 当前仍须补齐的复现缺口

1. **公开资产缺口：** 全部冻结权重／token池／选池与q未发布；仅靠本仓库无法逐位重训。
2. **来源固定缺口：** 实际ModelScope `master` 未映射到HF不可变commit；上游所有原始数据revision与许可闭包需继续核对。
3. **数值确认缺口：** 新旧Engine并非逐位等价，未隔离每项漂移来源；新共享路径通用OOM fallback也未实现成原版同等语义。
4. **因果证据缺口：** 没有五维KD-only正式分数，不能知道generic的边际贡献。
5. **外部验收缺口：** 无private Full、指定硬件官方时延验收或公开模型提交；本次GitHub报告发布不代表模型已参榜。

记录缺口是为让下一位工程师知道需要索取什么，而不是让其根据“应该差不多”重新发明一套数据后仍称同一个实验。
