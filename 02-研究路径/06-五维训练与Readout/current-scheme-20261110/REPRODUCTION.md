# 人工复现：需要什么资产，按什么顺序，核对哪些数

[返回总览](README.md)。本文说明依赖和操作顺序。归档脚本是历史实现，带固定路径和所有权保护，文中的步骤和命令均为模板，整理文档时没有执行。

## 结论

- **只凭本仓库，可以读懂并审计方案，也可以用聚合结果重算报告。**
- **重跑正式推理和精确重训都需要另取库外资产**：Engine 与 trainer 源码、冻结基座、joint final、共同初始化、选池与 token streams、Teacher 概率和关系数组。
- **五个缺口尚未补齐**：资产未公开、基座来源未固定到不可变 commit、两版 Engine 的数值差异未归因、缺少 KD-only 对照、没有外部验收。

## 1. 各层次能复现到什么程度

| 层次 | 本仓库提供 | 还缺什么 |
|---|---|---|
| 理解算法、逐处审计 | 详解、公式、配置摘录、聚合结果、源码文件清单 | 代码本体在库外，需另行取得 |
| 用现成结果重算报告 | scores、status、readout 聚合 JSON 和 CSV | 逐请求原始文件未公开，无法独立重算全部原生指标 |
| 重跑正式推理 | 两版 Engine 的算法说明、固定 kit commit、参数和命令形状 | 库外 Engine、冻结基座、joint adapter/head、tokenizer、原 suite 资产 |
| 精确重训 | 两段训练的算法、目标和组批定义、实际曝光与配置 | 库外 trainer、共同初始化、冻结的选池、索引、token streams、Teacher q、关系数组和构池资产 |
| 从公开来源另建一套近似实验 | 来源说明、历史 converter 参考 | 上游的不可变 revision、许可证和精确选池没有完整闭包，结果不保证相同 |

权重、原始语料和逐请求预测都没有发布。

## 2. 目录

```text
02-研究路径/06-五维训练与Readout/current-scheme-20261110/
  README.md                    总览和各版本的关系
  DATA-AND-TRAINING.md         数据、Teacher、loss、两段训练
  INFERENCE-AND-EVALUATION.md  两版 Engine 和计分
  REPRODUCTION.md              本文
  SOURCE-ARCHIVE.md            库外代码位置、二级依赖和公开范围
01-研究结论/06-五维训练与Readout/current-scheme-20261110/BOTTLENECKS.md
01-研究结论/06-五维训练与Readout/readout-study-20261108/FINAL-REPORT-20261110.md
03-证据与图表/06-五维训练与Readout/current-scheme-20261110/
  CURRENT_STATE.json           实际配置和完成计数摘录，路径已脱敏
  SOURCE_MANIFEST.json         库外源码清单与字节对照记录
03-证据与图表/06-五维训练与Readout/readout-study-20261108/final-20261110/
  RESULTS.json
  summary.csv
03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/
  scores.json / status.json / COMPARISON.json / RUNTIME.json 等
库外：jev-辅助归档/2026-11-10方案说明与近期报告/
  draft-before-pull/           拉取前 97 份材料的保全副本（含执行源码）
  extra-shared/                原生 FA3 加载与注册模块
```

`CURRENT_STATE.json` 是字段摘录，路径已脱敏，不能直接当作运行 config。库外源码保持历史原样，绝对路径没有改写。

## 3. 需要准备的资产

### 3.1 模型

1. 实际使用的 ModelScope 基座快照。它没有与 HF main 核对过，不能用同名的 HF 版本替代。
2. `Qwen3-0.6B-common-pair-init-r32`：未训练的 LoRA + 随机分类头 + config/tokenizer。只重设 seed 不能保证得到同一组初始化张量。
3. KnowLine Teacher 的本地权重、原生 `knowline_layout.py/knowline_server.py` 和发布 config。
4. 只做推理时：joint final 和实际基座快照，不需要 Teacher。
5. 重做 readout 时：原标签 stage2 final。它与 joint final 是两个不同的权重。

### 3.2 数据

1. 原 typed/native/global 池的来源清单、train 原件、inputs/candidates 字典、token streams 及其 offset。
2. 原四域各 250k selected pair 的六类数组：`state/candidate/target/source/group/length`。
3. 原四域的 Teacher q，以及已经混合和校准过的 effective targets。两者不同，都需要。
4. 原 Knowledge/Retrieval 的 `relations.npy` 和 `teacher_relation_target.npy`。
5. Language 九任务的原生 train 和 group 来源，尤其是 ANLI 三轮、英语 iSarcasm 核心原标签、ACOS 的完整 review 字段。
6. canonical zero-shot-label NLI 的五个原始 train 分片。
7. 固定的官方 public 0.3 套件及其 upstream 来源和校验信息。

每个来源还要核对许可证和用途。代码的 MIT/Apache 许可不覆盖训练数据和 Teacher 权重，这些资产没有重发授权。

## 4. 环境分三套

| 环境 | 关键版本 | 说明 |
|---|---|---|
| Student 训练与正式推理 | Torch 2.9.1+cu128、Transformers 4.57.6、PEFT 0.18.1、NumPy 2.4.6、safetensors | 训练用 `flash_attn3_hopper`（FA3），正式推理用 SDPA |
| Teacher | Transformers 5.12.1、Torch 2.9.1+cu128 | 独立 Python 环境；SDPA、BF16、原生 read-twice |
| Readout | 与 Student 同代的库 + FA3 | 显式关闭 TF32，使用 FP32 readout，数值设置与正式 Engine 不完全相同 |

FA3 由三项依赖组成：原生加载与注册模块 `student_flash_attention3.py`、确定性包装层 `student_flash_attention3_repro.py`，以及与 Torch 2.9、CUDA、Hopper 匹配的 kernel 构建。只复制 `repro.py` 不够，详见 [SOURCE-ARCHIVE.md](SOURCE-ARCHIVE.md)。

Teacher 的 Transformers 版本不要装进 Student 环境。加速运行的设备、驱动、Python 和包版本记录在 [RUNTIME.json](../../../03-证据与图表/05-Attention与效率/inference-prefix-graphs-20261110/results/RUNTIME.json)。

## 5. 重建第一段

1. 确认共同初始化、原四维 selected 和 Teacher 缓存的身份，原件保持只读。
2. 在新的独立 namespace 下布置第一段的 `code/` 和 config 路径。归档脚本用 `__file__` 的父目录确定 root，只改一个命令行参数不会带动其余路径。
3. 准备九任务的原生 group，运行 `prepare_five_dimension.py` 的冻结逻辑。
4. 检查 DATA_READY 中的逐域数量、Language 的完整 group、ANLI 身份恢复、unique 数和重复数。
5. 按 language-config 启动三个单卡 worker：`label_language.py --rank R --workers 3`。这一步会真实运行 Teacher。
6. 全部 worker 完成后 merge，读取 TEACHER_LABELS_READY。未完成的 NaN 不能用 hard label 填充。
7. 用 `prepare_five_dimension.py --targets` 固定 effective targets 和原子 plan。
8. 训练 `train_five_dimension.py --arm KD_HYBRID`，设置好数值环境变量、CUDA 映射和共享 scripts 的导入路径。
9. 核对终点：**1,250,308 pair、366,985,783 token、9,769 步、3,647 条关系**，并确认 final 已导出。日志出现 last step 不足以证明完成。

原协调器 `run_pipeline.py` 能自动处理资源和依赖，但含有历史目录和暂停条件，换机器后不能直接使用。

## 6. 重建第二段

1. 父模型必须是第一段五维 KD 的 final，不能是原标签 stage2 或共同初始化。
2. 准备 canonical 原始 train 的五个分片，按原映射和筛选规则生成 generic。
3. 固定五维全量回放和已写入 Tools bias 的 effective targets，Tools bias 不再重做。
4. 生成六个 stream 的数据、独立的 generic token streams、原子 plan 和 DATA_READY。
5. 加载父参数前先 upcast 分类头，再逐项比较 393 个张量的 dtype 和数值。
6. 新建 AdamW 和 scheduler，以 LR 2e-5 训练 `JOINT_HYBRID`，不恢复第一段的 optimizer。
7. 核对终点：**1,667,242 pair、458,731,913 token、13,026 步、尾批 42、3,647 条关系**，final 与完成回执一致。
8. 冻结 final、实际基座和 Engine 之后，再安排正式评测。

checkpoint 同时保存 optimizer、scheduler、RNG、采样身份、attention 计数和 epoch 游标。恢复时 run、数据和代码必须与保存时一致。

## 7. 正式评测的命令形状

历史路径已替换成变量：

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

加速版把 Engine 换成 `fast_pair_engine:FastPairClassifierEngine`，并保证 `prefix_graph_runtime.py` 可以导入；协调器和实际命令见 `run_public.py`。按已有两次运行的记录，完整评测一次约需 3.4 小时（加速版）或 9.3 小时（原版）。

完成后至少核对：

- runner 的 140,620 个请求全部 ok，scores 的 140,178 个全部 ok，且 `complete=true`。
- checkpoint、基座、Engine、suite、runtime 的绑定在运行前后一致。
- 没有截断、候选过滤、旧预测混用或按正确性重试。
- 原生 scores 与 completion receipt 一致。
- public 与 private Full、请求延迟与整套耗时分开报告。

## 8. 尚未补齐的缺口

| 缺口 | 现状 |
|---|---|
| 资产公开 | 冻结权重、token 池、选池和 Teacher q 都未发布，只凭本仓库无法逐位重训 |
| 来源固定 | ModelScope `master` 没有映射到 HF 的不可变 commit；上游原始数据的 revision 和许可证还需继续核对 |
| 数值确认 | 两版 Engine 不逐位一致，差异来源没有逐项归因；加速版共享路径的 OOM fallback 没有做到与原版同等 |
| 因果证据 | 没有五维 KD-only 的正式分数，generic 的边际贡献未知 |
| 外部验收 | 没有 private Full、指定硬件上的官方时延验收，也没有公开模型提交 |

复现时缺哪项资产，就向资产持有者索取哪一项。用自行重建的数据得到的结果属于另一个实验。
