# 执行记录：完整四科 Teacher 蒸馏

**已批准方案：**`reports/next-experiments-plan-20261030/REPORT.md` 与用户本轮直接执行要求；四科范围取远端 `ACTIVE_TRAINING_SCOPE.json`。

## 固定条件

- Parent：`runs/five-specialists-20261030/MIXED_ALL/final`，公共总分 22.82；三个 KD 臂从同一 LoRA/head 起点出发。
- 数据：Knowledge、Retrieval、Tools、Arts 共 18,173,706 pair / 4,042,053,269 Student 输入 tokens；不恢复 Language，不加上限。
- HARD：原始监督；SOFT：独立每 pair 的 Teacher Bernoulli KL；HYBRID：0.5 原始 CE + 0.5 Teacher KL。三组同 seed、顺序、全局 batch 128、lr 5e-5、完整一轮、新优化器。
- Teacher：PelaAI/KnowLine-31B-Gen1，Gemma4ForConditionalGeneration，BF16；使用原生 read-twice <=3000、label-token yes/no 概率，不重复施加已折入的温度。
- 复用 Student token cache，Teacher 读到 Student 实际截断后的同一文本。原公共37项五维评分不变，基线评分复用。
- 不额外运行测试、烟测、哈希；以实际下载、正式标注、正式训练、正式评测验证，不提交或推送。

## 执行决定

- HARD 不依赖 Teacher 标注，可在 GPU 0 提前正式续训；Teacher 在 GPU 1—3 标注，避免闲置等待。其他两臂标签齐备后运行；三组曝光和优化器设置不受调度顺序影响。
- Student 沿用 jev 环境。Teacher 需要 Gemma4，单独环境覆盖 Transformers 5.12.1，复用当前 Torch/CUDA，不升级 Student 环境。
- 项目盘可用约47 GiB，Teacher约63 GB；权重在本 Pod 空闲系统盘的专用实验目录承载，并通过项目 `models/KnowLine-31B-Gen1` 入口引用，不删除或压缩已有数据/成果，不扩大到其他主机或云资源。
- 本地只创建当前执行用的未跟踪控制脚本副本；远端新阶段独立目录和锁，不改动上一轮模型、数据、评分及其他任务。

## 接口

- 配置：`config/experiment-four-domain-distillation.json`；新 run：`runs/distillation-after-five-specialists-20261031`。
- Student：`train-four-domain-kd.py --arm KD_HARD|KD_SOFT|KD_HYBRID`，显式 `PairPool('MIXED_ALL', domains=cfg['active_training_domains'])`。
- Teacher 资产：`prepare-knowline-teacher.py`；成功输出 `TEACHER_READY.json`。
- Teacher worker：`label-knowline-pairs.py --rank 0|1|2 --workers 3`；只见对应 CUDA_VISIBLE_DEVICES；每 worker 的概率数组按 `global_index=rank+local_index*workers` 映射。
- 标注合并：`label-knowline-pairs.py --merge --workers 3`，生成 `teacher-labels/prob-match.npy`（canonical四科 pool 顺序）和 `TEACHER_LABELS_READY.json`。
- 评测：`run-distillation-evaluation.py`，复用现有正式 worker 与评分实现；三个 KD 臂＋既有 MIXED_ALL 基线图表。
- 控制器：`run-four-domain-distillation.py`，独占本阶段锁，已完成项只复用；失败写 PIPELINE blocked，监控按现有机制唤醒。

## 执行清单

- [x] 上一轮完成门槛、报告、四科配置、parent、缓存、空闲资源核实。
- [x] Teacher 独立环境准备，完整资产可续传下载已实际启动；13 片权重尚未全部就绪。
- [x] Teacher 完整四科、独立 yes/no、可续跑标注 worker 已实现部署；实际标注等待权重。
- [x] Student 三组续训脚本、正式评测与控制器已实现部署。
- [x] GPU 0 已实际启动 HARD，141,983 个更新步骤为正式完整一轮，不是烟测。
- [ ] Teacher 资产全部完成后自动启动 GPU 1—3 的正式标注，完成全量概率并合并。
- [x] 独立只读检查实现，修复三处进程/恢复/合并时序问题；核实正式训练与后台状态。
- [ ] 标签齐备后完成 SOFT/HYBRID；三组同37项正式评测及差值图表；最终完成标记。

## 独立评审与处置

- 逐臂训练增加子进程自身的 `TRAINING.lock` 和 `OWNER.json`，控制器可在 RUNTIME 丢失时按 PID/starttime/完整命令身份接管；现有 HARD 用真实 PID 身份登记防止重复启动，不重启它。
- Teacher 资产阶段在控制器恢复时先接管资产锁 owner，随后检查进程；正式运行中再次失败仍写 blocked，不做无界静默重启。
- 标签 merge 在独占 merge 锁后阻塞等待 worker 锁释放，完成标记与正常收尾的短窗口不再误判失败。
- 评审未发现可确认的数据缩减、索引错配或 Bernoulli KL 语义错误。静态评审不等于正式标注通过；原生 Teacher 的 GPU forward 仍等待真实权重完成后验证，不运行替代烟测。
- 同一全局 batch/数据顺序保持不变；Parent 实查 LoRA dropout 为 0.0。单卡 HARD 与后续两卡续训的硬件调度会写入运行身份，不宣称浮点执行逐位相同。

## 实际执行证据

- 2026-11-01 20:25:35 UTC：控制器 114582 启动，正式 HARD 进程 114583 从 MIXED_ALL/final 开始；无第二次确认，无样本/token 上限。
- Teacher 隔离环境实际就绪：Transformers 5.12.1 / 原 jev Torch 2.9.1+cu128 / CUDA12.8 / Gemma4 内置类，Student 环境未改变。四个常见默认 HF cache 中无此 Teacher 的现成缓存。
- 20:49:54 UTC：只交接控制器至 115779，HARD 114583 和 Teacher 资产准备 115004 的 PID/starttime 均保持不变，未给 GPU 作业发信号。
- 20:52:12 UTC：五个部署脚本逐字核对均一致，控制器持有锁且 running；HARD 已到 1,460 / 141,983 步、186,880 pair、41,593,212 tokens，GPU0 利用率100%。
- 同时 Teacher 资产 16/29 文件完成（这些为小文件，不代表权重完成率）；约62.58GB总资产仍下载中，环境已完成且 component_errors 为空。TEACHER_READY / LABELS_READY 均尚未生成，SOFT/HYBRID尚未启动，未假报完成。
- 当前项目盘约46.93GiB、Teacher独立承载盘约309.89GiB可用；旧模型、缓存与评分结果均未删除。
- 本地 watcher 2765321 存活，已有蒸馏执行唤醒回执；Teacher、标签、三个 KD 与评测、总完成继续使用原阶段通知/SCP机制。
- 证据文件：`launch-stdout.json`、`review-handoff-stdout.json`、`verification.json`。上述只证明正式训练/资产准备在执行，不代表三组已训完或 Teacher 已标完。
