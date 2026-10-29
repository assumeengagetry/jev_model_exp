## 1.根据现状来写 ： 目标 硬件 还有 ddl

## 2. 数据

|                                                                                                                   |                                                                                                                                                                                                                                           |                                                                                    |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 来源                                                                                                                | 怎么拿                                                                                                                                                                                                                                       | 量级                                                                                 |
| Mapika/decider 的 ~95 个公开数据集 + mixture v2 公开 60%<br><br><br>                                                       | `git clone` `https://github.com/Mapika/decider` `&& scripts/train.sh full` 的前半段（下载+转换，产出 `data/tasks.pkl` / `mixture_full.pkl`），写 50 行脚本转成我们的 schema                                                                                      | 1.47M 例/epoch 的池子（[README](https://github.com/Mapika/decider/blob/main/README.md)） |
| tasksource 全量<br><br><br>                                                                                         | `pip install tasksource`，遍历全部 MultipleChoice + Classification 任务（600+，缓存 >100 GB），每任务不设上限，只去重                                                                                                                                             | 数百万                                                                                |
| 榜单 37 个基准的源数据集 train split<br><br><br>                                                                            | 按 methodology.json 的 `sources` 字段逐个 `load_dataset(..., split="train")`，全部要；GSM8K 做数值选项，检索类做候选文档 choice                                                                                                                                    | 30–50 万                                                                            |
| Bobcat `dc13_skills.py`（17 个 licensed 公开源）、Jebadiah 的 ~80 个源、Together tev1 的 `fetch_sources.py`、Nimble converters | clone 后直接跑它们的构建脚本，再转 schema（[bobcat](https://github.com/foxl-ai/bobcat)、[jebadiah](https://github.com/getainode/jebadiah)、[tev1](https://github.com/togethercomputer/tev1)、[nimble](https://github.com/bespokelabsai/nimble)）<br><br><br> | 10–20 万<br><br><br>                                                                |
| LocalLLaMA/typed-decisions train、kev-suites、tasksource-jev-typed-decisions                                        | 原生 System One 格式直接入库                                                                                                                                                                                                                      | 万级                                                                                 |
| HelpSteer 1–3、civil_comments 等                                                                                    | 做 score 型<br><br><br>                                                                                                                                                                                                                     | 5 万                                                                                |

跳过 pngwn/typed-decisions-v2-system-one（榜单已标记与 MMLU-Pro test 重叠）。

全池处理顺序：转 schema → 精确去重 → 去污染（13-gram 命中任一 public suite 行即丢；Nimble 等按 ID 钉死）→ 选项顺序随机、noul 正反各一份、10% 样本复制成第二种选项顺序 → 32k varlen packing。目标 **3M+ 题、约 4B tokens**。

---

## 3. 合成：8 卡 vLLM

8 卡 vLLM 跑 Qwen3.5-27B（Mapika 用的 teacher，[README](https://github.com/Mapika/decider/blob/main/README.md)），每卡约 3k tok/s 生成，12 h ≈ 1B tokens ≈ 150 万条合成题。模板族全开、不挑：policy 规则应用、状态机/库存/工单跟踪、路由与工具选择、JSON 字段判断、对照对（改 ≤ 8 词翻转标签）、injected-answer、insufficient-evidence、50–255 选项大分类、3–10 级评分 rubric、把短题塞进 20k–60k 无关文本的长 state。gold 全部由代码或模拟器判定，teacher 只负责写文本；teacher 自己判的标签过一遍自洽过滤（Mapika 测到它对自己的通用选项标签只有 72% 自洽）。

换 base Qwen3.8-27B 用我们的渲染模板对全池做一次 prefill，存每题的 255 维 code logits（KL-to-base 项用，训练时不用第二份 27B 常驻显存）。4B tokens 纯 prefill 8 卡约 5 h。

---

## 4. 训练

- 基座 Qwen3.8-27B 文本部分，16 层 full-attn 改 noncausal、48 层 linear-attn 不动，[255×5120] readout 从 lm_head 行初始化，marker 处读出（结构见上一份计划第 5 节）。

- 第一次跑就是 27B full FT、32k pack、全池、全 loss。

```YAML
# configs/train/brute_27b.yamlbase: Qwen/Qwen3.8-27Battention_mode: prefix_bidirseq_len: 32768data: {shards: data/packed_all_32k}optim: {name: adamw, lr_backbone: 1.0e-5, lr_readout: 1.0e-4, betas: [0.9, 0.95], weight_decay: 0.0, warmup_steps: 300, schedule: cosine, min_lr_ratio: 0.1}loss: {soft_T: 8.0, w_kl: 0.5, w_brier: 0.3, w_rps: 0.3}   # S1 用 gold one-hot + KL-to-base；S2 起 target 换成 S1 的 soft labelepochs: 1micro_batch_packs: 1grad_accum: 4            # 4 × 32k × 8 卡 ≈ 1M tokens/stepfsdp: {shard: full, mixed_precision: {param: bf16, reduce: fp32}, ckpt_every_layer: 1}save_every: 1000         # 留最后 3 个 checkpoint 做 checkpoint soup
```

```Bash
torchrun --nproc_per_node=8 -m jevlab.train.sft --config configs/train/brute_27b.yaml --seed 1 --out runs/S1
```

- 算力：27B full FT 的优化器状态约 432 GB，8 卡 full shard 每卡 54 GB，32k pack 激活约 30 GB，141 GB 够。吞吐按 40% MFU ≈ 2 万 tok/s，4B tokens ≈ 55 h ≈ 2.3 天/seed。

- S2 起把 target 换成 S1 对全池打的 soft label（自蒸馏），比 Qwen3.5-27B letter-readout 的标签强得多；仍然保留 gold CE 项（权重 0.5）防止蒸馏错标签。

- 每个 run 结束：最后 3 个 checkpoint uniform 平均；多个 run 之间再 uniform 平均（StartLux 4 seed 平均、Perplexity souping 都是这么干的）。soup 掉分就换 greedy soup（按 kit 分数逐个加）。

- 温度：在不进训练的 5k 题 calib set 上拟合 per-type 温度 + T(n)=a+b·ln n，1 小时的事，写进 `decision_config.json`。

---

## 5. 评测：kit public suite 直接当 dev 集

- 每个 checkpoint 直接跑 kit 全量 public suite：`python -m decision_index pipeline --engine jevlab.eval.di_engine:JevlabEngine --option ckpt=runs/S1/soup --out runs/eval_S1`，8 卡 8 进程分片并行，约 12 万请求 1 h 内跑完（榜单只要求复跑一致，评分 run 可以多进程；延迟单独按协议测）。

- 只看三个数：public 总分、按 benchmark 的 skill 切片（找最差的几项补合成数据）、`dev_new_domain` 代理（typed-decisions test + Nimble 324 held-out，两者都不进训练）。

---

6.推理

## 7. 提交

1. 权重仓库：骨干 safetensors + `readout.safetensors` + `decision_config.json`（含 token_ids、温度、attention_mode、ensemble 成员）+ `jevlab` serving 代码。

2. 结果仓库：kit 产出的 `results.jsonl.gz`、`scores.json`、`environment.json` 等（[kit README](https://github.com/apolinario/decision-index)）。

3. 模型卡：数据源清单与 revision、去污染脚本与命中统计、开发期 public 运行次数、FP8 一致率、延迟自测。

4. PR 到 `submissions/README.md` 一行：模型名、Hub 仓库、结果数据集、engine/commit、硬件、精确推理命令。
