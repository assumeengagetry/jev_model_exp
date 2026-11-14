# 推理调度与计算内核 v2

用户于2026-11-11确认：**保留当前权重和评分规则，重做推理调度与计算内核**；所提4B半小时仅为已知参考，没有代码，不再要求用户提供实现。

## 最终结果：完整public已完成

**有界JIT版Public Index24.91，完整推理3436.6秒＝57分16.6秒；比v1再快3.538倍，比原始实现快9.780倍，仍未达到30分钟。** v1的24.99/12157.2秒及原始24.97/33609.1秒完整保留，没有原地替换或上传。

| 指标 | 原始实现 | v1共享prefix/Graph | v2调度/融合内核 |
|---|---:|---:|---:|
| Public Index | 24.97 | 24.99 | 24.91 |
| Raw Index | 44.15 | 44.18 | 44.07 |
| 完整推理墙钟 | 9h20m09.1s | 3h22m37.2s | **57m16.6s** |
| 请求中位数 | 70.5ms | 16.6ms | **8.0ms** |
| 请求均值 | 238.7ms | 86.2ms | **24.2ms** |
| 请求p95 | 676.5ms | 328.4ms | **79.4ms** |

- 140620个请求全部ok；官方排除442后计分140178条全部ok。实际成功attempt恰好140620条，无错误重试，无首个中断attempt或旧预测混用。
- 含加载/JIT/Graph初始化及官方评分的pipeline3505.258秒＝58m25.3s；之后两份只读全量比较与完成复核约37秒。成功attempt从启动到完成receipt3542.632秒＝59m02.6s。
- runner16:34:55UTC结束推理；16:36:19UTC（北京时间2026-11-12 00:36:19）发布最终receipt。若从首个主动停止的attempt15:29:42起计算，连同该partial和修正间隔约1h06m37s；它不计入成功完整run的57m16.6s性能指标，也没有被丢弃。
- 16:38:12UTC直接复核并于16:41:13UTC归档最终证据：完成输出文件绑定、原始/v1资产、新源码及冻结源码字节、实际运行时、两份完整对比、逻辑token等12组检查全部true。不是仅采用监视器布尔值。GPU0已释放，4卡均无显存占用。

## 全量数值与质量边界

与v1比较：

- 140620个run_id、请求payload、field及完整候选覆盖一致；逻辑输入token计数差0。
- 308559个choice字段中3814个选择变化，一致率98.7639%，变化率1.2361%。
- 14700个noul字段中155个跨过0.5阈值。
- 2506535个概率值平均绝对差0.001704067（0.1704个百分点），最大0.206436923（20.6437个百分点）。
- Public Index由24.99降至24.91，差−0.08；相对原始24.97为−0.06。不能说分数不变、逐位无损，不能仅凭平均差较小忽略尾部差异，也未证明全部漂移只由某一种舍入导致。

与原始实现比较：choice变化3978、一致率98.7108%；noul阈值变化145；平均/最大概率绝对差0.001708811/0.178211412。保持的是权重、输入与评分公式，而不是所有有限精度输出。

当前五维skill百分数：Knowledge9.20、Language32.26、Retrieval36.06、Tools31.80、Arts11.57。没有private Full或维护者指定硬件私测结论。

## 分任务收益与剩余成本

以下为实际请求计时之和，非额外benchmark；所有任务相对v1均无累计耗时退步。

| 任务 | v1 | v2 | 加速 |
|---|---:|---:|---:|
| RouterBench | 51m48.4s | **11m22.9s** | 4.55x |
| ToolRet | 14m22.3s | **8m58.7s** | 1.60x |
| CLINC150+OOS | 8m55.2s | **4m36.8s** | 1.93x |
| POP909-CL | 32m51.0s | **3m34.6s** | 9.18x |
| ACOS | 16m05.0s | **2m25.9s** | 6.62x |
| BRIGHT | 6m53.5s | **2m22.5s** | 2.90x |
| 完整BFCL | 3m43.0s | **38.6s** | 5.78x |

RouterBench/ToolRet/CLINC三项合计约24m58.4s，占成功请求时间之和约44.0%。30分钟仍需在当前基础上再快1.909倍；剩余任务成本不能用模型参数量或未知4B配置直接推断。没有额外profiler或分阶段模型诊断，尚未把剩余成本定量拆成CPU分词/调度/GPU计算，不能宣称某一项已被证实占固定百分比。没有自动开始下一轮修改或public。

## 最终实际内核/调度证据

- 全程compiled版本仍为RMS1、pointwise2、norm_rope2、fork_attention1，共6；不是只在启动时暂时有界。初版JIT膨胀问题在本轮全套实际输入上未重现。
- 67个Graph全部捕获；Graph启动10.414秒，模型加载含JIT/capture17.750秒。结束前reserved17.902GiB，非峰值声明。
- 计数含官方默认2次warmup：full Graph145732波/full eager7723波；prefix Graph22421次/prefix eager1次；fork Graph33481波/fork eager61波；跨field prefix请求12357次。
- 非padding计算计数含warmup：full231132455token、unique prefix33067778token、suffix20331658token。它们不包含bucket padding计算，也不是attention FLOPs或官方logical usage的替代口径。
- 上述模型调用合计209419，v1为443754，二者均含官方warmup；仅计数不能代替CUDA各kernel时间分解。

最终8份聚合证据已存入`03-证据与图表/05-Attention与效率/inference-kernel-runtime-20261111/results/`：原生scores/status/environment、COMPLETE、对原始和v1的两份COMPARISON、FINAL_VERIFICATION、TIMING_AUDIT。628MB逐请求结果保留远端，不加入研究Git。

### 首个生产attempt与有界JIT修正

- 15:29:42UTC首个新attempt实际启动，所有内核JIT/Graphcapture及请求执行成功；但长尾prefix长度被`_norm_rope`的constexpr特化，导致请求中不断编译新版本。
- 15:33只读cache查到426个norm_rope版本；终止后完整缓存共有586个。不是只凭猜测称CPU/JIT瓶颈，也没有追加模型benchmark。
- 15:34:25UTC核对启动PID/starttime/bootid后，仅SIGTERM本轮自有coordinator646593（自有child646602随其结束）；写NO_AUTOMATIC_RESUME，保留源、cache、日志和1942条ok预测，不混入替代运行。
- 修正仅把长度S与相关stride改成运行时标量，并显式`do_not_specialize`，不改kernel公式、分支寻址、舍入边界或权重。以Triton3.5.1官方`runtime/jit.py`核实该接口。

### 15:39:53UTC 首批真实生产快照（历史，不代替全量结果）

- 新coordinator651097；status1830请求全部ok/134.6秒，读取到完整结果1831条全部ok，处于BFCL结束后的ToolRet阶段。
- 实际加载含JIT/capture17.750秒；**67个Graph全部捕获并封存**，启动reserved13.004GiB。
- 实际compiled版本：RMS1、pointwise2、norm_rope2、fork_attention1，共6；norm_rope生成IR确认含运行时`%S: i32`。当时全程计数仍待复核，现最终证据如上。
- **完整BFCL1694请求：v1 223.050秒 → 新38.623秒，约5.78倍。** 旧原始203.633秒。该任务的v1多field退步已在本轮生产消除，不再只展示首500短题。
- 前500同案例：v1均值15.161ms → 新7.184ms；已执行的137条ToolRet同案例：141.315秒 →95.572秒，约1.48倍。不能把BFCL收益外推成整套半小时。
- 当时对1831条/8947字段，与v1 token计数差0、choice变化43、概率平均绝对差0.004206、最大0.177456；仅为历史首批样本，完整质量结论如上。
- 原始证据：`03-证据与图表/05-Attention与效率/inference-kernel-runtime-20261111/BOUNDED_EARLY_OBSERVED.json`；只读来自实际生产输出，没有额外forward。

## 本轮改动

1. 请求所有field先渲染，短候选跨field合批，修复v1逐field调用造成的ACOS等退步。
2. 同一请求可沿既有token ID的公共前缀切分并共享计算；不改变原分词边界或token序列。
3. full/prefix使用显式causal Flash/GQA；仅右padding且真实EOS前无padding未来依赖，因此不再生成dense4Dmask。
4. suffix用Triton在线softmax，直接读取单份prefixKV和当前分支suffix；不cat/expand整段prefix，不复制GQA KV。
5. 融合RMS、Q/Knorm+RoPE、SiLU×up和LoRA scale/add，保留原BF16中间舍入步骤、FP32归一化与softmax累积。
6. FP32保存的LoRA/head原张量不动；长期保留相同BF16 autocast运行副本，减少每次前向重复cast。不是LoRA merge或新量化。
7. request最终一次D2H；异步H2D的独立pinned源保留到该同步结束，防止输入提前释放/覆盖。图在启动封存，长尾仍执行同一内核。

## 不改变与不承诺

同一joint final/base、原输入渲染/独立分词/EOS、语义位置、最右非pad pooling、原二类margin及逐field全候选softmax。无训练、换head、prompt/温度改动、截断、删候选、上传或push。仅GPU0，不动readout/旧运行。

浮点归约与kernel实现会改变有限精度结果，不预称逐位等价。半小时是目标，不以未知4B配置或短样本作已验证依据。沿用no-extra-tests/smoke/model-or-corpus-hash边界；JIT与Graphwarmup只作为真实生产引擎启动，不另跑synthetic benchmark。

## 计划、源码和验收

- 计划：[2026-11-11-inference-kernel-runtime-v2.md](../../../02-研究路径/05-Attention与效率/2026-11-11-inference-kernel-runtime-v2.md)。
- 独立静态审查：[review-findings.md](../../../02-研究路径/05-Attention与效率/inference-kernel-runtime-20261111/review-findings.md)。完成重用证据缺口与提前声称数值已测已修；手工换解释器与非致命故障实例恢复的条件性Minor未扩展，实际受控parent/child均使用同一既有虚拟环境，遇错即停。
- 当前源码本机：`/tmp/opencode/jev-inference-kernels-20261111-bounded-local/code/`；按当前研究库约定不强行加执行脚本到Git。首attempt源码另留`/tmp/opencode/jev-inference-kernels-20261111-local/code/`。
- 当前独立远端：`/tmp/opencode/jev-inference-kernels-20261111-bounded-jit/`，部署后字节核对并封存；首attempt根`/tmp/opencode/jev-inference-kernels-20261111/`及partial不改。原模型/数据只读复用。
- 运行时版本：Torch2.9.1、Triton3.5.1、Transformers4.57.6、PEFT0.18.1；已有H800，无依赖安装。
- 完整public已达到140620执行/140178计分全部ok；原始/v1两份全量实际输出比较已完成。no-extra-tests边界下未另做故障注入或unit/smoke，不能写测试套件通过。
- 有界JIT监视器16:37:13UTC完成通知后已退出；随后有上述直接复核。初版监视器的SIGTERM阻塞通知是主动停止，与本轮最终成功分开记录。

Context7月额度耗尽；接口已核对版本匹配主源：[Triton normalization](https://github.com/triton-lang/triton/blob/v3.5.1/python/tutorials/05-layer-norm.py)、[online attention](https://github.com/triton-lang/triton/blob/v3.5.1/python/tutorials/06-fused-attention.py)、[Torch2.9SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)、[Qwen3源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)。没有执行这些教程的测试/benchmark代码。
