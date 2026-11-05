# Teacher原生协议直接评测：已完成

## 最终结果：2026-11-04 21:51已完成

**Teacher原生协议Index68.37，37项/5,935请求全部ok，coverage100%，step退出0。** Knowledge60.62、Language64.53、Retrieval70.58、Tools86.04、Arts61.55；比本轮rawStudent系列的最佳LS19.79高48.58分。

- **[最终report](report/REPORT.md)**
- [Teacher与Student对比图PNG](report/native-teacher-vs-students.png)
- [同口径CSV](report/native-teacher-vs-students.csv)

这证实Teacher在原始任务协议下很强，不自动证明此前genericpairtargets质量好或Student一定能学到。当前接着测同Teacher的pair协议，学生模型/训练缓存不改。

补充：完成JSON中latency_scope沿用了Student模板的“一张GPU”描述；真实allocation是两GPU/TP2，不把该字符串当作latency证据。上述质量分数、请求和coverage来自实际完成结果。

以下20:44等段落为历史过程，不表示本实验仍在运行。

2026-11-04 20:39启动step275711.67，使用现有allocation中的两张A800。20:44已实际完成首50/5,935请求；**当前没有最终Teacher质量分数**。

目的：在继续投入Student训练前，直接确认KnowLine31B在目标任务原始协议上是否强于当前Student。

- 同一冻结37项、4,786完整cases、5,935请求。
- 原始state/questions直接送官方KnowLine frontend，不逐candidate改写成独立noul判断；expected/gold/scoring信息不进入payload。
- 相同Teacher原始BF16权重、TP2/triton、fp32LMhead、read-twice≤3000、frontendT1和原per-type温度文件。权重本身已折叠readoutT1.44，不做额外改动。
- 不重启原训练标签producer，不写canonicalTeacher缓存，不训练Student、不安装环境、不跑测试/hash。
- 对照直接复用HARD18.46、HYBRID19.10、T2 19.17、LS19.79、CALSOFT18.72的同子集结果。
- 原生协议64questions/255criteria上限如触发会显式记unsupported，coverage如实报告，不删题或隐去失败。

NativeTeacher若更强，只说明原始协议下有更高能力，不证明此前pairtargets优质；下一步才会区分协议转换和Student学习问题。这里不会用评测回答训练模型。

完成/失败通知已挂上，最终report和PNG/SVG对比图会自动存入本目录。
