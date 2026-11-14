# Clean后续两探索臂：正式macro0/primary已过，资源guard阻断续跑

## 已实际验证的层级

1. 新9份trainer源码独立spec/quality审查通过；主代理全量读回并AST语法解析。`fresh_init.py`和`checkpoints.py`与实际旧v2基线逐字节一致，旧scope完全不编辑。
2. 新namespace独占部署，完整source stat/精确bytes绑定；新配置精确允许两个探索臂：state＋一个已有LF的packed prefix_block/CE，以及原full-pair causal/packed BCE＋hard-rank。
3. **先从唯一stock独立重新生成fresh0forward，完成后才加载旧v2共同fresh oracle比较**。全部393FP32张量同key/shape/dtype/bytes一致，196LoRA A/B对、B精确零、head2×1024、served616237056；未加载项目旧已训练adapter/head/optimizer/teacher，未初始化CUDA、未forward或更新。
4. final seed42 config绑定真实新init COMPLETE后独占封存；原manifest `trainer_code==ctx.code`校验未放宽，未把旧manifest改名复用。

实际回执`FRESH_PRIMARY_VERIFICATION.view.json`明确省略trainer sourcebytes，完整产物留远端；11份实际源（9trainer＋bootstrap＋独立CPU分析）均回读归档。未下载任何private case/token/target/weight。

## 实际正式macro0与资源阻断

- 两臂正式macro0实际完成并exit75/cursor1：各16case／19knownfield／60candidate、20947logical／41894forward／20947backward token；macro0只消费一次。
- 独立saved-only primary真实核对两遍bytes、完整目标/coef（rank27项book）、实际native分区/positions/EOS、393raw/clipped梯度、delta/Adam/scheduler及cursor1checkpoint。STATE336nativecalls/6wave、JPAIR168/6wave；JPAIR首遍logits/margins与原C_P_BCE bytes完全一致。
- STATE preclip60.0638275、JPAIR141.5658722，均clip1；rank18active/1zero/2soft。有限scientific gate不是新P跨backend/neighbor数值等价，也不是393模型梯度独立数学证明。
- 旧queue在PRIMARY_READY之后、MAIN授权前资源guard退出：可用115269525504bytes，完整预留要求116379385959bytes，差1109860455bytes。所有自有first/primary实际退出，**无AUTH、无resume、无重放macro0**。7份正式macro/primary/resourcefail聚合已归档。
- MAIN直接回读脚本在rootBLOCKED启动检查按设计拒绝，未读tensor或创建授权；脚本/log和原QUEUE_BLOCKED保留，不改写原失败。后续只能在独立新控制namespace显式分阶段续跑，25GiB/20GiB不降低，trainer/cfg/fresh/arm scope不变。

## 尚未完成、不得扩签

- 正式macro0和外部primary已经真实通过；其后独立external CTRLv2的MAIN实际回读、单独授权及显式续跑已完成准入，18:02回执到50/63macro；**完整100M/新dev仍未完成**。见`../clean-followup-staged-v2/README.md`。
- 新分阶段控制器绑定精确旧资源失败/退出、全部pins/最新cursor1，正确MAIN实际回读后才另行授权；原queue仍保持failed且从未重启。
- 不事后回填冻结cfg，不重复消费macro0；本次首macro无K1或unknown GPU字段，不冒称所有失败分支或unknown-forward路径已新实测。
- State只允许在真实分区/首macro通过后的有限fullrow科学训练，不签新P-specific跨后端数值／邻居隔离或cached-serve/production。JPAIR不是完整Jina/listwise/cosine多正例复现。
- 两臂均在看到10M/30M后提出，是探索性消融，不追记为原四臂预注册比较。未打开confirm/public，未选择winner。
