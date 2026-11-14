# Base-inspired state＋固定换行：独立v2边界设计

**状态：v2输入边界准备及独立primary已实际通过；新模型训练尚未启动。** 严格state末字符v1已实际完整投影失败并永久保留ready=false；本页不修改v1结果、不是给BPE跨界取整，也不重新分词或修改原case。

## 问题与最小选择

v1要求切点在`HEADER + helper.text(state)`字符末尾；实际41674train knownfield中28676、615select中427跨BPE。原renderer**总会在state后追加一个已有换行**，再追加`Question:`、`Available criteria:`或`Candidate:`标头。将这一固定输入分隔符归属P，是独立可归因的state/problem边界定义：

`P_text = 'Decide whether the candidate is a correct match.\nInput:\n' + helper.text(state) + '\n'`

**主代理裁定：全体field统一使用这一字符定义，不能依token形状／答案／成功与否在v1/v2之间挑选。** 不搜索正文中的标头，不取最邻近token边界，不循环尝试多个分隔符，不新增special token。P含state后一个已有换行，但不含任何后续问题／criteria／Candidate标头字符；case正文自身包含的问题字符串不在本定义的排除范围。

这仍是**Base-inspired state-boundary-only迁移**，不是Base原chat/slot/LM-head/Qwen3.5/GDN/CE+Brier配方复现。原conditional criteria renderer、原EOS二类head及全部input IDs不变。

## 验收合同

- 除state_end增加**恰好一个已存在换行字符**外，完整沿v1已冻结的严格offset谓词：Python字符坐标、原prefix独立tokenize与每ID相同、完整字符跨度并集、Unicode重复offset同侧、无跨界token、唯一相邻端点切分；任一known失败仍整版notready，禁止drop/round。
- 两版本使用同一24600train／375select原完整roster，unknown仍null且不tokenize，D_confirm不打开。v2必须重新全量投影和直接回读，不从v1成功子集起跑。
- source版本固定`base-state-boundary-fixed-separator-v2`，配置必须有`boundary_policy='state-plus-one-existing-newline-v2'`；case sidecar携同policy，避免与v1混用。新root`/tmp/opencode/jev-base-state-boundary-20261112-v2/`，v1文件不编辑／覆盖。
- 复用已实际执行v1准备器的代码时，只在独立新源中作显式、可枚举的版本／policy／state_prefix／report边界名修改；保留3项P2修正和外层preflight失败留痕。新src及cfg完整封存/readback/stat绑定，不能只写一个不受pin约束的动态wrapper。
- 独立primary用相邻offset端点枚举全量重算v2切点、原prefix IDs／所有row重拼、sidecar／失败计数及inputs/source/config/output pins；不是把v1primary.ready翻true。
- 只有v2真实全ready才建立新quality臂`STATE_NL_PB_P_CE`；同原C_P_CE控制的完整rows/position/slot/micro/曝光/393fresh/optimizer必须实证相同，只有maskP长度变；宏0报告作为冻结scope的外部cursor1续跑门禁，不中途回填cfg。
- 保留原controller的GPU0/1 BI/PB预留，只考虑基础四臂真正完整退出后空闲自有GPU2/3。不得在原四卡运行时抢卡或把新边界自动套到现PB_P_CE。
- 不因本变体更易切分就声称Base学习收益或cached-serving准入；首正式macro、完整100M、冻结dev/confirm/public协议同原设计。

## 最小实施计划

**Goal/Architecture/Tech Stack：** 原stock/tokenizer与原FA3链不变，仅在新namespace生成明确policy的无损token切点sidecar；不安装依赖、不运行模型或额外unit/smoke/hash。

- [x] 从已执行v1源生成并封存独立v2候选；8个producer及5个primary枚举delta经独立只读review，无接受谓词或sidecar/config guard放宽。
- [x] 新src完整CPU投影实际102.2980秒／峰值RSS201121792bytes；全24600train／375select的41674／615known field精确，ready=true；5753／90unknown原样保留不tokenize。
- [x] 独立primary相邻offset端点枚举全量回读实际48.7907秒，于14:19:32UTC确认原prefix IDs／所有row重拼／候选／EOS／字段／pins完全一致；3份聚合及2份实际源码已归档于results/base-state-boundary-v2/。
- [ ] 真实ready后才接新trainer/source/config/fresh／正式macro0/外部cursor1门禁；未做到不写已排队或已完成。继承`2026-11-12-base-state-boundary-plan.md`Task2/3的全部身份／曝光／保留要求，但臂名必须为STATE_NL_PB_P_CE且注明v2policy。

**代价与风险：** 固定分隔符归P改变其相对v1的mask归属，不能隐藏为原严格定义的“修复”；若仍跨界，停止v2，不继续通过round或删case凑覆盖。即使全ready，也只证明可恢复输入边界，不代表新模型数值或质量已通过。
