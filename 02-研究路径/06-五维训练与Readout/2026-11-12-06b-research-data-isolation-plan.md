# 0.6B研究：R2前原生case暴露闭包审计

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将新native validation从“精确state初筛库存”提升为有实际祖先微调暴露／原生case闭包证据的候选，或明确不足，不能使用旧dedup group伪证独立性。

**Architecture:** 独立CPU-only namespace，实际使用`/tmp/opencode/jev-06b-research-20261112-validation-audit-v2`；只读selected task state ID、generic原始provenance、原typed train／validation及完整public的提供输入文本视图。用私有SQLite存完整字符串键、紧凑union-find存case连通关系和历史／public／native-dev标记；先审计及输出聚合计数，不读test目标、不构造／运行模型。首版因public schema准入失败已保留，无图／候选输出。

**Tech Stack:** 已有Python／NumPy／PyArrow／SQLite；无新安装，CPU affinity≤4、Arrow单线程、nice／idle IO、scratch独立目录、明确预算。

**Spec:** [已批准研究设计](2026-11-12-06b-top-public-research-design.md)第4节；[来源编码新发现](../../../03-证据与图表/06-五维训练与Readout/06b-top-public-research-20261112/GROUP-PROVENANCE-DISCOVERY.md)。用户自主授权，此为无GPU的独立数据准入工作，不改变R1冻结数据或配方。

## Global Constraints

- `global-pool/groups.parquet.original_group_id`对typed数据是首次保留native id，不是原group_id；不得直接与validation group比较来宣称隔离。
- 历史暴露seed以实际已选state及generic原始row为依据；对原始所有typed训练记录恢复state／source-family／native group关联，传播旧dedup丢弃的其它出处和兄弟field。
- public只投影模型收到的state、question instructions及criteria描述文本作保守排除，不使用gold答案关联、预测正确性或benchmark ID指导新监督／engine路由。官方reader可能解析原row其它字段来保持edition语义，图逻辑仅接收新source-bound输入视图；不能声称整个JSON从未解码gold。
- 实际public有115532个dict state、25088个str state；按原模型JSON规范化后仍有3226个空state（Humicroedit2628／GPQA198／iSarcasm400），案例文本在instructions或criteria。不得跳过这些请求；所有140620请求均须至少一个非空提供输入文本。完整JSON规范化和嵌套输入文本作为保守排除键，彼此只单独标PUBLIC，不按同请求union通用标签词。
- native validation原group完整；不读取test目标，不把基座预训练未知说成从未见过；只声称“未进入本项目已核查微调暴露”。
- 字符串全值精确比较，不用模型／语料hash代替关系；模糊／语义近重复仍需明确残余限制，不能冒称穷尽所有泄漏。
- source family别名必须有来源依据；ANLI不同轮次的短id不得当全局身份。未知／缺失group来源进入quarantine而非主独立分析。
- native/anli未恢复来源覆盖anli/a1/a2/a3 quarantine，但各轮group namespace保持独立；未知父namespace视图仅拒绝为provenance unknown，不当case等价关系。所有family quarantine传播到已知component；missing-group family在扫描后遍历原group键补充root taint。
- 不修改任何原数据／源码／旧暂停任务；所有原始文本、SQLite、case映射留私有scratch，公开只聚合计数和拒绝原因。
- 准入耗时上限45分钟，临时磁盘预算20GiB、保留至少25GiB系统scratch余量；触限保留partial、标incomplete，不清理别人资产或悄悄降低扫描范围。

## Review Focus

1. 历史训练state的typed/new边界与generic sentinel2^31不能错位；处理全部已选暴露，不只百万task池的一部分。
2. union后 taint flags随root合并，状态碰撞／兄弟field不能跨D_select/D_confirm或新训练；原始native group不能被pair-row-ID替代。
3. 只扫描确切来源清单的train／validation，缺文件或schema差异阻断；不能用已暴露dev作无标签“补齐”。
4. 当前R1可并行只读原数据，但CPU／IO不抢占GPU训练源读取；不得向原文件、现有SQLite或原model cache写数据。
5. 只有完整扫描／来源不变才标ready；partial、未重建别名、缺group、public联通都在计数中显式归因，不混为干净库存。

## Tasks

- [ ] 冻结实际selected state／generic provenance与全部原typed train／validation源码和元数据；确认source-row编码，私有目录独占创建。
- [ ] 源schema／row数盘点及真实祖先state注入；原public state注入；全原typed train／validation建立精确state＋native source/group连通图，分离历史与public taint。
- [ ] 从实际validation逐行给出component／历史暴露／public关联／group不完备的原因，不输出原始内容到报告；case分组后冻结候选库存，尚不评模型。
- [ ] 独立静态审阅后正式CPU审计；回读聚合、source绑定、完整扫描计数、budget实际值；未通过则不给R2“独立dev”身份。
- [ ] 审计完成再在R2数据计划固定D_select／D_confirm和合法新train case清单；两臂共用且按case隔离。不在见到新public分数后移动验证集。

## 当前状态

首版2026-11-12T05:45:41UTC真实CPU启动，在public视图准入拒绝非字符串state，随后实际schema调查发现3226空state应从其它提供输入文本代表；没有SOURCE／SQLite或候选输出，原目录保留。v2修复schema及独立review3项Important/两项提升控制要求，尚待正式启动。前期107训练state／150public state及旧不可比group0仍仅是初筛侦察，不将15293题预填为独立dev。
