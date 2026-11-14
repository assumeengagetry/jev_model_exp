# Follow-up资源阻断后的显式分阶段续跑

## 已发生事实与授权边界

followup-v1两臂正式macro0实际完成、各cursor1/exit75，独立saved-only primary实际通过。旧queue随后在等待MAIN批准的资源guard退出：free115269525504bytes，要求116379385959bytes；差1109860455bytes。全部4个自有first/primary子进程已实际wait退出，原queue失败和leases metadata保留；未创建旧AUTHORIZE、未发生任何resume。MAIN只读脚本因根QUEUE_BLOCKED按设计拒绝，未读模型或签发授权。

原followup计划Task4b明确允许：双臂容量不足则BLOCKED，MAIN之后另建显式分阶段队列、重算总预算，不能自动retry本scope。本页落实该分支；不是删除失败、降低reserve、改trainer或重吃macro0。

## 冻结不变项

- 模型数据root仍`/tmp/opencode/jev-clean-trainer-20261112-followup-v1/`，9source/finalseed42cfg/fresh/两个arm OWN_SCOPE/actual macro0/两primary及cursor1checkpoint不编辑、不复制成新训练、不重新生成fresh。
- 每臂25GiB原上限、共享20GiB底线及5GiB evidence预算不降低；不删除任何checkpoint，不租卡、不抢卡、不commit/push/public。
- 3557calls/99824365logicaltokens、固定三select、所有训练数学及mask边界完全不变。仅资源准入从“同时预留两个新臂”改为“为实际准入臂预留；pending臂不宣称已分配容量”。
- 新独立控制root`/tmp/opencode/jev-clean-followup-staged-20261112-v1/`，本地同名`-local/`；旧QUEUE_BLOCKED及失败MAIN脚本/log全部保留，绝不重启旧queue。

## 前置验收与来源链

新manifest必须精确绑定旧QUEUE_BLOCKED stat/bytes及RuntimeError资源原因，旧queue launch身份确实已退出，children actual_wait_complete且first75/primary0、旧AUTH不存在。两个arm无BLOCKED/COMPLETE，ledger只macro0一次且最新pausedcursor1，无failed/dev/resume/未提交tail；原primary、fresh、source、cfg、DATA_READY与所有关键artifact pins仍匹配。

MAIN在新root的独立CPU只读程序可复用已review的213行direct-readback逻辑，但只允许这个**精确资源controller失败**，不允许泛化忽略BLOCKED；旧scope不能写MAIN success或AUTH。MAIN直接重读actualcache/replay/coef/dispatch/393raw/clipped/before-after/delta/Adam/latestcheckpoint，成功回执放新root；0forward/0更新，不制造测试。主代理读回后才原子发布新root的外置AUTH，绑定上述原因、两报告、完整scope/最新checkpoint与nextindex1。same-owner流程门禁不宣称抗恶意同UID。

## 资源与lease

先后顺序固定STATE再JPAIR。若启动时总预算已足以同时准入，允许同时恢复；否则先STATE，JPAIR明确pending，记录原因和未预留容量。为旧BI/PB尚未完成的各25GiB减实际保留用量继续预留；为每个实际admitted未完成新臂各25GiB减已用量预留，另加5+20GiB。每15秒只查真实磁盘/已持有GPU安全，不轮询训练进度；容量满足时可准入pending臂，无法满足则等旧mask或新臂的COMPLETE/pidfd事件，不隐式改变训练曝光。

观察旧BI/PB各自目录的原子COMPLETE事件，在watch设置后做一次初始回读；真实完整3557/3节点并source-bound才释放其未来磁盘预算。新臂从cursor1实际resume、退出并完整commit后释放其未来预算，所有实际文件继续保留。pending臂的既有cursor1文件已计入free，不能重复或隐匿其已用空间。

仅实际准入时取得目标GPU2或3的原全局flock。原lock文件非空metadata**不改、不删、不truncate**：在确认其旧scope/UUID/exactowner是本次已退出失败queue且旧FD已释放后，以flockEXNB复用同inode作互斥，另在新root独占写successor journal，绑定原metadata精确bytes及新owner。不是自动接管未知/stale scope；任何不匹配、外部占用或锁竞争都不抢占。未准入GPU不被本controller占租；外来任务占pending卡仅继续等待，不杀它。已持有GPU出现foreign则只安全暂停自有任务。

## 最小实现

允许只读导入已冻结旧`ops/side_queue.py`的经过review且实测的纯JSON/PID/已修复spawn-cleanup helper；其bytes/stat先验证。**不调用旧Queue.run、lease、approval或原双臂disk_budget**，不monkeypatch全局ROOT、不修改原module。继承/组合可复用spawn（输出仍实际armroot，resume日志此前不存在）、paused/primary绑定与actualwait屏障；新controller自行实现新root run/外部AUTH、分阶段预算、successor lease与事件调度。SOURCE/config/fresh不变，无需新fresh provenance bridge；新外部ops均独立manifest/stat/bytes绑定。

新控制器只能启动显式`--resume-checkpoint ... --resume-complete-pin ...`，无fresh、first、evaluate-only或重试入口。所有created Popen抗中断登记，exactpidfd signal/实际wait后才释放FD；0/1旧controller不改不signal。失败全部保留，不自动restart新controller。

## 验证与产物

先新ops源码独立review、AST语法、真实部署/回读；再MAIN实际保存产物验证及单独授权；之后真实cursor1恢复。只签有限科学fullrow，保留新P backend/neighbor/serve gap。checkpoint质量、原三select完整覆盖与saved-only独立回算沿原计划，不开confirm/public。

实际结果逐项写新progress；“两臂primary通过”不等于“MAIN已通过/已续跑”，不得改写旧失败为成功。

## 已执行版本裁定（18:02UTC实际回执）

初版external staged-v1真实部署后，MAIN在rank coef bitwise门禁失败；原训练producer/独立primary保持exact。实际saved-cache单因素诊断定位为MAIN先构建rank再BCE，造成12项coef最大3.7252902985e-9差异，改为冻结的BCE→rank构建顺序时MAX0。该失败及诊断全保留，不降低任何数值阈值、不改训练source/cfg/fresh/weights/optimizer。

独立external namespace`/tmp/opencode/jev-clean-followup-staged-20261112-v2/`仅修正MAIN节点构建顺序、host masks在tensor前及版本路径，绑定prior-v1四份stat证据。新source经过枚举delta独立review/AST后实际install/readback、MAIN实际tensor回读通过，主代理另行原子AUTH。

新controller17:59:32UTC显式启动；18:02:03UTC事件回读证明两臂均从原cursor1恢复到50/63macro，macro0和explicit ownresume各1，exactworker存活。原BI/PB此时都已complete，因此无需未来预留；实际容量足够两个新臂并发，25GiB/20GiB不变、旧files/失败/leasebytes不删除。原失败queue并未重启，模型scope从未改变。后续完整训练/三select仍待真实事件验证。
