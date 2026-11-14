# Explicit staged follow-up continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development；用户no-extra-unit/smoke/hash/no-clean/no-commit优先，沿既有自主授权不逐臂追问。

**Goal:** 不改任何trainer/fresh/scope，从已验收cursor1恢复两臂，以显式分阶段资源预留处理旧controller预算失败。

**Architecture:** 新external控制root，精确绑定旧资源失败和全部暂停证据；新MAIN CPU回读后单独授权。只读复用旧冻结queue已review/实测的spawn、paused/primary、safe cleanup及纯JSON helper，新controller自管事件、预算、successor lease与newroot状态，不调用旧run/approval/lease/双臂budget。

**Tech Stack:** 原SYSTEM Python3 pidfd/inotify、原模型Python CPU/Torch saved-only与显式resume，现有H8002/3，不安装。

**Spec:** `2026-11-12-followup-staged-resume-design.md`，builder-check nothing important left。

## Global Constraints

- DATA_ROOT=`/tmp/opencode/jev-clean-trainer-20261112-followup-v1`；CONTROL_ROOT=`/tmp/opencode/jev-clean-followup-staged-20261112-v1`；新本地同名`-local`。
- 原9source/cfg/fresh/armROOT/PRIMARY/firstmacro/checkpoints/QUEUE_BLOCKED及失败MAIN日志不改不删，旧queue不重启。
- 25GiB每admitted arm剩余额度＋旧BI/PB剩余额度＋5GiB evidence＋20GiB shared不降低；pending明确未预留，不伪称两臂已排他资源准入。
- 仅STATE/GPU2、JPAIR/GPU3显式own-checkpoint cursor1 resume；固定曝光、数学、三select、3557call/3556positive-LR不变，无新fresh/first/eval/public入口。
- 旧failure必须精确为已记录resourceRuntimeError、全部child实际退出，0resume/0旧AUTH，任何model/primary失败或uncommitted tail均不得借资源例外放行。

## Review Focus

- 历史pausedledger pin在第一臂恢复后正常变化：只对尚未启动臂重验完整primary/history，common staticpins不能递归验证active ledger。
- 旧helper globalROOT始终DATA_ROOT，新控制输出只CONTROL_ROOT；不monkeypatchROOT、不调用旧Queue.run/lease/approval/diskbudget。
- 旧非空lease证据不改：同inodeflock＋精确旧owner已死/UUID/root/bundle匹配后新root successor journal；未知记录不接管。
- 预算事件与异步退出：watch旧BI/PB各自COMPLETE原子文件且先watch再初读，实际completed未来额度才释放；只有resource timer，无训练进度poll。
- spawn登记/cleanup/MAIN授权：复用已修复exactpidfd等待屏障，实际退出后才关lease，AUTH只由MAIN实际读回后原子发布，sameUID局限保留。

### Task1：三external ops实现、独立review与部署

**Files:** 新本地`source/main_readback.py,source/staged_queue.py,deploy_staged.py,progress.md`；报告`implementation-report.md`。新root所有源及manifest/config独占stat+bytes绑定，旧scope只读。

**Consumes:** 旧QUEUE_BLOCKED/QUEUE_LAUNCH/PRIMARY_READY/twoPRIMARY/最新cursor1 pin、旧OPSBUNDLE/source/helper、review过的213行MAIN readback源码。

**Produces:** install-only deploy CLI、MAIN saved-only直接回读回执、exact新AUTH schema、source-only staged continuation入口。

- [x] 冻结完整接口/接受的旧失败谓词/静态与per-arm动态pin分类，先断言报告再代码。
- [x] MAIN回读新root独立写report，允许精确资源controller失败但要求旧owner已死、全部first/primary退出、0resumes/0AUTH、无armBLOCKED；所有原实际tensor/coef/dispatch检查不弱化，0modelforward。v1因MAIN节点构建顺序失败保留，external v2窄修正后真实回读通过，未改模型或数值阈值。
- [x] stagedcontroller实现STATE先、若两臂预算都fit可并发；否则JPAIR透明pending，容量合格且GPU空闲后才租用/启动。原scope完整ownresumeindex1、无重放、safe cleanup。
- [x] v1全实现独立spec/quality＋安全边界review及AST；v2枚举delta独立review/AST，主代理真实install/readback。源码评审未代替actualMAIN/worker验收。

### Task2：真实MAIN回读、外部授权、续跑与冻结节点

- [x] 部署newCTRLv2不launch；actualMAIN saved-only回读成功后主代理读回，再单独原子12fieldAUTH；旧DATArootAUTH未创建。
- [x] newcontroller17:59:32UTC显式启动；18:02:03UTC增量事件准入核验exactowner/两worker live、successorlease、两容量fit、actualrestoredcursor1、宏0和explicitresume各1，已前进50/63macro。原0/1任务已自行complete，不重启；未推算全部完成。
- [ ] 文件事件/pidfd等待真实三节点/退出，saved-only独立回算，不重启旧watcher/eval/queue；只归档聚合与实际源，无privatecase/target/token/tensor下载。

## Self-review

保持原模型scope不变；resource失败例外只绑定具体已退出controller，非泛化忽略失败；不更改25/20预算。动态ledger不跨恢复阶段误守护，未准入GPU不占租，剩余工作同原科学协议。没有toy/unit/smoke/hash/清理或auto-retry步骤。
