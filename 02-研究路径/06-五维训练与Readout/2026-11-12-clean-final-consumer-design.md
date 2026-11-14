# Frozen clean terminal：确认与完整public consumer设计

## 当前输入与范围

八个预定seed42终端训练全部结束，三节点和完整权重来源已实际核验；dense→packed两项固定权重consumer各34case违约，eligible=false保留。依据已冻结candidate/confirm/public policy，CPU freezer已独占封存`/tmp/opencode/jev-clean-final-eval-20261112-v1/FROZEN_CANDIDATES.json`：baseline=C_P_CE，challenger=JPAIR_P_BCE_RANK；两者均causal/packed。同一对模型的public列表也已固定，不能根据confirm或public再换权重/runtime/域路由。

本项目仅实现这份固定两候选的**新消费者**：输入codec/loader/后端/资源准入、一次250-case确认、预声明两臂完整public。没有新训练、seed、optimizer、Teacher、查表、模型/语料额外hash、unit/smoke、cache/fusion/Graph/dummy token，所有旧资料保留，不发布/commit/push/上传或租卡。

原fullrow科学准入不是自动生产资格；须新source/consumer封存、actual codec与393 trained参数身份、真实确认/官方Suite/请求完整性证据。没有全public几何等价、cache-serving或官方单卡latency保证。

## 目录与文件职责

- 现有ROOT=`/tmp/opencode/jev-clean-final-eval-20261112-v1/`已有policy/freezer/候选，绝不重建或改候选；新source子目录必须独占创建并作为source安装claim。
- LOCAL=`/tmp/opencode/jev-clean-final-eval-20261112-local/`；四public模块`common.py`（stdlib冻结身份与源配置）、`backend.py`（纯renderer＋lazy GPU backend＋SDK ducktype engine）、`evaluate.py`（codec/confirm/analysis/public prepare/worker/merge-score分支）、`ops.py`（install-only与one-shot manager）。各模块stdlib顶层，无Torch提前import。
- 新`SOURCE_SNAPSHOT/config/INSTALL_COMPLETE`绑定已存在candidate manifest/policy/freezer source和actual新源/SDK/helper/stock/core/native/producer文件，不能用旧暴露模型readiness。cfg/源独占chmod444/readback，无循环self-pin。confirm/public输出独立子目录，所有partial/失败保留。

## 来源与模型加载

只两个manifest roles、真实terminal m3557 checkpoint；stock仍唯一原ModelScope Qwen3-0.6B。每GPU worker只处理一个role，按该role实际producer root导入自己的原8或9source，核对__file__/stat/小源码bytes，避免同一进程混用两套`bindings`模块。

producer Context保持原scope，不篡改arm或cfg；两者原本均causal packed。child launcher在进程启动前设置CUBLAS/offline/determinism；`backend.configure_role(role)`在**首次Torch import前**调用原configure_cuda，然后才可进入official runner。SDK `run()`自身会在load_engine之前import Torch，所以不得在Engine构造器里才尝试原configure_cuda；构造器只接受本进程已配置且effective flags仍吻合的context。

build_stock(42)只作CPU架构构造，接着load_saved_adapter(terminal,fresh=False)一次，393FP32/196pairs/head2×1024/served616237056/unused2048实际审计；requires_grad保留393但eval/no_grad/.gradNone，无optimizer或backward。不合并或降精度LoRA/head。GPU move后与整个role执行结束后纯adapter_state byte比较同一immutable CPU trained oracle，**不重载掩盖变更**。

后端只原nativeFA3 FullRow，phase为codec之外的confirm/public，micro规划同原stable-length padded-admission16384/max32/multiple8，实际全realT、position reset、EOS151645、head-all-tokens-then-gather。P为Candidate prefix，causal不使用cache；没有prefixfork/SDPA fallback。两个role均不需要state边界confirm sidecar。

## 输入codec、读出与SDK

统一使用原已绑定`pair_data_common.text/criteria_context/render_pair`与stock tokenizer（local_files_only/trust_remote_code=False），prefix与candidate独立分词后append EOS，与训练完全相同。纯field renderer和request parser分开：request只接收state/questions，不允许_evaluation/gold/benchmark进入backend；choice保留全部named criteria，noul按旧接口默认true:`Yes.`/false:`No.`且只按通用true/yes/1键读正概率，不按benchmark路由。

SDK响应与旧API一致：每field在完整原candidate margin上FP64 stable softmax；choice按原criteria顺序首次argmax，noul输出指定正键概率，返回唯一frozen model名与真实usage input_tokens。CPU tokenizer cache可明确≤2048，不能有跨请求prediction cache。每个请求在任何forward前完整校验type/criteria/token长度≤40960；不截断、不删候选、不按OOM/正确性改micro或回退模型。

新增**input-only完整D_select codec准入**：只在已冻结candidate与source安装后，以同一stock tokenizer/renderer对原select所有615knownfield/5190candidate重建prefix/suffix/rows，与原prepared IDs逐项完全一致；unknown90保留不重新解释raw。只tokenizer/JSON、0模型forward，不是synthetic/smoke，也不重跑旧模型。shared data/tokenizer相同，两role共用一次实际codec证据。

官方SDK为既有pinned source snapshot（目录没有.git，不伪造git-head现查），其当前runner/base/Suite/pipeline/pyproject与原SDK source statpins一致；declared revision仍25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d。部署绑定canonical resolved SDK tree和各py/source静态data文件，保留alias→resolved身份。不得继承旧模型/预测/readiness。

官方`load_engine`接受module:Class ducktype，无需backend顶层import SDK/Torch。public worker先configure，再调用SDK `run(engine,..., resume=False,warm=False,limit=None,keep=None,halt_on_device_error=True)`：warm是可选参数，明确关闭它以不增加两次synthetic warmup；首个真实请求首调用成本照计。**官方强制Suite.verify(strict=True)及exclusions_match保留**；仅此必要corpus哈希允许，SDK自带runner源码hash也不是model/corpus额外hash。

## D_confirm

只有candidate与new source/loader/codec/资源实际准入才读`data/confirm.jsonl`，原native split validation、完整250case各域50。不能改名select绕过旧select-only函数，必须新eval函数，统计算法沿旧case→domain等权；unknown不forward、unsupported整field保留，所有known全候选，不造目标或改q归一化。

输出每role全250 records、真实field/candidate/unknown/unsupported/token/native计数、mass/NLL/normalizedNLL/Brier/coverage/exact的真实分母及纯权重before-after证明。真实完整case roster，partial保留不冒称complete。CPU独立从raw logits/原confirm重算并比较两个固定roles，五域50、whole-case paired10000 bootstrap/seed20261112。只有覆盖完整且challenger−baseline点>0、单一冻结contrast区间lower>0才称改善确认，否则“未确认”；不换候选、runtime或public列表，不重训/再开confirm。

## 完整public

确认之后独立准备official Suite0.3：verify(strict=True)和exclusions校验，流式`rows(apply_exclusions=False)`保存四个round-robin35155shard，完整原payload/metadata/criteria顺序不变，不Token预计算排除正式latency，不读取gold作路由。mandatory verify之外不新增corpus/model hash。

每role依次使用四GPU、四独立worker同一frozen predictor；SDK输出逐request保持modelid/engine/run_id/payload，所有status必须ok。log progress发现任何non-ok即拒绝并保留partial，不按正确性retry；source能力超限也不删候选或换model。合并按official完整原顺序逐条对齐，140620唯一输出且shard无extra，再官方score_run，排除442后140178全ok/complete。两role输出完全独立，禁跨model补旧预测。

报告每个预声明role的Public Index、mean/p80完整request latency及四卡pipeline、实际realT/native counters；模型load不计入SDK per-request但first-use forward成本计入。四卡pipeline不是官方单卡latency admission，public完成不是prefix-cache或全几何等价。所有private raw/case/q/prediction/tensor只留远端，仓库仅聚合和actual public source。

## 资源、退出与预算

原8训练worker及terminal consumer都已退出，申请真实空闲自有0/1/2/3；NVIDIA UUID/process在锁前后/每spawn前后与15秒资源guard查询。沿同global lease inode复用：0/1精确匹配terminal-admission predecessor，2/3精确匹配原followup metadata并绑定已完成staged-v2 successor；所有前owner/child实际退出，flockEXNB、UID600/nlink1/nofollow/inode/旧metadata bytes全核对，旧bytes永不改，newroot journal绑定新owner。不接管unknown记录、不抢占，foreign只停自有。

一个新manager持四锁至确认与两public arms整个链的所有own children真实wait退出；confirm只短用0/1，随后public四卡。stdlib manager不得importTorch，child TERM/INT登记mask、PDEATH恢复、exactpidfd signal及unregistered Popen wait屏障采用已修复模式；0loop sleep/训练进度poll/自动restart旧队列。Source/CFG/候选改变、模型/数据/计数不符就停。

所有旧模型文件保留。新ROOT总cap20GiB，保留shared20GiB，因原训练及短时consumer全complete不再为其未来增长预留；currentfree已含其实际占用。为避免SDK status.json频繁atomic replace与临时文件hardlink造成future credit错误，**全程固定再预留完整新20GiB，不用own-used抵扣future额度**（保守双重预留），并按唯一inode观察cap；仅已知SDK `.tmp`/新commit `.writing`消失可容忍，source/cfg等静态身份另严格守护。不是系统级磁盘/GPU独占或抗恶意sameUID。

## 完成与授权层级

设计/source→实际source install→codec→真实trained393 loader/confirm→独立confirm analysis→officialSuite/shard→每role fullpublic→官方score/独立合并对账逐层记账。任何中途失败不自动重跑confirm或public，不改阈值。当前只候选已封存，consumer_ready=false，未打开confirm/public；必须主代理在source review后显式启动这个one-shot链。
