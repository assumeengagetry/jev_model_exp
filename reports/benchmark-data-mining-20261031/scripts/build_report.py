#!/usr/bin/env python3
"""Generate the concise presentation report; keep full statistics as an appendix."""
import argparse
import collections
import csv
import html
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
AREA=['knowledge','language','retrieval','tools','arts']
ZH=dict(zip(AREA,['知识与推理','语言理解','检索与分类','工具与自动化','艺术与人类偏好']))

def load(name):
    with (ROOT/'tables'/name).open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            if value in ['True','False']:row[key]=value=='True'
            elif value=='':row[key]=None
            else:
                try:row[key]=float(value) if any(s in value.lower() for s in ['.','e']) else int(value)
                except (ValueError,TypeError):pass
    return rows

def num(x,small=False):
    if x is None or isinstance(x,float) and not math.isfinite(x):return '—'
    if not small and abs(x)>=1e8:return f'{x/1e8:.2f}'.rstrip('0').rstrip('.')+'亿'
    if not small and abs(x)>=1e4:return f'{x/1e4:.2f}'.rstrip('0').rstrip('.')+'万'
    return f'{x:,.2f}'.rstrip('0').rstrip('.') if x!=int(x) else f'{int(x):,}'

def pct(x):return '—' if x is None else num(x,True)+'%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v).replace('|','／').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'

NOTES={
1:('对话和函数 schema；逐工具 yes/no，但最终要求整组工具选择完全正确。','字段级 yes 为 35.53%；每请求最多 37 个工具判断。字段表现接近不代表整请求表现接近。'),
2:('Query 加检索到的工具说明，逐工具判断相关性，最终按 query 排序。','未判断不是已确认不相关；仅在已标注匹配对中计算的 50% 没有代表全候选池类别平衡。'),
3:('对话前缀加 53 个 API 的完整 catalog，选择下一 API 名称。','catalog 本身就很长。当前 Pair 口径下全部 508 条都超过 4,096；反复把 catalog 编进 53 个分支造成主要计算负担。'),
4:('银行短消息的 77 类意图分类，真正的用户文本主要在 instructions。','原生 77 类各 40 条，类别完全均衡；展开后却是 1 个匹配、76 个不匹配。不要用 1.30% 判断源数据失衡。'),
5:('150 个意图加 OOS，共 151 个候选。','各 in-scope 类只有 30 条，OOS 有 1,000 条；OOS 是单个普通意图类的 33.33 倍。151 候选也会把短句放大成 83 万个分支。'),
9:('家居设备状态、命令和策略，混合设备成员、处理状态、数值与结果判断。','每请求约 18 题，主指标要求全部正确；不能给这些混合语义统一定义正类。'),
11:('每份完整合同判断 17 个固定法律假设：蕴含、矛盾或未提及。','123 份合同并不等于 2,091 份独立长文；合同是关联单位。4,096 Pair 上限仍放不下 13 份合同。'),
12:('对抗性 premise/hypothesis 三分类；输入主要在 instructions。','三类约各三分之一，长度整体很短。观察不到长上下文负担不代表自然语言推断本身简单。'),
20:('原始打油诗与扰动版本二选一；目标是原作，不是人类幽默偏好投票。','选择较长候选的描述性命中率 79.07%；同长均分。这说明构造痕迹很强，不能把高成绩完全解释为审美能力。原生评分按扰动类别宏平均。'),
21:('两条经替词编辑的新闻标题，选择人类平均评分更幽默者。','标题文本主要在候选，state 为空；只测 state 会错误地得到零长度。原始同分对已不在当前范围。'),
22:('MIDI 符号音符上下文，选择 129 个固定和弦音级集合之一；不是音频任务。','129 类中本样本只观察到 87 个 gold 类；有的类只有 1 条。129 分支重复编码长音符序列，占全榜 Pair token 约 56%。原生按歌曲宏平均。'),
23:('用户历史配色评分加两个新配色，预测用户更偏好哪一个。','5,000 请求来自 669 个用户；按用户宏平均。相同 history state 重复不代表同一个完整问题重复。'),
25:('GPQA Diamond 原始 4 选 1 问题，主要内容在 instructions。','当前有效 196 题，不沿用 198；state 为空。知识难度与题面长度不是同一维度。'),
28:('填空句与两个原始候选；二选一，不是语义正负分类。','当前归入知识与推理。长度很短，正确答案位置接近平衡。'),
29:('场景文本与 4 个续写候选。','选择较长候选仅约 24%，接近四选一水平；与 BPoMP 的长度痕迹不同。'),
30:('1,319 道原始数学题，各做 4 选和 10 选两个版本。','2,638 请求不是 2,638 道独立数学题；132 个答案组还共享候选。两轨需分别机会校正后平均，不能只把 14.29% 当统一随机基线。'),
31:('棋局 FEN、棋盘与所有合法着法，接受所有并列最佳着法。','每题平均 29.53 个合法着法；289 个局面有多个可接受最佳着法。实际正确候选为 7,629 个，不是简单的 5,000 个。'),
32:('故事、问题与可变数量原始候选。','较长故事是主要输入负担；当前 752 题。全部问题内容可能在 instructions，不能从 state 估计长度。'),
33:('同一源题对多个答案分别判断 yes/no。','字段 yes 为 38.34%；指标却是整套答案集合完全正确率。约 9.40 个字段的错误会累积，而不是简单逐字段平均。'),
36:('Query 加 BM25 top-32 文档，按相关概率排序。','220 个可回答 query 的 Joint P95 超过 2.66 万 token，但最大单 Pair 的 P95 约 3,713。联合处理和逐文档分支是完全不同的上下文压力。'),
37:('查询—商品相关性：Exact、Substitute、Complement、Irrelevant。','Exact 占 61.64%，Complement 仅 3.28%，约 18.79 倍。majority accuracy 不能代替 macro-F1。'),
38:('评论中是否出现某个类别×情感组合；每条评论分成多个问题块。','91,092 个二元字段中 yes 只有 508 个（0.56%）。全答 no 的字段正确率达 99.44%，但不能识别任何正组合；原生看逐评论 F1。'),
39:('给定实体跨度，对金融文档中的实体做三类情感判断。','Neutral 约 53%；文档内多个实体相关。不是实体抽取任务，不能把给定跨度当作模型已识别实体。'),
40:('包含 A/B/C、英语/阿拉伯语的多个轨道。','主榜只取 A-English：1,400 条中讽刺 200 条（14.29%）。本报告长度覆盖仍包括完整 4,600 条套件请求；不要把 11,600 个全部字段当 headline 样本量。'),
41:('主题加帖子，判断 against / favor / neutral。','三类相对均衡；但发现相同完整可见输入对应不同标签的组，应把这些可见输入歧义与模型错误分开。'),
42:('声明加临床试验指定完整章节，判断蕴含或矛盾。','矛盾 66.53%、蕴含 33.47%；5,500 请求反复使用 249 个试验 ID。试验是比单条声明更大的来源关联单位。'),
43:('Python 函数、给定输入与错误输出干扰项，选择正确输出。','候选数 2–10，平均 3.20；不是开放代码生成，也不是执行通过率。'),
44:('背景、给定概率及因果问题，yes/no。','yes 49.90%，但完整可见输入有 9.10% 的重复展示；均衡不代表每条输入独立。'),
45:('仅 HLE 纯文本多选子集，5–26 个候选。','不是完整 HLE；题面中位数不长，但最大 Joint 超过 1.23 万 token。该项公开 Skill 的模型中位数很低，不能用长度解释知识难度。'),
50:('群体政策意见与候选共识文本，接受并列最优。','较长候选的描述性命中率 56.53%；146 个样本有并列最佳。1,676 个面板分属 728 个政策问题。'),
56:('邮件场景的九问题输入；只有 verdict 是主榜评分目标。','headline 正反各 1,000。五个信号问题没有 gold；另一个极简辅助问题只用选项键，当前 Pair 表示把两候选都变为空字符串。此局限不影响有完整描述的主 verdict 字段。'),
57:('14 学科原始 MMLU-Pro，多选数实际 3–10。','平均 9.47 个候选，不应固定当作十选一。部分题面重复，但候选不同，因此不能由 state 重复率推出完整请求重复。'),
58:('23 个文件的混合 BBH 任务，含不同固定答案集和多选。','任务家族、候选数及位置差异很大；先按轨道观察，不能把所有答案键当同一种语义类别。'),
59:('原生成 prompt 的上下文加待评 response，noul true 表示存在幻觉。','幻觉正类 943/2,700（34.93%）；6 个模型回答共享 450 个源 prompt。匹配正例 50% 不等于幻觉占一半。'),
61:('Claim 加官方支持文档完整导言，SUPPORTED / NOT_SUPPORTED。','正反各 2,000，但这是给定 oracle 文档的判断，不包含开放检索；4,000 请求共享 1,592 个源问题。'),
62:('问题和工具 schema，选择 direct / tool_call / request_for_info / cannot_answer。','direct 在 gold 中为零；另外三类为 1,295 / 1,062 / 1,295。全部 1,295 个唯一 BFCL 来源 ID 都出现在当前 BFCL 集，不能把两项当作独立数据来源。'),
64:('漫画文字描述、场景和实体，加 5 个 caption 选原配字幕。','528 请求来自 141 个 contest；不是图像输入，也不是选最幽默字幕。相同场景 state 重复不等于完整候选题重复。'),
}

def build_details():
    # The current panel overrides the legacy catalog's ACOS whole-case wording.
    for filename in ['main-37.csv','benchmarks.csv','data-vs-public-skill.csv']:
        path=ROOT/'tables'/filename
        with path.open() as f:
            reader=csv.DictReader(f);headers=reader.fieldnames;rows=list(reader)
        for row in rows:
            if row['benchmark_id']=='38':row['metric']='per-review F1'
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    summary=json.load(open(ROOT/'summary.json'));bench=load('main-37.csv');bench.sort(key=lambda r:(AREA.index(r['area']),r['benchmark_id']));byid={r['benchmark_id']:r for r in bench}
    areas=load('areas.csv');cover=load('coverage.csv');native=load('native-classes.csv');clusters=load('source-clusters.csv');content=load('input-content.csv');ambiguity=load('input-label-ambiguity.csv');tracks=load('tracks.csv');correlations=load('data-public-score-correlations.csv');scores=load('leaderboard-benchmark-summary.csv');board=json.load(open(ROOT/'leaderboard-public-context.json'));scope=json.load(open(ROOT/'source-overlap-summary.json'))
    lines=[]
    def add(s=''):lines.append(s+'\n')
    def fig(stem,title):add(f'![{title}](figures/{stem}.png)\n\n[打开 SVG 放大查看](figures/{stem}.svg)')
    add('# JEV Decision Index：37 项 benchmark 全量数据画像')
    add('**日期：2026-10-31｜公开面板：decision-index-0.3｜统一 tokenizer：Qwen3-0.6B**')
    add('这份报告分析的是**实际公开评测输入**，不是训练池，也不是只抽几条样例估计。37 项主榜和 6 项归档展示项分开；37 项的全部请求均进入统计。')
    add('## 先看最重要的结论')
    add(f'1. **全榜约 11 万请求、27 万道问题、219 万候选匹配对。** 分别约为 {num(summary["requests"])}、{num(summary["fields"])}、{num(summary["candidate_pairs"])}，完整整数见明细。候选展开以后，已知标签中匹配正例只有 **11.24%**，正例占编码 token 仅 **7.17%**；但这不是原生语义正类比例。')
    add('2. **4k 并不能完整覆盖所有任务。** 当前真实 Pair 输入还有 **590 条请求**放不下，其中 **508 条全是 API-Bank**。8k 仍有 **3 条**超长请求；16k 还有 **1 条**。32k 可覆盖本轮所有 Pair 输入。Joint 联合内容则有不同的长尾，不能混用。')
    add('3. **计算量极不均衡。** POP909 仅 2,000 请求，却占全榜 Pair token 的 **55.76%**；与 API-Bank 合计约 **73.68%**，两项公开分数权重合计只有 **5.73%**。这比按请求数均匀分配优化时间更值得关注。')
    add('4. **ACOS 的真正 yes 率仅 0.56%。** 相同数据展开成候选匹配后正负却是 50:50；其全答 no 的逐字段正确率可达 99.44%，但识别不出任何正组合。')
    add('5. **可见构造痕迹很强的任务存在。** BPoMP 仅“选择 token 较长的候选、同长均分”的描述性命中率就是 **79.07%**；Habermas 同一参照为 **56.53%**。这不是新模型效果，也不是官方宏平均成绩，而是候选长度与答案的结构性关联。')
    add('6. **名义样本数明显大于独立来源数。** RAGTruth 的 2,700 回答来自 450 个源 prompt；GSM8K 的 2,638 请求来自 1,319 题；ACOS 的 1,565 请求来自 400 评论；BPoMP 的 5,000 请求来自 811 原诗。')
    add('7. **两个工具 benchmark 的来源高度重合。** 当前 When2Call 的 1,295 个唯一 BFCL 来源 ID，全部出现在当前 BFCL。它们提供不同任务目标，但不能视为互相独立的新来源。')
    add('8. **还发现少量“同一可见输入、不同标签”。** VAST 为 10 组、21 次请求展示；BBH 为 2 组；NLI4CT、HoVer 各 1 组。另有 Pair 表示造成的辅助字段语义丢失，详见后文。')
    add('9. **长不等于难，短也不等于容易。** 接入榜单 247 个模型的已有逐项公开分数后，37 项之间的长度 P95 与完整覆盖模型的 Skill 中位数相关接近零；这是跨任务观察，不代表长文本在同一任务里不会掉分。')
    add('## 1. 统计范围和长度口径')
    add(table(['范围','数量','说明'],[['当前主榜','37 项','五大领域；下文所有主图均使用此范围'],['主榜有效请求',num(summary['requests']),'已应用当前版本选择与统一排除'],['主榜问题字段',num(summary['fields']),'一个请求可有多道问题'],['主榜候选匹配对',num(summary['candidate_pairs']),'一道 K 候选题展开为 K 对'],['其中没有可确认硬标签的 Pair',num(summary['unknown_pairs']),'检索未判断候选及无 gold 的辅助字段；不记为负例'],['归档完整范围',f'{num(summary["raw_cached_requests"])} 请求 / 43 项','6 项仅展示，不进入主榜统计'],['主榜排除前后',f'{num(summary["main_in_edition_requests"])} → {num(summary["requests"])}','当前主榜移除 433 条；不要再机械扣一次旧排除数']]))
    add(table(['长度口径','具体测量','如何使用'],[['State','仅 state 的统一 tokenizer 长度','解释文档体积；不能单独当输入长度'],['Joint 联合请求','紧凑 JSON 中的 state + 全部 questions，包括 instructions 与全部 criteria','跨任务统一内容长度；不是所有模型实际聊天模板'],['实际 Pair','现有分类器 prefix + candidate suffix + EOS 的缓存长度','当前候选匹配分类器的真实单次输入长度'],['Request Pair max','一条请求的全部候选 Pair 中的最大长度','判断整条请求是否能全部容纳'],['Pair token 总量','所有候选 Pair 长度相加','不共享前缀的编码负担；不是墙钟时间']]))
    add('**P50 / P95 / P99** 分别表示中位数、第 95 / 99 百分位。本文分位数按请求或 Pair 的实际完整数据计算；全部展示最多两位小数。完整请求编码不包含 expected、scoring 或来源信息。当前实测题型只有 choice / noul，**没有 score 输入题型**。')
    add('统计依据是服务器已落盘的公开 suite 和现有 Pair 缓存。GSM8K 是沿用官方语义的本地重建候选；不同引擎、tokenizer 或模板的长度会变化。当前 Full 来自私有任务，这份报告不把公开画像冒充私有题面统计。细节和一手链接见 [SOURCES.md](SOURCES.md)。')
    add('## 2. 五大领域：样本量、长度、候选量和分数权重')
    add(table(['领域','项数','请求','问题','Pair','Joint P95','分数权重','Pair token 占比'],[[a['area_zh'],a['benchmarks'],num(a['requests']),num(a['fields']),num(a['pairs']),num(a['joint_p95']),pct(a['weight_pct']),pct(a['pair_token_share_pct'])] for a in areas]))
    fig('07-five-domain-data-profile-radars','五领域数据画像雷达图')
    add('雷达图每个轴按五领域最大值归一化，仅比较**数据形状**：请求量、Joint P95、候选数/题、题数/请求、Joint 超 4k 比例、Pair/Joint 总 token 比。图形面积不是模型能力或综合难度评分。')
    fig('08-official-weight-vs-sample-token-burden','官方权重与数据计算负担对照')
    add('**不能按请求量或 Pair 数理解榜单权重。** 同领域 ★项权重 1.20，普通项 1.00，再乘固定领域权重。ContractNLI 只有 123 份合同，仍约占公开总指数 2.75%；POP909 的分支很多，也不会因此增加指数权重。')
    add('## 3. 输入长度：整体覆盖率与最值得处理的长尾')
    add(table(['上下文上限','Pair 整请求覆盖率','仍放不下的请求','Joint 内容覆盖率','仍放不下的 Joint'],[[num(c['cap'],True),pct(c['pair_request_covered_pct']),num(c['request_pair_over_count']),pct(c['joint_covered_pct']),num(c['joint_over_count'])] for c in summary['coverage'] if c['cap']>=512]))
    add('**显示为 100% 不总是零遗漏。** 例如 8k 的 Pair 覆盖率四舍五入后为 100%，实际还有 3 条；16k 还有 1 条。因此这里同时给整数超限数量，避免为了少保留小数而掩盖长尾。')
    add(table(['整体分布','P50','P90','P95','P99','最大值'],[['Joint / 请求',num(summary['joint_p50']),num(summary['joint_p90']),num(summary['joint_p95']),num(summary['joint_p99']),num(summary['joint_max'])],['Request Pair max',num(summary['request_pair_max_p50']),num(summary['request_pair_max_p90']),num(summary['request_pair_max_p95']),num(summary['request_pair_max_p99']),num(summary['pair_max'])],['Pair / 候选',num(summary['pair_p50']),num(summary['pair_p90']),num(summary['pair_p95']),num(summary['pair_p99']),num(summary['pair_max'])]]))
    fig('01-length-quantiles-heatmap','37 项输入长度热力图')
    fig('02-pair-request-coverage','实际 Pair 整请求覆盖率热力图')
    fig('03-joint-request-coverage','Joint 联合内容覆盖率热力图')
    add('### 4k 的 590 条 Pair 超限请求来自哪里')
    over=[c for c in cover if c['active'] and c['cap']==4096 and c.get('request_pair_over_count',0)>0];over.sort(key=lambda c:c['request_pair_over_count'],reverse=True)
    add(table(['Benchmark','超限请求','该项覆盖率','主要原因'],[[c['name'],num(c['request_pair_over_count']),pct(c['request_pair_covered_pct']),{3:'完整 53 API catalog',2:'个别工具文档极长',11:'完整合同',22:'个别音符窗口长',36:'个别 query/文档长',62:'工具 schema 长',33:'长段落',45:'极长题面/候选',1:'长 schema'}.get(c['benchmark_id'],'长尾输入')] for c in over]))
    add('**数据配置上的直接判断：**短输入主力和长尾回退应分开规划。512/1k/2k 是吞吐分桶，不是全榜统一硬裁剪。当前 Pair 需要至少覆盖 22,659 token 的极端请求；32k 有余量。Joint 最大为 32,575，接近 32k，换成带聊天模板的联合引擎后仍需另算模板余量，不能直接宣称它也必然完整容纳。')
    fig('10-joint-length-distribution','Joint 长度分桶')
    fig('11-pair-request-length-distribution','最大 Pair 长度分桶')
    add('## 4. 正反比例：必须区分原生语义与候选匹配')
    add('一个二元问题的 gold 若为 no，展开候选后，“no”是一个匹配正例，“yes”是一个匹配负例。因此 **yes/no 比例、匹配/不匹配比例、情感 Positive/Negative 比例不是同一个统计量**。普通 K 选 1 的匹配密度约为 1/K；NLI 三类不能随便合成正负。')
    native_examples=[]
    for bid,label,meaning in [(1,'yes','应调用工具'),(33,'yes','候选答案正确'),(38,'yes','评论出现该类别×情感'),(40,'yes','讽刺；仅 A-English'),(42,'Entailment','试验证据蕴含声明'),(44,'yes','因果问题回答 yes'),(56,'phishing','主 verdict 为 phishing'),(59,'true','回答存在幻觉'),(61,'SUPPORTED','证据支持 claim')]:
        xs=[r for r in native if r['benchmark_id']==bid and (r['scope']=='headline' if bid in {40,56} else True)]
        n=sum(r['count'] for r in xs);positive=sum(r['count'] for r in xs if r['label']==label)
        native_examples.append([byid[bid]['name'],meaning,num(positive),num(n-positive),pct(100*positive/n),pct(byid[bid]['match_positive_pct'])])
    add(table(['Benchmark','语义正类定义','语义正','语义负','原生正类率','匹配正例率'],native_examples))
    add('**ACOS 是最典型的例子：** 508 个语义 yes 对 90,584 个 no，而匹配展开后正反各 91,092。先改变 Pair 正负采样，并不会自动改变模型看到的 ACOS 原生 yes 稀疏性；要同时记录问题级类别、候选密度与 token 曝光。')
    add('**iSarcasm 主榜：**14.29% 的讽刺正类；如果总预测 no，准确率有 85.71%，但讽刺 F1 为零。**RAGTruth：**正类是“有幻觉”，不是“回答可靠”；当前机会校正基线是全判幻觉的 F1 51.77%，不是公平硬币的 50%。')
    add('### 多分类的原生类别分布')
    multi=[]
    for bid in [4,5,11,12,22,37,39,41,62]:
        c=collections.Counter()
        for r in native:
            if r['benchmark_id']==bid:c[str(r['label'])]+=r['count']
        n=sum(c.values());top=c.most_common();desc='；'.join(f'{k}: {pct(100*v/n)}' for k,v in top[:5])
        if bid==4:desc='77 类各 40 条（各 1.30%）'
        if bid==5:desc='OOS 1,000 条（18.18%）；150 个普通意图各 30 条（各 0.55%）'
        if bid==22:desc='129 个候选类中观察到 87 类；最大类 GM 102 条（5.10%），最小已观察类 1 条'
        multi.append([byid[bid]['name'],len(c),desc])
    add(table(['Benchmark','已观察语义类数','分布'],multi))
    add('When2Call 仍给出 direct 候选，但本样本 gold 为零；不能因为该类没有样本就推断现实请求里 direct 不存在。原生类别完整表见 [native-classes.csv](tables/native-classes.csv)，原始答案键及轨道表见 [answer-labels-by-track.csv](tables/answer-labels-by-track.csv)。')
    fig('04-matching-label-composition','候选匹配标签构成')
    fig('06-matching-vs-semantic-rates','匹配正例与字面 yes/true 对照')
    add('上图右列只识别字面键 yes/true，不猜测 A/B 的语义；所以 CLadder、HoVer 等应使用上面的显式语义表，而不是把图中灰色当成没有语义标签。各任务原生标签的定义始终以表中说明为准。')
    add('### 检索与没有标签的辅助题')
    add(table(['范围','有确认答案的字段','未判断/无标签字段','说明'],[[byid[bid]['name'],num(byid[bid]['known_fields']),num(byid[bid]['unknown_fields']),{2:'21,919 个候选相关性判断，仅 924 个有明确正相关答案',36:'7,040 个候选相关性判断，仅 447 个有明确正相关答案',56:'九问题中四个有 gold、五个信号没有 gold；主榜只用 verdict'}[bid]] for bid in [2,36,56]]))
    add('ToolRet / BRIGHT 在已标注字段里几乎只有相关文档；剩余候选是 **unjudged**，不能把它们写成可靠负例。nDCG 的候选池召回限制另列在 [retrieval-pools.csv](tables/retrieval-pools.csv)，按 query 解释，不能仅用候选池中的相关项重建理想分母。')
    add('## 5. 计算负担：候选数、问题数和重复编码')
    top=sorted(bench,key=lambda b:b['pair_sum'],reverse=True)
    add(table(['Benchmark','请求','平均候选/题','Pair token 总量','全榜 Pair token 占比','Pair/Joint token 比','公开权重'],[[r['name'],num(r['requests']),num(r['options_mean']),num(r['pair_sum']),pct(100*r['pair_sum']/summary['pair_total_tokens']),num(r['pair_to_joint_token_ratio'])+'×',pct(r['weight_pct'])] for r in top[:10]]))
    add(f'合计 Joint 内容约 **{num(summary["joint_total_tokens"])} token**，当前候选 Pair 展开编码约 **{num(summary["pair_total_tokens"])} token**，相差 **{num(summary["pair_to_joint_token_ratio"])}×**。这个比值同时包含格式差异、候选展开和 state 复用，并不等于某个共享前缀实现已经测得的加速倍数。')
    add('**优先方向：**POP909 和 API-Bank 最值得研究共享 state/candidate 编码、联合候选打分与前缀复用；CLINC/BANKING 则是大量很短分支。不能用同一 batch/request 策略覆盖这三种形态，也不应为省计算悄悄删除原始候选。这里给的是后续配置建议，本轮未修改训练或推理。')
    fig('09-candidates-fields-and-pairs-per-request','候选数、每请求题数和分支数')
    fig('13-total-candidate-pair-vs-joint-tokens','展开编码负担')
    add('### 整组完全正确指标会放大字段错误')
    add(table(['任务','平均字段/请求','最多字段','原生要求'],[[byid[i]['name'],num(byid[i]['fields_per_request_mean']),num(byid[i]['fields_per_request_max']),NOTES[i][0]] for i in [1,9,33]]))
    add('仅作数学示意：若每字段正确率为 95%，且错误独立，15 个字段全部正确约为 **46.33%**，18 个约为 **39.72%**。真实错误可能相关，所以这不是实际成绩预测；它说明字段准确率与整案例准确率为什么不能互换。ACOS 又不同，它按评论 F1，不应套用整组 exact。')
    add('## 6. 来源关联、重复展示和可见输入歧义')
    add(table(['Benchmark','来源关联键','来源组数','请求成员数','最大组','解释'],[[c['name'],c['cluster_key'],num(c['clusters']),num(c['request_memberships']),num(c['largest_cluster']),'一个请求可关联多个 trial，成员数不是独立请求数' if c['cluster_key']=='trial_ids' else '相同来源组可含多条请求'] for c in clusters]))
    add('**When2Call 与 BFCL：**全部 1,295 个唯一来源 ID 交集为 1,295，覆盖 When2Call 来源的 **100%**。输入格式和目标不同，所以没有发现相同非平凡 state 文本并不能否定这种关联。源 ID 连接比“直接比较 JSON 文本”更有解释力。')
    duplicates=sorted([r for r in content if r['active'] and r['duplicate_full_payload_pct']>0],key=lambda r:r['duplicate_full_payload_pct'],reverse=True)
    add(table(['Benchmark','请求','不同完整可见输入','重复展示比例'],[[r['name'],num(r['requests']),num(r['unique_full_payloads']),pct(r['duplicate_full_payload_pct'])] for r in duplicates]))
    add('完整输入相同的定义是 state + 全部 questions 相同，忽略答案与元数据。**state 相同不是样本重复**：例如 ANLI 的 state 都是空对象，真正内容在 instructions；New Yorker 共享漫画场景而 caption 候选不同。反过来，CLadder 9.10% 的完整输入重复是真实的重复展示，不能当作独立新题。')
    conflicts=[r for r in ambiguity if r['full_payload_conflicting_groups'] or r['actual_pair_conflicting_groups']]
    add(table(['Benchmark','完整输入标签冲突组','涉及请求','实际 Pair 标签冲突组','涉及 Pair 展示','解释'],[[r['name'],num(r['full_payload_conflicting_groups']),num(r['full_payload_conflicting_requests']),num(r['actual_pair_conflicting_groups']),num(r['actual_pair_conflicting_presentations']),'仅极简辅助字段；不是主 verdict' if r['benchmark_id']==56 else '同一可见输入或当前表示可能对应不同目标'] for r in conflicts]))
    add('**这里的“冲突”不自动等于上游标错。** 它可能来自原标签歧义、任务转换省略条件，或同一候选在不同完整选项集合中的角色变化。VAST 等是完整可见输入相同而 gold 不同；MMLU-Pro / When2Call 的部分冲突只在现有 Pair 表示中出现，不能直接归罪原始多选题。')
    add('**已确认的一个表示局限：**PhishNChips 的 `verdict_alt_minimal` 只有 phishing/legitimate 选项键、没有说明文字。现有候选匹配缓存只取说明文字，两个候选变成同一个空字符串，2,000 条该辅助题因此不能靠当前 Pair 输入区分两种选项。主评分 `verdict` 有完整的不同说明文字，缓存候选也不同，**没有这个问题**。本轮只记录发现，没有修改缓存或影响正在运行的会话。')
    add('### 只测 state 会漏掉大量真实输入')
    empty=[r for r in content if r['active'] and r['empty_state_pct']==100]
    add('以下任务的 state 全部为空，但真实题面在 instructions 或候选：'+ '、'.join(r['name'] for r in empty)+'。因此旧的 state 字符长度或混合 proxy_tokens 不适合作为统一输入长度分布。')
    add('语言覆盖另外保留了元数据 locale 分布和阿拉伯/CJK 字符出现率。字符出现只是脚本线索，不等于整条输入的语言分类；iSarcasmEval 的阿拉伯文轨道也不能混入 A-English headline。')
    add('## 7. 候选长度与答案位置：数据捷径有多强')
    add(table(['Benchmark','选最长候选命中率','选最短候选命中率','第一位置可接受率','说明'],[[byid[i]['name'],pct(byid[i]['longest_candidate_reference_pct']),pct(byid[i]['shortest_candidate_reference_pct']),pct(byid[i]['first_answer_pct']),{20:'二选一；长度痕迹最突出',50:'可并列最佳；请求微平均参照',57:'不同学科 3–10 选',29:'四选一；长度参照接近随机',62:'第一位置 direct 从未为 gold',5:'第一类不是多数类 OOS'}[i]] for i in [20,50,57,29,62,5]]))
    add('计算只看**候选文本 token 数**；最长或最短有并列时按并列候选均匀选择，正确答案允许官方 accepted 集合。它是数据结构参照，不是训练后的模型分数。BPoMP 原生按扰动类别宏平均，因此 79.07% 的请求微平均也不能直接当作官方成绩。')
    add('位置分布用于识别数据的答案先验，不表示应该给模型写逐 benchmark 固定答案。固定语义类（如 NLI、intent）与随机位置类（如一般 MCQ）分开解释；混合轨道的答案键熵也不当作语义类别熵。')
    fig('12-answer-position-and-candidate-length-bias','答案位置和候选长度关联')
    add('## 8. 与已有模型表现的关联：能说什么、不能说什么')
    add(f'读取官方榜单 **{board["generated_utc"]}** 快照，共 **{len(board["models"])} 个模型条目**，没有新增模型推理。每项取已有公开 Skill；完整覆盖模型的定义是该项 coverage 至少 99.99%。37 项之间使用 Spearman 相关，只作描述。')
    selected=[r for r in correlations if r['model_band']=='all' and r['outcome']=='complete_skill_median']
    labels={'joint_p95':'Joint P95','request_pair_max_p95':'实际最大 Pair 的 P95','options_mean':'平均候选数','fields_per_request_mean':'平均题数/请求','match_positive_pct':'匹配正例率','pair_to_joint_token_ratio':'Pair/Joint token 比','first_answer_pct':'第一位置答案比例'}
    add(table(['数据特征','与完整覆盖模型 Skill 中位数的相关','项数'],[[labels[r['data_feature']],num(r['spearman'],True),r['benchmark_count']] for r in selected]))
    add('这些数整体很弱：**跨任务的长度、候选数或标签密度，都不能单独解释模型表现。** 任务技能、原生指标、训练覆盖、模型结构和题面构造一起变化。分模型规模（<2B、2–10B、≥10B）的表也已保存；它们仍不是控制变量实验，更不是“长度没有影响”的证明。')
    score_rows=[s for s in scores if s['band']=='all' and s['benchmark_id'] in byid and s['complete_skill_median'] is not None];score_rows.sort(key=lambda r:r['complete_skill_median'])
    add(table(['当前较难项','完整覆盖模型 Skill 中位数','参与模型','完整覆盖模型'],[[r['name'],num(r['complete_skill_median']),r['models'],r['complete_models']] for r in score_rows[:10]]))
    hle=[r for r in load('leaderboard-model-benchmark.csv') if r['benchmark_id']==45 and r['coverage_pct']>=99.99]
    hle_raw=statistics.median(r['raw'] for r in hle)
    hle_chance=100*board['benchmarks']['45']['baseline']['value']
    add(f'HLE 的典型输入不特别长，但 {len(hle)} 个完整覆盖模型的原始准确率中位数仅 **{pct(hle_raw)}**，低于其机会基线 **{pct(hle_chance)}**。它的 Skill 中位数为 0；由于 Skill 会把低于随机的部分截到 0，**Skill=0 不能解释成原始表现恰好接近随机**。POP909 和 API-Bank 虽然编码很重，难点也不只有长度。逐项模型与数据连接表见 [data-vs-public-skill.csv](tables/data-vs-public-skill.csv)。')
    add('## 9. 37 项逐项数据卡')
    add('以下每张卡都使用完整公开输入范围。**iSarcasm 的长度是全部套件输入，headline 标签另取 A-English；Phish 的长度含九问题，但主指标只取 verdict。** 原生指标已按当前面板覆盖旧 catalog 文案，ACOS 使用逐评论 F1。')
    for area in AREA:
        add('### '+ZH[area])
        rows=[r for r in bench if r['area']==area]
        add(table(['Benchmark','请求','问题','平均候选','Joint P50 / P95','实际 Pair 最大值 P95','匹配正例率'],[[r['name'],num(r['requests']),num(r['fields']),num(r['options_mean']),f'{num(r["joint_p50"])} / {num(r["joint_p95"])}',num(r['request_pair_max_p95']),pct(r['match_positive_pct'])] for r in rows]))
        for r in rows:
            bid=r['benchmark_id'];description,note=NOTES[bid];metric='per-review F1' if bid==38 else r['metric']
            add(f'#### {bid}. {r["name"]}')
            add(f'**任务与指标：**{description} 原生指标：{metric}；公开权重约 **{pct(r["weight_pct"])}**。')
            add(f'- **数据规模：**{num(r["requests"])} 请求、{num(r["fields"])} 问题、{num(r["pairs"])} Pair；每请求平均 {num(r["fields_per_request_mean"])} 题、最多 {num(r["fields_per_request_max"])} 题；每题平均 {num(r["options_mean"])} 候选、最多 {num(r["options_max"])}。')
            add(f'- **Joint 长度：**P50 **{num(r["joint_p50"])}**；P95 **{num(r["joint_p95"])}**；P99 **{num(r["joint_p99"])}**；最大 **{num(r["joint_max"])}** token。')
            add(f'- **实际 Pair：**请求最大分支的 P50 **{num(r["request_pair_max_p50"])}**、P95 **{num(r["request_pair_max_p95"])}**、P99 **{num(r["request_pair_max_p99"])}**；该项最长单 Pair **{num(r["pair_max"])}** token。')
            add(f'- **匹配标签：**正 {num(r["positive_pairs"])}、负 {num(r["negative_pairs"])}、未标注 {num(r["unknown_pairs"])}；已标注内正例 **{pct(r["match_positive_pct"])}**，正例 token 份额 **{pct(r["positive_token_share_pct"])}**。')
            add(f'- **编码量：**Pair 共 **{num(r["pair_sum"])} token**，约为本项 Joint 总 token 的 **{num(r["pair_to_joint_token_ratio"])}×**。')
            cv={c['cap']:c for c in cover if c['benchmark_id']==bid}
            add('- **上下文余量：**'+ '；'.join(f'{num(cap)} 上限仍有 {num(cv[cap]["request_pair_over_count"])} 条 Pair 请求超限' for cap in [1024,2048,4096,8192])+ '。')
            add('- **数据解读：**'+note)
    add('## 10. 后续数据与模型配置优先级')
    add(table(['优先级','要处理的数据特征','建议方向','不应混淆的结果'],[['1','POP909 / API-Bank 的展开编码','优先研究共享前缀、联合候选打分、按 token 而非请求数分桶','统计编码量不是已实现的加速倍数'],['2','ACOS、iSarcasm、RAGTruth 的原生正例稀疏','原生类别曝光、候选 Pair 比例和 token 曝光同时记录；完整问题监督与候选匹配监督分开比较','把 Pair 调成 50:50 并不意味着原生正类已平衡'],['3','4k 以上长尾及模板差异','短主力桶 + 不裁剪的长尾路径；不同引擎分别确定上下文','8k 显示 100% 仍可能漏极少数请求'],['4','同源组和重复展示','按原诗/歌曲/用户/临床试验/源 prompt 等分组；区分重复请求与新问题','同 state 不等于同完整输入'],['5','并列最佳与没有标签的字段','保留 accepted 集合；unjudged 单列；不为凑二分类把 null 改成 no','模型没选代表 expected 不一定就是错'],['6','候选长度、位置和表示信息丢失','把捷径参照和输入表示可辨识性一起列入结果解释','公开高分不自动证明掌握目标技能'],['7','跨任务难度','先看逐项原生指标和同任务长度切片，再综合五维','跨 37 项弱相关不能推导同任务无长度效应']]))
    add('这些是由数据画像导出的后续建议；本轮没有改变其他会话正在使用的训练数据、模型、候选缓存、配置或进程。')
    add('## 附录 A. 不计入当前主榜的六项')
    add(table(['归档项','请求','Joint P95','实际 Pair 请求 P95'],[[r['name'],num(r['requests']),num(r['joint_p95']),num(r['request_pair_max_p95'])] for r in load('archived-6.csv')]))
    add('归档项用于保留范围说明，未混入五维雷达、37 项主图、整体正负比例或上下文覆盖率。')
    add('## 附录 B. 完整图表与数据入口')
    for stem,title in [('05-semantic-yes-no-composition','字面 yes/no 与其他标签构成'),('14-five-domain-request-coverage','五领域上下文覆盖曲线'),('15-feature-correlations-not-causation','数据特征之间的相关性'),('16-candidate-pair-length-distribution','全部候选 Pair 长度分桶')]:
        add(f'- [{title}（PNG）](figures/{stem}.png) / [SVG](figures/{stem}.svg)')
    add(table(['文件','内容'],[['[main-37.csv](tables/main-37.csv)','37 项完整规模、P50/P90/P95/P99/最大值、候选、标签、位置与 token 特征'],['[coverage.csv](tables/coverage.csv)','每项 128–40,960 token 的覆盖率及超限数量'],['[length-buckets.csv](tables/length-buckets.csv)','State / Joint / Request Pair max / Pair 四种长度分布'],['[native-classes.csv](tables/native-classes.csv)','原生语义类与主/辅助评分范围'],['[tracks.csv](tables/tracks.csv)','各轨道规模、长度和标签'],['[pair-label-by-length.csv](tables/pair-label-by-length.csv)','各长度桶的匹配正负与未标注比例'],['[source-clusters.csv](tables/source-clusters.csv)','原诗、歌曲、用户、trial、源 prompt 等聚类'],['[input-content.csv](tables/input-content.csv)','完整可见输入重复率、空 state、脚本与结构形式'],['[input-label-ambiguity.csv](tables/input-label-ambiguity.csv)','同输入标签冲突与当前 Pair 表示冲突'],['[requests.csv.gz](tables/requests.csv.gz)','逐请求的长度、候选、题数与轨道；用本轮序号'],['[fields.csv.gz](tables/fields.csv.gz)','逐题的候选密度、长短参照、最大 Pair 长度和已知标签'],['[leaderboard-benchmark-summary.csv](tables/leaderboard-benchmark-summary.csv)','247 模型已有公开表现，按参数规模分组'],['[data-public-score-correlations.csv](tables/data-public-score-correlations.csv)','数据与公开表现的描述性相关'],['[source-scope.csv](tables/source-scope.csv)','排除前后实际请求数量'],['[SOURCES.md](SOURCES.md)','输入口径、当前版本与官方定义链接']]))
    add('### 怎样复用本轮脚本')
    add('`scripts/analyze.py` 读取原始请求和已存在的分词缓存；`supplement.py` 补原生标签/来源关联并聚合轨道；`ambiguity.py` 统计相同输入的标签歧义；`visualize.py` 生成 PNG/SVG；`build_report.py` 从明细生成本文及浏览版。脚本没有模型调用，只有 CPU 数据分析。')
    add('## 来源')
    add('- [Decision Index 官方榜单](https://huggingface.co/spaces/multimodalart/jev-decision-index)；[当前榜单 JSON](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json)。')
    add('- [官方套件说明](https://github.com/apolinario/decision-index/blob/main/docs/suite.md)；[37 项面板与权重](https://github.com/apolinario/decision-index/blob/main/decision_index/data/index-0.3.json)。')
    add('- [版本选择与子集](https://github.com/apolinario/decision-index/blob/main/decision_index/editions.py)；[评分器](https://github.com/apolinario/decision-index/tree/main/decision_index/scoring)。')
    add('- 逐项构造器、当前缓存位置和模板见 [SOURCES.md](SOURCES.md)。')
    markdown='\n'.join(lines);(ROOT/'APPENDIX.md').write_text(markdown)
    write_html(markdown,summary,bench,filename='details.html')
    print('Generated APPENDIX.md and details.html; benchmark cards:',len(bench))

def build_brief():
    # Keep the previous complete report available without putting it in front
    # of a person who only needs the presentation conclusions.
    if not (ROOT/'APPENDIX.md').exists() and (ROOT/'REPORT.md').exists():
        (ROOT/'APPENDIX.md').write_text((ROOT/'REPORT.md').read_text())
    if not (ROOT/'details.html').exists() and (ROOT/'index.html').exists():
        previous=(ROOT/'index.html').read_text().replace('href="REPORT.md"','href="APPENDIX.md"')
        (ROOT/'details.html').write_text(previous)
    summary=json.load(open(ROOT/'summary.json'))
    bench=load('main-37.csv');byid={row['benchmark_id']:row for row in bench}
    covers={row['cap']:row for row in summary['coverage']}
    token_share=100*sum(byid[i]['pair_sum'] for i in [22,3])/summary['pair_total_tokens']
    weight=sum(byid[i]['weight_pct'] for i in [22,3])
    native=load('native-classes.csv')
    def positive(bid,label,headline=False):
        rows=[r for r in native if r['benchmark_id']==bid and (not headline or r['scope']=='headline')]
        return 100*sum(r['count'] for r in rows if r['label']==label)/sum(r['count'] for r in rows)
    acos=positive(38,'yes');sarcasm=positive(40,'yes',True);rag=positive(59,'true')
    rows=[
        ['输入以短文本为主，少量任务有长尾',f'4k 覆盖 {pct(covers[4096]["pair_request_covered_pct"])}；仍超限 {num(covers[4096]["request_pair_over_count"])} 条，其中 API-Bank {num(byid[3]["requests"])} 条','短输入分桶，长尾单独处理'],
        ['原生类别比例差异很大',f'ACOS 正类 {pct(acos)}；英语讽刺 {pct(sarcasm)}；RAGTruth 幻觉类 {pct(rag)}','候选展开后的 50:50 不代表原生类别均衡'],
        ['编码负担集中在少数任务',f'POP909＋API-Bank 占逐候选编码量 {pct(token_share)}，分数权重仅 {pct(weight)}','优先减少候选之间的重复编码'],
        ['请求数不等于独立样本数','RAGTruth 的 2,700 条回答来自 450 个源 prompt；GSM8K 每道原题有两个版本','数据划分与结果分析保留来源分组'],
        ['部分任务存在明显构造线索',f'BPoMP 按“选较长候选”即可命中 {pct(byid[20]["longest_candidate_reference_pct"])}','高分不一定完全来自目标能力'],
    ]
    parts=[
        '# JEV Decision Index 数据分析｜汇报版',
        '**范围：37 项公开 benchmark，约 11 万条请求、27 万道问题、219 万个候选匹配对。**',
        '**核心结论：重点不只是加长上下文，更要关注类别失衡、候选重复编码和同源样本关联。**',
        '## 主要发现',
        table(['关键发现','核心数字','意义'],rows),
        '长度按当前 Qwen3 候选匹配模型的实际输入统计；BPoMP 的长度规则是请求级描述性参照，不是模型成绩。',
        '## 图表概览',
        '![五领域上下文覆盖率](figures/14-five-domain-request-coverage.png)',
        '多数领域在较短上下文下已能覆盖大部分请求，工具任务的长尾更突出。',
        '![五领域数据画像雷达图](figures/07-five-domain-data-profile-radars.png)',
        '不同领域的候选量、问题量和输入长度差异明显；雷达图描述数据结构，不代表模型能力。',
        '## 后续重点',
        '1. **先提效：**优先优化 POP909、API-Bank 的重复编码。',
        '2. **再配数据：**按原生类别安排曝光，不只调整候选匹配对的正负比例。',
        '3. **分组分析：**按源问题、文档或用户看效果，避免同源样本掩盖泛化差异。',
        '[完整统计与 37 项明细见附件](APPENDIX.md)。',
    ]
    markdown='\n\n'.join(parts)+'\n'
    (ROOT/'REPORT.md').write_text(markdown)
    write_html(markdown,summary,bench,brief=True)
    print('Generated concise REPORT.md and index.html; full details kept in APPENDIX.md and details.html.')

def inline(text):
    text=html.escape(text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',lambda m:f'<a href="{m[2]}">{m[1]}</a>',text)
    return text

def write_html(markdown,summary,bench,*,filename='index.html',brief=False):
    blocks=[];toc=[];i=0;lines=markdown.splitlines();hcount=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));title=line[n:].strip();hcount+=1;anchor='section-'+str(hcount)
            blocks.append(f'<h{n} id="{anchor}">{inline(title)}</h{n}>')
            if n==2:toc.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
        elif line.startswith('|'):
            header=[x.strip() for x in line.strip('|').split('|')];i+=2;rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                values=[x.strip() for x in lines[i].strip().strip('|').split('|')];rows.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in values)+'</tr>');i+=1
            blocks.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in header)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>');continue
        elif line.startswith('!['):
            match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
            if match:blocks.append(f'<figure><a href="{match[2]}" target="_blank"><img loading="lazy" src="{match[2]}" alt="{html.escape(match[1])}"></a><figcaption>{html.escape(match[1])} · 点击原图放大</figcaption></figure>')
        elif re.match(r'^(- |\d+\. )',line):blocks.append('<p class="bullet">'+inline(line)+'</p>')
        else:blocks.append('<p>'+inline(line)+'</p>')
        i+=1
    if brief:
        css='''*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#203047;font:16px/1.75 system-ui,-apple-system,"Microsoft YaHei",sans-serif}main{max-width:1160px;margin:28px auto;padding:32px 38px;background:white;border:1px solid #e0e6ef;border-radius:16px}h1{font-size:28px;margin:0 0 18px}h2{font-size:21px;margin:28px 0 12px;border-bottom:2px solid #e8eef5;padding-bottom:8px}a{color:#176bb1;text-decoration:none}a:hover{text-decoration:underline}.table-wrap{overflow:auto;margin:18px 0;border:1px solid #dce4ee;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.65}th,td{padding:13px 15px;border-bottom:1px solid #e5eaf1;text-align:left;vertical-align:top}th{background:#eaf1f8;color:#1e4566}td:first-child{font-weight:600;width:24%}tr:nth-child(even){background:#f8fafc}figure{margin:20px 0;padding:10px;border:1px solid #e5eaf1;border-radius:10px}img{width:100%;height:auto;display:block}figcaption{text-align:center;font-size:13px;color:#657b90}strong{font-weight:650}.bullet{margin:10px 0}@media(max-width:760px){main{margin:0;padding:22px 16px;border:0;border-radius:0}h1{font-size:24px}th,td{padding:10px;font-size:14px}}@media print{body{background:white}main{margin:0;padding:0;border:0}figure{break-inside:avoid}h2{break-after:avoid}}'''
        page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Decision Index 数据分析｜汇报版</title><style>'+css+'</style></head><body><main>'+''.join(blocks)+'</main></body></html>'
        page=page.replace('href="APPENDIX.md"','href="details.html"')
        (ROOT/filename).write_text(page)
        return
    filters='<div class="filters"><input id="benchmark-search" placeholder="筛选下表：搜索 benchmark 或领域…"><span id="visible-count">37 项</span></div>'
    headers=['Benchmark','领域','请求','Joint P50','Joint P95','Pair max P95','候选/题','匹配正例','Pair token']
    rows=[]
    for b in bench:
        vals=[b['name'],b['area_zh'],num(b['requests']),num(b['joint_p50']),num(b['joint_p95']),num(b['request_pair_max_p95']),num(b['options_mean']),pct(b['match_positive_pct']),num(b['pair_sum'])]
        raw=[b['name'],b['area_zh'],b['requests'],b['joint_p50'],b['joint_p95'],b['request_pair_max_p95'],b['options_mean'],b['match_positive_pct'],b['pair_sum']]
        rows.append('<tr>'+''.join(f'<td data-sort="{html.escape(str(v))}">{html.escape(str(label))}</td>' for v,label in zip(raw,vals))+'</tr>')
    interactive=filters+'<div class="table-wrap"><table id="benchmark-table"><thead><tr>'+''.join('<th tabindex="0">'+h+' ↕</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6fa;color:#203047;font:16px/1.8 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}a{color:#176bb1;text-decoration:none}a:hover{text-decoration:underline}.layout{max-width:1680px;margin:auto;display:grid;grid-template-columns:258px minmax(0,1fr);gap:28px;padding:32px}aside{position:sticky;top:24px;height:calc(100vh - 48px);overflow:auto;padding:20px;background:#fff;border:1px solid #e0e6ef;border-radius:16px}aside strong{font-size:20px}aside a{display:block;font-size:13px;line-height:1.5;padding:7px 0}main{min-width:0}.hero{background:linear-gradient(135deg,#122f50,#1f6d83);color:white;padding:34px;border-radius:20px;margin-bottom:24px}.hero h1{font-size:32px;margin:0 0 14px}.hero p{color:#dcebf6;margin:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:25px}.stat{background:#ffffff14;border:1px solid #ffffff22;border-radius:12px;padding:14px}.stat b{display:block;font-size:28px}.stat span{font-size:13px;color:#d5e6f5}.document,.quick{background:#fff;border:1px solid #e0e6ef;border-radius:16px;padding:30px;margin-bottom:24px}h1{font-size:28px}h2{font-size:25px;margin:48px 0 16px;border-bottom:2px solid #e7edf5;padding-bottom:10px;scroll-margin-top:20px}h3{font-size:21px;margin-top:34px}h4{font-size:19px;border-left:4px solid #2b8292;background:#f0f7f9;padding:8px 14px;margin-top:30px;scroll-margin-top:20px}strong{font-weight:650}code{background:#edf2f6;font-size:.9em;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}.table-wrap{overflow:auto;margin:18px 0 26px;border:1px solid #dce4ee;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65}th,td{padding:11px 13px;border-bottom:1px solid #e5eaf1;text-align:left;vertical-align:top}th{background:#eaf1f8;color:#1e4566;white-space:nowrap}tr:nth-child(even){background:#f8fafc}tr:hover{background:#eef5fb}figure{margin:26px 0;padding:16px;border:1px solid #e1e7ef;border-radius:12px;background:#fff}img{width:100%;height:auto;display:block}figcaption{text-align:center;font-size:13px;color:#667b91;margin-top:10px}.filters{display:flex;gap:16px;align-items:center}.filters input{flex:1;padding:13px 16px;border:1px solid #cbd8e7;border-radius:8px;font:inherit}#benchmark-table th{cursor:pointer}.bullet{padding-left:14px;border-left:2px solid #dce6ef}footer{font-size:13px;color:#738299;padding:20px 0 40px}@media(max-width:1000px){.layout{display:block;padding:16px}aside{position:static;height:auto;margin-bottom:20px}aside nav{columns:2}.document,.quick{padding:20px}.stats{grid-template-columns:repeat(2,1fr)}}@media print{aside,.filters{display:none}.layout{display:block;padding:0}.hero{color:#203047;background:white}.hero p,.stat span{color:#203047}.document{border:0;padding:0}.table-wrap{overflow:visible}h2,h4{break-after:avoid}figure{break-inside:avoid}}'''
    js='''const search=document.querySelector('#benchmark-search'),body=document.querySelector('#benchmark-table tbody');function filter(){let n=0;for(const row of body.rows){row.hidden=!row.innerText.toLowerCase().includes(search.value.toLowerCase());if(!row.hidden)n++}document.querySelector('#visible-count').textContent=n+' 项'}search.addEventListener('input',filter);document.querySelectorAll('#benchmark-table th').forEach((th,index)=>{let asc=true;function sort(){const rows=[...body.rows];rows.sort((a,b)=>{let x=a.cells[index].dataset.sort,y=b.cells[index].dataset.sort;let v=Number.isNaN(Number(x))?x.localeCompare(y):Number(x)-Number(y);return asc?v:-v});rows.forEach(row=>body.appendChild(row));asc=!asc}th.addEventListener('click',sort);th.addEventListener('keydown',e=>{if(e.key==='Enter')sort()})});'''
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Decision Index · 37 项数据画像</title><style>'+css+'</style></head><body><div class="layout"><aside><strong>37 项数据画像</strong><p>2026-10-31 · 全量公开输入</p><nav>'+''.join(toc)+'</nav><hr><a href="REPORT.md">Markdown 报告</a><a href="tables/main-37.csv">37 项明细 CSV</a><a href="SOURCES.md">口径与来源</a></aside><main><header class="hero"><h1>Decision Index · 数据画像</h1><p>从长度、类别、候选和来源结构，解释评测数据如何影响模型。</p><div class="stats">'+''.join('<div class="stat"><b>'+n+'</b><span>'+label+'</span></div>' for n,label in [('37','公开 benchmark'),('11万+','有效请求'),('27万+','问题字段'),('219万','候选匹配对')])+'</div></header><section class="quick"><h2 style="margin-top:0">37 项快速查阅</h2><p>可按名称筛选，点击表头排序。Pair max P95 是每请求最大分支的第 95 百分位，不是总 token 量。</p>'+interactive+'</section><article class="document">'+''.join(blocks)+'</article><footer>所有展示最多两位小数；细节以 CSV 整数计数与明确分母为准。图表点击后可查看原始分辨率。</footer></main></div><script>'+js+'</script></body></html>'
    if filename=='details.html':page=page.replace('href="REPORT.md"','href="APPENDIX.md"')
    (ROOT/filename).write_text(page)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details',action='store_true',help='Regenerate the detailed appendix instead of the brief report.')
    args=parser.parse_args()
    build_details() if args.details else build_brief()
