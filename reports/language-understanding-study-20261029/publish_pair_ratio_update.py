"""Publish the completed ratio round without replacing the earlier 91-endpoint snapshot."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import csv,json,os

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SystemExit('CPU publication only')
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[1]
PREP=PROJECT/'research/lu-pair-ratio-prep-20261029'
ROOT=Path(json.loads((PREP/'formal-round-launched.json').read_text())['root'])
ARMS=['NEG2','NATURAL','POS2','POS4']
NAMES={'NEG2':'不匹配重复两次','NATURAL':'自然展开对照','POS2':'匹配重复两次','POS4':'匹配重复四次'}
IDS=['11','12','29','38','39','40','41','42','59']


def read(path):return json.loads(Path(path).read_text())
def new_text(name,text):
    path=OUT/name;assert not path.exists(),path;path.write_text(text)
def new_json(name,value):new_text(name,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def new_csv(name,rows,fields=None):
    path=OUT/name;assert not path.exists(),path
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    report=read(ROOT/'round-report.json');review=read(ROOT/'RATIO-REVIEW.json');state=read(ROOT/'state.json');plan=read(ROOT/'plan.json')
    generated=read(PREP/'generated/schedule-summary.json');dev=read(ROOT/'natural-dev-manifest.json')
    inventory=read(PROJECT/'reports/length-label-benchmark-20261029/data-inventory.json')
    assert inventory['native_english_sarcasm_original_questions']=={'sarcastic':664,'not_sarcastic':2062}
    with (OUT/'runs.csv').open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
    assert len(old)==91 and state['state']=='complete' and report['comparable_training_and_natural_validation']
    new=[];details={};summary=[];native_rows=[];timings=[];coverage={};sar={};decisions={}
    reference=review['natural_validation_results']['NATURAL'];reference_skill=next(x['public_Language']['language_skill'] for x in report['arms'] if x['arm']=='NATURAL')
    for row in report['arms']:
        arm=row['arm'];train=row['train'];exposure=row['actual_exposure'];natural=row['natural_validation'];public=row['public_Language']
        assert arm in ARMS and train['passed'] and exposure['full_schedule_committed']
        assert natural['scored_pairs']==67202 and public['language_complete'] and public['counts']=={'ok':31715}
        item={k:'' for k in fields};item.update(stage='pair_ratio',round=ROOT.name,code=arm,name=NAMES[arm],seed=3407,rank=32,LoRA_alpha=64,
            new_tokens=train['actual_new_tokens'],training_compute_tokens=train['actual_new_tokens'],optimizer_steps=train['optimizer_steps'],
            language_skill=public['language_skill'],coverage=public['coverage'],requests=31715,checkpoint=train['checkpoint'],
            source=str(ROOT/'runs'/arm/'arms'/arm/'evaluation/result.json'))
        item.update({f'skill_{cid}':public['per_benchmark'][cid]['skill_0_100'] for cid in IDS});new.append(item)
        metrics=natural['pair_metrics'];native=natural['native_question_metrics']
        summary.append({'arm':arm,'name':NAMES[arm],'positive_row_fraction':exposure['positive_row_fraction'],
            'positive_token_fraction':exposure['positive_token_fraction'],'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],
            **{f'{key}_class{label}':metrics['classes'][label][key] for label in ['0','1'] for key in ['precision','recall','f1','support']},
            'native_macro':native['macro_native_independent_dev'],'D0_task_macro_pair_f1':natural['D0_task_macro_f1'],
            'language_skill':public['language_skill'],'language_skill_delta':public['language_skill']-reference_skill})
        for task,m in native['native_independent_dev_metrics'].items():
            native_rows.append({'arm':arm,'task':task,**m,'candidate_accuracy':native['original_question_accuracy_per_task'][task]['accuracy']})
        with __import__('numpy').load(ROOT/'runs'/arm/'schedule.npz',allow_pickle=False) as archive:
            questions=set(archive['question_code'][archive['draw_id']>=0].tolist())
        groups={(generated['question_metadata'][q]['task'],generated['question_metadata'][q]['source_group']) for q in questions}
        draws=exposure['native_question_draws_fully_committed'];unique=len(questions)
        coverage[arm]={'complete_question_draws':draws,'unique_candidate_questions':unique,
            'unique_candidate_question_population':len(generated['question_metadata']),
            'unique_task_source_groups':len(groups),'mean_draws_per_unique_candidate_question':draws/unique,
            'unique_source_pairs_D0_D2':exposure['unique_input_pairs_by_pool'],
            'unit_note':'ACOS candidate questions are property fields; neither draws nor unique candidate questions are independent reviews.'}
        stages={stage:Counter() for stage in ['core','tail']}
        for x in exposure['native_by_task_gold_stage']:
            if x['task']=='isarcasm/intended-en':stages['tail' if x['tail'] else 'core'][x['gold']]+=x['draws']
        combined=stages['core']+stages['tail']
        sar[arm]={'core_not_sarcastic':stages['core'][0],'core_sarcastic':stages['core'][1],
            'tail_not_sarcastic':stages['tail'][0],'tail_sarcastic':stages['tail'][1],
            'core_sarcastic_fraction':stages['core'][1]/sum(stages['core'].values()),
            'all_draw_sarcastic_fraction':combined[1]/sum(combined.values())}
        delta={'pair_macro_f1':metrics['macro_f1']-reference['pair_metrics']['macro_f1'],
            'recall0':metrics['classes']['0']['recall']-reference['pair_metrics']['classes']['0']['recall'],
            'recall1':metrics['classes']['1']['recall']-reference['pair_metrics']['classes']['1']['recall'],
            'native_macro':native['macro_native_independent_dev']-reference['native_question_metrics']['macro_native_independent_dev'],
            'D0_task_macro_pair_f1':natural['D0_task_macro_f1']-reference['D0_task_macro_f1']}
        decisions[arm]={'deltas':delta,'checks':{'pair_gain_at_least_0.005':delta['pair_macro_f1']>=.005,
            'recall0_drop_at_most_0.02':delta['recall0']>=-.02,'recall1_drop_at_most_0.02':delta['recall1']>=-.02,
            'native_drop_at_most_0.01':delta['native_macro']>=-.01,'D0_drop_at_most_0.01':delta['D0_task_macro_pair_f1']>=-.01}}
        decisions[arm]['composite_pass']=all(decisions[arm]['checks'].values()) if arm!='NATURAL' else None
        details[arm]={'train':train,'exposure':exposure,'natural_validation':natural,'official_Language':public,
            'recipe':read(ROOT/'runs'/arm/'recipe.json'),'source_directory':str(ROOT/'runs'/arm)}
        timings.append({'arm':arm,'training_wall_minutes':train['wall_seconds']/60,'training_loop_minutes':train['training_loop_seconds']/60,
            'natural_dev_seconds':natural['seconds'],'formal_execution_minutes':report['elapsed_minutes'],
            'aggregate_evaluation_minutes':review['aggregate_evaluation_minutes']})
    assert len(new)==4 and len({str(Path(x['checkpoint']).resolve()) for x in old+new})==95
    assert all(not decisions[a]['composite_pass'] for a in ARMS if a!='NATURAL')
    broad_lower_bound=(datetime.fromisoformat(report['time'])-datetime.fromisoformat(generated['time'])).total_seconds()/60
    snapshot={'time':datetime.now(timezone.utc).isoformat(),'old_snapshot_endpoint_count':91,'new_endpoint_count':4,'cumulative_complete_Language_endpoints':95,
        'old_91_snapshot_files_unchanged':['runs.csv','experiment-facts.json','APPENDIX.md'],
        'new_endpoints':new,'round_report':report,'review':review,'plan':plan,'state':state,'natural_dev_manifest':dev,
        'native_SAR_static_population_source':str(PROJECT/'reports/length-label-benchmark-20261029/data-inventory.json'),
        'native_SAR_static_population':inventory['native_english_sarcasm_original_questions'],
        'native_SAR_draw_counts':sar,'candidate_question_coverage':coverage,'promotion_checks_recomputed':decisions,
        'per_task_native_metrics':native_rows,'arms':details,
        'timing_scope':{'formal_execution_minutes':report['elapsed_minutes'],'training_max_minutes':max(x['training_wall_minutes'] for x in timings),
            'evaluation_minutes':review['aggregate_evaluation_minutes'],'prior_schedule_generation_completed_utc':generated['time'],
            'minutes_from_prior_schedule_completion_to_initial_report_lower_bound':broad_lower_bound,
            'whole_direction_preparation_to_report_within120_proven':False},
        'no_new_training_or_inference_during_publication':True,'no_recomputation_of_official_request_scores':True,
        'sources':{'execution_root':str(ROOT),'CPU_preparation':str(PREP),'original_91_facts':str(OUT/'experiment-facts.json')}}
    new_json('pair-ratio-update.json',snapshot)
    new_csv('pair-ratio-runs.csv',new,fields);new_csv('runs-through-pair-ratio.csv',old+new,fields)
    new_csv('pair-ratio-summary.csv',summary);new_csv('pair-ratio-native-metrics.csv',native_rows);new_csv('pair-ratio-timing.csv',timings)
    for source,target in [('task-metrics.csv','pair-ratio-task-metrics.csv'),('public-benchmarks.csv','pair-ratio-public-benchmarks.csv'),
                          ('actual-exposure-by-task-label.csv','pair-ratio-exposure-by-task-label.csv')]:new_text(target,(ROOT/source).read_text())
    new_text('pre-pair-ratio-report.md',(PROJECT/'reports/language-understanding-study-20261029.md').read_text())
    s={x['arm']:x for x in summary}
    lines=['# 正负pair比例首轮：匹配召回提高，不代表总体匹配分类改善','',
        '**范围：2026-10-29已完成的一轮、一个续训seed；下列数值来自实际worker回执和落盘预测。**',
        '四组均完成67202对自然dev和31715请求的完整Language九项，覆盖100%、零unsupported、零错误。复核与本报告整理仅用CPU，不重训、不重跑推理或官方逐请求评分。','',
        '## 一、提高匹配pair曝光带来召回取舍，未通过预注册的保留条件','',
        '正类0是entailment/compatible，负类1是contradiction/incompatible，不是原生讽刺类别，也不是交换premise/hypothesis。只在D2完整原题候选内重复指定标签的pair，明确属于pair层重采样；D0不做该重采样。各组CE权重均为[1,1]。','',
        '| 训练条件 | 实际正pair占比 | 实际正token占比 | 自然dev Macro-F1 | 匹配召回 | 不匹配召回 |',
        '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        x=s[a];lines.append(f"| {NAMES[a]} | {x['positive_row_fraction']:.4%} | {x['positive_token_fraction']:.4%} | {x['macro_f1']:.4%} | {x['recall_class0']:.4%} | {x['recall_class1']:.4%} |")
    lines+=['',
        '相对自然展开对照，匹配重复两次、四次使匹配召回分别增加2.8399、5.3884个百分点，不匹配召回分别下降4.1870、10.3824个百分点；Macro-F1分别下降0.8229、2.8636个百分点。不匹配重复两次的Macro-F1只低0.0741个百分点，但匹配召回下降2.8243个百分点。','',
        '| 相对自然展开对照 | Macro-F1差值，百分点 | 95%条件配对区间，百分点 |',
        '|---|---:|---:|']
    for a in ['NEG2','POS2','POS4']:
        lo,hi=review['pair_Macro_F1_paired_intervals'][a]['Macro_F1_delta_percentile95']
        lines.append(f"| {NAMES[a]} | {100*decisions[a]['deltas']['pair_macro_f1']:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] |")
    lines+=['',
        '区间采用1000次配对bootstrap，seed20261029，按(source, source_group)聚类、按source分层；D0为2837组，D2为3635组，候选字段始终随源组共同重抽样，不把67202对视为独立样本。区间只反映本评估人口的条件抽样变化，不含训练seed、共享R0、选池、反复使用dev或多重筛选不确定性。','',
        '预注册要求相对对照的pair Macro-F1增益至少0.005，两类recall各降不超过0.02，D2原生任务宏均值与D0任务宏pair-F1各降不超过0.01。三个处理都未共同满足：不匹配重复两次违反pair收益与匹配召回线，两个提高匹配曝光的处理违反pair收益与不匹配召回线。因此不自动追加seed；自然展开只是在这四个单seed条件中Macro-F1最高，精确全局50:50并未测试，不能宣布最佳比例。','',
        '## 二、pair分类、原生任务和公开Language的变化不同步','',
        '| 训练条件 | pair accuracy | D2原生10任务宏均值 | D0任务宏pair-F1 | Language Skill | ΔLanguage |',
        '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        x=s[a];lines.append(f"| {NAMES[a]} | {x['accuracy']:.4%} | {x['native_macro']:.6f} | {x['D0_task_macro_pair_f1']:.6f} | {x['language_skill']:.4f} | {x['language_skill_delta']:+.4f} |")
    lines+=['',
        '匹配重复两次的Language比对照高1.7413分，同时pair Macro-F1低0.8229个百分点；原生任务宏均值仅增加0.002200，不能将这些量视为同一指标。公开分项中，VAST提高8.45 Skill，ACOS下降1.5170、RAGTruth下降2.20，收益并不均匀。公开Language只报告，不用于推翻预注册判断或事后挑比例。','',
        '对所有候选分数加同一个常数不改变原题argmax，但本次训练也可能改变候选间相对分数；当前结果不能把全部差异归因为常数校准，也不能用pair召回增加直接证明原题选择能力提高。原生指标中ANLI使用Macro-F1、HellaSwag用accuracy、SAR用讽刺F1、RAGTruth用幻觉F1、ACOS用每review presence-F1；它们的宏均值不是Language Skill。原字段名中的independent_dev仅为历史命名，本自然dev仍是复用验证集。','',
        '## 三、真实token预算一致，但原题覆盖和重复程度随pair重采样改变','',
        '| 训练条件 | 已提交pair行数 | 已提交token | 完整候选题draw | 唯一候选题 | 唯一D2源pair |',
        '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        x=review['actual_exposure_audit'][a];c=coverage[a]
        lines.append(f"| {NAMES[a]} | {x['actual_rows']} | {x['actual_nonpadding_tokens']} | {c['complete_question_draws']} | {c['unique_candidate_questions']} | {c['unique_source_pairs_D0_D2'][1]} |")
    lines+=['',
        '逐更新核对了实际attention_mask回执的task×标签行数与token，并与源row标签/长度及计划中的相应更新计数对照；实际源pair计数、每task token和D0样本multiset均一致。所有候选题draw完整，未提交累积、未完成draw均为0。这里没有重新播放模型前向，也没有声称逐microbatch内部顺序已被独立重放。','',
        '四组共同预算为19999954非padding token，比20M少46；固定每(pool,task)预算和D0实际出现次数，不保证共同分桶后D0位置完全相同。四表实际pair出现总数471110，四份静态池保守上限469412，均不超过500000，但都不是独立新语料数。原自动报告“固定pair预算”应读作固定token预算，四组行数并不相同。','',
        '候选题draw是一次完整候选题呈现，唯一候选题按源question ID去重；ACOS的一道候选题可只是同一review的一个属性字段，不能把这些数量称为独立review数。固定token后重复倍率会改变唯一覆盖与重复率，因此本实验不是保持独立原题数不变的纯先验干预。','',
        '原生SAR的静态训练人口为664条sarcastic和2062条not_sarcastic；各组沿用同一个原生core平衡策略。core的讽刺draw约50%，包含精确填尾后的全流约48.19%；这与全局pair正类36.46%—64.11%、正token34.51%—71.65%以及CE权重[1,1]分别使用不同分母。精确core/尾部计数见附录。','',
        '## 附录：设置、完整数值与比较边界','',
        '### 冻结设置和评估人口','',
        '- 真实Qwen3-0.6B，causal BF16 SEQ_CLS LoRA rank32/alpha64，已训练的普通Linear(1024,2,bias=False)头；共同expanded-R0-r32起点，续训/采样seed3407。',
        '- microbatch32×accumulation4，有效batch128，fresh AdamW、peak LR5e-5、3% token warmup后cosine、clip1、激活重算开启、真实padding mask；不把r16/3M batch/LR pilot外推为本轮最优配置。',
        '- 复用未改变的117353对Arrow池，训练不新增截断、缓存最长2895。固定每(pool,task) token、D0出现次数、原生类别策略及输入；只改变D2正/负pair重复次数。',
        '- 自然dev为D0 Language validation原文2840对，加未截断、完整候选的D2 dev64362对/30331候选题。无标签/长度再平衡或抽样，不用旧5904或2006分层面板作自然主验证。该人口与协作1667对加权宽D0长度面板不同，不能直接横比。',
        '- 公开只评IDs 11、12、29、38、39、40、41、42、59；保留全部criteria，temperature1，真实上下文40960内不额外截断、不写benchmark答案规则。',
        '- 本轮自然控制重新训练，因为冻结task调度不同于历史20M的随机task采样；历史20M公开分数不是本轮匹配控制。不同池、预算和面板的历史分数也不并为本轮效应。','',
        '### 九项公开Skill，仅作报告','',
        '| ID／数据集 | 不匹配重复两次 | 自然展开对照 | 匹配重复两次 | 匹配重复四次 |',
        '|---|---:|---:|---:|---:|']
    for cid in IDS:
        name=details['NATURAL']['official_Language']['per_benchmark'][cid]['dataset']
        values=' | '.join(f"{details[a]['official_Language']['per_benchmark'][cid]['skill_0_100']:.4f}" for a in ARMS)
        lines.append(f'| {cid} {name} | {values} |')
    lines+=['','Language总分采用官方权重，不是上表简单平均。原始指标与完整精度保存在pair-ratio-public-benchmarks.csv和pair-ratio-update.json。','',
        '### 原生SAR draw单独计数','',
        '| 训练条件 | core非讽刺／讽刺 | 尾部非讽刺／讽刺 | 全流讽刺draw比例 |',
        '|---|---:|---:|---:|']
    for a in ARMS:
        x=sar[a];lines.append(f"| {NAMES[a]} | {x['core_not_sarcastic']}／{x['core_sarcastic']} | {x['tail_not_sarcastic']}／{x['tail_sarcastic']} | {x['all_draw_sarcastic_fraction']:.4%} |")
    lines+=['','### 计时与范围','',
        f"正式配置/CPU预检从13:33:42 UTC计时，初报14:30:40 UTC落盘，正式执行{report['elapsed_minutes']:.3f}分钟；各组训练wall为{min(x['training_wall_minutes'] for x in timings):.2f}—{max(x['training_wall_minutes'] for x in timings):.2f}分钟，整轮自然dev加公开评测{review['aggregate_evaluation_minutes']:.2f}分钟。没有因超时停止有效任务。",
        f"此前CPU设计、采样表和自然dev准备不包含在57分钟中。从已知的采样表生成完成时间11:55:59到初报也至少{broad_lower_bound:.2f}分钟；若将这些前置工作并入本轮准备，不能声称满足120分钟全流程上限。正式执行时长不得替代整个方向的端到端准备成本。CPU独立复核另于14:58完成，没有重新推理。",'',
        '### 文件与版本','',
        '- [pair-ratio-summary.csv](pair-ratio-summary.csv)：全局两类precision/recall/F1、Macro-F1、accuracy和各类指标。',
        '- [pair-ratio-task-metrics.csv](pair-ratio-task-metrics.csv)：四模型每task两类指标；[原生任务表](pair-ratio-native-metrics.csv)按原生metric另列。',
        '- [pair-ratio-exposure-by-task-label.csv](pair-ratio-exposure-by-task-label.csv)：实际task×标签行数/token；[时间表](pair-ratio-timing.csv)保留阶段口径。',
        '- [pair-ratio-runs.csv](pair-ratio-runs.csv)：本轮4个新终点；[runs-through-pair-ratio.csv](runs-through-pair-ratio.csv)：累计95终点。',
        '- [pair-ratio-update.json](pair-ratio-update.json)：完整精度、全部原聚合结果、配对区间、独立曝光核对、题目覆盖与预注册条件逐项结果。',
        '- 旧91终点的runs.csv、experiment-facts.json、APPENDIX.md原样保留；增补前主报告另存[pre-pair-ratio-report.md](pre-pair-ratio-report.md)。8个dev-only batch/LR pilot仍单列，不计入95。',
        f'- 原始训练、权重、逐更新回执、预测和自动初报保留于`{ROOT}`；CPU复核入口`{PREP}/review_ratio_round.py`。没有修改已执行worker或旧实验产物。','',
        '| 正文名称 | 目录代号 | D2正／负重复次数 |','|---|---|---|',
        '| 不匹配重复两次 | NEG2 | 1／2 |','| 自然展开对照 | NATURAL | 1／1 |','| 匹配重复两次 | POS2 | 2／1 |','| 匹配重复四次 | POS4 | 4／1 |','',
        '本轮不据公开最高分替换配方、不自动追加seed。若随后研究CE类别权重，应逐数组复用自然展开采样流，保持真实数据曝光完全一致；权重变化另称loss干预，不与pair行数比例混称。目前该诊断尚未实施。']
    new_text('PAIR-RATIO-ADDENDUM.md','\n'.join(lines)+'\n')
    print(json.dumps({'time':snapshot['time'],'new_complete_Language_endpoints':4,'cumulative':95,
        'old91_unchanged':True,'formal_execution_minutes':report['elapsed_minutes'],
        'prep_to_initial_report_lower_bound_minutes':broad_lower_bound,'GPU_jobs_started':0},ensure_ascii=False))


if __name__=='__main__':main()
