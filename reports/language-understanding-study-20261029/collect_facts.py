"""Read-only consolidation of completed artifacts. No model, training, inference or scoring rerun."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import csv,json,re,statistics

PROJECT=Path('/home/jovyan/assumeengage')
RESEARCH=PROJECT/'research'
OUT=Path(__file__).resolve().parent
IDS=['11','12','29','38','39','40','41','42','59']
DATASETS=['ContractNLI','ANLI','HellaSwag','ACOS','FinEntity','iSarcasmEval','VAST','NLI4CT','RAGTruth']
SOURCES=[
 ('controlled','lu-replication-20261027/three-seed-results.json'),
 ('D2','lu-d2-studies-20261027/study-results.json'),
 ('mixture','lu-token-mixtures-20261027/mixture-results.json'),
 ('native-sampling','lu-sarcasm-sampling-20261027/study-results.json'),
 ('joint','lu-joint-confirmation-20261028/joint-results.json'),
 ('rank','lu-lora-capacity-20261028/capacity-results.json')]
NAMES={
 'R1':'D0原池／causal','R2':'D0原池／双向attention','R3':'D0＋ANLI/HellaSwag目标数据／causal','R4':'D0＋ANLI/HellaSwag目标数据／双向attention',
 'R5_DATA':'扩充D2目标数据／保留原采样控制','Q1':'D2原题级任务采样／指数1','Q05':'D2原题级任务采样／指数0.5','Q0':'D2原题级任务采样／指数0',
 'M80_20':'D0:目标token=80:20','M20_80':'D0:目标token=20:80',
 'SCB':'原生讽刺核心类别平衡','ST5':'讽刺占目标池token5%','SCB_T5':'原生讽刺平衡＋目标池token5%',
 'JOINT':'20:80＋讽刺平衡/5%联合／rank16','LR32':'20:80＋讽刺平衡/5%联合／rank32'}
FAST_NAMES={
 'F_BASE':'小池参考','F_NLI40':'ANLI占目标池40%','F_HS30':'HellaSwag占目标池30%','F_HS15':'HellaSwag占目标池15%',
 'F_NLI25_HS25':'ANLI25%＋HellaSwag25%',
 'P_H30_S10':'HellaSwag30%＋讽刺10%','P_H30_R50':'HellaSwag30%＋RAGTruth50%',
 'P_H30_S10_R50':'HellaSwag30%＋讽刺10%＋RAGTruth50%','P_H15_S10':'HellaSwag15%＋讽刺10%',
 'C_BASE_R32':'小池参考／rank32','C_BASE_R64':'小池参考／rank64',
 'C_R50_R32':'HellaSwag30%＋RAGTruth50%／rank32','C_R50_R64':'HellaSwag30%＋RAGTruth50%／rank64',
 'M35_BASE':'D0:目标35:65／参考','M35_R50':'D0:目标35:65／常识-grounding联合',
 'M50_BASE':'D0:目标50:50／参考','M50_R50':'D0:目标50:50／常识-grounding联合',
 'M10_BASE':'D0:目标10:90／参考','M10_R50':'D0:目标10:90／常识-grounding联合',
 'M00_BASE':'纯目标池／参考','M00_R50':'纯目标池／常识-grounding联合',
 'S25_BAL':'讽刺2.5%／原生核心平衡','S25_NAT':'讽刺2.5%／原生自然比例',
 'S5_NAT':'讽刺5%／原生自然比例','J5_NAT':'常识-grounding联合／讽刺5%原生自然比例',
 'T0_BASE':'任务指数0／参考','T1_BASE':'任务指数1／参考','T0_R50':'任务指数0／常识-grounding联合','T1_R50':'任务指数1／常识-grounding联合',
 'A05_CTRL':'seed3408新对照／D0与D2指数0.5','A1_BOTH':'D0与D2指数1／新seed复核',
 'A075_BOTH':'D0与D2指数0.75','A1_TARGET':'D0指数0.5／D2指数1',
 'E075_BOTH':'D0与D2指数0.75／seed3407复核','E075_TARGET':'仅D2指数0.75','E075_D0':'仅D0指数0.75'}


def read(path):return json.loads(Path(path).read_text())
def stat(values):return {'values':values,'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None}
def macro(dev):return dev.get('macro_native_independent_dev') if isinstance(dev,dict) else None


def main():
    assert not (OUT/'experiment-facts.json').exists(),'Preserve an existing published snapshot'
    raw={};runs=[];dev_rows=[]
    r0path=RESEARCH/'lu-controlled-20261027/r0-complete-summary.json';r0=read(r0path);raw[str(r0path)]=r0
    def add(row):
        assert row['coverage']==1.0
        assert set(row['per_benchmark'])==set(IDS)
        for v in row['per_benchmark'].values():assert v.get('errors',0)==0 and v.get('unsupported',0)==0 and v.get('coverage',1)==1
        row['requests']=31715;row['request_count_basis']='complete Language0.3 coverage records, not a new recount of prediction lines'
        assert Path(row['checkpoint']).exists();runs.append(row)
    add({'stage':'initial','code':'R0','name':'起始Qwen3-0.6B分类基线','seed':3407,'rank':16,'LoRA_alpha':32,
         'new_tokens':r0['selected_checkpoint_actual_token_exposure'],'training_compute_tokens':r0['r0_training_compute_actual_tokens'],
         'language_skill':r0['language_skill'],'coverage':r0['coverage'],'per_benchmark':r0['per_benchmark'],
         'checkpoint':r0['starting_checkpoint'],'source':str(r0path),'formal_directional_experiment':True})
    three_review=read(RESEARCH/'lu-replication-20261027/three-seed-review.json');raw[str(RESEARCH/'lu-replication-20261027/three-seed-review.json')]=three_review
    def add_dev(run,panel,dev,source,note=''):
        if not isinstance(dev,dict):return
        for task,m in dev.get('native_independent_dev_metrics',{}).items():
            dev_rows.append({'checkpoint':run['checkpoint'],'stage':run['stage'],'code':run['code'],'seed':run['seed'],
                 'panel':panel,'task':task,'metric':m['metric'],'value':m['value'],'questions_or_fields':m.get('questions'),
                 'pairs':dev.get('pairs'),'support_note':note,'source':source})
    for stage,rel in SOURCES:
        path=RESEARCH/rel;j=read(path);raw[str(path)]=j
        for item in j['ledger']:
            code=re.sub(r'-seed\d+$','',item.get('experiment','LR32'))
            rank=item.get('rank',32 if code=='LR32' else 16)
            row={'stage':stage,'code':code,'name':NAMES[code],'seed':item['seed'],'rank':rank,'LoRA_alpha':item.get('alpha',rank*2),
                 'new_tokens':item.get('new_actual_tokens',item.get('new_tokens')),'language_skill':item['language_skill'],
                 'coverage':item['coverage'],'per_benchmark':item['per_benchmark'],'checkpoint':item['checkpoint'],
                 'source':str(path),'formal_directional_experiment':True}
            assert row['new_tokens']==50_000_000
            add(row)
            if stage=='controlled':
                entry=three_review['clean_dev_reaudits'][code];position=[3407,3408,3409].index(item['seed'])
                row['clean_native_macro']=entry['macro_native']['values'][position]
                for task,m in entry['per_task'].items():
                    dev_rows.append({'checkpoint':row['checkpoint'],'stage':stage,'code':code,'seed':row['seed'],
                       'panel':'D2-clean-64362-reaudit','task':task,'metric':m['metric'],'value':m['values'][position],
                       'questions_or_fields':None,'pairs':64362,'support_note':'Review F1 uses reviews; field counts are not independent review counts.',
                       'source':str(RESEARCH/'lu-replication-20261027/three-seed-review.json')})
            else:
                dev=item.get('native_dev') or item.get('clean_dev') or item.get('clean_validation') or item.get('independent_dev',{}).get('target',{})
                if dev.get('pairs')==64362:
                    row['clean_native_macro']=macro(dev);add_dev(row,'D2-clean-64362',dev,str(path),'ACOS questions/fields are not independent reviews.')
    assert len(runs)==48
    grouped=defaultdict(list)
    for x in runs[1:]:grouped[x['code']].append(x)
    long_groups={k:{'name':NAMES[k],'seeds':[x['seed'] for x in a],
                   'language_skill':stat([x['language_skill'] for x in a]),
                   'clean_native_macro':stat([x['clean_native_macro'] for x in a]) if all(x.get('clean_native_macro') is not None for x in a) else None,
                   'per_benchmark':{cid:stat([x['per_benchmark'][cid]['skill'] for x in a]) for cid in IDS}} for k,a in grouped.items()}
    rounds=[]
    for file in sorted(RESEARCH.glob('lu-fastscreen-*/round-report.json')):
        root=file.parent;j=read(file);plan=read(root/'plan.json');prepared=read(root/'prepared.json');state=read(root/'state.json')
        raw[str(file)]=j
        reviews={}
        for name in ['target-metric-review.json','quota-review.json','capacity-review.json','mixture-review.json','sarcasm-review.json','task-alpha-review.json','alpha-confirm-review.json','alpha-attribution-review.json']:
            if (root/name).exists():reviews[name]=read(root/name);raw[str(root/name)]=reviews[name]
        model_rows=[]
        for item in j['candidates']:
            train=item['training'];ev=item['evaluation'];assert ev and ev.get('language_complete') and ev['counts']=={'ok':31715}
            code=item['candidate'];recipe=read(root/'runs'/code/'recipe.json');rank=read(root/'runs'/code/'arms'/code/'main/final/adapter_config.json')['r']
            bench={cid:{'skill':v['skill_0_100'],'raw':v.get('raw_coverage_adjusted'),'coverage':v['coverage'],
                        'errors':v.get('errors',0),'unsupported':v.get('unsupported',0)} for cid,v in ev['per_benchmark'].items()}
            engineering=bool(j.get('emergency_pipeline_validation'))
            row={'stage':'engineering' if engineering else 'fastscreen','round':root.name,'code':code,'name':FAST_NAMES[code],
                 'seed':train['seed'],'rank':rank,'LoRA_alpha':rank*2,'new_tokens':train['actual_new_tokens'],
                 'optimizer_steps':train['optimizer_steps'],'training_seconds_including_small_dev':train['elapsed_seconds'],
                 'language_skill':ev['language_skill'],'coverage':ev['coverage'],'per_benchmark':bench,'checkpoint':train['checkpoint'],
                 'source':str(file),'formal_directional_experiment':not engineering,'recipe':recipe,
                 'automatic_round_hint_not_final_decision':item.get('preliminary_decision'),'review_sources':[str(root/k) for k in reviews]}
            add(row);model_rows.append(row)
            add_dev(row,'small-D2-1476',train.get('small_dev'),str(root/'runs'/code/'arms'/code/'main/complete.json'),
                    'Most tasks32 questions; ACOS only1 review/domain, not39/363 independent reviews.')
        ends=[datetime.fromisoformat(j['time']).timestamp()]
        for name in ['confirmation/summary.json','d0-validation/summary.json','aggregate-eval-cap.json']:
            if (root/name).exists():ends.append(datetime.fromisoformat(read(root/name)['time']).timestamp())
        starts=[x['evaluation_started'] for x in state['tasks'] if 'evaluation_started' in x]
        launch=min(x['launch_epoch'] for x in state['tasks'])
        rounds.append({'root':str(root),'round_name':root.name,'seed':j['seed'],'models':len(model_rows),
           'tokens_per_model':[x['new_tokens'] for x in model_rows],'pool_unique_pairs':j['shared_unique_training_pairs'],
           'candidatewise_pool_upper_pairs':prepared['candidatewise_upper_count'],'preparation_minutes_recorded':j['preparation_minutes'],
           'preparation_minutes_to_effective_final_attempt':(launch-plan['round_start_epoch'])/60,
           'primary_report_minutes':j['elapsed_minutes'],'all_evaluations_completed_minutes':(max(ends)-plan['round_start_epoch'])/60,
           'aggregate_evaluation_minutes':(max(ends)-min(starts))/60 if starts else None,
           'hard120_primary_met':j['elapsed_minutes']<=120,'preparation_target_met_recorded':j['preparation_10minute_budget_met'],
           'engineering_only':bool(j.get('emergency_pipeline_validation')),'reviews':reviews})
    # Attach stored confirmation metrics by checkpoint, including immutable cached copies; do not rescore predictions.
    by_checkpoint={str(Path(x['checkpoint']).resolve()):x for x in runs};assert len(by_checkpoint)==len(runs)
    panel_seen={}
    for round_item in rounds:
        root=Path(round_item['root'])
        for folder,panel in [('confirmation','D2-confirmation-5904'),('d0-validation','D0-Language-pair-proxy-2006')]:
            for file in (root/folder).glob('*/result.json'):
                value=read(file)
                if not value.get('complete'):continue
                key=str(Path(value['checkpoint']).resolve());assert key in by_checkpoint
                pair=(key,panel);measurement=value.get('metrics') if folder=='confirmation' else {'task_macro_pair_accuracy':value['task_macro_pair_accuracy'],'per_task':value['per_task']}
                if pair in panel_seen:
                    assert panel_seen[pair]['measurement']==measurement,('Different same-panel copies',file)
                    panel_seen[pair]['sources'].append(str(file));continue
                panel_seen[pair]={'checkpoint':key,'panel':panel,'measurement':measurement,'sources':[str(file)]}
                if folder=='confirmation':add_dev(by_checkpoint[key],panel,value['metrics'],str(file),'128 native-label-stratified questions/task; ACOS4 reviews/domain.')
                else:
                    row=by_checkpoint[key]
                    for task,m in value['per_task'].items():dev_rows.append({'checkpoint':row['checkpoint'],'stage':row['stage'],'code':row['code'],'seed':row['seed'],
                         'panel':panel,'task':task,'metric':'pair_accuracy','value':m['pair_accuracy'],'questions_or_fields':None,'pairs':m['pairs'],
                         'support_note':'Pair-CE transfer proxy, not native Language Skill. Fixed label-stratified subset of an existing validation source.','source':str(file)})
    assert len(runs)==91
    count=Counter(x['stage'] for x in runs);normal=[x for x in runs if x['stage']=='fastscreen'];emergency=[x for x in runs if x['stage']=='engineering']
    assert len(normal)==39 and len(emergency)==4
    facts={'collected_utc':datetime.now(timezone.utc).isoformat(),'scope':'Completed results only; no new training/inference/scoring/hash audit.',
       'verification':'Source JSON values, arithmetic, checkpoint path existence, coverage/error metadata and cached-copy agreement; weights and request-level scoring not rerun.',
       'counts_by_stage':dict(count),'unique_reported_checkpoints':len(runs),'formal_research_checkpoints_including_R0':len(runs)-len(emergency),
       'unique_public_requests_per_full_evaluation':31715,'checkpoint_request_records_derived':31715*len(runs),
       'formal50M_continuations':47,'normal_fastscreen_models':len(normal),'engineering_validation_models':len(emergency),
       'normal_fastscreen_training_tokens':sum(x['new_tokens'] for x in normal),'emergency_validation_training_tokens':sum(x['new_tokens'] for x in emergency),
       'long_groups':long_groups,'fastscreen_rounds':rounds,'normalized_runs':runs,'stored_confirmation_measurements':list(panel_seen.values()),
       'source_documents':raw,'baseline_internal_test_source':str(PROJECT/'runs/language-baseline-20261027T061200Z/internal-test.json'),
       'diagnostics_manifest':read(RESEARCH/'lu-dev-diagnostics-20261028/diagnostic-results.json')}
    (OUT/'experiment-facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
    fields=['stage','round','code','name','seed','rank','LoRA_alpha','new_tokens','training_compute_tokens','optimizer_steps','language_skill','coverage','requests']+[f'skill_{cid}' for cid in IDS]+['checkpoint','source']
    with (OUT/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in runs:writer.writerow({**{k:row.get(k) for k in fields if not k.startswith('skill_')},**{f'skill_{cid}':row['per_benchmark'][cid]['skill'] for cid in IDS}})
    with (OUT/'dev-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['checkpoint','stage','code','seed','panel','task','metric','value','questions_or_fields','pairs','support_note','source']);writer.writeheader();writer.writerows(dev_rows)
    with (OUT/'round-timing.csv').open('w',newline='') as f:
        fields=[k for k in rounds[0] if k not in ['reviews']];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:x[k] for k in fields} for x in rounds)
    lines=['# 完整结果索引（本地已完成记录，非新增实验）','',f'共{len(runs)}个不同checkpoint；完整精度与逐项dev指标分别见runs.csv、dev-metrics.csv和experiment-facts.json。',
           '工程应急1M结果不进入方向性结论；任何复用控制均不重复计为新checkpoint。','',
           '## 50M受控研究：逐seed结果','','| 配置 | seed | Language Skill |','|---|---:|---:|']
    for row in runs[1:48]:lines.append(f"| {row['name']}（{row['code']}） | {row['seed']} | {row['language_skill']:.8f} |")
    lines+=['','## 每轮小池快筛：所有新模型','','配额百分比均指目标D2池token份额；比例D0:目标按实际token。Native类平衡不等于pair标签比例。']
    for item in rounds:
        lines+=['',f"### {item['round_name']}",f"准备记录{item['preparation_minutes_recorded']:.3f}分钟；主报告{item['primary_report_minutes']:.3f}分钟；全部评测{item['all_evaluations_completed_minutes']:.3f}分钟。",'',
                '| 配置（追溯代号） | seed | 实际新token | Language Skill |','|---|---:|---:|---:|']
        for row in runs:
            if row.get('round')==item['round_name']:lines.append(f"| {row['name']}（{row['code']}） | {row['seed']} | {row['new_tokens']} | {row['language_skill']:.8f} |")
        lines.append(f"原始主报告：`{item['root']}/ROUND-REPORT.md`；最终判断另见该目录复核文件，自动主报告提示不直接作为最终选择。")
    lines+=['','## 来源与数据保留','',
            'JSON保留源聚合文档及原始ledger字段；旧independent_dev字段名不保证早期面板独立，受控研究采用另存的clean-dev复核。',
            '所有逐请求原始结果和权重仍在各checkpoint对应的原实验目录，本次未复制巨型预测文件、重算官方分数或更改旧产物。']
    (OUT/'APPENDIX.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'collected_utc':facts['collected_utc'],'checkpoints':len(runs),'counts_by_stage':dict(count),'fastscreen_rounds':len(rounds),
       'normal_fastscreen_tokens':facts['normal_fastscreen_training_tokens'],'engineering_tokens':facts['emergency_validation_training_tokens'],
       'rounds':[{k:x[k] for k in ['round_name','models','primary_report_minutes','all_evaluations_completed_minutes','aggregate_evaluation_minutes','preparation_minutes_recorded','preparation_minutes_to_effective_final_attempt']} for x in rounds]},ensure_ascii=False))


if __name__=='__main__':main()
