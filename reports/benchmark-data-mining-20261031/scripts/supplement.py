#!/usr/bin/env python3
"""Native classes, source clusters and optional public-score associations."""
import collections
import gzip
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HOME=Path('/home/dev/assumeengage')
ROOT=HOME/'reports/benchmark-data-mining-20261031'
RAW=HOME/'datasets/evaluation-only/decision-index-suite-0.3/candidate-cache/normalized/rows-for-evaluation.jsonl.gz'
FIXED={1,4,5,11,12,22,33,37,38,39,40,41,42,44,56,59,61,62}
CLUSTERS={20:['group_id'],22:['song_id'],23:['user_id'],30:['group_id','answer_group'],38:['group_id'],42:['trial_ids'],50:['group_id','question_id'],59:['source_task_id'],61:['hpqa_id'],62:['bfcl_source_id'],64:['contest_number']}

def txt(v):return v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,sort_keys=True)
def keylabel(bid,key,description,metadata):
    if bid==62:return metadata.get('answer_key_map',{}).get(key,key)
    if bid==12:
        low=description.lower()
        if 'not necessarily' in low or 'neutral' in low or 'neither' in low:return 'Neutral'
        if 'contradict' in low:return 'Contradiction'
        if 'entail' in low:return 'Entailment'
    if bid in {4,5,22,41,44}:return description
    if bid==37:return {'E':'Exact','S':'Substitute','C':'Complement','I':'Irrelevant'}.get(key,key)
    return key

def main():
    classes=collections.Counter();denom=collections.Counter();roles=collections.Counter();clusters=collections.defaultdict(collections.Counter);shapes=collections.Counter();locales=collections.Counter();bfcl=set();when=set();live_source_counts=collections.Counter();yes_per_request=[]
    state_bench=collections.defaultdict(collections.Counter);payloads=collections.defaultdict(set)
    with gzip.open(RAW,'rt') as stream:
        for rid,line in enumerate(stream):
            r=json.loads(line);e=r['_evaluation'];bid=int(e['catalog_id']);m=r.get('metadata',{});state=r['state'];track=str(e.get('track',m.get('track','default')));source=txt(state)
            payload=txt({'state':state,'questions':r['questions']})
            payloads[bid].add(payload)
            shapes[(bid,'requests')]+=1
            for label,flag in [('empty_state',not state),('arabic_script',bool(re.search('[\u0600-\u06ff]',payload))),('cjk_script',bool(re.search('[\u4e00-\u9fff]',payload))),('structured_state',isinstance(state,(dict,list)))]:shapes[(bid,label)]+=int(flag)
            if len(source)>=80:state_bench[source][bid]+=1
            locale=m.get('locale',m.get('language',state.get('locale','unspecified') if isinstance(state,dict) else 'unspecified'));locales[(bid,str(locale))]+=1
            if bid==1:bfcl.add(str(m.get('provenance',{}).get('source_id','')))
            if bid==62:when.add(str(m.get('bfcl_source_id','')))
            for ck in CLUSTERS.get(bid,[]):
                value=m.get(ck)
                if value is None:continue
                if ck=='answer_group':value=value.get('id')
                values=value if isinstance(value,list) else [value]
                for value in values:clusters[(bid,ck)][str(value)]+=1
            n_yes=n_no=n_unknown=0
            for field,q in r['questions'].items():
                gold=r.get('expected',{}).get(field)
                if gold is None:n_unknown+=1;continue
                if isinstance(gold,bool):gold='true' if gold else 'false'
                gold=str(gold);options=q.get('criteria',{});description=txt(options.get(gold,gold))
                if gold in {'yes','true'}:n_yes+=1
                elif gold in {'no','false'}:n_no+=1
                if bid in FIXED:
                    label=keylabel(bid,gold,description,m)
                    # Phish headline is verdict; other mirrored/signal fields are
                    # retained separately instead of changing its class balance.
                    scope='headline' if (bid==56 and field=='verdict') or (bid==40 and track=='iSarcasmEval-A-En') else ('auxiliary' if bid==56 else 'all')
                    classes[(bid,track,scope,label)]+=1;denom[(bid,track,scope)]+=1
                if bid==9:
                    role=m.get('question_roles',{}).get(field,'unspecified');role=txt(role)
                    roles[(bid,role,gold,description[:120])]+=1
            if n_yes+n_no:yes_per_request.append(dict(request=rid,benchmark_id=bid,track=track,yes=n_yes,no=n_no,unknown=n_unknown,yes_pct=100*n_yes/(n_yes+n_no)))
    bdf=pd.read_csv(ROOT/'tables/benchmarks.csv');names=bdf.set_index('benchmark_id').name.to_dict();active=set(bdf[bdf.active].benchmark_id)
    bdf.loc[bdf.benchmark_id==38,'metric']='per-review F1'
    def save(name,rows):pd.DataFrame(rows).to_csv(ROOT/'tables'/name,index=False,float_format='%.2f')
    # Aggregate answer-key entropy across tracks: tracks are not new classes.
    label_rows=pd.read_csv(ROOT/'tables/answer-labels-by-track.csv')
    for bid,g in label_rows.groupby('benchmark_id'):
        counts=g.groupby('label')['count'].sum();p=counts/counts.sum()
        bdf.loc[bdf.benchmark_id==bid,'answer_key_entropy_bits']=float(-(p*np.log2(p)).sum())
        bdf.loc[bdf.benchmark_id==bid,'answer_key_majority_pct']=100*counts.max()/counts.sum()
    save('benchmarks.csv',bdf);save('main-37.csv',bdf[bdf.active]);save('archived-6.csv',bdf[~bdf.active])
    save('native-classes.csv',[dict(benchmark_id=b,name=names[b],active=b in active,track=t,scope=s,label=k,count=n,pct=100*n/denom[(b,t,s)]) for (b,t,s,k),n in classes.items()])
    save('home-question-roles.csv',[dict(benchmark_id=b,role=role,label=k,description=description,count=n) for (b,role,k,description),n in roles.items()])
    save('source-clusters.csv',[dict(benchmark_id=b,name=names[b],active=b in active,cluster_key=ck,clusters=len(counter),request_memberships=sum(counter.values()),largest_cluster=max(counter.values()),median_cluster=np.median(list(counter.values())),repeated_membership_pct=100*sum(v for v in counter.values() if v>1)/sum(counter.values())) for (b,ck),counter in clusters.items()])
    save('input-content.csv',[dict(benchmark_id=b,name=names[b],active=b in active,requests=shapes[(b,'requests')],unique_full_payloads=len(payloads[b]),duplicate_full_payload_pct=100*(1-len(payloads[b])/shapes[(b,'requests')]),empty_state_pct=100*shapes[(b,'empty_state')]/shapes[(b,'requests')],arabic_script_pct=100*shapes[(b,'arabic_script')]/shapes[(b,'requests')],cjk_script_pct=100*shapes[(b,'cjk_script')]/shapes[(b,'requests')],structured_state_pct=100*shapes[(b,'structured_state')]/shapes[(b,'requests')]) for b in names])
    save('locale-metadata.csv',[dict(benchmark_id=b,name=names[b],active=b in active,locale=l,requests=n) for (b,l),n in locales.items()])
    save('yes-per-request.csv.gz',yes_per_request)
    cross=collections.Counter()
    for value in state_bench.values():
        bids=sorted(b for b in value if b in active)
        for ai,a in enumerate(bids):
            for b in bids[ai+1:]:cross[(a,b)]+=1
    save('cross-benchmark-identical-states.csv',[dict(benchmark_a=a,name_a=names[a],benchmark_b=b,name_b=names[b],shared_distinct_nontrivial_states=n) for (a,b),n in cross.items()] or [dict(benchmark_a='',name_a='',benchmark_b='',name_b='',shared_distinct_nontrivial_states=0)])
    bfcl.discard('');when.discard('');overlap=bfcl&when
    context=dict(bfcl_unique_source_ids=len(bfcl),when2call_unique_bfcl_source_ids=len(when),exact_source_id_overlap=len(overlap),when2call_source_overlap_pct=round(100*len(overlap)/max(1,len(when)),2),nontrivial_cross_benchmark_state_pairs=len(cross),definition='Exact original source-ID intersection only; no approximate or semantic overlap inference.')
    (ROOT/'source-overlap-summary.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
    rdf=pd.read_csv(ROOT/'tables/requests.csv.gz');fdf=pd.read_csv(ROOT/'tables/fields.csv.gz')
    cv=pd.read_csv(ROOT/'tables/coverage.csv')
    for (bid,cap),ix in cv.groupby(['benchmark_id','cap']).groups.items():
        xs=rdf[rdf.benchmark_id==bid]
        cv.loc[ix,'request_pair_over_count']=int((xs.pair_max>cap).sum())
        cv.loc[ix,'joint_over_count']=int((xs.joint_tokens>cap).sum())
    save('coverage.csv',cv)
    summary=json.load(open(ROOT/'summary.json'));main_requests=rdf[rdf.active]
    for row in summary['coverage']:
        cap=row['cap'];row['request_pair_over_count']=int((main_requests.pair_max>cap).sum());row['joint_over_count']=int((main_requests.joint_tokens>cap).sum())
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    track_rows=[]
    for (bid,track),xs in rdf.groupby(['benchmark_id','track']):
        fs=fdf[fdf.request.isin(xs.request)]
        track_rows.append(dict(benchmark_id=bid,name=names[bid],active=bid in active,track=track,requests=len(xs),fields=len(fs),joint_p50=xs.joint_tokens.median(),joint_p95=xs.joint_tokens.quantile(.95),request_pair_max_p95=xs.pair_max.quantile(.95),semantic_yes_fields=int(fs.semantic_yes.sum()),semantic_no_fields=int(fs.semantic_no.sum()),unknown_fields=int((~fs.known).sum()),positive_pairs=int(fs.accepted.sum()),known_pairs=int(fs.loc[fs.known,'options'].sum()),options_mean=fs.options.mean()))
    save('tracks.csv',track_rows)
    leader=ROOT/'tables/leaderboard-benchmark-summary.csv'
    if leader.exists():
        scores=pd.read_csv(leader)
        merged=bdf[bdf.active].merge(scores,on='benchmark_id',suffixes=('','_board'))
        save('data-vs-public-skill.csv',merged)
        metrics=['joint_p95','request_pair_max_p95','options_mean','fields_per_request_mean','match_positive_pct','pair_to_joint_token_ratio','first_answer_pct']
        correlations=[]
        for band,g in merged.groupby('band'):
            for metric in metrics:
                for score in ['skill_median','complete_skill_median','coverage_mean_pct']:
                    subset=g[[metric,score]].dropna()
                    r=float(spearmanr(subset[metric],subset[score]).statistic) if len(subset)>2 and subset[score].nunique()>1 else np.nan
                    correlations.append(dict(model_band=band,data_feature=metric,outcome=score,benchmark_count=len(subset),spearman=r))
        save('data-public-score-correlations.csv',correlations)
    print(json.dumps(context,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
