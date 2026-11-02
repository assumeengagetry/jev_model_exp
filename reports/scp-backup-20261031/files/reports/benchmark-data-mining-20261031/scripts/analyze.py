#!/usr/bin/env python3
"""Read-only CPU data mining of staged Decision Index requests and token caches."""
import os
os.environ.setdefault('RAYON_NUM_THREADS', '6')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import collections
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
from tokenizers import Tokenizer

HOME = Path('/home/dev/assumeengage')
OUT = HOME / 'reports/benchmark-data-mining-20261031'
DATA = HOME / 'datasets/evaluation-only/decision-index-suite-0.3'
CACHE = DATA / 'candidate-cache'
KIT = HOME / 'src/decision-index-main'
CAPS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 40960]
EDGES = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 10**9]
BUCKETS = ['<128','128–255','256–511','512–1023','1024–2047','2048–4095','4096–8191','8192–16383','16384–32767','≥32768']
ZH = {'knowledge':'知识与推理','language':'语言理解','retrieval':'检索与分类','tools':'工具与自动化','arts':'艺术与人类偏好'}
START = time.monotonic()

def dump(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, (float,np.floating)): return round(float(v),2) if math.isfinite(v) else None
        return v
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2)+'\n')

def save(name, rows):
    frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/'tables'/name, index=False, float_format='%.2f')

def txt(v):
    if v is None:return ''
    return v.strip() if isinstance(v,str) else json.dumps(v,ensure_ascii=False,sort_keys=True)

def compact(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def lengths(tokenizer, values):
    result = np.empty(len(values),np.int32)
    for start in range(0,len(values),128):
        result[start:start+128] = [len(x.ids) for x in tokenizer.encode_batch(values[start:start+128],add_special_tokens=False)]
    return result

def stat(a,prefix):
    a=np.asarray(a)
    if len(a)==0:return {}
    q=np.quantile(a,[.5,.9,.95,.99])
    return {prefix+'_mean':a.mean(),prefix+'_p50':q[0],prefix+'_p90':q[1],prefix+'_p95':q[2],prefix+'_p99':q[3],prefix+'_max':a.max(),prefix+'_sum':a.sum()}

def expected_keys(row,field,question):
    gold=row.get('expected',{}).get(field)
    accepted=row.get('scoring',{}).get('accepted')
    if isinstance(accepted,dict): accepted=accepted.get(field)
    if accepted and isinstance(accepted,list):return set(map(str,accepted))
    if isinstance(gold,bool):return {'true' if gold else 'false'}
    if isinstance(gold,(str,int,float)):return {str(gold)}
    if isinstance(gold,list):return set(map(str,gold))
    return set()

def main():
    (OUT/'tables').mkdir(parents=True,exist_ok=True)
    panel=json.load(open(KIT/'decision_index/data/index-0.3.json'))
    catalog={x['catalog_id']:x for x in json.load(open(KIT/'decision_index/data/benchmarks.json'))}
    areas={int(i):area['id'] for area in panel['areas'] for i in area['benchmarks']}
    active=set(areas)
    # Added 0.2/0.3 benchmarks live in the index panel, not the legacy catalog.
    for bid in active:
        if bid not in catalog:
            catalog[bid]={'dataset':panel['chance'][str(bid)]['name'],'area':areas[bid],
                          'index_metric':{59:'response-level hallucination F1'}.get(bid,'accuracy')}
        override=panel.get('metrics',{}).get(str(bid),{})
        if override.get('name'):catalog[bid]['index_metric']=override['name']
    tokenizer=Tokenizer.from_file(str(HOME/'models/Qwen3-0.6B/tokenizer.json'))
    offsets={role:np.memmap(CACHE/'tokenized'/(role+'_offsets.bin'),dtype=np.uint64,mode='r') for role in ['prefix','suffix']}
    cached_lengths={role:np.diff(x).astype(np.int64) for role,x in offsets.items()}
    requests=[]; fields=[]; field_map={}; accepted=[]; schemas=collections.defaultdict(collections.Counter)
    state_lookup={}; states=[]; joint_text=[]; tracks=[]; retrieval=[]
    label_counts=collections.Counter(); native_counts=collections.Counter(); content_counts=collections.Counter()
    with gzip.open(CACHE/'normalized/rows-for-evaluation.jsonl.gz','rt') as stream:
        for ordinal,line in enumerate(stream):
            row=json.loads(line);e=row['_evaluation'];bid=int(e['catalog_id']);run=e['run_id']
            state=txt(row['state']);sid=state_lookup.get(state)
            if sid is None:sid=len(states);state_lookup[state]=sid;states.append(state)
            qtypes=collections.Counter(q['type'] for q in row['questions'].values())
            metadata=row.get('metadata',{}); scoring=row.get('scoring',{})
            track=str(e.get('track',metadata.get('track','default')))
            group=str(e.get('group_id',metadata.get('group_id',row.get('id',run))))
            rid=len(requests)
            requests.append(dict(request=rid,benchmark_id=bid,name=catalog[bid]['dataset'],active=bid in active,area=areas.get(bid,catalog[bid]['area']),state_id=sid,group=group,track=track,fields=len(row['questions']),state_kind=type(row['state']).__name__,state_chars=len(state),instructions_chars=sum(len(txt(q.get('instructions',''))) for q in row['questions'].values()),locale=metadata.get('locale',metadata.get('language','unspecified'))))
            joint_text.append(compact({'state':row['state'],'questions':row['questions']}))
            tracks.append((bid,track))
            for field,q in row['questions'].items():
                fid=len(fields);field_map[(run,field)]=fid
                options=q.get('criteria') or {'true':'Yes.','false':'No.'}
                options={str(k):txt(v) for k,v in options.items()}
                good=expected_keys(row,field,q)
                relevance=None; judged=False
                if scoring.get('type')=='retrieval_ranking':
                    doc=scoring.get('field_to_document',{}).get(field)
                    qrels=scoring.get('qrels',{})
                    judged=doc in qrels
                    if judged:
                        relevance=qrels[doc];good={'yes' if float(relevance)>0 else 'no'}
                    # Unjudged documents deliberately stay unknown, even if evaluation
                    # ranks them with zero gain. They are not confirmed negatives.
                    elif not good:good=set()
                good=good.intersection(options)
                accepted.append(good)
                keys=list(options);pos=[keys.index(k) for k in good]
                instructions=txt(q.get('instructions',''))
                fields.append(dict(field=fid,request=rid,benchmark_id=bid,kind=q['type'],options=len(options),known=bool(good),accepted=len(good),first_correct=0 in pos,last_correct=(len(keys)-1) in pos,correct_position_mean=np.mean(pos) if pos else np.nan,normalized_position=np.mean(pos)/max(1,len(keys)-1) if pos else np.nan,semantic_yes=bool(good.intersection({'yes','true'})),semantic_no=bool(good.intersection({'no','false'})),judged=judged,relevance=relevance,instructions_chars=len(instructions)))
                schema=tuple(options)
                schemas[bid][schema]+=1
                for k in good:
                    label_counts[(bid,track,k,options[k][:160])]+=1/len(good)
                    native_counts[(bid,track,k)]+=1/len(good)
            if scoring.get('type')=='retrieval_ranking':
                retrieved=set(scoring.get('retrieved_ids',[]));qrels=scoring.get('qrels',{})
                relevant={k for k,v in qrels.items() if float(v)>0}
                retrieval.append(dict(benchmark_id=bid,request=rid,group=group,track=track,candidate_recall=float(scoring.get('candidate_recall',0)),pool_candidates=len(retrieved),qrels_relevant=len(relevant),pool_relevant=len(retrieved&relevant),scorable_candidates=len(scoring.get('scorable_ids',[]))))
            if (ordinal+1)%20000==0:print('ROWS',ordinal+1,flush=True)
    request_df=pd.DataFrame(requests);field_df=pd.DataFrame(fields)
    print('REQUESTS',len(requests),'FIELDS',len(fields),'UNIQUE_STATES',len(states),flush=True)
    # Reuse actual separated prefix/suffix tokenization of the current pair classifier.
    pairs=pq.read_table(CACHE/'normalized/field-options.parquet').to_pydict()
    pfield=np.fromiter((field_map[(r,f)] for r,f in zip(pairs['run_id'],pairs['field'])),dtype=np.int32,count=len(pairs['run_id']))
    plen=cached_lengths['prefix'][pairs['state_id']]+cached_lengths['suffix'][pairs['candidate_id']]+1
    psuffix=cached_lengths['suffix'][pairs['candidate_id']]
    ptarget=np.fromiter(((-1 if not accepted[f] else int(str(k) in accepted[f])) for f,k in zip(pfield,pairs['option_key'])),dtype=np.int8,count=len(plen))
    pbid=field_df.benchmark_id.to_numpy()[pfield]
    prid=field_df.request.to_numpy()[pfield]
    pactive=np.isin(pbid,list(active))
    prefix_values=cached_lengths['prefix'][pairs['state_id']]
    field_df['pair_token_sum']=np.bincount(pfield,weights=plen,minlength=len(fields)).astype(np.int64)
    field_df['pair_max']=np.zeros(len(fields),dtype=np.int64);np.maximum.at(field_df['pair_max'].values,pfield,plen)
    field_df['pair_min']=np.full(len(fields),np.iinfo(np.int32).max,dtype=np.int64);np.minimum.at(field_df['pair_min'].values,pfield,plen)
    field_df['candidate_max']=np.zeros(len(fields),dtype=np.int64);np.maximum.at(field_df['candidate_max'].values,pfield,psuffix)
    field_df['candidate_min']=np.full(len(fields),np.iinfo(np.int32).max,dtype=np.int64);np.minimum.at(field_df['candidate_min'].values,pfield,psuffix)
    field_df['candidate_mean']=np.bincount(pfield,weights=psuffix,minlength=len(fields))/field_df.options
    gold_len_sum=np.bincount(pfield[ptarget==1],weights=psuffix[ptarget==1],minlength=len(fields))
    field_df['gold_candidate_mean']=np.divide(gold_len_sum,field_df.accepted,out=np.full(len(fields),np.nan),where=field_df.accepted>0)
    longest=psuffix==field_df.candidate_max.to_numpy()[pfield];shortest=psuffix==field_df.candidate_min.to_numpy()[pfield]
    for name,mask in [('longest',longest),('shortest',shortest)]:
        den=np.bincount(pfield[mask],minlength=len(fields));num=np.bincount(pfield[mask&(ptarget==1)],minlength=len(fields))
        field_df[name+'_answer_reference']=np.divide(num,den,out=np.full(len(fields),np.nan),where=den>0)
        field_df.loc[~field_df.known,name+'_answer_reference']=np.nan
    print('PAIR_LENGTHS_READY',len(plen),flush=True)
    # Two exact, engine-independent text measurements. No expected/scoring/metadata
    # is ever placed into tokenized input.
    slen=lengths(tokenizer,states);print('STATE_TOKENIZATION_READY',flush=True)
    request_df['state_tokens']=slen[request_df.state_id]
    joint_len=lengths(tokenizer,joint_text);del joint_text;del states;del state_lookup
    request_df['joint_tokens']=joint_len
    req_group=field_df.groupby('request',sort=False)
    for target,col,op in [('pair_max','pair_max','max'),('pair_token_sum','pair_token_sum','sum'),('pairs','options','sum'),('options_max','options','max')]:
        request_df[target]=getattr(req_group[col],op)().reindex(request_df.request).to_numpy()
    print('JOINT_TOKENIZATION_READY',flush=True)
    weights={}
    for area in panel['areas']:
        denom=sum(float(panel['gold'].get(str(i),1)) for i in area['benchmarks'])
        for i in area['benchmarks']:weights[i]=100*panel['area_sizing']['fixed'][area['id']]*float(panel['gold'].get(str(i),1))/denom
    benchmark_rows=[];coverage=[];buckets=[];native_table=[];label_table=[];pair_buckets=[]
    for bid,rdf in request_df.groupby('benchmark_id',sort=True):
        fdf=field_df[field_df.benchmark_id==bid];mask=pbid==bid;pa=plen[mask];tar=ptarget[mask];suffix=psuffix[mask]
        known=fdf[fdf.known];truth=fdf[fdf.semantic_yes|fdf.semantic_no]
        state_counts=rdf.state_id.value_counts();group_counts=rdf.group.value_counts()
        by_key=collections.Counter()
        for (b,t,k),v in native_counts.items():
            if b==bid:by_key[k]+=v
        counts=np.array(list(by_key.values()));dist=counts/counts.sum() if counts.sum() else np.array([])
        entropy=-sum(dist*np.log2(dist)) if len(dist) else np.nan
        known_pair=tar>=0;pos=tar==1;neg=tar==0
        auc=np.nan
        if pos.any() and neg.any():
            ranks=rankdata(pa[known_pair]);n1=pos.sum();n0=neg.sum();auc=(ranks[tar[known_pair]==1].sum()-n1*(n1+1)/2)/(n1*n0)
        item=dict(benchmark_id=int(bid),name=catalog[bid]['dataset'],active=bid in active,area=areas.get(bid,catalog[bid]['area']),area_zh=ZH[areas.get(bid,catalog[bid]['area'])],metric=catalog[bid]['index_metric'],requests=len(rdf),fields=len(fdf),pairs=len(pa),groups=len(group_counts),unique_states=len(state_counts),repeated_state_pct=100*(1-len(state_counts)/len(rdf)),largest_state_reuse=state_counts.max(),group_mean=group_counts.mean(),group_max=group_counts.max(),weight_pct=weights.get(bid,0),tracks=rdf.track.nunique(),known_fields=len(known),unknown_fields=len(fdf)-len(known),known_pairs=known_pair.sum(),positive_pairs=pos.sum(),negative_pairs=neg.sum(),unknown_pairs=(~known_pair).sum(),match_positive_pct=100*pos.sum()/max(1,known_pair.sum()),semantic_yes_fields=len(truth[truth.semantic_yes]),semantic_no_fields=len(truth[truth.semantic_no]),semantic_yes_pct=100*truth.semantic_yes.mean() if len(truth) else np.nan,noul_fields=(fdf.kind=='noul').sum(),choice_fields=(fdf.kind=='choice').sum(),multi_answer_fields=(fdf.accepted>1).sum(),option_schema_count=len(schemas[bid]),answer_key_entropy_bits=entropy,answer_key_majority_pct=100*counts.max()/counts.sum() if len(counts) else np.nan,first_answer_pct=100*known.first_correct.mean(),last_answer_pct=100*known.last_correct.mean(),normalized_gold_position_pct=100*known.normalized_position.mean(),longest_candidate_reference_pct=100*known.longest_answer_reference.mean(),shortest_candidate_reference_pct=100*known.shortest_answer_reference.mean(),length_only_match_auc=auc,positive_token_share_pct=100*pa[pos].sum()/max(1,pa[known_pair].sum()),positive_pair_mean=pa[pos].mean() if pos.any() else np.nan,negative_pair_mean=pa[neg].mean() if neg.any() else np.nan,gold_candidate_mean=known.gold_candidate_mean.mean(),negative_candidate_mean=suffix[neg].mean() if neg.any() else np.nan,structured_state_pct=100*rdf.state_kind.isin(['dict','list']).mean())
        for label,values in [('state',rdf.state_tokens),('joint',rdf.joint_tokens),('request_pair_max',rdf.pair_max),('pair',pa),('options',fdf.options),('fields_per_request',rdf.fields),('pairs_per_request',rdf.pairs),('candidate',suffix)]:item.update(stat(values,label))
        item['pair_to_joint_token_ratio']=pa.sum()/max(1,rdf.joint_tokens.sum())
        for cap in [26,62,128]:item[f'fields_options_gt{cap}_pct']=100*(fdf.options>cap).mean()
        for cap in CAPS:
            coverage.append(dict(benchmark_id=bid,name=item['name'],active=item['active'],area=item['area'],cap=cap,request_pair_covered_pct=100*(rdf.pair_max<=cap).mean(),joint_covered_pct=100*(rdf.joint_tokens<=cap).mean(),field_pair_covered_pct=100*(fdf.pair_max<=cap).mean(),pair_covered_pct=100*(pa<=cap).mean(),state_covered_pct=100*(rdf.state_tokens<=cap).mean(),request_pair_over_count=int((rdf.pair_max>cap).sum()),joint_over_count=int((rdf.joint_tokens>cap).sum()),pair_over_count=int((pa>cap).sum())))
        for label,vals in [('joint',rdf.joint_tokens),('request_pair_max',rdf.pair_max),('pair',pa),('state',rdf.state_tokens)]:
            hist=np.histogram(vals,bins=EDGES)[0]
            for bucket,n in zip(BUCKETS,hist):buckets.append(dict(benchmark_id=bid,name=item['name'],active=item['active'],area=item['area'],unit=label,bucket=bucket,count=n,pct=100*n/len(vals)))
        for lower,upper,bucket in zip(EDGES[:-1],EDGES[1:],BUCKETS):
            ix=(pa>=lower)&(pa<upper);n=ix.sum();nk=(ix&known_pair).sum()
            pair_buckets.append(dict(benchmark_id=bid,name=item['name'],active=item['active'],bucket=bucket,pairs=n,positive=(ix&pos).sum(),negative=(ix&neg).sum(),unknown=(ix&~known_pair).sum(),match_positive_pct=100*(ix&pos).sum()/nk if nk else np.nan))
        benchmark_rows.append(item)
        for (b,track,k,description),n in label_counts.items():
            if b==bid:label_table.append(dict(benchmark_id=bid,name=item['name'],active=item['active'],track=track,label=k,description=description,count=n))
        for track,tdf in fdf.assign(track=rdf.set_index('request').track.reindex(fdf.request).to_numpy()).groupby('track'):
            tc={k:v for (b,t,k),v in native_counts.items() if b==bid and t==track}
            for k,n in tc.items():native_table.append(dict(benchmark_id=bid,name=item['name'],active=item['active'],track=track,label=k,count=n,pct=100*n/max(1,sum(tc.values()))))
    bdf=pd.DataFrame(benchmark_rows);main_df=bdf[bdf.active].copy()
    area_rows=[]
    for area in ZH:
        bids=set(main_df[main_df.area==area].benchmark_id);rdf=request_df[request_df.benchmark_id.isin(bids)];fdf=field_df[field_df.benchmark_id.isin(bids)];pmask=np.isin(pbid,list(bids));p=plen[pmask];t=ptarget[pmask];truth=fdf[fdf.semantic_yes|fdf.semantic_no]
        item=dict(area=area,area_zh=ZH[area],benchmarks=len(bids),requests=len(rdf),fields=len(fdf),pairs=len(p),positive_pairs=(t==1).sum(),negative_pairs=(t==0).sum(),unknown_pairs=(t<0).sum(),match_positive_pct=100*(t==1).sum()/max(1,(t>=0).sum()),semantic_yes_pct=100*truth.semantic_yes.mean() if len(truth) else np.nan,weight_pct=100*panel['area_sizing']['fixed'][area],joint_token_share_pct=100*rdf.joint_tokens.sum()/request_df[request_df.active].joint_tokens.sum(),pair_token_share_pct=100*p.sum()/plen[pactive].sum())
        for prefix,values in [('joint',rdf.joint_tokens),('pair',p),('options',fdf.options),('fields_per_request',rdf.fields)]:item.update(stat(values,prefix))
        for cap in [512,1024,2048,4096,8192,16384]:item[f'pair_request_over_{cap}_pct']=100*(rdf.pair_max>cap).mean();item[f'joint_over_{cap}_pct']=100*(rdf.joint_tokens>cap).mean()
        area_rows.append(item)
    save('benchmarks.csv',bdf);save('main-37.csv',main_df);save('archived-6.csv',bdf[~bdf.active]);save('areas.csv',area_rows);save('coverage.csv',coverage);save('length-buckets.csv',buckets);save('pair-label-by-length.csv',pair_buckets);save('answer-labels.csv',label_table);save('answer-labels-by-track.csv',native_table);save('retrieval-pools.csv',retrieval)
    save('requests.csv.gz',request_df.drop(columns=['group','state_id']))
    save('fields.csv.gz',field_df.drop(columns=['field']))
    # Within each benchmark: length-label association, not a causal performance claim.
    correlations=[]
    numeric=['joint_p95','request_pair_max_p95','options_mean','fields_per_request_mean','match_positive_pct','first_answer_pct','pair_to_joint_token_ratio']
    for a in numeric:
        for b in numeric:
            subset=main_df[[a,b]].dropna() if a!=b else main_df[[a]].dropna()
            if a==b:r=1.;n=len(subset)
            else:r=float(spearmanr(subset[a],subset[b]).statistic);n=len(subset)
            correlations.append(dict(x=a,y=b,spearman=r,benchmarks=n))
    save('feature-correlations.csv',correlations)
    # Actual in-edition row counts (including exclusions), kept separately from
    # completed candidate cache to expose the source scope, not infer parity.
    sys.path.insert(0,str(KIT))
    from decision_index.suite.io import Suite
    before=collections.Counter(int(r['_evaluation']['catalog_id']) for r in Suite(DATA,edition='0.3').rows(apply_exclusions=False))
    source_rows=[dict(benchmark_id=int(bid),name=catalog[bid]['dataset'],active=bid in active,in_edition_requests=int(before[bid]),cached_requests=int(len(rdf)),removed_requests=int(before[bid]-len(rdf))) for bid,rdf in request_df.groupby('benchmark_id')]
    save('source-scope.csv',source_rows)
    rdf=request_df[request_df.active];fdf=field_df[field_df.benchmark_id.isin(active)];p=plen[pactive];t=ptarget[pactive]
    summary=dict(date='2026-10-31',tokenizer='Qwen3-0.6B',public_benchmarks=len(main_df),archive_benchmarks=len(bdf),requests=len(rdf),fields=len(fdf),candidate_pairs=len(p),positive_pairs=int((t==1).sum()),negative_pairs=int((t==0).sum()),unknown_pairs=int((t<0).sum()),match_positive_pct=100*(t==1).sum()/max(1,(t>=0).sum()),positive_token_share_pct=100*p[t==1].sum()/max(1,p[t>=0].sum()),noul_fields=int((fdf.kind=='noul').sum()),choice_fields=int((fdf.kind=='choice').sum()),score_fields=int((fdf.kind=='score').sum()),raw_cached_requests=len(request_df),raw_cached_fields=len(field_df),raw_cached_pairs=len(plen),main_in_edition_requests=sum(before[i] for i in active),cache_removed_requests=sum(before[i] for i in active)-len(rdf),joint_total_tokens=int(rdf.joint_tokens.sum()),pair_total_tokens=int(p.sum()),pair_to_joint_token_ratio=p.sum()/rdf.joint_tokens.sum(),pair_max=int(p.max()),joint_max=int(rdf.joint_tokens.max()),elapsed_seconds=time.monotonic()-START,areas=area_rows,coverage=[])
    summary.update(stat(rdf.joint_tokens,'joint'));summary.update(stat(p,'pair'));summary.update(stat(rdf.pair_max,'request_pair_max'))
    for cap in CAPS:summary['coverage'].append(dict(cap=cap,pair_request_covered_pct=100*(rdf.pair_max<=cap).mean(),joint_covered_pct=100*(rdf.joint_tokens<=cap).mean(),pair_covered_pct=100*(p<=cap).mean(),benchmark_macro_pair_request_covered_pct=float(pd.DataFrame(coverage).query('active and cap == @cap').request_pair_covered_pct.mean())))
    dump(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['areas','coverage']},ensure_ascii=False,default=lambda x:int(x) if isinstance(x,np.integer) else round(float(x),2)),flush=True)

if __name__=='__main__':main()
