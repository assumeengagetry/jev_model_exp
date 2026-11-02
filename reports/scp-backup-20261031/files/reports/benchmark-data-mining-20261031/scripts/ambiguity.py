#!/usr/bin/env python3
"""Measure exact-input reuse and inconsistent hard targets, without model calls."""
import collections
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HOME=Path('/home/dev/assumeengage')
ROOT=HOME/'reports/benchmark-data-mining-20261031'
CACHE=HOME/'datasets/evaluation-only/decision-index-suite-0.3/candidate-cache/normalized'
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def main():
    main=pd.read_csv(ROOT/'tables/main-37.csv');names=main.set_index('benchmark_id').name.to_dict();active=set(names)
    groups=collections.defaultdict(dict);field_lookup={};field_info=[]
    with gzip.open(CACHE/'rows-for-evaluation.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);e=row['_evaluation'];bid=int(e['catalog_id'])
            if bid not in active:continue
            payload=canonical({'state':row['state'],'questions':row['questions']})
            signature={};scoring=row.get('scoring',{})
            for field,q in row['questions'].items():
                raw=row.get('expected',{}).get(field);accepted=scoring.get('accepted')
                if isinstance(accepted,dict):accepted=accepted.get(field)
                if isinstance(accepted,list) and accepted:good=set(map(str,accepted))
                elif isinstance(raw,bool):good={'true' if raw else 'false'}
                elif isinstance(raw,(str,int,float)):good={str(raw)}
                elif isinstance(raw,list):good=set(map(str,raw))
                else:good=set()
                options=q.get('criteria') or {'true':'Yes.','false':'No.'};good&=set(options)
                if scoring.get('type')=='retrieval_ranking':
                    doc=scoring.get('field_to_document',{}).get(field);qrels=scoring.get('qrels',{})
                    good=({'yes' if float(qrels[doc])>0 else 'no'} if doc in qrels else good)
                field_lookup[(e['run_id'],field)]=len(field_info);field_info.append((bid,good))
                if good:signature[field]=sorted(good)
            # Ranking relevance has unknown negatives and a different scoring
            # unit, so do not call differing qrels "contradictory class labels".
            if signature and bid not in {2,36}:
                target=canonical(signature);counter=groups[bid].setdefault(payload,collections.Counter());counter[target]+=1
    pairs=pq.read_table(CACHE/'field-options.parquet').to_pydict();keys=collections.defaultdict(list);labels=collections.defaultdict(list)
    for run,field,option,sid,cid in zip(pairs['run_id'],pairs['field'],pairs['option_key'],pairs['state_id'],pairs['candidate_id']):
        fid=field_lookup.get((run,field))
        if fid is None:continue
        bid,good=field_info[fid]
        if not good:continue
        keys[bid].append((int(sid)<<32)|int(cid));labels[bid].append(int(str(option) in good))
    rows=[]
    for bid,name in names.items():
        gs=groups[bid];n=sum(sum(c.values()) for c in gs.values());conf=[c for c in gs.values() if len(c)>1]
        key=np.asarray(keys[bid],dtype=np.uint64);target=np.asarray(labels[bid],dtype=np.int8)
        if len(key):
            unique,inverse=np.unique(key,return_inverse=True);pos=np.bincount(inverse,weights=target,minlength=len(unique));total=np.bincount(inverse,minlength=len(unique));neg=total-pos;conflict=(pos>0)&(neg>0)
            conflicting_pairs=int(total[conflict].sum());pair_conflict_groups=int(conflict.sum());pair_bound=100*np.maximum(pos,neg).sum()/len(key)
        else:unique=[];conflicting_pairs=pair_conflict_groups=0;pair_bound=np.nan
        rows.append(dict(benchmark_id=bid,name=name,labeled_requests=n,distinct_labeled_full_payloads=len(gs),full_payload_conflicting_groups=len(conf),full_payload_conflicting_requests=sum(sum(c.values()) for c in conf),label_vector_agreement_ceiling_pct=100*sum(max(c.values()) for c in gs.values())/n if n else np.nan,known_candidate_pairs=len(key),unique_actual_pair_inputs=len(unique),repeated_actual_pair_pct=100*(1-len(unique)/len(key)) if len(key) else np.nan,actual_pair_conflicting_groups=pair_conflict_groups,actual_pair_conflicting_presentations=conflicting_pairs,actual_pair_hard_label_accuracy_ceiling_pct=pair_bound))
    out=pd.DataFrame(rows);out.to_csv(ROOT/'tables/input-label-ambiguity.csv',index=False,float_format='%.2f')
    print(out[['name','full_payload_conflicting_groups','actual_pair_conflicting_groups']].query('full_payload_conflicting_groups>0 or actual_pair_conflicting_groups>0').to_string(index=False),flush=True)

if __name__=='__main__':main()
