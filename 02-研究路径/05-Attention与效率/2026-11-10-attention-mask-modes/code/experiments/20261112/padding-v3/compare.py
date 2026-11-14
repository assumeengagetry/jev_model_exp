"""Direct streaming comparison of two new full outputs; never selects a winner checkpoint."""
from collections import Counter
import json
import math
import sys
from pf_common import ROOT,KIT,read,atomic,require,metadata,verify


def main():
    verify()
    sys.path.insert(0,str(KIT))
    from decision_index.suite.io import read_jsonl
    import numpy as np
    dense=ROOT/'eval/dense';packed=ROOT/'eval/packed'
    receipts={name:read(ROOT/'eval'/name/'COMPLETE.json') for name in ('dense','packed')}
    for value in receipts.values():
        require(value['complete'] and value['executed_requests']==140620 and value['scored_requests']==140178,'Incomplete full output')
        for record in value['files']:require(metadata(record['path'])==record,'Bound official result changed')
    right=iter(read_jsonl(packed/'results.jsonl'))
    margins={name:[iter(read_jsonl(ROOT/'eval'/name/'workers'/str(rank)/'margins.jsonl')) for rank in range(4)] for name in ('dense','packed')}
    counts=Counter();latencies={'dense':[],'packed':[]};max_probability=0.;sum_probability=0.;probability_count=0
    max_margin_error=0.;worst_margin_relative_rms=0.
    path=ROOT/'PREDICTION_DIFFERENCES.jsonl'
    with path.open('x') as differences:
        for a in read_jsonl(dense/'results.jsonl'):
            require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'PF stop requested during comparison')
            b=next(right,None)
            require(b is not None and a['run_id']==b['run_id'] and a['payload']==b['payload'],'Dense/packed input/order differs')
            require(a['status']==b['status']=='ok','Non-ok actual output')
            counts['requests']+=1
            ordinal=counts['requests']-1;rank=ordinal%4;shard_ordinal=ordinal//4
            ma=next(margins['dense'][rank],None);mb=next(margins['packed'][rank],None)
            require(ma is not None and mb is not None and ma['request_ordinal']==mb['request_ordinal']==shard_ordinal and ma['fields']==mb['fields'],'Margin sidecar request/field misalignment')
            va=np.asarray(ma['margins'],dtype=np.float64);vb=np.asarray(mb['margins'],dtype=np.float64)
            require(va.shape==vb.shape and va.ndim==1 and np.isfinite(va).all() and np.isfinite(vb).all(),'Invalid raw margins')
            delta=vb-va;rms=float(np.sqrt(np.mean(va*va)));error=float(np.sqrt(np.mean(delta*delta)))
            maxabs=float(np.max(np.abs(va)));maxerr=float(np.max(np.abs(delta)))
            max_margin_error=max(max_margin_error,maxerr);worst_margin_relative_rms=max(worst_margin_relative_rms,error/max(rms,1e-3))
            if error>.01*max(rms,1e-3) or maxerr>.02+.02*maxabs:counts['numeric_contract_violations']+=1
            for name,row in [('dense',a),('packed',b)]:
                value=row['model_request_wall_ms']
                require(isinstance(value,(float,int)) and math.isfinite(value) and value>=0,'Missing/invalid measured request latency')
                latencies[name].append(float(value))
            aa=a['response']['answers'];bb=b['response']['answers']
            require(list(aa)==list(bb),'Field order differs')
            changed=[]
            for field_index,(field,answer) in enumerate(aa.items()):
                other=bb[field];require(answer['type']==other['type'],'Answer type differs')
                pa=answer['probabilities'];pb=other['probabilities'];require(list(pa)==list(pb),'Criteria differ')
                errs=[abs(pa[k]-pb[k]) for k in pa]
                max_probability=max(max_probability,max(errs));sum_probability+=sum(errs);probability_count+=len(errs)
                counts['fields']+=1
                if answer.get('choice')!=other.get('choice'):
                    counts['choice_flips']+=1
                    field_meta=ma['fields'][field_index]
                    require(field_meta['name']==field and list(pa)==field_meta['keys'],'Raw margin criteria alignment changed')
                    begin=field_meta['begin'];raw=va[begin:begin+len(pa)];ordered=np.sort(raw)
                    gap=float(ordered[-1]-ordered[-2]) if len(raw)>1 else None
                    band=.04+.04*float(np.max(np.abs(raw)))
                    outside=gap is not None and gap>band
                    if outside:counts['outside_band_choice_flips']+=1
                    changed.append({'field':field,'dense':answer,'packed':other,'dense_top_margin':gap,
                        'frozen_numerical_band':band,'outside_band':outside,'dense_raw_margins':raw.tolist(),
                        'packed_raw_margins':vb[begin:begin+len(pa)].tolist()})
            if changed:
                counts['requests_with_choice_flips']+=1
                differences.write(json.dumps({'run_id':a['run_id'],'changes':changed},allow_nan=False)+'\n')
        require(next(right,None) is None and counts['requests']==140620 and all(next(s,None) is None for values in margins.values() for s in values),'Missing/extra compared requests or sidecars')
    scores={name:read(ROOT/'eval'/name/'scores.json') for name in ('dense','packed')}
    timing={name:read(ROOT/'eval'/name/'TIMING.json') for name in ('dense','packed')}
    require(all(len(v)==140620 for v in latencies.values()),'Incomplete actual latency measurements')
    latency={name:{'field':'model_request_wall_ms','count':len(values),'mean_ms':float(np.mean(values)),'p80_ms':float(np.percentile(values,80))} for name,values in latencies.items()}
    accepted=counts['numeric_contract_violations']==0 and counts['outside_band_choice_flips']==0
    verify()
    atomic(ROOT/'COMPARISON.json',{'complete':accepted,'coverage_complete':True,'numerical_acceptance':accepted,'counts':dict(counts),
        'public_index':{k:v['decision_index'] for k,v in scores.items()},
        'max_abs_probability_difference':max_probability,'mean_abs_probability_difference':sum_probability/probability_count,
        'max_abs_margin_difference':max_margin_error,'worst_margin_relative_rms':worst_margin_relative_rms,
        'latency':latency,'timing':timing,'prediction_differences':metadata(path),
        'interpretation':'same-weight layout/numerics diagnostic, not training improvement or clean ranking',
        'new_training_score':None,'official_single_gpu_latency_admission':False})
    print(json.dumps(read(ROOT/'COMPARISON.json'),ensure_ascii=False,indent=2),flush=True)
    require(accepted,'Full coverage complete, numerical contract violated; investigate preserved outputs')


if __name__=='__main__':main()
