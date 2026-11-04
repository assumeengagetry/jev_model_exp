#!/usr/bin/env python3
"""Scientific analysis of existing targets; no extra model inference or relabeling."""
import json
import time
import numpy as np
from kd_runtime import ROOT,DOMAINS,pool,atomic_json


def summarize(y,q):
    clipped=q.clip(1e-7,1-1e-7)
    determinate=np.abs(y-0.5)>1e-7
    conflict=((y-0.5)*(q-0.5)<0)&determinate
    agree=(y-0.5)*(q-0.5)>0
    bins=[];ece=0.0
    for i in range(10):
        chosen=(q>=i/10)&((q<(i+1)/10) if i<9 else (q<=1))
        if not chosen.any():continue
        predicted=float(q[chosen].mean());gold=float(y[chosen].mean());n=int(chosen.sum())
        ece+=n/len(q)*abs(predicted-gold)
        bins.append(dict(interval=[i/10,(i+1)/10],pairs=n,teacher_mean=predicted,original_target_mean=gold))
    return dict(pairs=len(y),original_target_mean=float(y.mean()),teacher_match_probability_mean=float(q.mean()),
      teacher_probability_quantiles=np.quantile(q,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
      original_soft_label_fraction=float(((y>0)&(y<1)).mean()),
      original_label_conflict_fraction=float(conflict.sum()/max(1,determinate.sum())),
      high_confidence_conflict_fraction=float((conflict&(np.maximum(q,1-q)>=.9)).mean()),
      agreement_gate_fraction=float(agree.mean()),teacher_uncertain_fraction=float(((q>.4)&(q<.6)).mean()),
      brier_vs_original_targets=float(np.mean((q-y)**2)),
      cross_entropy_vs_original_targets=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
      teacher_binary_entropy_nats=float(np.mean(-clipped*np.log(clipped)-(1-clipped)*np.log1p(-clipped))),
      probability_calibration_gap_vs_original_targets=float(ece),bins=bins)


def main():
    started=time.time();dataset=pool()
    fields=dataset.fetch(np.arange(len(dataset),dtype=np.int64))
    probabilities=np.load(ROOT/'outputs/teacher/probabilities.npy',mmap_mode='r')
    if probabilities.shape!=(len(dataset),2) or not np.isfinite(probabilities).all():
        raise RuntimeError('The actual experiment requires complete matching cached Teacher targets')
    y=fields['target'].astype(np.float64);q=np.asarray(probabilities[:,1],dtype=np.float64)
    value=dict(scope='training_target_diagnostic_not_teacher_benchmark_score',
               note='Agreement/Brier/CE use original training targets, which are not guaranteed flawless ground truth.',
               global_metrics=summarize(y,q),domains={},elapsed_seconds=0)
    for i,domain in enumerate(DOMAINS):
        selected=fields['domain']==i
        value['domains'][domain]=summarize(y[selected],q[selected])
    value['elapsed_seconds']=time.time()-started
    atomic_json(ROOT/'teacher-diagnostic.json',value)
    print(json.dumps(dict(event='teacher_target_diagnostic',global_metrics=value['global_metrics'],
                          domains={d:{k:v for k,v in m.items() if k in ['pairs','original_target_mean','teacher_match_probability_mean','original_label_conflict_fraction','high_confidence_conflict_fraction','agreement_gate_fraction','brier_vs_original_targets']} for d,m in value['domains'].items()})),flush=True)


if __name__=='__main__':main()
