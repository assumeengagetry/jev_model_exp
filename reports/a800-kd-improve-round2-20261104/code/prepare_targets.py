#!/usr/bin/env python3
"""Fit eight calibration scalars and preserve/shuffle instance-level Teacher information."""
import json
import time
import numpy as np
from kd_runtime import ROOT,DOMAINS,pool,atomic_json


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))


def fit_calibration(x,y):
    theta=np.array([1.,0.]);ridge=1e-6
    def loss(t):
        z=t[0]*x+t[1]
        return float(np.mean(np.logaddexp(0,z)-y*z)+0.5*ridge*np.sum((t-np.array([1.,0.]))**2))
    iterations=0
    for iterations in range(30):
        p=sigmoid(theta[0]*x+theta[1]);v=p*(1-p);res=p-y
        gradient=np.array([np.mean(res*x),np.mean(res)])+ridge*(theta-np.array([1.,0.]))
        if np.linalg.norm(gradient)<1e-8:break
        hessian=np.array([[np.mean(v*x*x),np.mean(v*x)],[np.mean(v*x),np.mean(v)]])+ridge*np.eye(2)
        direction=np.linalg.solve(hessian,gradient);current=loss(theta)
        rate=1.;accepted=False
        for _ in range(20):
            candidate=theta-rate*direction
            candidate[0]=np.clip(candidate[0],0.05,10);candidate[1]=np.clip(candidate[1],-20,20)
            if loss(candidate)<current:
                theta=candidate;accepted=True;break
            rate*=0.5
        if not accepted:break
    return theta,iterations+1


def metrics(y,q):
    q=np.clip(q,1e-7,1-1e-7)
    return dict(mean=float(q.mean()),brier=float(np.mean((q-y)**2)),
                cross_entropy=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))))


def main():
    if (ROOT/'TARGETS_READY.json').exists():return
    started=time.time();dataset=pool();fields=dataset.fetch(np.arange(len(dataset),dtype=np.int64))
    original=np.load(ROOT/'outputs/teacher/probabilities.npy',mmap_mode='r')
    y=fields['target'].astype(np.float64);q=np.asarray(original[:,1],dtype=np.float64)
    calibrated=np.empty(len(y),dtype=np.float64);epsilon=np.empty(len(y),dtype=np.float64);shuffled=np.empty(len(y),dtype=np.float64)
    rng=np.random.default_rng(20261104);summary=dict(domains={},fit_scope='original training targets only; no benchmark calibration',
       adaptive_scope='rank confidence inside each(domain, exact original target) group; matched shuffled histogram',
       ridge=1e-6,slope_bounds=[.05,10],intercept_bounds=[-20,20],seed=20261104)
    for d,domain in enumerate(DOMAINS):
        positions=np.flatnonzero(fields['domain']==d);local_y=y[positions];local_q=q[positions]
        clipped=local_q.clip(1e-6,1-1e-6);x=np.log(clipped)-np.log1p(-clipped)
        theta,it=fit_calibration(x,local_y);corrected=sigmoid(theta[0]*x+theta[1]);calibrated[positions]=corrected
        confidence=local_y*corrected+(1-local_y)*(1-corrected)
        groups=0
        for label in np.unique(local_y):
            local=np.flatnonzero(local_y==label);selected=positions[local];order=np.argsort(confidence[local],kind='stable')
            _,starts,counts=np.unique(confidence[local][order],return_index=True,return_counts=True)
            ranks=np.repeat(starts+(counts-1)/2,counts)
            sorted_epsilon=.15-.1*(ranks+.5)/len(local)
            epsilon[selected[order]]=sorted_epsilon
            shuffled[selected]=epsilon[selected][rng.permutation(len(local))]
            groups+=1
        summary['domains'][domain]=dict(pairs=len(positions),slope=float(theta[0]),intercept=float(theta[1]),iterations=it,
           original_target_mean=float(local_y.mean()),before=metrics(local_y,local_q),after=metrics(local_y,corrected),
           exact_target_groups=groups,adaptive_epsilon_mean=float(epsilon[positions].mean()),
           shuffled_epsilon_mean=float(shuffled[positions].mean()),adaptive_epsilon_range=[float(epsilon[positions].min()),float(epsilon[positions].max())])
    probabilities=np.stack((1-calibrated,calibrated),axis=-1).astype(np.float32)
    np.save(ROOT/'calibrated-probabilities.npy',probabilities)
    np.save(ROOT/'adaptive-epsilon.npy',epsilon.astype(np.float32));np.save(ROOT/'shuffled-epsilon.npy',shuffled.astype(np.float32))
    summary['elapsed_seconds']=time.time()-started
    atomic_json(ROOT/'target-preparation.json',summary)
    atomic_json(ROOT/'TARGETS_READY.json',dict(complete=True,pairs=len(y),calibration_parameters=8,
                dropped_pairs=0,benchmark_inputs_used=False,elapsed_seconds=summary['elapsed_seconds']))
    print(json.dumps(dict(event='round2_targets_prepared',**summary)),flush=True)


if __name__=='__main__':main()
