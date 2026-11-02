#!/usr/bin/env python3
"""Keep only public aggregate model metrics; no checkpoint identifiers."""
import argparse
import csv
import json
import statistics
from pathlib import Path
import urllib.request

URL='https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    with urllib.request.urlopen(URL,timeout=90) as response:data=json.load(response)
    models=[];scores=[];benchmarks={}
    for key,b in data.get('benchmarks',{}).items():
        benchmarks[key]={k:b.get(k) for k in ['id','dataset','short','metric','explainer','baseline','cases','group','in_suite','gold','references']}
    for m in data['models']:
        meta=m.get('meta',{});params=meta.get('served_params')
        if params is None:band='unknown'
        elif params<2e9:band='<2B'
        elif params<10e9:band='2–10B'
        else:band='≥10B'
        name=m.get('name',m.get('engine',''))
        models.append(dict(name=name,band=band,params_b=round(params/1e9,2) if params else None,kind=meta.get('kind'),technique=meta.get('technique'),public_skill=m.get('scores',{}).get('public_skill')))
        for key,s in m.get('benchmarks',{}).items():
            result=m.get('results',{}).get(str(key),{})
            if s.get('skill') is not None:
                scores.append(dict(model=name,band=band,benchmark_id=int(key),skill=round(100*s['skill'],2),raw=round(100*s['raw'],2) if s.get('raw') is not None else None,coverage_pct=round(100*s.get('coverage',0),2),median_ms=result.get('median_ms')))
    clean=dict(source=URL,generated_utc=data.get('generated_utc'),edition=data.get('suite',{}).get('label'),models=models,benchmarks=benchmarks)
    args.root.mkdir(parents=True,exist_ok=True);(args.root/'tables').mkdir(exist_ok=True)
    (args.root/'leaderboard-public-context.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n')
    with (args.root/'tables/leaderboard-model-benchmark.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader()
        for row in scores:
            if row['median_ms'] is not None:row['median_ms']=round(row['median_ms'],2)
            w.writerow(row)
    aggregate=[]
    for bid in sorted({x['benchmark_id'] for x in scores}):
        for band in ['all','<2B','2–10B','≥10B']:
            xs=[x for x in scores if x['benchmark_id']==bid and (band=='all' or x['band']==band)]
            complete=[x for x in xs if x['coverage_pct']>=99.99]
            aggregate.append(dict(benchmark_id=bid,name=benchmarks.get(str(bid),{}).get('short'),band=band,models=len(xs),complete_models=len(complete),skill_median=round(statistics.median(x['skill'] for x in xs),2) if xs else None,skill_mean=round(statistics.mean(x['skill'] for x in xs),2) if xs else None,complete_skill_median=round(statistics.median(x['skill'] for x in complete),2) if complete else None,zero_coverage_models=sum(x['coverage_pct']==0 for x in xs),coverage_mean_pct=round(statistics.mean(x['coverage_pct'] for x in xs),2) if xs else None))
    with (args.root/'tables/leaderboard-benchmark-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    print(json.dumps(dict(generated_utc=clean['generated_utc'],edition=clean['edition'],models=len(models),benchmark_metrics=len(scores)),ensure_ascii=False))

if __name__=='__main__':main()
