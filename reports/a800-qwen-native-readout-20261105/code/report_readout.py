#!/usr/bin/env python3
import csv
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kd_runtime import ROOT,SOURCE_ROOT,atomic_json

NEW=['QWEN_RAW_LM','QWEN_HARD_LM','QWEN_LS_LM']
AXES=['knowledge','language','retrieval','tools','arts']


def main():
    scores={}
    for name in NEW:
        folder=ROOT/'outputs/evaluation'/name
        completion=json.loads((folder/'complete.json').read_text())
        if completion.get('requests')!=5935 or completion.get('scope')!='pilot_subset':raise RuntimeError('Native Qwen scope mismatch')
        scores[name]=json.loads((folder/'scores.json').read_text())
    refs={
      'RAW_CLASSIFIER':('kd-raw-qwen-1m-20261103','BASELINE'),
      'HARD_CLASSIFIER':('kd-raw-qwen-1m-20261103','KD_HARD'),
      'LS_CLASSIFIER':('kd-improve-round1-20261104','KD_LS10'),
      'TEACHER_NATIVE':('teacher-native-diagnostic-20261104','TEACHER_NATIVE'),
      'TEACHER_PAIR':('teacher-pair-diagnostic-20261104','TEACHER_PAIR'),
    }
    for label,(experiment,name) in refs.items():
        scores[label]=json.loads((SOURCE_ROOT/'experiments'/experiment/'outputs/evaluation'/name/'scores.json').read_text())
    out=ROOT/'outputs/report';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'score-records.json',scores)
    with (out/'native-readout-comparison.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['condition',*AXES,'frozen_subset_index','coverage'])
        for name,score in scores.items():
            areas={a['id']:100*a['skill'] for a in score['areas']}
            writer.writerow([name,*(areas[a] for a in AXES),score['index'],score['coverage']])
    names=list(scores);fig,ax=plt.subplots(figsize=(11,5.5))
    bars=ax.barh(names,[scores[n]['index'] for n in names],color=['#2563eb' if n in NEW else '#94a3b8' for n in names])
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.invert_yaxis();ax.set_xlim(left=0)
    ax.set_xlabel('Frozen-subset Index (not full-public Index)');ax.set_title('Qwen native LM readout vs saved classifier pipeline')
    for suffix in ['png','svg']:fig.savefig(out/f'native-readout-comparison.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    effects=dict(hard_backbone_native_minus_raw_native=scores['QWEN_HARD_LM']['index']-scores['QWEN_RAW_LM']['index'],
      ls_backbone_native_minus_raw_native=scores['QWEN_LS_LM']['index']-scores['QWEN_RAW_LM']['index'],
      raw_native_minus_random_classifier=scores['QWEN_RAW_LM']['index']-scores['RAW_CLASSIFIER']['index'],
      ls_native_minus_ls_classifier=scores['QWEN_LS_LM']['index']-scores['LS_CLASSIFIER']['index'])
    lines=['# 原始Qwen LM-head/native协议与现有classifier读出诊断','',
      '本次只有已有checkpoint导出及推理，没有新梯度更新。所有条件同37项、4,786完整cases、5,935请求；不是完整公共榜单或私测Full。','',
      '| 条件 | Knowledge | Language | Retrieval | Tools | Arts | Index | coverage |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,score in scores.items():
        areas={a['id']:100*a['skill'] for a in score['areas']}
        lines.append('| '+name+' | '+' | '.join(f'{areas[a]:.2f}' for a in AXES)+f" | {score['index']:.2f} | {100*score['coverage']:.2f}% |")
    lines+=['','## 实际条件','',
      '- QWEN_RAW_LM：原始发布Qwen3-0.6B，保留原LMhead，未做本项目四域训练。此前RAW_CLASSIFIER1.76是随机二分类head，不是这个条件。',
      '- QWEN_HARD_LM/LS_LM：已完成raw1M HARD/LS的LoRA在FP32中合并，严格恢复全部decoder backbone到原始CausalLM，保留原LMhead；原分类head不用于预测。没有原生协议SFT或再训练。',
      '- 三组都使用原始state/questions、自己的Qwen tokenizer/chat template、thinkingoff和native label-token scoring，BF16/triton/TP1，每组一张卡；前端T1/read-twice≤3000，不使用Gemma专属折叠温度或temperatures文件。',
      '- 跟旧classifier相比，同时改变输入包装、读出位置和输出head，不将差值单独归因为head权重。三组新native条件之间保持这些因素一致，仅backbone历史不同。',
      '- 不把Teacher分数当Student必须达到的数学上界；不发送expected/gold标签，不在评测数据训练，不筛候选；coverage如实报告。', '',
      '## 描述性交互差值','']
    for key,value in effects.items():lines.append(f'- {key}: **{value:+.2f}**')
    lines+=['','若原生LM明显更强，下一步应考虑原生输出协议的matched CE/KD训练，而不是把这次推理读出变化叫作KD胜利；若仍低，优先查容量、训练分布和泛化。单次固定子集诊断不等于因果组件隔离或显著性证明。','',
      '![读出对比](native-readout-comparison.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    marker=dict(complete=True,scope='qwen_native_readout_diagnostic',report=str(out/'REPORT.md'),
                scores={n:s['index'] for n,s in scores.items()},effects=effects,new_training_steps=0,unix_time=time.time())
    atomic_json(ROOT/'COMPLETE.json',marker);print(json.dumps(dict(event='qwen_native_readout_complete',**marker)),flush=True)


if __name__=='__main__':main()
