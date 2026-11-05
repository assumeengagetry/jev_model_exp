#!/usr/bin/env python3
"""Compare actual native-protocol Teacher quality with saved matched Student scores."""
import csv
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kd_runtime import ROOT,SOURCE_ROOT,atomic_json

AXES=['knowledge','language','retrieval','tools','arts']


def main():
    teacher=ROOT/'outputs/evaluation/TEACHER_NATIVE'
    completion=json.loads((teacher/'complete.json').read_text())
    if completion.get('requests')!=5935 or completion.get('scope')!='pilot_subset':
        raise RuntimeError('Teacher diagnostic must retain the exact frozen request scope')
    scores={'TEACHER_NATIVE':json.loads((teacher/'scores.json').read_text())}
    sources={
      'KD_HARD':'kd-raw-qwen-1m-20261103','KD_SOFT':'kd-raw-qwen-1m-20261103','KD_HYBRID':'kd-raw-qwen-1m-20261103',
      'KD_TEMP2_50':'kd-improve-round1-20261104','KD_LS10':'kd-improve-round1-20261104',
      'KD_CALSOFT':'kd-improve-round2-20261104',
    }
    for name,experiment in sources.items():
        scores[name]=json.loads((SOURCE_ROOT/'experiments'/experiment/'outputs/evaluation'/name/'scores.json').read_text())
    out=ROOT/'outputs/report';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'score-records.json',scores)
    with (out/'native-teacher-vs-students.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['model',*AXES,'frozen_subset_index','coverage'])
        for name,score in scores.items():
            area={a['id']:100*a['skill'] for a in score['areas']}
            writer.writerow([name,*(area[a] for a in AXES),score['index'],score['coverage']])
    difference=scores['TEACHER_NATIVE']['index']-scores['KD_LS10']['index']
    names=list(scores)
    fig,ax=plt.subplots(figsize=(11,5))
    bars=ax.barh(names,[scores[n]['index'] for n in names],color=['#ea580c' if n=='TEACHER_NATIVE' else '#2563eb' if n=='KD_LS10' else '#94a3b8' for n in names])
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.invert_yaxis();ax.set_xlim(left=0)
    ax.set_xlabel('Frozen-subset Index; not an equal-compute comparison')
    ax.set_title('Native-protocol KnowLine Teacher vs saved Students')
    for suffix in ['png','svg']:fig.savefig(out/f'native-teacher-vs-students.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    lines=['# Teacher原生协议直接评测','',
      f"**Teacher native Index：{scores['TEACHER_NATIVE']['index']:.2f}；coverage：{100*scores['TEACHER_NATIVE']['coverage']:.2f}%；相对LS0.1：{difference:+.2f}。**",'',
      '| 模型 | Knowledge | Language | Retrieval | Tools | Arts | Index | coverage |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,score in scores.items():
        area={a['id']:100*a['skill'] for a in score['areas']}
        lines.append('| '+name+' | '+' | '.join(f'{area[a]:.2f}' for a in AXES)+f" | {score['index']:.2f} | {100*score['coverage']:.2f}% |")
    lines+=['','## 实验意义','',
      'Teacher使用原始row.state及row.questions，保留原问题type/instructions/criteria，通过官方KnowLine frontend回答；不把各candidate改写为独立的correct-match二分类。expected/gold/scoring标签从未发给Teacher。',
      '本实验检验Teacher在目标任务原生协议上的能力上界。即便native Teacher更强，也不能自动推出之前独立pair概率足够好；那需要另做协议转换对照。若native Teacher不强，则不应继续假定31B一定能提供优质标签。',
      '这不是等预算模型比较，也不是Student训练或新标注数据生产；不会把这些评测回答用于训练。','',
      '## 口径与设置','',
      '同37项、4,786完整cases、5,935请求，和Student冻结子集一致；不是完整公共榜单或私测Full。Student参考复用保存的结果，不重新推理。',
      'Teacher原始BF16权重、TP2、triton attention、fp32 LM head、frontend temperature1、read-twice≤3000、原temperatures.json。发布权重已经折叠1.44的readout温度，没有手动改回或重新拟合。原生box渲染使用gemma/yxyx。',
      '原生服务使用8个scoring threads以匹配16CPU资源，HTTP并发32；仅操作新实验自己的服务进程。没有新环境安装、单测、smoke、hash或全局Teacher cache写入。',
      f"请求状态计数：`{json.dumps(completion.get('counts',{}),ensure_ascii=False)}`。超过官方64questions/255criteria限制的请求显式记unsupported，而不是删题或删候选；coverage按实际结果报告。",'',
      '## 解释边界','',
      '原生任务能力与独立candidate-pair二分类能力不是同一个测量。这里不根据模型参数量预设Teacher必须赢，不将coverage不足悄悄隐藏为高分，也不把原训练targets诊断冒充benchmark成绩。','',
      '![原生Teacher与Student](native-teacher-vs-students.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    marker=dict(complete=True,scope='native_teacher_frozen_subset_diagnostic',report=str(out/'REPORT.md'),
                teacher_index=scores['TEACHER_NATIVE']['index'],coverage=scores['TEACHER_NATIVE']['coverage'],
                counts=completion.get('counts'),minus_ls10=difference,scores={n:s['index'] for n,s in scores.items()},unix_time=time.time())
    atomic_json(ROOT/'COMPLETE.json',marker)
    print(json.dumps(dict(event='native_teacher_diagnostic_complete',**marker)),flush=True)


if __name__=='__main__':main()
