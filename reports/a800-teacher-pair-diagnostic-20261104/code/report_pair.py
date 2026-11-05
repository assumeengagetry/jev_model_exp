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
    teacher=ROOT/'outputs/evaluation/TEACHER_PAIR'
    completion=json.loads((teacher/'complete.json').read_text())
    if completion.get('requests')!=5935 or completion.get('scope')!='pilot_subset':
        raise RuntimeError('Teacher diagnostic must retain the exact frozen request scope')
    scores={'TEACHER_PAIR':json.loads((teacher/'scores.json').read_text())}
    sources={
      'TEACHER_NATIVE':'teacher-native-diagnostic-20261104',
      'KD_HARD':'kd-raw-qwen-1m-20261103','KD_SOFT':'kd-raw-qwen-1m-20261103','KD_HYBRID':'kd-raw-qwen-1m-20261103',
      'KD_TEMP2_50':'kd-improve-round1-20261104','KD_LS10':'kd-improve-round1-20261104',
      'KD_CALSOFT':'kd-improve-round2-20261104',
    }
    for name,experiment in sources.items():
        scores[name]=json.loads((SOURCE_ROOT/'experiments'/experiment/'outputs/evaluation'/name/'scores.json').read_text())
    out=ROOT/'outputs/report';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'score-records.json',scores)
    with (out/'teacher-protocol-comparison.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['model',*AXES,'frozen_subset_index','coverage'])
        for name,score in scores.items():
            area={a['id']:100*a['skill'] for a in score['areas']}
            writer.writerow([name,*(area[a] for a in AXES),score['index'],score['coverage']])
    difference=scores['TEACHER_PAIR']['index']-scores['KD_LS10']['index']
    protocol_difference=scores['TEACHER_PAIR']['index']-scores['TEACHER_NATIVE']['index']
    names=list(scores)
    fig,ax=plt.subplots(figsize=(11,5))
    bars=ax.barh(names,[scores[n]['index'] for n in names],color=['#ea580c' if n=='TEACHER_PAIR' else '#0d9488' if n=='TEACHER_NATIVE' else '#2563eb' if n=='KD_LS10' else '#94a3b8' for n in names])
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.invert_yaxis();ax.set_xlim(left=0)
    ax.set_xlabel('Frozen-subset Index; not an equal-compute comparison')
    ax.set_title('Same KnowLine weights: native vs pair protocol, same frozen cases')
    for suffix in ['png','svg']:fig.savefig(out/f'teacher-protocol-comparison.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    lines=['# 同一Teacher：原生任务协议与candidate-pair改写对照','',
      f"**Teacher pair Index：{scores['TEACHER_PAIR']['index']:.2f}；coverage：{100*scores['TEACHER_PAIR']['coverage']:.2f}%；pair−native：{protocol_difference:+.2f}；相对LS0.1：{difference:+.2f}。**",'',
      '| 模型 | Knowledge | Language | Retrieval | Tools | Arts | Index | coverage |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,score in scores.items():
        area={a['id']:100*a['skill'] for a in score['areas']}
        lines.append('| '+name+' | '+' | '.join(f'{area[a]:.2f}' for a in AXES)+f" | {score['index']:.2f} | {100*score['coverage']:.2f}% |")
    lines+=['','## 实验意义','',
      'Pair条件从同一eval cache的input/question/criteria/candidate重建render_pair文本，并用原训练标签生成时的通用noul提示独立判断correct match。没有expected/gold/scoring标签进入Teacher payload。',
      'Native条件直接复用已完成的原始state/questions协议结果。两者权重相同、case IDs相同；本实验改变的是问题序列化、通用二分类问法/输出类型及候选聚合，不单独隔离其中某一个因素。不能把pairedTeacher称作所有Student的数学上界。',
      '这不是等预算模型比较，也不是Student训练或新标注数据生产；不会把这些评测回答用于训练。','',
      '## 口径与设置','',
      '同37项、4,786完整cases、5,935请求，和Student冻结子集一致；不是完整公共榜单或私测Full。Student参考复用保存的结果，不重新推理。',
      'Teacher原始BF16权重、TP2、triton attention、fp32 LM head、frontend temperature1、read-twice≤3000、原temperatures.json。发布权重已经折叠1.44的readout温度，没有手动改回或重新拟合。Pair均为noul；Native保留原question types及对应官方per-type温度。',
      'Pair使用两个TP2副本、每副本8scoring threads、HTTP并发64，合计现有四GPU。相同input state的candidates交给同一副本，复用完全相同的(state,candidate)输入。没有安装环境、单测、smoke、hash或canonical训练概率cache写入。',
      '本次保留全部evaluation输入，不使用训练时4096token head-tail cap，因此不是长输入上的逐字节训练label管线重放。它给pair条件保留更多原始信息；该差异明确披露，不静默丢弃长样本。',
      '概率按原annotation的float32[P(notmatch),P(match)]两列保存。margin使用log(Pmatch)−log(Pnotmatch)，数值下限1e−30；不从round-to1的单个Pmatch列重建小概率。随后复用原Student候选softmax及native scorer。',
      f"请求状态计数：`{json.dumps(completion.get('counts',{}),ensure_ascii=False)}`。所有请求/candidates均保留，scope及coverage按实际完成记录报告。",'',
      '## 解释边界','',
      '若pair也强，下一步应关注Student容量/优化/数据覆盖；若pair明显下降，优先修正监督目标或任务协议，而非继续盲扫loss。这只是机制诊断，未证明任何新Student一定获益。评测回答不会用于训练。','',
      '![协议对照](teacher-protocol-comparison.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    marker=dict(complete=True,scope='teacher_pair_protocol_diagnostic',report=str(out/'REPORT.md'),
                teacher_pair_index=scores['TEACHER_PAIR']['index'],teacher_native_index=scores['TEACHER_NATIVE']['index'],
                pair_minus_native=protocol_difference,coverage=scores['TEACHER_PAIR']['coverage'],
                counts=completion.get('counts'),minus_ls10=difference,scores={n:s['index'] for n,s in scores.items()},unix_time=time.time())
    atomic_json(ROOT/'COMPLETE.json',marker)
    print(json.dumps(dict(event='teacher_pair_protocol_complete',**marker)),flush=True)


if __name__=='__main__':main()
