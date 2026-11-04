#!/usr/bin/env python3
"""Report all prespecified KD hypotheses, including the no-Teacher smoothing control."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from kd_runtime import ROOT,SOURCE_ROOT,atomic_json

NEW=['KD_MIX25','KD_AGREE50','KD_TEMP2_50','KD_LS10']
REFERENCE=SOURCE_ROOT/'experiments/kd-raw-qwen-1m-20261103'
REF=['BASELINE','KD_HARD','KD_SOFT','KD_HYBRID']
AXES=['knowledge','language','retrieval','tools','arts']


def main():
    out=ROOT/'outputs/report';out.mkdir(parents=True,exist_ok=True)
    scores={};training={}
    for name,root in [(n,REFERENCE) for n in REF]+[(n,ROOT) for n in NEW]:
        folder=root/'outputs/evaluation'/name
        complete=json.loads((folder/'complete.json').read_text())
        if not complete.get('complete') or complete.get('scope')!='pilot_subset' or complete.get('counts')!={'ok':5935}:
            raise RuntimeError('A matched frozen-subset evaluation is not complete: '+name)
        scores[name]=json.loads((folder/'scores.json').read_text())
        if scores[name]['coverage']!=1:raise RuntimeError('Incomplete selected-subset coverage')
        if name in NEW:
            training[name]=json.loads((ROOT/'outputs'/name/'complete.json').read_text())
            if any(training[name][k]!=v for k,v in [('pairs_seen',1000000),('optimizer_steps',7813),('effective_tokens',228021289)]):
                raise RuntimeError('Training exposure mismatch: '+name)
    values={name:{a['id']:100*a['skill'] for a in score['areas']} for name,score in scores.items()}
    differences={name:dict(minus_hard=scores[name]['index']-scores['KD_HARD']['index'],
      minus_hybrid50=scores[name]['index']-scores['KD_HYBRID']['index'],
      minus_label_smoothing=scores[name]['index']-scores['KD_LS10']['index'],
      areas_minus_hard={a:values[name][a]-values['KD_HARD'][a] for a in AXES}) for name in NEW}
    atomic_json(out/'differences.json',differences)
    atomic_json(out/'all-score-records.json',scores)
    with (out/'all-model-five-area.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['model',*AXES,'frozen_subset_index','coverage'])
        for name in REF+NEW:writer.writerow([name,*(values[name][a] for a in AXES),scores[name]['index'],scores[name]['coverage']])
    names=REF[1:]+NEW
    colors=['#94a3b8']*3+['#2563eb','#0d9488','#ea580c','#7c3aed']
    fig,ax=plt.subplots(figsize=(11,5.5))
    bars=ax.barh(names,[scores[n]['index'] for n in names],color=colors)
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.invert_yaxis();ax.set_xlim(left=0)
    ax.set_xlabel('Frozen-subset Index (not full-public Index)');ax.set_title('Matched raw-Qwen 1M: all four prespecified variants')
    ax.grid(axis='x',alpha=.2)
    for suffix in ['png','svg']:fig.savefig(out/f'variant-index-comparison.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    x=np.arange(len(NEW));fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(x-.18,[differences[n]['minus_hard'] for n in NEW],.36,label='vs HARD',color='#2563eb')
    ax.bar(x+.18,[differences[n]['minus_hybrid50'] for n in NEW],.36,label='vs existing HYBRID0.5',color='#ea580c')
    ax.axhline(0,color='black',linewidth=1);ax.set_xticks(x,NEW);ax.set_ylabel('Index gain (points)');ax.legend()
    ax.set_title('Exploratory hypothesis round: gains over fixed references')
    for container in ax.containers:ax.bar_label(container,fmt='%+.2f',padding=3)
    for suffix in ['png','svg']:fig.savefig(out/f'gains-over-controls.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    diag=json.loads((ROOT/'teacher-diagnostic.json').read_text())
    lines=['# 蒸馏改进：第一轮假设验证','',
      '**所有预先指定的四组均完整报告，不只展示赢家。** 本轮使用已知冻结子集进行探索性模型选择，不是独立held-out确认。','',
      '## 完整结果','', '| 条件 | Knowledge | Language | Retrieval | Tools | Arts | Index | coverage |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in REF+NEW:
        lines.append('| '+name+' | '+' | '.join(f'{values[name][a]:.2f}' for a in AXES)+f" | {scores[name]['index']:.2f} | {100*scores[name]['coverage']:.1f}% |")
    lines+=['','## 相对固定对照的收益','', '| 新条件 | vs HARD | vs HYBRID0.5 | vs LS0.1 |','|---|---:|---:|---:|']
    for name,d in differences.items():lines.append(f"| {name} | {d['minus_hard']:+.2f} | {d['minus_hybrid50']:+.2f} | {d['minus_label_smoothing']:+.2f} |")
    lines+=['','## 本轮假设与目标函数','',
      '- MIX25：0.75CE＋0.25KL(T1)，检验Teacher权重过大。',
      '- AGREE50：(1−0.5w)CE＋0.5wKL，w为Teacher与原标签方向一致指示；冲突时回到完整CE，不丢样本，不改原始连续标签。改善也可能来自平均KD强度降低，不能单独归因于过滤。',
      '- TEMP2_50：0.5CE＋0.5×4KL(q_T||p_T)，T2。对缓存二类概率重新温度缩放，不声称恢复原始LM vocabulary logits；T²保留标准梯度尺度，不保证错误Teacher梯度更小。',
      '- LS10：针对0.9×原始分布＋0.1×uniform做CE，不使用Teacher。它区分Teacher信息与普通smoothing，但单个epsilon不代表所有smoothing设置。','',
      '## Teacher-target科学诊断','',
      '以下是Teacher与原训练target的对照，不是Teacher在目标benchmark上的准确率；原target本身也可能有噪声。','',
      '| 领域 | pairs | 原target均值 | Teacher匹配概率均值 | 判断冲突率 | 高置信冲突占比 |','|---|---:|---:|---:|---:|---:|']
    for domain,d in diag['domains'].items():
        lines.append(f"| {domain} | {d['pairs']:,} | {100*d['original_target_mean']:.2f}% | {100*d['teacher_match_probability_mean']:.2f}% | {100*d['original_label_conflict_fraction']:.2f}% | {100*d['high_confidence_conflict_fraction']:.2f}% |")
    lines+=['','## 共同实验条件','',
      '原始Qwen3-0.6B＋同一份未训练LoRA/head；同1M样本和epoch顺序，r32/alpha64，BF16/SDPA，batch128，lr5e-5，AdamW/cosine，warmup3%，wd0.01，seed42，一遍。每组7,813steps、228,021,289 effective tokens。复用所有已标Teacher概率，不重新调用Teacher。',
      '评测同37项、4,786完整cases、5,935请求、163,671完整候选pairs。未用训练截断过滤评测候选；不是完整公共榜单或私测Full。原始四组对照直接复用已完成结果，没有重跑。','',
      '## 数学与经验依据','',
      '二分类margin梯度：CE为p−y，T1 KL为p−q；hybrid等价于对(1−alpha)y＋alphaq做CE，差一个不影响Student梯度的常数。q有偏时，纯KD不保证优于原标签。',
      '- Yuan et al., Revisiting Knowledge Distillation via Label Smoothing Regularization: https://arxiv.org/abs/1909.11723',
      '- Cho & Hariharan, On the Efficacy of Knowledge Distillation: https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html',
      '- Hinton et al., Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531','',
      '## 图表','', '![所有条件](variant-index-comparison.png)','', '![对照增益](gains-over-controls.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    best=max(NEW,key=lambda n:scores[n]['index'])
    marker=dict(complete=True,scope='exploratory_kd_hypothesis_round',report=str(out/'REPORT.md'),
      scores={n:scores[n]['index'] for n in REF+NEW},differences=differences,best_new_arm=best,
      best_new_arm_minus_hard=differences[best]['minus_hard'],best_new_arm_minus_hybrid50=differences[best]['minus_hybrid50'])
    atomic_json(ROOT/'outputs/pilot-complete.json',marker)
    print(json.dumps(dict(event='kd_improvement_report_complete',**marker)),flush=True)


if __name__=='__main__':main()
