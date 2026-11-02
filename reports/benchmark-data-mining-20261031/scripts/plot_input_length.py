#!/usr/bin/env python3
"""Draw a presentation-ready input-length histogram from the existing counts."""
import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

BUCKETS=['<128','128–255','256–511','512–1023','1024–2047','2048–4095','4096–8191','≥8192']
TAIL={'8192–16383','16384–32767','≥32768'}

def readable_count(n):
    return f'{n/10000:.2f}'.rstrip('0').rstrip('.')+'万' if n>=10000 else f'{n:,}'

def percent(value):
    return '<0.01%' if 0<value<0.01 else f'{value:.2f}'.rstrip('0').rstrip('.')+'%'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    root=parser.parse_args().root
    counts=collections.Counter()
    with (root/'tables/length-buckets.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['active'].lower()!='true' or row['unit']!='request_pair_max':continue
            bucket='≥8192' if row['bucket'] in TAIL else row['bucket']
            counts[bucket]+=int(row['count'])
    total=sum(counts.values())
    if not total:raise ValueError('No active request-length counts were found.')
    summary=json.loads((root/'summary.json').read_text())
    if total!=summary['requests']:raise ValueError('Length counts and request scope differ.')
    rows=[{'bucket':b,'requests':counts[b],'share_pct':round(100*counts[b]/total,2),
           'display_share':percent(100*counts[b]/total)} for b in BUCKETS]
    with (root/'tables/input-length-bar.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    font=root/'assets/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size':13,'axes.unicode_minus':False,'svg.fonttype':'path',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.spines.left':False,'figure.facecolor':'white',
                         'axes.labelcolor':'#31475D','text.color':'#203047'})
    fig,ax=plt.subplots(figsize=(14,7.6))
    values=[100*counts[b]/total for b in BUCKETS]
    palette=['#2767A4']*4+['#418D99']*2+['#D48A42']*2
    bars=ax.bar(range(len(BUCKETS)),values,width=.66,color=palette,zorder=3)
    ax.set_xticks(range(len(BUCKETS)),['0–127']+BUCKETS[1:],fontsize=12)
    ax.set_xlabel('输入长度（token）',labelpad=16)
    ax.set_ylabel('请求占比（%）',labelpad=14)
    ax.set_ylim(0,max(values)*1.25)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_:f'{x:.0f}'))
    ax.grid(axis='y',color='#E6EBF1',linewidth=.8,zorder=0)
    ax.tick_params(axis='both',length=0,pad=9)
    ax.spines['bottom'].set_color('#CFD8E3')
    for bar,row,value in zip(bars,rows,values):
        y=max(value+.9,1.2)
        ax.text(bar.get_x()+bar.get_width()/2,y,
                row['display_share']+'\n'+readable_count(row['requests'])+'条',
                ha='center',va='bottom',fontsize=12,linespacing=1.6)
    short=100*sum(counts[b] for b in BUCKETS[:3])/total
    fig.suptitle('输入长度分布',x=.09,y=.965,ha='left',fontsize=25,fontweight='bold')
    fig.text(.09,.902,'37 项公开 benchmark · 约11万条请求',fontsize=13,color='#667D92')
    fig.text(.91,.91,f'{percent(short)} 的请求少于 512 token',ha='right',fontsize=15,
             color='#2767A4',fontweight='bold')
    fig.text(.09,.035,'口径：每条请求取全部候选匹配输入的最长长度；Qwen3 tokenizer，未截断。',
             fontsize=11,color='#667D92')
    fig.subplots_adjust(left=.09,right=.97,bottom=.16,top=.82)
    out=root/'figures/17-input-length-distribution-bar'
    for ext in ['png','svg']:fig.savefig(str(out)+'.'+ext,dpi=200,bbox_inches='tight',pad_inches=.22)
    plt.close(fig)
    print(json.dumps({'requests':total,'less_than_512_pct':round(short,2),'buckets':rows},ensure_ascii=False))

if __name__=='__main__':main()
