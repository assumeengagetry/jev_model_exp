#!/usr/bin/env python3
"""Generate requested rule-labelled training coverage, not model tests."""
import collections
import json
import random
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from pair_data_common import text, criteria_context

HOME = Path('/home/dev/assumeengage')
OUT = HOME / 'cache/pair-training/generated-pairs'
OUT.mkdir(parents=True, exist_ok=True)
SCHEMA = pa.schema([(k, pa.string()) for k in ['state','question','criteria','candidate']]
                   + [('label', pa.float32())] + [(k,pa.string()) for k in ['domain','source','source_repo','group_id']])
REPORT = []


class Writer:
    def __init__(self, name, domain):
        self.name, self.domain = name, domain; self.path = OUT / (name + '.parquet')
        self.writer = pq.ParquetWriter(self.path.with_suffix('.parquet.preparing'), SCHEMA, compression='zstd')
        self.buffer=[]; self.cases=0; self.pairs=0
    def add(self,state,question,options,positive):
        options=[text(x) for x in options]; context=criteria_context(question,options)
        for i,candidate in enumerate(options):
            self.buffer.append(dict(state=text(state),question=question,criteria=context,candidate=candidate,
                                    label=float(i in positive),domain=self.domain,source=self.name,
                                    source_repo='local closed-form training generator',group_id=f'{self.name}:{self.cases}'))
            self.pairs+=1
        self.cases+=1
        if len(self.buffer)>=8192:self.writer.write_table(pa.Table.from_pylist(self.buffer,schema=SCHEMA));self.buffer.clear()
    def finish(self):
        if self.buffer:self.writer.write_table(pa.Table.from_pylist(self.buffer,schema=SCHEMA))
        self.writer.close();self.path.with_suffix('.parquet.preparing').replace(self.path)
        REPORT.append(dict(source=self.name,domain=self.domain,cases=self.cases,pairs=self.pairs,path=str(self.path.relative_to(HOME))))
        print(self.name,self.cases,'cases',self.pairs,'pairs',flush=True)


rng=random.Random(42)
w=Writer('workflow_policy','tools')
policy='Suspended or non-members are denied. Otherwise require MFA. A viewer may only read. Delete requires admin. Sensitive writes/deletes require recorded approval. All remaining requests are approved.'
for _ in range(100000):
    state=dict(role=rng.choice(['viewer','editor','admin']),action=rng.choice(['read','write','delete']),
               member=rng.random()<0.9,suspended=rng.random()<0.1,mfa=rng.random()<0.75,
               sensitive=rng.random()<0.5,approved=rng.random()<0.5,resource_id=rng.randrange(1000000),policy=policy)
    if state['suspended'] or not state['member']:label=1
    elif not state['mfa']:label=2
    elif (state['role']=='viewer' and state['action']!='read') or (state['action']=='delete' and state['role']!='admin'):label=1
    elif state['sensitive'] and state['action']!='read' and not state['approved']:label=3
    else:label=0
    w.add(state,'Which workflow action follows the stated policy?',['Approve the request.','Deny the request.','Request multi-factor authentication.','Request approval.'],{label})
w.finish()

w=Writer('numeric_constraints','knowledge')
for _ in range(50000):
    values=[rng.randint(-100,100) for _ in range(rng.randint(3,12))];threshold=rng.randint(-100,100)
    operation=rng.choice(['sum','maximum','count_greater'])
    answer=sum(values) if operation=='sum' else max(values) if operation=='maximum' else sum(v>threshold for v in values)
    wrong=[]
    while len(wrong)<3:
        value=answer+rng.choice([-11,-7,-3,-1,1,2,4,9,13])
        if value not in wrong:wrong.append(value)
    w.add(dict(values=values,threshold=threshold),'Compute '+operation+' from the supplied values; count_greater means strictly above the threshold.',[str(answer)]+list(map(str,wrong)),{0})
w.finish()

w=Writer('symbolic_music','arts')
names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
qualities={'major':(0,4,7),'minor':(0,3,7),'diminished':(0,3,6),'augmented':(0,4,8),
           'dominant seventh':(0,4,7,10),'major seventh':(0,4,7,11),'minor seventh':(0,3,7,10),'suspended fourth':(0,5,7)}
catalog=[(f'{names[root]} {quality}',tuple(sorted((root+x)%12 for x in intervals))) for root in range(12) for quality,intervals in qualities.items()]
for _ in range(50000):
    bars=[];classes=[]
    for beat in range(4):
        name,pcs=rng.choice(catalog);notes=[pc+12*rng.randint(3,6) for pc in pcs];rng.shuffle(notes)
        bars.append(notes);classes.append(pcs)
    target=rng.randrange(4);correct=classes[target];candidates=[(next(name for name,pcs in catalog if pcs==correct),correct)]
    while len(candidates)<4:
        item=rng.choice(catalog)
        if all(item[1]!=x[1] for x in candidates):candidates.append(item)
    w.add(dict(bars_midi_notes=bars,target_bar=target+1,pitch_class_rule='MIDI note modulo 12; C=0 through B=11'),
          'Which chord pitch-class set is present in the target bar?',
          [name+'; pitch classes '+text(pcs) for name,pcs in candidates],{0})
w.finish()

w=Writer('poetry_structure','arts')
families=[['cat','hat','bat','mat'],['light','night','bright','sight'],['day','play','stay','way'],
          ['blue','true','new','dew'],['stone','alone','known','tone'],['tree','sea','free','bee'],
          ['rain','train','plain','grain'],['slow','glow','snow','flow']]
openings=['A wandering bird remembered','The quiet child could hear','Beyond the hill we found','A distant bell recalled','The silver moon revealed','Our small boat followed']
schemes=['AABB','ABAB','ABBA']
for _ in range(50000):
    a,b=rng.sample(range(len(families)),2);scheme=rng.choice(schemes)
    endings={key:rng.sample(families[value],4) for key,value in [('A',a),('B',b)]};used=collections.Counter()
    lines=[]
    for symbol in scheme:
        word=endings[symbol][used[symbol]];used[symbol]+=1
        lines.append(rng.choice(openings)+' the '+word+'.')
    w.add('\n'.join(lines),'Which end-rhyme scheme do the four lines follow?',schemes,{schemes.index(scheme)})
w.finish()

w=Writer('chess_move_legality','knowledge');board=chess.Board()
for case in range(30000):
    if board.is_game_over() or board.fullmove_number>80:board=chess.Board()
    legal=list(board.legal_moves)
    if not legal:board=chess.Board();legal=list(board.legal_moves)
    selected=rng.sample(legal,min(2,len(legal)));options=[move.uci() for move in selected];positive=set(range(len(options)))
    while len(options)<4:
        move=chess.Move(rng.randrange(64),rng.randrange(64))
        if move.uci() not in options and not board.is_legal(move):options.append(move.uci())
    w.add(dict(fen=board.fen(),board=str(board),side_to_move='white' if board.turn else 'black'),
          'Which supplied moves are legal under standard chess rules? Multiple candidates can be legal.',options,positive)
    board.push(rng.choice(legal))
w.finish()
(OUT/'MANIFEST.json').write_text(json.dumps(dict(complete=True,seed=42,sources=REPORT,
    purpose='requested poetry/music/rules/workflow/chess training coverage; not a replacement for canonical NLI',
    tests_run=False,file_hash_checks=False),ensure_ascii=False,indent=2))
