"""Apply the attention-mode edits to copies of the three scripts (text only; nothing is imported)."""
import sys
from pathlib import Path

original, patched = Path(sys.argv[1]), Path(sys.argv[2])

EDITS = {
'pair_training_runtime.py': [
("""def assemble(store,state,candidate,max_length,eos):
""", """def assemble(store,state,candidate,max_length,eos,boundary=False):
"""),
("""    return np.concatenate((prefix,suffix,np.array([eos],dtype=np.int32)))
""", """    ids=np.concatenate((prefix,suffix,np.array([eos],dtype=np.int32)))
    return (ids,len(prefix)) if boundary else ids
"""),
("""def collate(store,batch,pad,eos,device):
    ids=[assemble(store,state,candidate,int(limit),eos)
         for state,candidate,limit in zip(batch['state'],batch['candidate'],batch['max_length'])]
""", """def collate(store,batch,pad,eos,device,layout=False):
    rows=[assemble(store,state,candidate,int(limit),eos,boundary=True)
          for state,candidate,limit in zip(batch['state'],batch['candidate'],batch['max_length'])]
    ids=[row for row,_ in rows]
"""),
("""    return {'input_ids':input_ids.to(device,non_blocking=True),'attention_mask':mask.to(device,non_blocking=True)},targets.to(device,non_blocking=True),sum(map(len,ids))
""", """    result=({'input_ids':input_ids.to(device,non_blocking=True),'attention_mask':mask.to(device,non_blocking=True)},targets.to(device,non_blocking=True),sum(map(len,ids)))
    if not layout:return result
    lengths=lambda values:torch.tensor(values,dtype=torch.int32,device=device)
    return (*result,(lengths([prefix for _,prefix in rows]),lengths([len(row) for row in ids])))
"""),
],
'train_five_dimension.py': [
("""    restore_attention_record,
)
""", """    restore_attention_record,
)
from attention_modes import MODES, attention_layout, enable_bidirectional, install_prefix_block_flash
"""),
("""        raise RuntimeError('Student training requires explicit FlashAttention-3; SDPA fallback is forbidden')
""", """        raise RuntimeError('Student training requires explicit FlashAttention-3; SDPA fallback is forbidden')
    attention_mode=cfg.get('pair_attention_mode','causal')
    if attention_mode not in MODES:raise RuntimeError('Unknown pair_attention_mode')
"""),
("""            raise RuntimeError('Checkpoint belongs to another exploration policy; preserved')
""", """            raise RuntimeError('Checkpoint belongs to another exploration policy; preserved')
        if state.get('pair_attention_mode','causal')!=attention_mode:
            raise RuntimeError('Checkpoint was trained with another attention mode; preserved')
"""),
("""    assert_student_uses_fa3(model)
""", """    assert_student_uses_fa3(model)
    if attention_mode=='bidirectional':enable_bidirectional(model)
    elif attention_mode=='prefix_block':install_prefix_block_flash(BACKEND)
    model.config.pair_attention_mode=attention_mode
"""),
("""                              exploration_policy=cfg['exploration_policy'],
                              sampler_identity""", """                              exploration_policy=cfg['exploration_policy'],pair_attention_mode=attention_mode,
                              sampler_identity"""),
("""                    inputs,targets,_=collate(store,micro,pad,eos,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(**inputs).logits
                        margin=output[:,1].float()-output[:,0].float()
                        loss=group_micro_loss(margin,micro,plan['teacher_relation_target'],cfg,device)
                    (loss/len(indices)).backward();local_loss+=loss.detach().double()
""", """                    inputs,targets,_,layout=collate(store,micro,pad,eos,device,layout=True)
                    with attention_layout(*layout):
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            output=model(**inputs).logits
                            margin=output[:,1].float()-output[:,0].float()
                            loss=group_micro_loss(margin,micro,plan['teacher_relation_target'],cfg,device)
                        (loss/len(indices)).backward();local_loss+=loss.detach().double()
"""),
],
'decision_engine.py': [
("""class PairClassifierEngine(Engine):
""", """def dense_mask(mode,row_len,width,prefix_len):
    # True = may attend. Padded query rows keep the real keys, so no row is fully masked.
    pos=torch.arange(width,device=row_len.device);key_ok=pos[None]<row_len[:,None]
    if mode=='bidirectional':return key_ok[:,None,None,:]
    candidate=pos[None]>=prefix_len[:,None]
    return ((candidate[:,:,None]|~candidate[:,None,:])&key_ok[:,None,:])[:,None]


class PairClassifierEngine(Engine):
"""),
("""        config=AutoConfig.from_pretrained(model,trust_remote_code=False)
""", """        config=AutoConfig.from_pretrained(model,trust_remote_code=False)
        self.attention=getattr(config,'pair_attention_mode','causal')
        if self.attention not in ('causal','bidirectional','prefix_block'):
            raise ValueError('Unknown pair_attention_mode in saved config')
"""),
("""            'inference_attention':'sdpa','dtype'""", """            'inference_attention':'sdpa','attention_mask':self.attention,'dtype'"""),
("""    def score(self,rows):
        lengths=np.asarray([len(row) for row in rows],dtype=np.int64)
""", """    def score(self,rows,prefixes):
        lengths=np.asarray([len(row) for row in rows],dtype=np.int64)
        prefixes=np.asarray(prefixes,dtype=np.int64)
"""),
("""            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):
""", """            try:
                if self.attention!='causal':
                    mask=dense_mask(self.attention,torch.from_numpy(lengths[chosen]).to(self.device),width,
                                    torch.from_numpy(prefixes[chosen]).to(self.device))
                with torch.autocast('cuda',dtype=torch.bfloat16):
"""),
("""        rows=[];fields=[];state=text(state)
""", """        rows=[];prefixes=[];fields=[];state=text(state)
"""),
("""            for value in values:rows.append(prefix_ids+self.tokens(value)+(self.eos,))
""", """            for value in values:rows.append(prefix_ids+self.tokens(value)+(self.eos,));prefixes.append(len(prefix_ids))
"""),
("""        margins,count=self.score(rows);answers={}
""", """        margins,count=self.score(rows,prefixes);answers={}
"""),
],
}

for name, edits in EDITS.items():
    text = (original / name).read_text()
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f'{name}: expected exactly one match for:\n{old}')
        text = text.replace(old, new)
    (patched / name).write_text(text)
    print('patched', name, len(edits), 'edits')
