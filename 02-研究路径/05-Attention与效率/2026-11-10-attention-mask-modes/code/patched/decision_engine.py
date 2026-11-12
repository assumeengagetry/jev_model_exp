"""Portable Decision Index Engine for the trained pair classifier, not an LM readout."""
from functools import lru_cache
import gc
import json
import math
import re
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from decision_index.engines.base import Engine, Unsupported

REFERENCES=re.compile(r'(all|none|both|neither).{0,32}(above|other|option|answer|criteri)|(?:option|answer|choice)\s+[A-Z](?:\b|\))',re.I)
RELATIVE=re.compile(r'better|preferred|preferable|funnier|more helpful|more appropriate|which.*(?:prefer|quality)',re.I)


def text(value):
    if value is None:return ''
    if isinstance(value,str):return value.strip()
    return json.dumps(value,ensure_ascii=False,sort_keys=True)


def criteria_context(question,options):
    if not (any(REFERENCES.search(option) for option in options)
            or (len(options)<=8 and RELATIVE.search(question))):return ''
    labels=[chr(65+i) if i<26 else str(i+1) for i in range(len(options))]
    return '\n'.join(f'{label}: {option}' for label,option in zip(labels,options))


def dense_mask(mode,row_len,width,prefix_len):
    # True = may attend. Padded query rows keep the real keys, so no row is fully masked.
    pos=torch.arange(width,device=row_len.device);key_ok=pos[None]<row_len[:,None]
    if mode=='bidirectional':return key_ok[:,None,None,:]
    candidate=pos[None]>=prefix_len[:,None]
    return ((candidate[:,:,None]|~candidate[:,None,:])&key_ok[:,None,:])[:,None]


class PairClassifierEngine(Engine):
    name='qwen3-0.6b-five-dimension-distilled'
    latency='Single-process in-request rendering, tokenization and independent candidate scoring; excludes model load.'

    def __init__(self,model,base_model='Qwen/Qwen3-0.6B',device='cuda:0',
                 max_microbatch_pairs=32,max_microbatch_tokens=32768,**options):
        super().__init__(model=model,base_model=base_model,device=device,
            max_microbatch_pairs=max_microbatch_pairs,max_microbatch_tokens=max_microbatch_tokens,**options)
        self.device=torch.device(device)
        self.max_pairs=int(max_microbatch_pairs);self.token_budget=int(max_microbatch_tokens)
        if self.max_pairs<1 or self.token_budget<1:raise ValueError('Positive batching limits are required')
        if self.device.type!='cuda':raise ValueError('This submitted inference configuration requires CUDA')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
        self.tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
        config=AutoConfig.from_pretrained(model,trust_remote_code=False)
        self.attention=getattr(config,'pair_attention_mode','causal')
        if self.attention not in ('causal','bidirectional','prefix_block'):
            raise ValueError('Unknown pair_attention_mode in saved config')
        base=AutoModelForSequenceClassification.from_pretrained(base_model,config=config,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False)
        self.model=PeftModel.from_pretrained(base,model,is_trainable=False).to(self.device).eval()
        self.model.config.use_cache=False
        self.context=int(config.max_position_embeddings)
        self.pad=self.tokenizer.pad_token_id;self.eos=self.tokenizer.eos_token_id
        if self.pad is None or self.eos is None:raise ValueError('Saved tokenizer must supply pad and EOS IDs')
        self.provenance={'kind':'LoRA rank32 with trained binary classification head',
            'base_model':base_model,'checkpoint':model,'rendering':'candidate-pair-v1',
            'readout':'softmax of logit(match)-logit(nonmatch) over every original field criterion',
            'truncation':False,'option_filtering':False,'context_tokens_per_candidate':self.context,
            'inference_attention':'sdpa','attention_mask':self.attention,'dtype':'bf16 base, saved PEFT head/adapter precision retained',
            'teacher_used_for_inference':False}

    @lru_cache(maxsize=2048)
    def tokens(self,value):
        return tuple(self.tokenizer(value,add_special_tokens=False,truncation=False)['input_ids'])

    def runtime(self):
        return {**self.provenance,'device':str(self.device),'max_microbatch_pairs':self.max_pairs,
            'max_microbatch_tokens':self.token_budget,'cpu_token_cache_entries':2048}

    def synchronize(self):torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def score(self,rows,prefixes):
        lengths=np.asarray([len(row) for row in rows],dtype=np.int64)
        prefixes=np.asarray(prefixes,dtype=np.int64)
        if np.any(lengths>self.context):
            raise Unsupported('Whole candidate input exceeds model context window; nothing truncated')
        order=np.argsort(lengths,kind='stable');scores=np.empty(len(rows),dtype=np.float64)
        start=0
        while start<len(order):
            end=start+1
            while end<len(order) and end-start<self.max_pairs:
                width=(int(lengths[order[end]])+7)//8*8
                if width*(end-start+1)>self.token_budget:break
                end+=1
            chosen=order[start:end]
            width=(int(lengths[chosen].max())+7)//8*8
            ids=torch.full((len(chosen),width),self.pad,dtype=torch.long)
            mask=torch.zeros_like(ids,dtype=torch.bool)
            for i,index in enumerate(chosen):
                row=rows[int(index)];ids[i,:len(row)]=torch.tensor(row,dtype=torch.long);mask[i,:len(row)]=True
            oom=False;output=None
            try:
                if self.attention!='causal':
                    mask=dense_mask(self.attention,torch.from_numpy(lengths[chosen]).to(self.device),width,
                                    torch.from_numpy(prefixes[chosen]).to(self.device))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=self.model(input_ids=ids.to(self.device),attention_mask=mask.to(self.device)).logits
                values=(output[:,1].float()-output[:,0].float()).cpu().numpy()
                if not np.isfinite(values).all():raise ValueError('Nonfinite pair-classifier margins')
                scores[chosen]=values;start=end
            except torch.cuda.OutOfMemoryError:
                oom=True
            finally:
                output=ids=mask=None
            if oom:
                gc.collect();torch.cuda.empty_cache()
                if len(chosen)==1:
                    raise Unsupported('Whole single-candidate input exceeds available GPU memory; request rejected intact, no candidate removed')
                self.token_budget=max(1,self.token_budget//2)
        return scores,int(lengths.sum())

    def __call__(self,state,questions):
        rows=[];prefixes=[];fields=[];state=text(state)
        for field,question in questions.items():
            kind=question['type']
            if kind not in ('choice','noul'):raise Unsupported('Unsupported question type '+str(kind))
            options=question.get('criteria')
            if kind=='noul' and not options:options={'true':'Yes.','false':'No.'}
            if not isinstance(options,dict) or not options:raise ValueError('Every field needs named criteria')
            keys=list(options);values=[text(value) for value in options.values()]
            instructions=text(question.get('instructions',''))
            context=criteria_context(instructions,values)
            prefix='Decide whether the candidate is a correct match.\nInput:\n'+state
            if instructions:prefix+='\nQuestion:\n'+instructions
            if context:prefix+='\nAvailable criteria:\n'+context
            prefix+='\nCandidate:\n'
            prefix_ids=self.tokens(prefix)
            begin=len(rows)
            for value in values:rows.append(prefix_ids+self.tokens(value)+(self.eos,));prefixes.append(len(prefix_ids))
            fields.append((field,kind,keys,begin,len(rows)))
        if not rows:raise ValueError('An inference request must contain at least one field')
        margins,count=self.score(rows,prefixes);answers={}
        for field,kind,keys,begin,end in fields:
            scores=margins[begin:end];probabilities=np.exp(scores-scores.max());probabilities/=probabilities.sum()
            distribution={key:float(value) for key,value in zip(keys,probabilities)}
            if kind=='choice':
                answers[field]={'type':'choice','choice':keys[int(np.argmax(probabilities))],
                    'probabilities':distribution}
            else:
                positive=[key for key in keys if str(key).lower() in ('true','yes','1')]
                if len(keys)!=2 or len(positive)!=1:raise ValueError('noul needs one named true/yes criterion')
                answers[field]={'type':'noul','noul':distribution[positive[0]],'probabilities':distribution}
        return {'model':self.name,'answers':answers,'usage':{'input_tokens':count}},None

    def close(self):
        self.tokens.cache_clear()
        self.model=None
        gc.collect();torch.cuda.empty_cache()
