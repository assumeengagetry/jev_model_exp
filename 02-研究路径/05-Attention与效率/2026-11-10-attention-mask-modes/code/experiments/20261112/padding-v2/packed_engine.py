"""Original complete renderer/readout; new real-token execution only."""
from functools import lru_cache
import json
import os
import sys
import torch
from transformers import AutoTokenizer
from pf_common import ROOT, OLD

# Read-only inheritance: no edits to historical sealed numerical/renderer files.
REFERENCE='/tmp/opencode/jev-06b-research-20261112-r1-eval-recovery/code/inference'
sys.path.insert(0,REFERENCE)
from kernel_engine import KernelPairClassifierEngine
from packed_model import load_model, PackedBackbone, install_inference
from packed_runtime import PackedRuntime


class PaddingFreeEngine(KernelPairClassifierEngine):
    latency='Complete request, all fields/candidates; eager dynamic shape JIT included in first use.'
    def __init__(self,model,base_model,layout,device='cuda:0',max_microbatch_pairs=32,
                 max_microbatch_tokens=32768,**options):
        if layout not in ('dense','packed'):raise ValueError('Explicit controlled layout required')
        self.device=torch.device(device);self.max_pairs=int(max_microbatch_pairs);self.token_budget=int(max_microbatch_tokens)
        if not 1<=self.max_pairs<=32 or self.token_budget<1:raise ValueError('Invalid wave limits')
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        from pathlib import Path
        self.tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
        self.model,backbone,head,fa3=load_model(Path(model),Path(base_model),self.device)
        self.model.requires_grad_(False)
        config=backbone.config
        self.context=int(config.max_position_embeddings)
        self.pad,self.eos=self.tokenizer.pad_token_id,self.tokenizer.eos_token_id
        if config.id2label!={0:'not_match',1:'match'} or self.pad!=151643 or self.eos!=151645:
            raise ValueError('Frozen label/token identity changed')
        head=install_inference(backbone,head)
        core=PackedBackbone(backbone,head,fa3,fused_inference=False)
        self.execution=PackedRuntime(core,self.pad,self.eos,self.context,self.device,layout)
        self.installation={'real_token_attention':'FA3 varlen / immutable packed prefix fork',
            'same_cached_lora_bf16_weights_both_layouts':True,'extra_math_fusions':False,'trainable_fused_path':False}
        self.name='qwen3-0.6b-pf-mechanism-'+layout
        self.margin_stream=open(os.environ['PF_MARGINS_PATH'],'x') if 'PF_MARGINS_PATH' in os.environ else None
        self.margin_ordinal=0
        self.provenance={'checkpoint':model,'base_model':base_model,'layout':layout,
            'kind':'same historical R1 adapter/head; known input exposure, diagnostic only',
            'rendering':'unchanged candidate-pair-v1','truncation':False,'option_filtering':False,
            'teacher_used_for_inference':False,'cross_request_prediction_cache':False,
            'lora_merged':False,'experiment_stage':'pf-'+layout,
            'numerical_equivalence':'pending actual controlled full public comparison',
            'evaluation_scope':'full public0.3 diagnostic, not clean ranking or single-GPU latency admission'}

    def _waves(self,rows):
        order=sorted(range(len(rows)),key=lambda i:len(rows[i]));start=0
        while start<len(order):
            end=start;tokens=0
            while end<len(order) and end-start<self.max_pairs:
                proposed=tokens+len(rows[order[end]])
                if end>start and proposed>self.token_budget:break
                tokens=proposed;end+=1
            chosen=order[start:end]
            yield chosen,[rows[i] for i in chosen]
            start=end

    def score_request(self,fields):
        margins,tokens=super().score_request(fields)
        if self.margin_stream is not None:
            self.margin_stream.write(json.dumps({'request_ordinal':self.margin_ordinal,
                'fields':[{'name':f.name,'begin':f.begin,'keys':f.keys} for f in fields],
                'margins':margins.tolist()},allow_nan=False)+'\n')
            self.margin_ordinal+=1
        return margins,tokens

    def close(self):
        if self.margin_stream is not None:
            self.margin_stream.flush();os.fsync(self.margin_stream.fileno());self.margin_stream.close()
            self.margin_stream=None
        super().close()
