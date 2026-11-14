"""Differentiable Qwen3 over real tokens; PEFT modules/parameter format unchanged."""
import math
import sys
import torch
from torch import nn
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def load_model(path,base,device,trainable=False):
    from pf_common import FA3
    sys.path.insert(0,'/home/dev/assumeengage/scripts')
    from student_flash_attention3_repro import register_student_fa3, BACKEND
    from student_flash_attention3 import load_student_fa3
    register_student_fa3(FA3)
    config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if (config.model_type!='qwen3' or config.num_hidden_layers!=28 or config.hidden_size!=1024
        or config.num_attention_heads!=16 or config.num_key_value_heads!=8 or config.head_dim!=128
        or config.rope_scaling is not None or config.attention_dropout!=0 or config.attention_bias
        or any(t!='full_attention' for t in config.layer_types)):
        raise ValueError('Only the actual admitted Qwen3 configuration is supported')
    model=AutoModelForSequenceClassification.from_pretrained(base,config=config,
        torch_dtype=torch.bfloat16,attn_implementation=BACKEND,local_files_only=True,trust_remote_code=False)
    model.score.float()
    model=PeftModel.from_pretrained(model,path,is_trainable=trainable,local_files_only=True)
    saved=load_file(str(path/'adapter_model.safetensors'),device='cpu')
    actual=get_peft_model_state_dict(model)
    if len(saved)!=393 or set(saved)!=set(actual) or any(v.dtype!=actual[k].dtype or not torch.equal(v,actual[k].detach().cpu()) for k,v in saved.items()):
        raise RuntimeError('393 saved adapter/head tensors must load exactly')
    adapter=model.peft_config['default']
    if adapter.r!=32 or adapter.lora_alpha!=64 or adapter.lora_dropout!=0:
        raise ValueError('Expected original r32/alpha64/dropout0 adapter')
    model.config.use_cache=False
    model.to(device)
    model.train(trainable)
    classifier=model.base_model.model
    return model,classifier.model,classifier.score.modules_to_save['default'],load_student_fa3(FA3)


class PackedBackbone(nn.Module):
    def __init__(self,backbone,head,fa3,fused_inference=False):
        super().__init__();self.backbone=backbone;self.head=head;self.fa3=fa3
        self.fused_inference=fused_inference

    def forward(self,batch,prefix=None,capture_prefix=False):
        if capture_prefix and (len(batch.lengths)!=1 or prefix is not None):
            raise ValueError('One immutable real prefix owner required')
        x=self.backbone.embed_tokens(batch.ids)
        pos=batch.positions.reshape(1,-1)
        cos,sin=self.backbone.rotary_emb(x,pos)
        cos=cos.reshape(*batch.ids.shape,128).unsqueeze(-2)
        sin=sin.reshape(*batch.ids.shape,128).unsqueeze(-2)
        cache=[]
        for layer in self.backbone.layers:
            residual=x;x=layer.input_layernorm(x);attn=layer.self_attn
            if self.fused_inference:
                if torch.is_grad_enabled():raise RuntimeError('Fused inference cannot train')
                from fused_kernels import norm_rope
                q=norm_rope(attn.q_proj(x).reshape(1,-1,2048),attn.q_norm.weight,
                    cos.reshape(-1,128),sin.reshape(-1,128),16,attn.q_norm.variance_epsilon)
                k=norm_rope(attn.k_proj(x).reshape(1,-1,1024),attn.k_norm.weight,
                    cos.reshape(-1,128),sin.reshape(-1,128),8,attn.k_norm.variance_epsilon)
                q=q.transpose(1,2).reshape(*batch.ids.shape,16,128)
                k=k.transpose(1,2).reshape(*batch.ids.shape,8,128)
            else:
                q=attn.q_norm(attn.q_proj(x).view(*batch.ids.shape,16,128))
                k=attn.k_norm(attn.k_proj(x).view(*batch.ids.shape,8,128))
                q=q*cos+rotate_half(q)*sin;k=k*cos+rotate_half(k)*sin
            v=attn.v_proj(x).view(*batch.ids.shape,8,128)
            q=batch.real(q).contiguous();k=batch.real(k).contiguous();v=batch.real(v).contiguous()
            if capture_prefix:cache.append((k.detach(),v.detach()))
            if prefix is None:
                out=self.fa3.flash_attn_varlen_func(q,k,v,batch.cu,batch.cu,
                    batch.max_length,batch.max_length,softmax_scale=attn.scaling,
                    causal=True,deterministic=True)
            else:
                if torch.is_grad_enabled():raise RuntimeError('Fork is inference-only; training uses native FA3 backward')
                from packed_fork import fork_attention
                pk,pv=prefix[attn.layer_idx]
                out=fork_attention(q,k,v,pk,pv,batch.cu,batch.max_length)
            out=batch.restore(out.reshape(batch.total,-1))
            x=residual+attn.o_proj(out)
            x=x+layer.mlp(layer.post_attention_layernorm(x))
        x=self.backbone.norm(x)
        eos=x.reshape(-1,x.shape[-1]).index_select(0,batch.eos_indices)
        return self.head(eos),tuple(cache) if capture_prefix else None


def install_inference(backbone,head):
    """Only tokenwise fusion; attention stays the same packed FA3/fork contract."""
    from kernel_model import CachedLoRA, CachedLinear, FusedRMS, FusedMLP
    from peft.tuners.lora.layer import Linear
    projections=[(n,m) for n,m in backbone.named_modules() if isinstance(m,Linear)]
    if len(projections)!=196:raise ValueError('196 LoRA projections required')
    for name,module in projections:
        parent,_,leaf=name.rpartition('.')
        setattr(backbone.get_submodule(parent),leaf,CachedLoRA(module))
    for layer in backbone.layers:
        layer.input_layernorm=FusedRMS(layer.input_layernorm)
        layer.post_attention_layernorm=FusedRMS(layer.post_attention_layernorm)
        layer.mlp=FusedMLP(layer.mlp)
    backbone.norm=FusedRMS(backbone.norm)
    return CachedLinear(head)


class RowTrace:
    """External evidence hook, not a production-mode switch or capacity estimate."""
    def __init__(self,backbone,batch):self.backbone=backbone;self.batch=batch;self.records=[];self.handles=[]
    def __enter__(self):
        for name,module in self.backbone.named_modules():
            if not isinstance(module,(nn.Linear,nn.Embedding)) and module.__class__.__name__ not in ('Qwen3RMSNorm','FusedRMS','FusedMLP','CachedLoRA'):continue
            def hook(mod,args,label=name):
                x=args[0];rows=x.numel() if isinstance(mod,nn.Embedding) else math.prod(x.shape[:-1])
                heads=16 if '.q_norm' in label else 8 if '.k_norm' in label else 1
                expected=self.batch.ids.numel()*heads
                self.records.append({'module':label,'shape':list(x.shape),'rows':rows,'expected_rows':expected})
                if self.batch.layout=='packed' and rows!=self.batch.total*heads:
                    raise RuntimeError('Packed tokenwise module processed non-real rows: '+label)
            self.handles.append(module.register_forward_pre_hook(hook))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
    def result(self):
        return {'layout':self.batch.layout,'real_tokens':self.batch.total,'capacity_rows':self.batch.ids.numel(),
            'layers':len({r['module'].split('.')[1] for r in self.records if r['module'].startswith('layers.')}),
            'records':self.records}
