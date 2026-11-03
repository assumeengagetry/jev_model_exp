#!/usr/bin/env python3
"""Coefficient-only SAMerging for common-origin binary Qwen3 specialists."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from official_sam_utils import SAM

SOURCE = Path('/data/run01/scwb721/jev-kd-20261101')
WORK = SOURCE/'workspace'
ROOT = SOURCE/'experiments/samerging-four-experts-20261103'
COMMON = WORK/'models/Qwen3-0.6B-common-pair-init-r32'
NAME = 'MERGE_SAMERGING'
DOMAINS = ['knowledge','retrieval','tools','arts']
EXPERTS = ['SPECIALIST_'+d.upper() for d in DOMAINS]
sys.path.insert(0, str(SOURCE/'staged/scale-series-20261102/code'))
os.environ['JEV_KD_ROOT'] = str(ROOT)
from kd_runtime import TokenStore, POOL, legacy, atomic_json


def emit(event, **fields):
    fields.update(event=event, unix_time=time.time())
    print(json.dumps(fields), flush=True)


def status(phase, **fields):
    value = dict(phase=phase, unix_time=time.time(), **fields)
    atomic_json(ROOT/'status.json', value)
    emit('samerging_status', **value)


def load_effective(adapter):
    config = AutoConfig.from_pretrained(COMMON, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(WORK/'models/Qwen3-0.6B', config=config,
             torch_dtype=torch.float32, attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    return model.merge_and_unload().requires_grad_(False).eval()


class Merger:
    def __init__(self):
        self.model = load_effective(COMMON)
        initial = self.model.state_dict()
        states = []
        for name in EXPERTS:
            model = load_effective(ROOT/'assets'/name/'final')
            states.append({k:v.detach().cpu().float().contiguous() for k,v in model.state_dict().items()})
            del model
        if any(s.keys() != initial.keys() for s in states):
            raise RuntimeError('Expert effective tensor keys disagree')
        self.keys = [k for k,v in initial.items() if any(not torch.equal(v,s[k]) for s in states)]
        if 'score.weight' not in self.keys:
            raise RuntimeError('The shared binary classification head must be included')
        self.initial = {}
        self.delta = {}
        for key in self.keys:
            self.initial[key] = initial[key].float().to('cuda')
            self.delta[key] = torch.stack([s[key]-initial[key] for s in states]).to('cuda')
        del states, initial
        self.model.to(device='cuda', dtype=torch.bfloat16)
        self.alpha = torch.nn.Parameter(torch.full((4,len(self.keys)),0.5,device='cuda',dtype=torch.float32))
        self.tokenizer = AutoTokenizer.from_pretrained(COMMON, local_files_only=True)
        atomic_json(ROOT/'coefficient-map.json', dict(experts=EXPERTS, tensor_keys=self.keys,
                    coefficients=self.alpha.numel(), initialization=0.5, clamp=False,
                    construction='effective W0 + sum_t lambda[t,k]*(Wt-W0), including one shared binary score head'))
        emit('samerging_model_ready', changed_tensors=len(self.keys), coefficients=self.alpha.numel(),
             allocated_GiB=torch.cuda.memory_allocated()/2**30)

    def weights(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        return {k:(self.initial[k]+torch.tensordot(alpha[:,i],self.delta[k],dims=([0],[0]))).to(torch.bfloat16)
                for i,k in enumerate(self.keys)}

    def logits(self, inputs, alpha=None):
        # Rebuild for EVERY forward; SAM's perturbed pass cannot reuse cached weights.
        return functional_call(self.model,self.weights(alpha),(),inputs,tie_weights=True,strict=False).logits.float()

    @torch.no_grad()
    def export(self):
        final = ROOT/'outputs'/NAME/'final'
        final.mkdir(parents=True,exist_ok=True)
        values = {k:v.detach().cpu().contiguous() for k,v in self.model.state_dict().items()}
        values.update({k:v.detach().cpu().contiguous() for k,v in self.weights().items()})
        temporary = final/'model.safetensors.writing'
        save_file(values,str(temporary),metadata={'format':'pt'})
        temporary.replace(final/'model.safetensors')
        self.model.config.architectures = ['Qwen3ForSequenceClassification']
        self.model.config.use_cache = False
        self.model.config.save_pretrained(final)
        self.tokenizer.save_pretrained(final)
        return final


def calibration():
    path = ROOT/'calibration.npz'
    if not path.exists():
        rng = np.random.default_rng(42)
        data = {}
        for domain in DOMAINS:
            states = np.load(POOL/'pairs'/f'{domain}_state.npy',mmap_mode='r')
            data[domain] = rng.choice(len(states),size=1600,replace=False)
            data[domain+'_schedule'] = np.concatenate([rng.permutation(1600).reshape(100,16) for _ in range(6)])
        np.savez(path,**data)
        atomic_json(ROOT/'calibration-manifest.json',dict(domains=DOMAINS,unique_inputs_per_domain=1600,
                    total_unique_inputs=6400,batch_per_domain=16,steps=600,cycles=6,seed=42,
                    labels_used=False,evaluation_inputs_used=False,source=str(POOL)))
    saved = np.load(path)
    limits = json.loads((POOL/'TRAINING_READY.json').read_text())['domains']
    pools = []
    for domain in DOMAINS:
        ids = saved[domain]
        fields = {key:np.asarray(np.load(POOL/'pairs'/f'{domain}_{key}.npy',mmap_mode='r')[ids])
                  for key in ['state','candidate','length']}
        fields.update(target=np.zeros(1600,dtype=np.float32),
                      max_length=np.full(1600,limits[domain]['max_length'],dtype=np.uint32))
        pools.append(fields)
    return saved,pools


def minibatches(batch, indices, store, merger):
    selected = {k:v[indices] for k,v in batch.items()}
    selected['position'] = np.asarray(indices,dtype=np.int64)
    for micro in legacy.microbatches(selected,16,8192):
        inputs,_,tokens = legacy.collate(store,micro,merger.tokenizer.pad_token_id,
                                        merger.tokenizer.eos_token_id,'cuda')
        yield inputs,micro['position'],tokens


@torch.no_grad()
def teacher_logits(merger, pools, store):
    path = ROOT/'expert-logits.npy'
    if path.exists():
        logits = np.load(path)
        if logits.shape != (4,1600,2) or not np.isfinite(logits).all():
            raise RuntimeError('Invalid cached expert logits; preserved for diagnosis')
        return torch.from_numpy(logits).to('cuda')
    logits = np.empty((4,1600,2),dtype=np.float32)
    for task,batch in enumerate(pools):
        alpha = torch.zeros_like(merger.alpha)
        alpha[task] = 1
        for inputs,positions,_ in minibatches(batch,np.arange(1600),store,merger):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction = merger.logits(inputs,alpha)
            logits[task,positions] = prediction.cpu().numpy()
        emit('samerging_expert_logits_cached', expert=EXPERTS[task], pairs=1600,
             sample_alignment='exact fixed calibration positions')
    if not np.isfinite(logits).all():
        raise RuntimeError('Nonfinite expert logits')
    np.save(path,logits)
    return torch.from_numpy(logits).to('cuda')


def train():
    if (ROOT/'MERGE_COMPLETE.json').exists():
        return
    started = time.monotonic()
    status('constructing_effective_task_vectors')
    merger = Merger()
    saved,pools = calibration()
    store = TokenStore()
    status('caching_aligned_expert_logits')
    experts = teacher_logits(merger,pools,store)
    optimizer = SAM([merger.alpha],torch.optim.SGD,lr=1e-3,rho=0.07,adaptive=True,
                    momentum=0.99,weight_decay=5e-4)
    checkpoint = ROOT/'calibration-checkpoint.pt'
    first = 0
    if checkpoint.exists():
        prior = torch.load(checkpoint,map_location='cuda',weights_only=True)
        with torch.no_grad():merger.alpha.copy_(prior['alpha'])
        optimizer.base_optimizer.load_state_dict(prior['optimizer'])
        optimizer.param_groups = optimizer.base_optimizer.param_groups
        first = prior['step']
        emit('samerging_resumed',step=first)

    def backward(step):
        total = 0.0
        for task,domain in enumerate(DOMAINS):
            positions = saved[domain+'_schedule'][step]
            for inputs,local,_ in minibatches(pools[task],positions,store,merger):
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction = merger.logits(inputs)
                target = F.softmax(experts[task,local],dim=-1)
                loss = F.kl_div(F.log_softmax(prediction,dim=-1),target,reduction='sum')/(16*4)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite calibration KL')
                loss.backward()
                total += float(loss.detach())
        return total

    status('optimizing_merge_coefficients',step=first,total_steps=600)
    for step in range(first,600):
        optimizer.zero_grad()
        clean = backward(step)
        grad_norm = float(merger.alpha.grad.norm())
        if not torch.isfinite(merger.alpha.grad).all() or grad_norm == 0:
            raise RuntimeError('SAMerging coefficients have no finite nonzero gradient')
        original = merger.alpha.detach().clone()
        optimizer.first_step(zero_grad=True)
        perturbation = float((merger.alpha-original).norm())
        if perturbation == 0:raise RuntimeError('SAM failed to perturb actual coefficients')
        perturbed = backward(step)
        optimizer.second_step(zero_grad=True)
        if not torch.isfinite(merger.alpha).all():raise RuntimeError('Nonfinite merge coefficients')
        if step == first or (step+1)%25 == 0:
            value = dict(step=step+1,total_steps=600,clean_kl=clean,perturbed_kl=perturbed,
                         coefficient_gradient_norm=grad_norm,sam_coefficient_perturbation_norm=perturbation,
                         coefficient_min=float(merger.alpha.min()),coefficient_max=float(merger.alpha.max()),
                         coefficient_mean=float(merger.alpha.mean()),elapsed_seconds=time.monotonic()-started)
            atomic_json(ROOT/'calibration-progress.json',value)
            emit('samerging_calibration_progress',**value)
            temporary = checkpoint.with_suffix('.writing.pt')
            torch.save(dict(alpha=merger.alpha.detach(),optimizer=optimizer.base_optimizer.state_dict(),step=step+1),temporary)
            temporary.replace(checkpoint)
    final = merger.export()
    torch.save(merger.alpha.detach().cpu(),ROOT/'merging-coefficients.pt')
    record = dict(complete=True,model=NAME,checkpoint=str(final),experts=EXPERTS,
                  method='SAMerging_layer_wise_effective_weights',steps=600,unique_calibration_inputs=6400,
                  batch_per_task=16,task_loss_weight=0.25,temperature=1,init_coefficients=0.5,
                  optimizer='official SAM with SGD',lr=1e-3,momentum=0.99,weight_decay=5e-4,
                  rho=0.07,adaptive=True,clamp=False,classification_head_merged=True,
                  optimized_parameters='merge coefficients only',coefficient_count=merger.alpha.numel(),
                  calibration_labels_used=False,evaluation_inputs_used_for_calibration=False,
                  forward_and_export_dtype='BF16',merge_arithmetic_dtype='FP32',
                  elapsed_seconds=time.monotonic()-started,
                  upstream_commit='a194e8ed8ab790dcae9c02a8d1467e00e86b53e5',
                  adaptations=['Qwen3 shared binary head instead of GPT2 task-specific heads',
                    'fixed input-ID/logit pairing instead of advancing shuffled loaders twice',
                    'fresh merged weights after SAM coefficient perturbation',
                    'stable log_softmax KL and gradient accumulation preserving task batch16'])
    atomic_json(ROOT/'MERGE_COMPLETE.json',record)
    emit('samerging_calibration_complete',**record)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'allocation.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),
                pid=os.getpid(),devices=os.environ.get('CUDA_VISIBLE_DEVICES'),physical_step_gpus=os.environ.get('SLURM_STEP_GPUS')))
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        train()
        gc.collect()
        torch.cuda.empty_cache()
        from evaluate_and_report import main as evaluate
        evaluate()
    except BaseException as error:
        atomic_json(ROOT/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),unix_time=time.time()))
        traceback.print_exc()
        raise


if __name__ == '__main__':main()
