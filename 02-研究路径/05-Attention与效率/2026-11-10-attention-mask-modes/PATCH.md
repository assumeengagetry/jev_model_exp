# Qwen3-0.6B pair 分类器：attention mask 三种模式（补丁草案）

日期：2026-11-10。**状态：补丁草案。没有应用到集群，没有在 GPU 上运行，没有训练或评测。** 本机只做了 CPU 检查（随机初始化的小模型＋仿真的 flash 内核），见文末“已验证／未验证”。

榜单调研见 [attention-survey.md](../../../01-研究结论/05-Attention与效率/2026-11-09-attention-audit/attention-survey.md)。榜单上两条 Qwen3-0.6B-Base 记录都保留 causal，没有在这个底座上放开 mask 的先例；这里是一个待验证的实验臂，不是已知有效的改进。

## 范围

- 对象：五维蒸馏那套 pair 分类器。`AutoModelForSequenceClassification`（Qwen3-0.6B）＋LoRA r32＋二分类 head；每个候选单独成行 `prefix + candidate + EOS`；训练用 FA3（`flash_attn3_hopper`），推理用 SDPA。
- Torch 2.9.1、Transformers 4.57.6、PEFT 0.18.1 不变，不新增依赖。
- 新增一个配置键 `pair_attention_mode`，取值 `causal`（缺省，行为与现在完全相同）、`bidirectional`、`prefix_block`。
- 不改：行的拼接与截断、readout、loss、LoRA、batch、FA3 守卫、禁止 SDPA 回退的规则。

## 三种模式

| 模式 | 谁能看谁 | 训练侧（FA3） | 推理侧（SDPA） |
|---|---|---|---|
| `causal` | 每个 token 只看自己和前面 | 不变 | 不变 |
| `bidirectional` | 一行里所有真实 token 互相可见 | 每层 `is_causal=False`，仍是每层一次内核调用 | 显式 4D mask `[B,1,1,L]`，只挡 padding |
| `prefix_block` | prefix 内部互看；candidate＋EOS 看整行；prefix 看不到 candidate | 每层两次无 mask 的 varlen 调用：prefix→prefix、candidate→整行 | 显式 4D mask `[B,1,L,L]` |

`prefix_block` 的边界现在放在 `Candidate:\n` 之后：prefix＝指令＋场景＋问题（＋可选标准），后一块＝候选＋EOS。Mammoth／d3-turbo 的边界在“场景”和“问题＋选项”之间。代码只认每行一个 `prefix_len`，要换成后者，把场景部分的 token 数传进去即可；训练侧的 token store 目前只存 `prefix`／`suffix` 两段，需要另外保存场景长度。

## Transformers 4.57.6 里实际起作用的开关

- flash 后端：是否 causal 只取决于每层的 `module.is_causal`（`Qwen3Attention`、`Qwen2Attention` 里写死为 `True`）。有 padding 的 batch 走 varlen 内核，没有 padding 的走 dense 内核。
- SDPA：只把 `is_causal` 设成 `False` 不够。有 padding 的 batch 仍然是 causal，没有 padding 的 batch 才变成双向，同一个模型在不同 batch 上 mask 不一致。所以 SDPA 一侧一律传显式的 4D bool mask（`True`＝可见）。
- `config.is_causal=False` 在 4.57.6 不起任何作用；vLLM 和 transformers `main` 分支会读它。
- 分类 head 取每行最右边的非 pad token（按 `input_ids` 判断），不受 mask 形状影响。
- 非重入的 gradient checkpointing 会在 `backward()` 里把各层重跑一遍，所以 `prefix_block` 的长度信息必须覆盖到 `backward()` 结束。

## 文件与放置位置

| 文件 | 集群位置 | 改动 |
|---|---|---|
| `attention_modes.py`（新增） | `ROOT/code/`（与 trainer 同目录）或 `HOME/scripts/` | 新模块，全文见下 |
| `pair_training_runtime.py` | `HOME/scripts/` | `assemble`／`collate` 可选返回边界；缺省返回值不变，其它 trainer 不受影响 |
| `train_five_dimension.py` | `ROOT/code/` | 6 处 |
| `decision_engine.py` | `ROOT/code/` | 8 处；仍是单文件，不 import 新模块（正式评测只冻结这一个文件和 requirements） |
| `config.json` | 实验目录 | 加 `"pair_attention_mode"` |

基准版本：`pair_training_runtime.py` 取自归档的 `cluster-followup-20261031/four-specialists-change/after/`，另两个取自 `five-dimension-distilled-20261109/source/`。三个基准文件的拷贝在 `code/baseline/`。应用前先核对集群上的文件与基准一致。

## attention_modes.py

```python
"""Attention-mask modes for the Qwen3 pair classifier: causal / bidirectional / prefix_block.

Row layout (unchanged): prefix tokens, candidate tokens, EOS, right padding.
  causal         stock behaviour, nothing here is used.
  bidirectional  every real token sees every real token (padding-only mask).
  prefix_block   prefix tokens see the prefix; candidate+EOS tokens see the whole row;
                 the prefix never sees the candidate (the Mammoth / d3-turbo layout).
Written against transformers 4.57.6 (Qwen3Attention / Qwen2Attention, AttentionInterface, flash wrapper).
"""
import contextlib
import contextvars

import torch

MODES = ('causal', 'bidirectional', 'prefix_block')
ATTENTION_LAYERS = ('Qwen3Attention', 'Qwen2Attention')    # Qwen3 and Qwen2.5: every layer is softmax attention
_LAYOUT = contextvars.ContextVar('pair_attention_layout', default=None)


def attention_layers(model):
    layers = [m for m in model.modules() if type(m).__name__ in ATTENTION_LAYERS]
    if len(layers) != model.config.num_hidden_layers:
        raise RuntimeError('Expected one attention layer per hidden layer')
    return layers


# ---- SDPA: an explicit 4D mask, passed as `attention_mask` -------------------------------------

def dense_mask(mode, row_len, width, prefix_len=None):
    """Bool mask, True = may attend. [B,1,1,L] for bidirectional, [B,1,L,L] for prefix_block.

    Padded query rows keep the real keys, so no row is fully masked (that would give NaN).
    """
    pos = torch.arange(width, device=row_len.device)
    key_ok = pos[None] < row_len[:, None]                          # real (non-padding) keys
    if mode == 'bidirectional':
        return key_ok[:, None, None, :]
    if mode == 'prefix_block':
        candidate = pos[None] >= prefix_len[:, None]               # candidate, EOS and padding
        allow = candidate[:, :, None] | ~candidate[:, None, :]     # query after the prefix, or key inside it
        return (allow & key_ok[:, None, :])[:, None]
    raise ValueError(f'No dense mask for attention mode {mode!r}')


# ---- Flash backends: no dense mask exists, the kernel takes a causal flag and lengths ----------

def enable_bidirectional(model):
    """Whole-row bidirectional on a flash backend: its wrapper reads `module.is_causal`."""
    for layer in attention_layers(model):
        layer.is_causal = False
    model.config.is_causal = False     # read by transformers v5 mask creation and by vLLM, not by 4.57.6


@contextlib.contextmanager
def attention_layout(prefix_len, row_len):
    """Per-row prefix length and total length for prefix_block.

    With gradient checkpointing the layers run again inside backward(), so keep this open over it.
    """
    token = _LAYOUT.set((prefix_len.to(torch.int32), row_len.to(torch.int32)))
    try:
        yield
    finally:
        _LAYOUT.reset(token)


def _cumulative(lengths):
    return torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))


def prefix_block_flash_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                               sliding_window=None, **kwargs):
    """prefix_block as two unmasked flash calls per layer: prefix -> prefix, candidate -> whole row."""
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    layout = _LAYOUT.get()
    if layout is None:
        raise RuntimeError('prefix_block attention needs attention_layout(...) around forward and backward')
    if sliding_window is not None:
        raise RuntimeError('prefix_block attention does not cover sliding-window layers')
    prefix_len, row_len = (lengths.to(query.device) for lengths in layout)
    candidate_len = row_len - prefix_len
    if len(row_len) != query.shape[0] or int(prefix_len.min()) < 1 or int(candidate_len.min()) < 1:
        raise RuntimeError('Every row needs a non-empty prefix and a non-empty candidate part')
    if query.dtype == torch.float32 and torch.is_autocast_enabled():   # same cast as the stock flash wrapper
        dtype = torch.get_autocast_gpu_dtype()
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    (_, varlen, _, _), flash_kwargs_for = lazy_import_flash_attention(module.config._attn_implementation)
    flash_kwargs = flash_kwargs_for(query_length=query.shape[2], key_length=key.shape[2], is_causal=False,
                                    dropout=dropout, softmax_scale=scaling)
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)      # [B, L, heads, D]
    pos = torch.arange(q.shape[1], device=q.device)[None]
    in_prefix = pos < prefix_len[:, None]
    in_row = pos < row_len[:, None]
    in_candidate = in_row & ~in_prefix

    def attend(queries, keys, query_len, key_len):
        out = varlen(q[queries], k[keys], v[keys],
                     cu_seqlens_q=_cumulative(query_len), cu_seqlens_k=_cumulative(key_len),
                     max_seqlen_q=int(query_len.max()), max_seqlen_k=int(key_len.max()), **flash_kwargs)
        return out[0] if isinstance(out, tuple) else out

    out = torch.zeros_like(q)
    out[in_prefix] = attend(in_prefix, in_prefix, prefix_len, prefix_len)
    out[in_candidate] = attend(in_candidate, in_row, candidate_len, row_len)
    return out, None


def install_prefix_block_flash(backend):
    """Point an already registered flash backend (e.g. 'flash_attn3_hopper') at the prefix_block forward."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(backend, prefix_block_flash_forward)
```

## pair_training_runtime.py

```diff
--- a/pair_training_runtime.py
+++ b/pair_training_runtime.py
@@ -91,24 +91,29 @@
     return np.concatenate((ids[:head],ids[-tail:] if tail else np.empty(0,dtype=np.int32)))
 
 
-def assemble(store,state,candidate,max_length,eos):
+def assemble(store,state,candidate,max_length,eos,boundary=False):
     prefix=store.get('prefix',int(state));suffix=store.get('suffix',int(candidate))
     prefix_reserve=min(len(prefix),min(1024,max_length//2))
     suffix=clip_tokens(suffix,max_length-1-prefix_reserve)
     prefix=clip_tokens(prefix,max_length-1-len(suffix))
-    return np.concatenate((prefix,suffix,np.array([eos],dtype=np.int32)))
+    ids=np.concatenate((prefix,suffix,np.array([eos],dtype=np.int32)))
+    return (ids,len(prefix)) if boundary else ids
 
 
-def collate(store,batch,pad,eos,device):
-    ids=[assemble(store,state,candidate,int(limit),eos)
-         for state,candidate,limit in zip(batch['state'],batch['candidate'],batch['max_length'])]
+def collate(store,batch,pad,eos,device,layout=False):
+    rows=[assemble(store,state,candidate,int(limit),eos,boundary=True)
+          for state,candidate,limit in zip(batch['state'],batch['candidate'],batch['max_length'])]
+    ids=[row for row,_ in rows]
     width=((max(len(row) for row in ids)+7)//8)*8
     input_ids=torch.full((len(ids),width),pad,dtype=torch.long,pin_memory=True)
     mask=torch.zeros((len(ids),width),dtype=torch.bool,pin_memory=True)
     for i,row in enumerate(ids):
         input_ids[i,:len(row)]=torch.from_numpy(row.astype(np.int64,copy=False));mask[i,:len(row)]=True
     targets=torch.from_numpy(np.asarray(batch['target'],dtype=np.float32).copy()).pin_memory()
-    return {'input_ids':input_ids.to(device,non_blocking=True),'attention_mask':mask.to(device,non_blocking=True)},targets.to(device,non_blocking=True),sum(map(len,ids))
+    result=({'input_ids':input_ids.to(device,non_blocking=True),'attention_mask':mask.to(device,non_blocking=True)},targets.to(device,non_blocking=True),sum(map(len,ids)))
+    if not layout:return result
+    lengths=lambda values:torch.tensor(values,dtype=torch.int32,device=device)
+    return (*result,(lengths([prefix for _,prefix in rows]),lengths([len(row) for row in ids])))
 
 
 def microbatches(batch,max_pairs,max_tokens):
```

## train_five_dimension.py

```diff
--- a/train_five_dimension.py
+++ b/train_five_dimension.py
@@ -25,6 +25,7 @@
     BACKEND, register_student_fa3, assert_student_uses_fa3, attention_record,
     restore_attention_record,
 )
+from attention_modes import MODES, attention_layout, enable_bidirectional, install_prefix_block_flash
 
 from pair_data_common import DOMAINS
 from pair_training_runtime import HOME, POOL, PairPool, TokenStore, collate, microbatches
@@ -128,6 +129,8 @@
     cfg=json.loads((HOME/os.environ.get('JEV_FRESH_DISTILL_CONFIG',str(EXP_ROOT/'config.json'))).read_text())
     if cfg.get('attention')!='flash_attention_3' or cfg.get('student_attention_implementation')!=BACKEND:
         raise RuntimeError('Student training requires explicit FlashAttention-3; SDPA fallback is forbidden')
+    attention_mode=cfg.get('pair_attention_mode','causal')
+    if attention_mode not in MODES:raise RuntimeError('Unknown pair_attention_mode')
     if cfg['student_parent']!=cfg['common_initialization'] or not 1250000<=cfg['pairs']<1260000:
         raise RuntimeError('Five-domain extension requires untrained initialization and admitted native-language exposure')
     if args.arm not in cfg['arms']:raise RuntimeError('Unknown distillation arm')
@@ -203,6 +206,8 @@
             raise RuntimeError('Checkpoint / pool / common Student parent mismatch; preserved')
         if state.get('exploration_policy')!=cfg['exploration_policy']:
             raise RuntimeError('Checkpoint belongs to another exploration policy; preserved')
+        if state.get('pair_attention_mode','causal')!=attention_mode:
+            raise RuntimeError('Checkpoint was trained with another attention mode; preserved')
     common=HOME/cfg['common_initialization'];base=HOME/cfg['base_model'];parent=HOME/cfg['student_parent']
     register_student_fa3(HOME/cfg['student_attention_kernel_directory'])
     model_config=AutoConfig.from_pretrained(common,local_files_only=True)
@@ -210,6 +215,9 @@
                                                             attn_implementation="flash_attn3_hopper",local_files_only=True)
     model=PeftModel.from_pretrained(model,checkpoint or parent,is_trainable=True)
     assert_student_uses_fa3(model)
+    if attention_mode=='bidirectional':enable_bidirectional(model)
+    elif attention_mode=='prefix_block':install_prefix_block_flash(BACKEND)
+    model.config.pair_attention_mode=attention_mode
     model.config.use_cache=False
     for parameter in model.parameters():
         if parameter.requires_grad:parameter.data=parameter.data.float()
@@ -284,7 +292,7 @@
                 folder.mkdir(parents=True,exist_ok=True)
                 model.save_pretrained(folder,safe_serialization=True);model.config.save_pretrained(folder)
                 payload=dict(model=args.model,student_parent=cfg['student_parent'],pool_pairs=len(dataset),step=step,pairs_seen=pairs_seen,tokens_done=tokens_done,
-                              exploration_policy=cfg['exploration_policy'],
+                              exploration_policy=cfg['exploration_policy'],pair_attention_mode=attention_mode,
                               sampler_identity=cfg['group_plan_identity'],seen_relations=seen_relations,
                              elapsed_seconds=elapsed,seen_domains=dict(seen_domains),
                               micro_tokens=micro_tokens,gradient_checkpointing=checkpointing,
@@ -349,12 +357,13 @@
                         print(json.dumps(dict(event='atomic_unit_over_micro_budget',step=step,
                             rows=len(micro['state']),padded_tokens=padded_width*len(micro['state']),
                             budget=micro_tokens)),flush=True)
-                    inputs,targets,_=collate(store,micro,pad,eos,device)
-                    with torch.autocast('cuda',dtype=torch.bfloat16):
-                        output=model(**inputs).logits
-                        margin=output[:,1].float()-output[:,0].float()
-                        loss=group_micro_loss(margin,micro,plan['teacher_relation_target'],cfg,device)
-                    (loss/len(indices)).backward();local_loss+=loss.detach().double()
+                    inputs,targets,_,layout=collate(store,micro,pad,eos,device,layout=True)
+                    with attention_layout(*layout):
+                        with torch.autocast('cuda',dtype=torch.bfloat16):
+                            output=model(**inputs).logits
+                            margin=output[:,1].float()-output[:,0].float()
+                            loss=group_micro_loss(margin,micro,plan['teacher_relation_target'],cfg,device)
+                        (loss/len(indices)).backward();local_loss+=loss.detach().double()
                     inputs=targets=output=loss=margin=q=kl=entropy=None
             except torch.cuda.OutOfMemoryError:
                 failed.fill_(1);inputs=targets=output=loss=margin=q=kl=entropy=None;gc.collect();torch.cuda.empty_cache()
```

`install_prefix_block_flash(BACKEND)` 必须在 `register_student_fa3(...)` 之后：后者每次调用都会把后端重新指回原版 forward。

## decision_engine.py

```diff
--- a/decision_engine.py
+++ b/decision_engine.py
@@ -29,6 +29,14 @@
     return '\n'.join(f'{label}: {option}' for label,option in zip(labels,options))
 
 
+def dense_mask(mode,row_len,width,prefix_len):
+    # True = may attend. Padded query rows keep the real keys, so no row is fully masked.
+    pos=torch.arange(width,device=row_len.device);key_ok=pos[None]<row_len[:,None]
+    if mode=='bidirectional':return key_ok[:,None,None,:]
+    candidate=pos[None]>=prefix_len[:,None]
+    return ((candidate[:,:,None]|~candidate[:,None,:])&key_ok[:,None,:])[:,None]
+
+
 class PairClassifierEngine(Engine):
     name='qwen3-0.6b-five-dimension-distilled'
     latency='Single-process in-request rendering, tokenization and independent candidate scoring; excludes model load.'
@@ -44,6 +52,9 @@
         torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
         self.tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
         config=AutoConfig.from_pretrained(model,trust_remote_code=False)
+        self.attention=getattr(config,'pair_attention_mode','causal')
+        if self.attention not in ('causal','bidirectional','prefix_block'):
+            raise ValueError('Unknown pair_attention_mode in saved config')
         base=AutoModelForSequenceClassification.from_pretrained(base_model,config=config,
             torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False)
         self.model=PeftModel.from_pretrained(base,model,is_trainable=False).to(self.device).eval()
@@ -55,7 +66,7 @@
             'base_model':base_model,'checkpoint':model,'rendering':'candidate-pair-v1',
             'readout':'softmax of logit(match)-logit(nonmatch) over every original field criterion',
             'truncation':False,'option_filtering':False,'context_tokens_per_candidate':self.context,
-            'inference_attention':'sdpa','dtype':'bf16 base, saved PEFT head/adapter precision retained',
+            'inference_attention':'sdpa','attention_mask':self.attention,'dtype':'bf16 base, saved PEFT head/adapter precision retained',
             'teacher_used_for_inference':False}
 
     @lru_cache(maxsize=2048)
@@ -69,8 +80,9 @@
     def synchronize(self):torch.cuda.synchronize(self.device)
 
     @torch.inference_mode()
-    def score(self,rows):
+    def score(self,rows,prefixes):
         lengths=np.asarray([len(row) for row in rows],dtype=np.int64)
+        prefixes=np.asarray(prefixes,dtype=np.int64)
         if np.any(lengths>self.context):
             raise Unsupported('Whole candidate input exceeds model context window; nothing truncated')
         order=np.argsort(lengths,kind='stable');scores=np.empty(len(rows),dtype=np.float64)
@@ -89,6 +101,9 @@
                 row=rows[int(index)];ids[i,:len(row)]=torch.tensor(row,dtype=torch.long);mask[i,:len(row)]=True
             oom=False;output=None
             try:
+                if self.attention!='causal':
+                    mask=dense_mask(self.attention,torch.from_numpy(lengths[chosen]).to(self.device),width,
+                                    torch.from_numpy(prefixes[chosen]).to(self.device))
                 with torch.autocast('cuda',dtype=torch.bfloat16):
                     output=self.model(input_ids=ids.to(self.device),attention_mask=mask.to(self.device)).logits
                 values=(output[:,1].float()-output[:,0].float()).cpu().numpy()
@@ -106,7 +121,7 @@
         return scores,int(lengths.sum())
 
     def __call__(self,state,questions):
-        rows=[];fields=[];state=text(state)
+        rows=[];prefixes=[];fields=[];state=text(state)
         for field,question in questions.items():
             kind=question['type']
             if kind not in ('choice','noul'):raise Unsupported('Unsupported question type '+str(kind))
@@ -122,10 +137,10 @@
             prefix+='\nCandidate:\n'
             prefix_ids=self.tokens(prefix)
             begin=len(rows)
-            for value in values:rows.append(prefix_ids+self.tokens(value)+(self.eos,))
+            for value in values:rows.append(prefix_ids+self.tokens(value)+(self.eos,));prefixes.append(len(prefix_ids))
             fields.append((field,kind,keys,begin,len(rows)))
         if not rows:raise ValueError('An inference request must contain at least one field')
-        margins,count=self.score(rows);answers={}
+        margins,count=self.score(rows,prefixes);answers={}
         for field,kind,keys,begin,end in fields:
             scores=margins[begin:end];probabilities=np.exp(scores-scores.max());probabilities/=probabilities.sum()
             distribution={key:float(value) for key,value in zip(keys,probabilities)}
```

引擎从 checkpoint 目录里保存的 config 读 `pair_attention_mode`，不需要新的命令行选项；`run_official.py` 不用改。模块里的 `dense_mask` 是 SDPA 一侧的参考实现，供检查脚本使用；引擎必须是单文件，所以自带一份等价的拷贝，检查脚本对比过这两份。

## 配置

```json
"pair_attention_mode": "prefix_block"
```

用新的实验目录或新的 arm。已有的 causal checkpoint 不能换模式续训：模式写进了 checkpoint 记录和保存的 config，不一致会直接报错。训练和推理的 mask 必须一致，引擎按保存的 config 自动对齐。

## 清单

- [x] 本机 CPU：三种模式下 flash 路径与 SDPA 显式 mask 的输出、梯度一致（Qwen3、Qwen2.5 各 31 项）。
- [x] 本机 CPU：打过补丁的 `collate`／`assemble` 缺省返回值与原版相同；打过补丁的引擎三种模式下批量打分与逐行参考一致（20 项）。
- [ ] 真实 FA3 Hopper 内核上的 `prefix_block`（见下）。
- [ ] 应用到集群。
- [ ] 训练与评测。

## 已验证／未验证

已验证，全部在本机 CPU 上，Torch 2.9.1、Transformers 4.57.6、PEFT 0.18.1，float64 的随机小模型，flash 内核是按 FA3 接口写的仿真实现：

- `bidirectional`：flash 路径（改 `is_causal`）与 SDPA 显式 mask 一致；有 padding 走 varlen，没有 padding 走 dense。
- `prefix_block`：两次调用的实现与 SDPA `[B,1,L,L]` mask 一致；每层正好两次 varlen 调用；缺少长度信息或行数对不上会报错。
- `prefix_block` 下 prefix 的 hidden state 与候选无关；`bidirectional` 下有关。
- 梯度一致；非重入 gradient checkpointing 下可用，长度信息离开 `backward()` 就失效会报错。
- 穿过 PEFT LoRA 包装后仍然生效；`pair_attention_mode` 能随 config 保存和读回。
- 引擎：三种模式、batch 3 和 32，与逐行无 padding 的参考结果一致（差 < 1e-6）；三种模式的分数确实不同。

未验证：

- **真实的 FA3 Hopper 内核一次都没跑过。** `prefix_block` 依赖 `flash_attn_varlen_func` 在 `causal=False`、`cu_seqlens_q` 与 `cu_seqlens_k` 不相同的情况下前向和反向都正确，并且在 bf16 autocast、`FLASH_ATTENTION_DETERMINISTIC=1`、适配器的计数包装之下都正常。如果这里有偏差，训练不会报错，但训练用的 mask 会与推理用的 SDPA mask 不一致。能把这一点坐实的检查是：同一批数据、同一份权重，比较 FA3 `prefix_block` 与 SDPA 显式 mask 的 logits。`bidirectional` 只用到原版 wrapper 已有的路径，风险小得多。
- `train_five_dimension.py` 的补丁只做了语法检查；它调用的每个部件由上面的仿真检查间接覆盖，整个脚本没有运行过。
- 真实权重、真实数据、训练效果、速度、显存都没有测。

## 代价与注意

- 底座是 causal 预训练的，放开 mask 后权重需要重新适应。Qwen3-0.6B 的 28 层全是 softmax attention，一改就是全部层；Qwen3.5 系列只改其中的 full-attention 层。前者的分布偏移应当更大（推断，没有实验）。
- `prefix_block` 训练时每层两次内核调用；适配器的 `fa3_forward_calls` 会按每层两次累加，没有代码依赖这个计数。
- 现在的引擎不在候选之间复用 prefix 的计算，所以 `prefix_block` 目前没有速度收益。它保留的是“prefix 表示与候选无关”这个性质，以后做 prefix 复用时才用得上；`bidirectional` 没有这个性质。
- 推理侧 `prefix_block` 的 mask 是 `B×L×L` 的 bool 张量；现有的 OOM 减半重试照常适用。
- `enable_bidirectional` 会把 `is_causal: false` 写进保存的 config。4.57.6 忽略它；vLLM 会据此把整段输入当双向处理，这与 `bidirectional` 一致，与 `prefix_block` 不一致。
- transformers `main` 分支的 `create_causal_mask` 已经读 `config.is_causal`，并且支持用 `block_sequence_ids` 在 causal 之上叠加块内双向。升级之后可以改走上游接口；4.57.6 没有这些。这一条来自 2026-11-10 对 `main` 源码的一次阅读，升级前需要重新核对。

## 换底座

- Qwen2.5（`Qwen2Attention`）：同一个模块直接可用，本机检查同样通过。`assert_student_uses_fa3` 里写死了 `Qwen3Attention` 和 `head_dim==128`，换底座要改这个守卫。
- Qwen3.5／3.6／3.8：混合架构，只有 `full_attention` 层能改 mask，Gated DeltaNet 层保持 causal。模块里的 `ATTENTION_LAYERS` 要加上对应的类名，并且“每层一个 attention”的检查不再成立。本机的 Transformers 4.57.6 没有这些模型，没有验证；榜单上的做法见 [mammoth-d3-findings.md](../2026-11-09-attention-audit/mammoth-d3-findings.md)。

## 不改 mask 的替代做法

- 把输入读两遍（KnowLine）：第二遍的场景能用到第一遍的问题，mask 仍是 causal，输入长度翻倍。
- causal 骨干后面加一个全局 attention head（clef-flash、clef-omni、Nari Decision 4B）。
- 只换 readout（Kev、Sieve 的 pointer）：与 mask 无关。

## 代码文件与复跑

用户要求（2026-11-10）：代码放在仓库里，不放库外辅助归档。文件在本目录的 `code/` 下：

| 路径 | 内容 |
|---|---|
| [code/attention_modes.py](code/attention_modes.py) | 新模块，与上文全文相同 |
| `code/patched/` | 打过补丁的三个文件；集群文件与基准一致时可以直接替换同名文件 |
| `code/baseline/` | 三个基准文件，与库外归档里的来源逐字节相同，用来和集群文件比对 |
| `code/checks/` | `check_attention_modes.py`、`check_patched.py`、`make_patched.py`，以及引擎检查用的 `decision_index` 桩 |

复跑只用 CPU，不读权重和数据。在本目录下执行：

```bash
python -B code/checks/check_attention_modes.py qwen3
```

```bash
python -B code/checks/check_attention_modes.py qwen2
```

```bash
python -B code/checks/check_patched.py
```

预期输出分别以 `31/31 checks passed`、`31/31 checks passed`、`20/20 checks passed` 结尾。`make_patched.py` 从基准重新生成补丁文件，任何一处找不到唯一匹配就报错退出：

```bash
mkdir -p /tmp/attn-regen && python -B code/checks/make_patched.py code/baseline /tmp/attn-regen && diff -r /tmp/attn-regen code/patched
```

检查时的环境是 uv 建的临时虚拟环境，不在仓库里：Python 3.12.13、Torch 2.9.1、Transformers 4.57.6、PEFT 0.18.1、NumPy 2.5.4。

仓库的 `.gitignore` 按资料库规则忽略所有 `*.py`。用户要求（2026-11-12）为本目录加例外：`.gitignore` 末尾新增一条规则，只放行本目录 `code/` 下的 `*.py`，其它位置的规则不变。这 13 个文件现在可以被 Git 跟踪。尚未提交；本目录的索引 README 和[整理记录](../../../整理记录.md)里“脚本进入库外辅助归档”的说明没有更新。
