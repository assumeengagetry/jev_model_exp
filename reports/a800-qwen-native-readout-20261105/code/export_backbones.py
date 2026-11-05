#!/usr/bin/env python3
"""Inference preparation only: retain original LMhead, restore completed classifier backbones."""
import gc
import json
from pathlib import Path
import torch
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import PeftModel
from kd_runtime import ROOT,SOURCE_ROOT,WORKSPACE,atomic_json


def main():
    marker=ROOT/'native-model-paths.json'
    if marker.exists():return
    torch.set_num_threads(4)
    base=WORKSPACE/'models/Qwen3-0.6B';common=WORKSPACE/'models/Qwen3-0.6B-common-pair-init-r32'
    paths={'QWEN_RAW_LM':str(base)}
    adapters={
      'QWEN_HARD_LM':SOURCE_ROOT/'experiments/kd-raw-qwen-1m-20261103/outputs/KD_HARD/final',
      'QWEN_LS_LM':SOURCE_ROOT/'experiments/kd-improve-round1-20261104/outputs/KD_LS10/final',
    }
    for name,adapter in adapters.items():
        config=AutoConfig.from_pretrained(common,local_files_only=True)
        sequence=AutoModelForSequenceClassification.from_pretrained(base,config=config,torch_dtype=torch.float32,
                                                                  attn_implementation='sdpa',local_files_only=True)
        sequence=PeftModel.from_pretrained(sequence,adapter,is_trainable=False,local_files_only=True).merge_and_unload()
        language=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float32,
                                                    attn_implementation='sdpa',local_files_only=True)
        language.model.load_state_dict(sequence.model.state_dict(),strict=True)
        language.to(dtype=torch.bfloat16)
        language.config.architectures=['Qwen3ForCausalLM'];language.config.use_cache=True
        folder=ROOT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
        language.save_pretrained(folder,safe_serialization=True)
        AutoTokenizer.from_pretrained(base,local_files_only=True).save_pretrained(folder)
        paths[name]=str(folder)
        print(json.dumps(dict(event='native_readout_backbone_exported',name=name,source_adapter=str(adapter),
              original_lm_head_retained=True,classification_head_used=False,backbone_keys_strict=True,
              new_training_steps=0,checkpoint=str(folder))),flush=True)
        del sequence,language;gc.collect()
    atomic_json(marker,paths)


if __name__=='__main__':main()
