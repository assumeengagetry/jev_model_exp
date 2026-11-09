#!/usr/bin/env python3
"""Reuse the native resumable Teacher implementation on unique language pairs."""
import importlib.util
import json
from pathlib import Path
import sys

HOME=Path('/home/dev/assumeengage')
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HOME/'scripts'))
CFG=json.loads((ROOT/'language-config.json').read_text())
spec=importlib.util.spec_from_file_location('native_language_labeler',HOME/'scripts/label-knowline-pairs.py')
LABEL=importlib.util.module_from_spec(spec);spec.loader.exec_module(LABEL)
LABEL.DOMAIN_ORDER=['language'];LABEL.DOMAIN_PAIRS=dict(CFG['selected_domain_pairs']);LABEL.TOTAL_PAIRS=CFG['pairs']


def load_configuration(workers):
    cfg=json.loads((ROOT/'language-config.json').read_text())
    for actual,expected,name in [
        (cfg['run_root'],str(ROOT/'language-labeling'),'namespace'),
        (cfg['active_training_domains'],['language'],'domains'),
        (cfg['teacher'],'PelaAI/KnowLine-31B-Gen1','Teacher'),
        (cfg['teacher_workers'],workers,'worker count'),
        (cfg['student_parent'],cfg['common_initialization'],'untrained initialization'),
        (cfg['teacher_read_twice'],True,'native read-twice'),
        (cfg['teacher_twice_max_tokens'],3000,'native cutoff'),
        (cfg['teacher_dtype'],'bfloat16','native dtype')]:
        LABEL.require_equal(actual,expected,name)
    if not 1<=cfg['teacher_max_pairs']<=16 or not 1<=cfg['teacher_max_padded_tokens']<=8192:
        raise RuntimeError('Teacher batching differs from approved native settings')
    pool=LABEL.PairPool('MIXED_ALL',root=Path(cfg['training_pool']),domains=['language'])
    LABEL.require_equal(len(pool),cfg['pairs'],'unique language pairs')
    LABEL.require_equal(pool.ready['effective_student_tokens'],cfg['effective_student_tokens'],'token exposure')
    root=Path(cfg['run_root']);labels=root/'teacher-labels';labels.mkdir(exist_ok=True)
    return cfg,pool,root,labels


original_identity=LABEL.common_identity


def common_identity(cfg,pool,workers):
    identity=original_identity(cfg,pool,workers)
    identity.update(producer='E24-five-dimension-label_language.py',
        selection_metadata=str(ROOT/'language-pool/SELECTION.json'),
        global_source_rows=str(ROOT/'language-pool/language-global-source-rows.npy'),
        language_only=True,old_four_teacher_cache_untouched=True)
    return identity


LABEL.load_configuration=load_configuration;LABEL.common_identity=common_identity
if __name__=='__main__':LABEL.main()
