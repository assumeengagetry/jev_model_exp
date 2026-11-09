#!/usr/bin/env python3
"""Freeze the approved E24-plus-language pool; no model inference or tests."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

HOME = Path('/home/dev/assumeengage')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOME / 'scripts'))
from pair_training_runtime import PairPool
from group_training_runtime import build_plan

OLD = HOME / 'runs/distillation-group-20261106/E24_GROUP_MIXED_RANK'
FOUR = ['knowledge', 'retrieval', 'tools', 'arts']
DOMAINS = FOUR + ['language']
KEYS = ['state', 'candidate', 'target', 'source', 'group', 'length']
NATIVE = ['anli', 'hellaswag', 'acos_laptop-acos', 'acos_restaurant-acos',
          'ragtruth', 'contractnli', 'nli4ct', 'vast', 'isarcasm']


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def preserve(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError('Existing identity differs; preserved: ' + str(path))
    else:
        atomic(path, value)


def array(path, value):
    path = Path(path)
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), value):
            raise RuntimeError('Existing array differs; preserved: ' + str(path))
        return
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def metadata(path):
    stat = Path(path).stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def indexed_rows(path, ids, boundary=0):
    """Read only requested dictionary rows; global input IDs span two files."""
    ids=np.asarray(sorted(set(map(int,ids))),dtype=np.int64)
    result={};start=boundary
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        stop=start+batch.num_rows;lo,hi=np.searchsorted(ids,[start,stop])
        chosen=ids[lo:hi]
        for index,item in zip(chosen,batch.take(chosen-start).to_pylist()):result[int(index)]=item
        start=stop
    return result


def native_anli_groups(pool, source_id):
    """Repair legacy round-colliding group IDs using original train file + row."""
    table=pool.tables['language'];rows=np.flatnonzero(table['source']==source_id)
    order=rows[np.argsort(table['state'][rows],kind='stable')]
    states,begins,counts=np.unique(table['state'][order],return_index=True,return_counts=True)
    manifest=read(pool.root/'MANIFEST.json');boundary=manifest['typed_prefix_boundary']
    inputs=indexed_rows(Path(manifest['typed_normalized'])/'inputs.parquet',states[states<boundary])
    inputs.update(indexed_rows(pool.root/'inputs.parquet',states[states>=boundary],boundary))
    candidates=np.unique(table['candidate'][rows]);cb=manifest['typed_suffix_boundary']
    values=indexed_rows(Path(manifest['typed_normalized'])/'candidates.parquet',candidates[candidates<cb])
    values.update(indexed_rows(pool.root/'candidates.parquet',candidates[candidates>=cb],cb))
    raw={};files=[];raw_rounds=collections.Counter()
    relation={'e':'entailment','n':'neutral','c':'contradiction'}
    paths=sorted((HOME/'datasets/benchmarks-train/language/anli/plain_text').glob('train_*.parquet'))
    if len(paths)!=3:raise RuntimeError('Exactly three native ANLI train-round files are required')
    for path in paths:
        files.append(metadata(path));pf=pq.ParquetFile(path)
        features=json.loads(pf.schema_arrow.metadata[b'huggingface'])['info']['features']
        labels=features['label'].get('names') or ['entailment','neutral','contradiction']
        labels=[relation.get(name,name) for name in labels]
        for i,item in enumerate(pf.read(columns=['premise','hypothesis','label']).to_pylist()):
            state=json.dumps({'premise':item['premise'],'hypothesis':item['hypothesis']},ensure_ascii=False,sort_keys=True)
            origin={'file':str(path),'row_index':i,'round':path.name.split('-')[0],
                    'label':labels[int(item['label'])]}
            raw.setdefault(state,[]).append(origin);raw_rounds[origin['round']]+=1
    admitted=[];rejected=collections.Counter();rounds=collections.Counter()
    for state,begin,count in zip(states,begins,counts):
        indices=order[int(begin):int(begin+count)];entry=inputs.get(int(state),{})
        origins=raw.get(entry.get('state'),[])
        if (len(origins)!=1 or entry.get('question')!='What is the relation of the hypothesis to the premise?'
                or entry.get('criteria')!=''):
            rejected['ambiguous_or_unmapped_native_question']+=1;continue
        origin=origins[0]
        names=[values[int(candidate)]['candidate'] for candidate in table['candidate'][indices]]
        targets=np.asarray(table['target'][indices])
        expected=np.asarray([float(name==origin['label']) for name in names])
        if len(indices)!=3 or set(names)!={'entailment','neutral','contradiction'} or not np.array_equal(targets,expected):
            rejected['incomplete_or_changed_native_candidates']+=1;continue
        gid=2**31+int(state)
        if gid>np.iinfo(np.uint32).max:raise RuntimeError('Native question ID exceeds cache dtype')
        admitted.append((gid,indices,origin));rounds[origin['round']]+=1
    return admitted,files,{'raw_train_questions_by_round':dict(raw_rounds),
        'intact_admitted_questions_by_round':dict(rounds),'rejected':dict(rejected),
        'identity':'native train filename + row index, exact premise/hypothesis and complete labeled criteria',
        'cache_group_id':'2**31 + state_id; independent of legacy round-colliding group IDs'}


def grouped_rows(pool):
    table = pool.tables['language']
    sources = read(pool.root / 'SOURCES.json')['sources']
    source_ids = {}
    for name in NATIVE:
        ids = [i for i, item in enumerate(sources) if item['name'] == 'native/' + name]
        if len(ids) != 1:
            raise RuntimeError('Missing or ambiguous native source: ' + name)
        source_ids[name] = ids[0]
    rows = np.flatnonzero(np.isin(table['source'], list(source_ids.values())))
    order = rows[np.argsort(table['group'][rows], kind='stable')]
    gids, begins, counts = np.unique(table['group'][order], return_index=True, return_counts=True)
    decoded = {}
    start = 0
    for batch in pq.ParquetFile(pool.root / 'groups.parquet').iter_batches(batch_size=65536):
        stop = start + batch.num_rows
        lo, hi = np.searchsorted(gids, [start, stop])
        selected = gids[lo:hi]
        for gid, item in zip(selected, batch.take((selected - start).astype(np.int64)).to_pylist()):
            decoded[int(gid)] = item
        start = stop
    if len(decoded) != len(gids):
        raise RuntimeError('Incomplete global group provenance')
    by_source = {name: [] for name in NATIVE}
    raw_files = []
    expected = {}
    for name in NATIVE:
        path = HOME / 'cache/pair-training/extra-pairs/language' / (name + '.parquet')
        raw_files.append(metadata(path))
        group_counts = collections.Counter()
        positives = collections.Counter()
        for batch in pq.ParquetFile(path).iter_batches(columns=['group_id', 'label']):
            for item in batch.to_pylist():
                group_counts[item['group_id']] += 1
                positives[item['group_id']] += float(item['label'])
        expected[name] = (group_counts, positives)
    english = HOME / 'datasets/benchmarks-train/language/isarcasm/train/train.En.csv'
    raw_files.append(metadata(english))
    sarcasm_classes = {}
    with english.open(encoding='utf-8-sig', newline='') as stream:
        for i, item in enumerate(csv.DictReader(stream)):
            value = str(item.get('sarcastic', '')).strip()
            if value in ('0', '1'):
                sarcasm_classes[f'isarcasm:train.En.csv:{i}:sarcastic'] = int(value)
    if not sarcasm_classes:
        raise RuntimeError('English native sarcasm-class metadata is unavailable')
    reverse = {v: k for k, v in source_ids.items()}
    dropped = collections.Counter()
    for gid, begin, count in zip(gids, begins, counts):
        indices = order[int(begin):int(begin + count)]
        name = reverse.get(int(table['source'][indices[0]]))
        item = decoded[int(gid)]
        original = item['original_group_id']
        if item['source_id'] != source_ids[name] or not np.all(table['source'][indices] == source_ids[name]):
            raise RuntimeError('Native group source identity mismatch')
        if name=='anli':
            continue  # Restore real round/question identity below, never trust legacy group IDs.
        if name == 'isarcasm' and original not in sarcasm_classes:
            dropped['sarcasm_non_english_or_non_core'] += 1
            continue
        expected_count = expected[name][0].get(original)
        if expected_count != len(indices):
            dropped[name + '_incomplete_after_global_dedup'] += 1
            continue
        target = np.asarray(table['target'][indices])
        if not np.isin(target, [0, 1]).all() or float(target.sum()) != expected[name][1][original]:
            dropped[name + '_changed_labels_after_global_dedup'] += 1
            continue
        by_source[name].append((int(gid), indices, sarcasm_classes.get(original)))
    by_source['anli'],anli_files,anli_identity=native_anli_groups(pool,source_ids['anli'])
    raw_files.extend(anli_files)
    if any(not groups for groups in by_source.values()):
        raise RuntimeError('A required native task has no admitted intact training groups')
    return by_source, raw_files, dict(dropped),anli_identity


def allocate(groups, total):
    weights = {name: math.sqrt(len(items)) for name, items in groups.items()}
    capacities = {name: sum(len(item[1]) for item in items) for name, items in groups.items()}
    if sum(capacities.values()) < total:
        raise RuntimeError('Insufficient intact native inventory; no silent replacement')
    pending = set(groups)
    quotas = {}
    remaining = total
    while pending:
        denominator = sum(weights[name] for name in sorted(pending))
        capped = [name for name in sorted(pending)
                  if capacities[name] <= remaining * weights[name] / denominator]
        if capped:
            for name in capped:
                quotas[name] = capacities[name]
                remaining -= quotas[name]
                pending.remove(name)
            continue
        fractions = {name: remaining * weights[name] / denominator for name in sorted(pending)}
        for name, value in fractions.items():
            quotas[name] = int(value)
        left = remaining - sum(quotas[name] for name in pending)
        for name in sorted(pending, key=lambda x: (-(fractions[x] - int(fractions[x])), x))[:left]:
            quotas[name] += 1
        break
    return quotas


def select_language(pool):
    groups, raw_files, dropped,anli_identity = grouped_rows(pool)
    rng = np.random.default_rng(42)
    selected = []
    report = {}
    sarcasm = groups.pop('isarcasm')
    quotas = allocate(groups, 237500)
    for name, items in groups.items():
        chosen = []
        seen = 0
        for i in rng.permutation(len(items)):
            if seen >= quotas[name]:
                break
            item = items[int(i)]
            chosen.append(item)
            seen += len(item[1])
        selected.extend(chosen)
        report[name] = {'quota_pairs': quotas[name], 'pairs': seen, 'unique_groups': len(chosen),
                        'available_groups': len(items), 'replacement': False}
        if name=='anli':
            report[name]['selected_questions_by_round']=dict(collections.Counter(item[2]['round'] for item in chosen))
    for original_class in (0, 1):
        items = [item for item in sarcasm if item[2] == original_class]
        if not items:
            raise RuntimeError('A native English sarcasm class is empty')
        seen = 0
        chosen = []
        while seen < 6250:
            for i in rng.permutation(len(items)):
                if seen >= 6250:
                    break
                item = items[int(i)]
                if len(item[1]) != 2:
                    raise RuntimeError('Core native sarcasm must retain its complete No/Yes pair')
                chosen.append(item)
                seen += len(item[1])
        selected.extend(chosen)
        report['isarcasm_native_class_' + str(original_class)] = {
            'quota_pairs': 6250, 'pairs': seen, 'group_exposures': len(chosen),
            'unique_groups': len({item[0] for item in chosen}), 'available_groups': len(items),
            'replacement': len(chosen) > len({item[0] for item in chosen})}
    indices = np.concatenate([item[1] for item in selected]).astype(np.int64)
    return indices, report, raw_files, dropped,anli_identity


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'DATA_READY.json').exists():
        ready = read(ROOT / 'DATA_READY.json')
        if ready.get('complete') is not True or ready.get('run_root') != str(ROOT):
            raise RuntimeError('Existing data readiness is not this experiment')
        print(json.dumps({'event': 'frozen_data_reused', **ready}), flush=True)
        return
    preserve(ROOT / 'ADMISSION.json', {'run_root': str(ROOT), 'seed': 42, 'old_four_pairs': 1000000,
        'language_pair_target': 250000, 'language_teacher_fraction': 0.5,
        'user_approved_native_execution': True, 'official_public_run_authorized': True,
        'extra_tests': False, 'extra_hashes': False, 'automatic_publish': False})
    cfg = read(OLD / 'config.json')
    old = PairPool('MIXED_ALL', root=HOME / cfg['training_pool'], domains=FOUR)
    if len(old) != 1000000:
        raise RuntimeError('The preserved E24 four-domain pool differs')
    source = PairPool('MIXED_ALL', domains=['language'])
    indices, sampling, raw_files, dropped,anli_identity = select_language(source)
    unique, inverse = np.unique(indices, return_inverse=True)
    pool = ROOT / 'pool'
    teacher_pool = ROOT / 'language-pool'
    for folder in [pool, teacher_pool]:
        (folder / 'pairs').mkdir(parents=True, exist_ok=True)
    summaries = {}
    for domain in FOUR:
        summaries[domain] = dict(old.ready['domains'][domain])
        if summaries[domain]['pairs'] != 250000:
            raise RuntimeError('Old domain exposure differs')
        for key in KEYS:
            target = pool / 'pairs' / f'{domain}_{key}.npy'
            original = old.root / 'pairs' / target.name
            if target.is_symlink():
                if target.resolve() != original.resolve():
                    raise RuntimeError('Foreign pool link preserved')
            elif target.exists():
                raise RuntimeError('Foreign pool file preserved')
            else:
                target.symlink_to(original)
    for folder, rows in [(pool, indices), (teacher_pool, unique)]:
        for key in KEYS:
            value=np.asarray(source.tables['language'][key][rows]).copy()
            if key=='group':
                source_info=read(source.root/'SOURCES.json')['sources']
                anli_id=next(i for i,item in enumerate(source_info) if item['name']=='native/anli')
                mask=np.asarray(source.tables['language']['source'][rows])==anli_id
                value[mask]=2**31+np.asarray(source.tables['language']['state'][rows[mask]],dtype=np.uint32)
            array(folder / 'pairs' / f'language_{key}.npy',value)
    lengths = np.asarray(source.tables['language']['length'][indices], dtype=np.uint64)
    summaries['language'] = {'pairs': len(indices), 'max_length': 4096,
        'effective_training_tokens': int(lengths.sum()), 'unique_pairs': len(unique)}
    tokens = sum(item['effective_training_tokens'] for item in summaries.values())
    total = 1000000 + len(indices)
    selection = {'complete': True, 'run_root': str(ROOT), 'selected_pairs': total,
        'selection_seed': 42, 'domains': summaries, 'active_training_domains': DOMAINS,
        'effective_student_tokens': tokens, 'student_parent': cfg['student_parent'],
        'source_index_order': 'unchanged K/R/Tools/Arts million, then selected language multiset',
        'language_sampling': sampling, 'language_source_files': raw_files, 'dropped_groups': dropped,
        'native_anli_identity':anli_identity,
        'old_four_pool': str(old.root), 'old_teacher': metadata(HOME / cfg['teacher_label_cache_source'])}
    array(pool / 'language-global-source-rows.npy', indices)
    array(teacher_pool / 'language-global-source-rows.npy', unique)
    array(ROOT / 'language-teacher-index.npy', inverse.astype(np.int64))
    preserve(pool / 'TRAINING_READY.json', selection)
    preserve(pool / 'SELECTION.json', selection)
    teacher_selection = {**selection, 'selected_pairs': len(unique), 'active_training_domains': ['language'],
        'domains': {'language': {'pairs': len(unique), 'max_length': 4096,
            'effective_training_tokens': int(np.asarray(source.tables['language']['length'][unique],dtype=np.uint64).sum())}}}
    teacher_selection['effective_student_tokens'] = teacher_selection['domains']['language']['effective_training_tokens']
    preserve(teacher_pool / 'TRAINING_READY.json', teacher_selection)
    preserve(teacher_pool / 'SELECTION.json', teacher_selection)
    policy = {'method': 'E24_five_dimension_extension', 'base_recipe': cfg['exploration_policy']['base_recipe'],
        'language_teacher_fraction': 0.5, 'language_sampling': 'native-9task-sqrt-capped-core-English-SAR5-pair-v1',
        'relation_target': 'mixed', 'relation_lambda': 0.05, 'relative_teacher_fraction': 0.5}
    cfg.update(run_root=str(ROOT), training_pool=str(pool), active_training_domains=DOMAINS,
        pairs=total, pair_cap=total, milestone_pairs=total, authorized_milestones=[total],
        selected_domain_pairs={d: summaries[d]['pairs'] for d in DOMAINS}, effective_student_tokens=tokens,
        exploration_policy=policy, arms=['KD_HYBRID'], optimizer_steps=math.ceil(total/128),
        effective_target_path=str(ROOT/'effective-targets.npy'),
        effective_target_ready_path=str(ROOT/'DISTILLATION_TARGETS_READY.json'),
        group_plan_root=str(ROOT/'plan'), relation_count=3647, relative_teacher_fraction=0.5,
        checkpoint_storage_root=str(ROOT/'checkpoints'),
        numerical_controls={'torch_deterministic_algorithms':True,'warn_only':False,'fa3_backward_deterministic':True},
        official_public_run_authorized=True, automatic_publish=False)
    cfg['group_plan_identity'] = {'version':'E24-native-language-extension-v1','seed':42,'pairs':total,
        'relations':3647,'optimizer_steps':cfg['optimizer_steps'],'global_batch_pairs':128}
    cfg['checkpoint_storage_identity'] = {'run_root':str(ROOT),'arm':'KD_HYBRID','sampler_identity':cfg['group_plan_identity']}
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    preserve(ROOT/'STORAGE_OWNER.json',cfg['checkpoint_storage_identity'])
    preserve(ROOT/'config.json',cfg)
    label_cfg={**cfg,'run_root':str(ROOT/'language-labeling'),'training_pool':str(teacher_pool),
        'pairs':len(unique),'active_training_domains':['language'],'selected_domain_pairs':{'language':len(unique)},
        'effective_student_tokens':teacher_selection['effective_student_tokens'],'teacher_workers':3}
    (ROOT/'language-labeling').mkdir(exist_ok=True)
    preserve(ROOT/'language-config.json',label_cfg)
    preserve(ROOT/'DATA_READY.json',selection)
    print(json.dumps({'event':'five_dimension_pool_frozen','pairs':total,'effective_tokens':tokens,
        'language_pairs':len(indices),'new_teacher_pairs':len(unique),'sampling':sampling,
        'native_anli_identity':anli_identity,'dropped_groups':dropped}),flush=True)


def targets():
    cfg=read(ROOT/'config.json')
    label_ready=read(ROOT/'language-labeling/TEACHER_LABELS_READY.json')
    if label_ready.get('complete') is not True:
        raise RuntimeError('Native language Teacher labeling is not complete')
    old_cfg=read(OLD/'config.json')
    old_targets=np.load(HOME/old_cfg['effective_target_path'])
    old_teacher=np.load(HOME/old_cfg['teacher_label_cache_source'])
    pool=PairPool('MIXED_ALL',root=ROOT/'pool',domains=DOMAINS)
    mapping=np.load(ROOT/'language-teacher-index.npy')
    language_q=np.load(ROOT/'language-labeling/teacher-labels/prob-match.npy')
    if language_q.shape!=(label_ready['pairs'],) or not np.isfinite(language_q).all() or np.any((language_q<0)|(language_q>1)):
        raise RuntimeError('Native Teacher probabilities are incomplete or invalid')
    if old_targets.shape!=(1000000,) or old_teacher.shape!=(1000000,):
        raise RuntimeError('Original E24 cache identity differs')
    language_y=np.asarray(pool.tables['language']['target'],dtype=np.float32)
    effective=np.concatenate([old_targets.astype(np.float32),0.5*language_y+0.5*language_q[mapping]])
    if effective.shape!=(cfg['pairs'],) or not np.isfinite(effective).all() or np.any((effective<0)|(effective>1)):
        raise RuntimeError('Five-domain targets are incomplete or invalid')
    array(ROOT/'effective-targets.npy',effective.astype(np.float32))
    old_plan=HOME/'runs/distillation-group-20261106/plan'
    relations=np.load(old_plan/'relations.npy')
    relative=np.load(old_plan/'teacher_relation_target.npy')
    if relations.shape!=(3647,2) or relative.shape!=(3647,) or np.any(relations>=1000000):
        raise RuntimeError('Preserved K/R relation identity differs')
    positive=pool.fetch(relations[:,0]);negative=pool.fetch(relations[:,1])
    if not np.all(positive['target']==1) or not np.all(negative['target']==0):
        raise RuntimeError('Relation supervision changed')
    for key in ['source','group','state','domain']:
        if not np.array_equal(positive[key],negative[key]):
            raise RuntimeError('Relation endpoints no longer share an intact audited question')
    lengths=np.concatenate([np.asarray(pool.tables[d]['length']) for d in DOMAINS])
    order,tags,offsets=build_plan(lengths,relations,seed=42,batch_size=128)
    plan=ROOT/'plan';plan.mkdir(exist_ok=True)
    for name,value in [('order',order),('relation_tag',tags),('macro_offsets',offsets),
                       ('relations',relations),('teacher_relation_target',relative)]:
        array(plan/(name+'.npy'),value)
    preserve(plan/'PLAN_READY.json',{'complete':True,'identity':cfg['group_plan_identity']})
    ready={'complete':True,'student_parent':cfg['student_parent'],'training_pool':cfg['training_pool'],
        'exploration_policy':cfg['exploration_policy'],'pairs':cfg['pairs'],
        'sampler_identity':cfg['group_plan_identity'],'effective_target_path':cfg['effective_target_path'],
        'mixing_statistics':{'old_four_source_pairs_retained':1000000,'language_teacher_fraction':0.5,
            'language_pairs':len(language_y),'new_teacher_unique_pairs':len(language_q)},
        'label_provenance':label_ready}
    preserve(ROOT/'DISTILLATION_TARGETS_READY.json',ready)
    print(json.dumps({'event':'five_dimension_targets_and_plan_ready','pairs':len(pool),
        'steps':len(offsets)-1,'tail_pairs':int(offsets[-1]-offsets[-2]),'relations':len(relations)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--targets',action='store_true');args=parser.parse_args()
    targets() if args.targets else freeze()
