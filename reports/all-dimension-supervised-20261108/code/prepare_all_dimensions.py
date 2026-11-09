#!/usr/bin/env python3
"""Create the real two-stage training schedules from existing token caches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

HOME = Path('/home/dev/assumeengage')
sys.path.insert(0, str(HOME / 'scripts'))
from pair_data_common import DOMAINS
from pair_training_runtime import PairPool, TokenStore


def write_json(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def bucket_order(order, pool, rng, macro):
    rng.shuffle(order)
    for begin in range(0, len(order), 8192):
        block = order[begin:begin + 8192].copy()
        block = block[np.argsort(pool.fetch(block)['length'], kind='stable')]
        full = len(block) // macro
        if full:
            groups = block[:full * macro].reshape(full, macro)
            block[:full * macro] = groups[rng.permutation(full)].reshape(-1)
        order[begin:begin + len(block)] = block
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    root = Path(cfg['run_root'])
    schedule = Path(cfg['schedule_root'])
    schedule.mkdir(parents=True, exist_ok=True)
    if (schedule / 'READY.json').is_file():
        return
    pool = PairPool('MIXED_ALL')
    sources = json.loads((pool.root / 'SOURCES.json').read_text())['sources']
    excluded = {'canonical_nli/isarcasm/intended', 'canonical_nli/ragtruth/grounding',
                'canonical_nli/semeval2014-absa/laptops',
                'canonical_nli/semeval2014-absa/restaurants'}
    ids_file = np.lib.format.open_memmap(schedule / 'source-rows.npy', mode='w+',
                                        dtype=np.uint32, shape=(len(pool),))
    lengths = np.lib.format.open_memmap(schedule / 'lengths.npy', mode='w+',
                                       dtype=np.uint32, shape=(len(pool),))
    metadata = []
    warmup_rows = []
    cursor = 0
    offset = 0
    for domain in DOMAINS:
        table = pool.tables[domain]
        length = table['length']
        lengths[offset:offset + len(length)] = length
        ordered = np.argsort(table['source'], kind='stable')
        ordered_sources = table['source'][ordered]
        boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_sources)) + 1, len(ordered)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            local = ordered[begin:end]
            sid = int(ordered_sources[begin])
            record = sources[sid]
            global_rows = (local + offset).astype(np.uint32)
            ids_file[cursor:cursor + len(local)] = global_rows
            item = {'source_id': sid, 'name': record['name'], 'domain': domain,
                    'start': cursor, 'end': cursor + len(local), 'pairs': len(local),
                    'tokens': int(length[local].sum()),
                    'original_groups': int(len(np.unique(table['group'][local])))}
            metadata.append(item)
            if record['name'].startswith('canonical_nli/') and record['name'] not in excluded:
                warmup_rows.append(global_rows)
            cursor += len(local)
        offset += len(length)
        print(json.dumps({'event': 'domain_training_index_ready', 'domain': domain,
                          'pairs': len(length)}), flush=True)
    ids_file.flush()
    lengths.flush()
    write_json(schedule / 'sources.json', metadata)
    rng = np.random.default_rng(cfg['seed'])
    generic = np.concatenate(warmup_rows)
    np.save(schedule / 'generic-rows.npy', generic)
    phase1 = bucket_order(generic.copy(), pool, rng, cfg['batch_pairs'])
    np.save(schedule / 'stage1-order.npy', phase1)
    stage1 = {'name': 'stage1', 'rank': 16, 'alpha': 32, 'learning_rate': 1e-4,
              'pairs': int(len(phase1)), 'tokens': int(lengths[phase1].sum()),
              'order_path': str(schedule / 'stage1-order.npy'), 'epochs': 1}
    write_json(schedule / 'stage1.json', stage1)
    print(json.dumps({'event': 'generic_training_schedule_ready', **stage1}), flush=True)

    def draw(rows, token_budget):
        if not len(rows):
            raise RuntimeError('A selected training source is empty')
        average = float(lengths[rows].mean())
        count = max(128, int(token_budget / max(1, average) * 1.10) + 128)
        chosen = rows[rng.integers(0, len(rows), size=count)]
        cumulative = np.cumsum(lengths[chosen], dtype=np.int64)
        if cumulative[-1] < token_budget:
            return np.concatenate((chosen, draw(rows, token_budget - int(cumulative[-1]))))
        return chosen[:int(np.searchsorted(cumulative, token_budget, side='left')) + 1]

    # Native English sarcasm uses its original class, not pair match/non-match.
    sarcasm = [m for m in metadata if m['name'] == 'native/isarcasm']
    sarcasm_classes = []
    if sarcasm:
        from transformers import AutoTokenizer
        sid = sarcasm[0]['source_id']
        group_sources = pq.read_table(pool.root / 'groups.parquet', columns=['source_id'])['source_id'].to_numpy()
        group_ids = np.flatnonzero(group_sources == sid)
        group_names = pq.read_table(pool.root / 'groups.parquet', filters=[('source_id', '=', sid)],
                                    columns=['original_group_id'])['original_group_id'].to_pylist()
        english_groups = [int(gid) for gid, name in zip(group_ids, group_names)
                          if ':train.En.csv:' in name and name.endswith(':sarcastic')]
        m = sarcasm[0]
        all_rows = np.asarray(ids_file[m['start']:m['end']])
        table = pool.fetch(all_rows)
        mask = np.isin(table['group'], english_groups)
        rows = all_rows[mask]
        batch = pool.fetch(rows)
        tokenizer = AutoTokenizer.from_pretrained(cfg['common_initialization'], local_files_only=True)
        store = TokenStore()
        yes_ids = [int(cid) for cid in np.unique(batch['candidate'])
                   if tokenizer.decode(store.get('suffix', int(cid)), skip_special_tokens=True)
                   .strip().casefold() in {'yes', 'yes.'}]
        class_by_group = {int(g): float(y) for g, c, y in
                          zip(batch['group'], batch['candidate'], batch['target']) if int(c) in yes_ids}
        native_class = np.array([class_by_group.get(int(g), -1) for g in batch['group']])
        sarcasm_classes = [rows[native_class == v] for v in [0, 1]]
        if any(not len(r) for r in sarcasm_classes):
            raise RuntimeError('English sarcasm native-class mapping is incomplete')

    phase2_parts = []
    exposure = []
    generic_budget = int(cfg['target_tokens_per_domain'] * len(DOMAINS) * 0.20 / 0.80)
    part = draw(generic, generic_budget)
    phase2_parts.append(part)
    exposure.append({'kind': 'generic', 'requested_tokens': generic_budget,
                     'actual_tokens': int(lengths[part].sum()), 'pairs': len(part)})
    for domain in DOMAINS:
        budget = cfg['target_tokens_per_domain']
        entries = [m for m in metadata if m['domain'] == domain
                   and not m['name'].startswith('canonical_nli/') and m['name'] != 'native/isarcasm']
        if not entries:
            raise RuntimeError('No task training sources for ' + domain)
        if domain == 'language' and sarcasm_classes:
            for cls, rows in enumerate(sarcasm_classes):
                requested = int(budget * 0.05 / 2)
                chosen = draw(rows, requested)
                phase2_parts.append(chosen)
                exposure.append({'kind': 'task', 'domain': domain, 'source': 'native/isarcasm/english',
                                 'native_sarcastic': cls, 'requested_tokens': requested,
                                 'actual_tokens': int(lengths[chosen].sum()), 'pairs': len(chosen)})
            budget = int(budget * 0.95)
        weights = np.sqrt([m['original_groups'] for m in entries])
        weights /= weights.sum()
        for entry, weight in zip(entries, weights):
            requested = max(1, int(round(budget * float(weight))))
            chosen = draw(ids_file[entry['start']:entry['end']], requested)
            phase2_parts.append(chosen)
            exposure.append({'kind': 'task', 'domain': domain, 'source': entry['name'],
                             'requested_tokens': requested, 'actual_tokens': int(lengths[chosen].sum()),
                             'pairs': len(chosen)})
    phase2 = bucket_order(np.concatenate(phase2_parts), pool, rng, cfg['batch_pairs'])
    np.save(schedule / 'stage2-order.npy', phase2)
    stage2 = {'name': 'stage2', 'rank': 32, 'alpha': 64, 'learning_rate': 5e-5,
              'pairs': int(len(phase2)), 'tokens': int(lengths[phase2].sum()),
              'order_path': str(schedule / 'stage2-order.npy'), 'epochs': None,
              'generic_task_token_ratio': '20:80', 'task_domain_token_allocation': 'equal',
              'source_weight': 'sqrt(original_group_count)',
              'english_sarcasm_fraction_of_language_task_tokens': 0.05 if sarcasm_classes else 0.0,
              'sarcasm_balance': 'native classes by token budget; natural candidate pairs retained'}
    write_json(schedule / 'stage2.json', stage2)
    write_json(schedule / 'exposure.json', exposure)
    stages = [s for s in [stage1, stage2] if s['name'] not in cfg.get('skip_stages', [])]
    write_json(schedule / 'READY.json', {'complete': True, 'stages': stages,
                                        'all_training_domains': DOMAINS, 'new_downloads': 0})
    write_json(root / 'DATA_READY.json', {'complete': True, 'stages': stages,
                                        'all_training_domains': DOMAINS})
    print(json.dumps({'event': 'all_training_schedules_ready', 'stages': stages}), flush=True)


if __name__ == '__main__':
    main()
