"""Frozen two-role consumer phases. No work, Torch or SDK import at module load."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import stat
import struct
import sys
import time

sys.dont_write_bytecode = True
from common import (ROOT, ROLES, Guard, canonical, commit, entry, exact, identity,
                    need, pin, read, same, verify)

DOMAINS = ('knowledge', 'language', 'retrieval', 'tools', 'arts')
ENGINE = 'backend:CleanEngine'
PHASES = ('codec', 'confirm', 'confirm-analysis', 'public-prepare',
          'public-worker', 'public-score')
METRICS = ('coverage_adjusted_argmax_target_mass',
           'supported_known_field_argmax_target_mass', 'supported_known_field_NLL',
           'supported_known_field_normalized_NLL', 'supported_known_field_Brier_sum',
           'known_field_coverage')
NON_OK_DETECTION = {
    'mechanism': 'frozen SDK event callback and final all-ok validation',
    'ordinary_progress_every_requests': 10,
    'max_additional_requests_after_ordinary_non_ok': 9,
    'first_non_ok_prevents_next_forward': False,
    'device_errors_or_sdk_failure_budget_may_halt_earlier': True,
    'automatic_retry': False,
    'complete_requires_all_ok': True,
}


def key(phase, role=None, rank=None):
    return phase + (('-' + role) if role is not None else '') + (
        ('-' + str(rank)) if rank is not None else '')


def output_for(phase, role=None, rank=None):
    if phase == 'confirm':
        return ROOT / 'confirm' / role
    if phase == 'public-worker':
        return ROOT / 'public' / role / 'workers' / str(rank)
    if phase == 'public-score':
        return ROOT / 'public' / role
    return ROOT


def absent(path):
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), 'Existing partial preserved: ' + str(path))


def directory(path):
    path = Path(path)
    s = path.lstat()
    need(stat.S_ISDIR(s.st_mode) and not path.is_symlink() and path.resolve() == path,
         'Fixed nonsymlink directory required: ' + str(path))


def jsonl(path):
    """Reject blank/truncated/duplicate-key/nonfinite rows, not a recovery reader."""
    def pairs(items):
        result = {}
        for name, value in items:
            need(name not in result, 'Duplicate JSON key')
            result[name] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            need(line.endswith('\n') and line.strip(), 'Incomplete/empty JSONL row')
            yield json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)


def ordered(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def flush(handle, readonly=False):
    handle.flush()
    if readonly:
        os.fchmod(handle.fileno(), 0o444)
    os.fsync(handle.fileno())


def durable_files(paths):
    """Fsync stopped immutable output dependencies and ancestry, THEN bind pins.

    Only fixed ROOT outputs are accepted. No SDK status/temp file is frozen and
    no content is rewritten, reloaded, or hashed. Failures prevent COMPLETE.
    """
    paths = [Path(path) for path in paths]
    need(paths and len(set(paths)) == len(paths), 'Nonempty unique output dependency list required')
    directory(ROOT)
    held, parents = [], set()
    try:
        for path in paths:
            need(path.is_absolute() and path.is_relative_to(ROOT) and path.resolve() == path and
                 path.name != 'status.json' and not path.name.endswith(('.tmp', '.writing')),
                 'Only fixed nonsymlink stopped output files may be synchronized')
            parent = path.parent
            while True:
                directory(parent)
                parents.add(parent)
                if parent == ROOT:
                    break
                parent = parent.parent
            ls = path.lstat()
            need(stat.S_ISREG(ls.st_mode), 'Regular output dependency required')
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
            held.append((path, fd, ls))
            fs = os.fstat(fd)
            need(stat.S_ISREG(fs.st_mode) and (fs.st_dev, fs.st_ino, fs.st_size, fs.st_mtime_ns) ==
                 (ls.st_dev, ls.st_ino, ls.st_size, ls.st_mtime_ns), 'Output changed before dependency fsync')
            os.fsync(fd)
        # Bottom-up: persist each file entry, then every newly-created parent
        # entry up through ROOT, before COMPLETE can refer to those names.
        for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            directory(parent)
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                fs, ls = os.fstat(fd), parent.lstat()
                need(stat.S_ISDIR(fs.st_mode) and (fs.st_dev, fs.st_ino) == (ls.st_dev, ls.st_ino),
                     'Output ancestry changed before directory fsync')
                os.fsync(fd)
                now = parent.lstat()
                need((fs.st_dev, fs.st_ino) == (now.st_dev, now.st_ino), 'Output ancestry changed during fsync')
            finally:
                os.close(fd)
        pins = []
        for path, fd, before in held:
            fs, ls = os.fstat(fd), path.lstat()
            expected = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            need(stat.S_ISREG(ls.st_mode) and path.resolve() == path and
                 (fs.st_dev, fs.st_ino, fs.st_size, fs.st_mtime_ns) == expected ==
                 (ls.st_dev, ls.st_ino, ls.st_size, ls.st_mtime_ns), 'Output changed during dependency synchronization')
            value = pin(path)  # First bound pin is strictly after all dependency fsyncs.
            after = path.lstat()
            need(value['path'] == value['resolved'] == str(path) and
                 value['bytes'] == fs.st_size and value['mtime_ns'] == fs.st_mtime_ns and
                 stat.S_ISREG(after.st_mode) and
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == expected,
                 'Output changed while binding synchronized dependency pin')
            pins.append(value)
        return pins
    finally:
        for path, fd, before in held:
            os.close(fd)


def installed(guard):
    guard.check()
    done = read(ROOT / 'INSTALL_COMPLETE.json')
    need(done['complete'] is True and done['no_workers_launched'] is True,
         'Actual install-only completion required')
    same(done['config'], pin(ROOT / 'config.json'), 'Installed config differs')
    same(done['source_binding'], guard.cfg['source_binding'], 'Installed source snapshot differs')
    same(done['source_files'], guard.cfg['source_files'], 'Installed source inventory differs')
    verify([done['config'], done['source_binding'], *done['source_files']])


def grant(guard, phase, role=None, rank=None):
    """Every launcher child, including CPU readers, must have its exact live parent."""
    installed(guard)
    path = ROOT / 'grants' / (key(phase, role, rank) + '-GRANT.json')
    need(os.environ.get('JEV_FINAL_GRANT') == str(path), 'Missing/misdirected worker grant')
    directory(path.parent)
    item = entry(path)
    s = path.lstat()
    need(s.st_uid == os.getuid() and s.st_nlink == 1 and stat.S_IMODE(s.st_mode) == 0o444,
         'Unsafe grant file')
    g = read(path)
    same({name: g[name] for name in ('version', 'root', 'phase', 'role', 'rank', 'output')},
         {'version': 'clean-final-phase-grant-v1', 'root': str(ROOT), 'phase': phase,
          'role': role, 'rank': rank, 'output': str(output_for(phase, role, rank))},
         'Wrong phase/role/rank/output grant')
    same(g['config'], pin(ROOT / 'config.json'), 'Grant config changed')
    same(g['source_files'], guard.cfg['source_files'], 'Grant source changed')
    same(identity(os.getppid()), g['owner'], 'Grant parent is not actual live manager')
    same(read(ROOT / 'OWNER.json')['owner'], g['owner'], 'Grant owner is not one-shot manager')
    if phase in ('confirm', 'public-worker'):
        gpu = ROLES.index(role) if phase == 'confirm' else rank
        same(g['gpu'], gpu, 'Wrong physical device index grant')
        same(g['device_uuid'], guard.cfg['predecessors'][str(gpu)]['uuid'], 'Wrong physical device UUID grant')
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == g['device_uuid'], 'Visible GPU is not grant UUID')
    else:
        need(g['gpu'] is None and g['device_uuid'] is None and
             os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU phase must hide CUDA')
        need('torch' not in sys.modules, 'CPU launcher phase unexpectedly imported Torch')
    exact(item)
    return g


def child_exit(phase, role=None, rank=None):
    path = ROOT / (key(phase, role, rank) + '-EXIT.json')
    value = read(path)
    need(value['phase'] == phase and value['role'] == role and value['rank'] == rank and
         value['created'] is True and value['actual_wait_complete'] is True and
         type(value['exit_code']) is int and value['exit_code'] == 0, 'Actual successful child wait required')
    same(value['owner'], read(ROOT / 'OWNER.json')['owner'], 'Foreign child exit receipt')
    return pin(path)


def codec_receipt(guard):
    value = read(ROOT / 'CODEC_COMPLETE.json')
    need(value['complete'] is True and value['cases'] == 375 and value['known_fields'] == 615 and
         value['candidates'] == 5190 and value['unknown_fields_retained_no_forward'] == 90 and
         value['model_forward_calls'] == 0 and value['all_ids_exact'] is True,
         'Full real input-only codec admission required')
    same(value['frozen_candidates'], guard.frozen_pin, 'Codec frozen candidates differ')
    same(value['source_files'], guard.cfg['source_files'], 'Codec source differs')
    verify(value['input_pins'])
    return pin(ROOT / 'CODEC_COMPLETE.json')


def validate_case(case):
    """Independent native validation used by CPU analysis (no producer imports)."""
    need(case['split'] == 'validation' and case['primary_domain'] in DOMAINS and
         case['case_id'] == 'clean-semantic-' + str(case['component']) and case['fields'] and
         not case['rejection_reasons'] and case['truncated'] is False and
         case['unknown_forward_rule'] == 'skip-all-unknown-fields-v1', 'Illegal complete native case')
    need(case['known_mask'] == [f['known'] for f in case['fields']], 'Native mask differs')
    coords, total, maximum, known = set(), 0, 0, 0
    for field in case['fields']:
        coord = canonical(field['coordinate'])
        need(coord not in coords and field['domain'] in DOMAINS and type(field['known']) is bool,
             'Duplicate/invalid field')
        coords.add(coord)
        if not field['known']:
            need(field['target'] is None and field['prefix_ids'] == field['suffix_ids'] ==
                 field['pair_lengths'] == [] and 'raw_options' in field and 'raw_target' in field and
                 field['forward_policy'] == 'masked_unknown_no_forward' and
                 field['static_unsupported_reason'] == 'unknown_supervision_masked_no_forward',
                 'Unknown raw must remain masked, not reinterpreted')
            continue
        known += 1
        q = field['target']
        need(isinstance(q, list) and len(q) == len(field['options']) == len(field['suffix_ids']) >= 1 and
             all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in q) and
             math.isclose(math.fsum(q), 1.0, rel_tol=1e-6, abs_tol=1e-6), 'Invalid native q')
        need(field['prefix_ids'] and all(type(t) is int and t >= 0 for t in field['prefix_ids']) and
             field['forward_policy'] == 'known_complete_candidates', 'Invalid complete prefix')
        need(all(s and s[-1] == 151645 and all(type(t) is int and t >= 0 for t in s)
                 for s in field['suffix_ids']), 'Invalid suffix/EOS')
        lengths = [len(field['prefix_ids']) + len(s) for s in field['suffix_ids']]
        same(lengths, field['pair_lengths'], 'Native row lengths differ')
        need(field['static_unsupported_reason'] == ('pair_context>40960' if max(lengths) > 40960 else None),
             'Static whole-field support changed')
        total += sum(lengths)
        maximum = max(maximum, max(lengths))
    need(known > 0 and total == case['logical_supervised_pair_tokens'] ==
         case['all_field_pair_tokens_once'] == case['scheduled_forward_pair_tokens_once'] and
         maximum == case['max_pair_tokens'] and case['unknown_field_forward_tokens_once'] == 0 and
         case['unknown_fields_no_forward'] == len(case['fields']) - known, 'Native accounting differs')


def codec(guard):
    grant(guard, 'codec')
    need(os.environ.get('USE_TORCH') == '0' and 'torch' not in sys.modules, 'Tokenizer-only process required')
    from transformers import AutoTokenizer
    from backend import field_tokens
    need('torch' not in sys.modules, 'Tokenizer import unexpectedly loaded Torch')
    tokenizer = AutoTokenizer.from_pretrained(guard.cfg['stock_directory'],
                                             local_files_only=True, trust_remote_code=False)
    need(tokenizer.eos_token_id == 151645, 'Original tokenizer EOS required')
    path = Path(guard.cfg['data_root']) / 'data' / 'select.jsonl'
    source_pin = pin(path)
    count, domains, seen = Counter(), Counter(), set()
    for case in jsonl(path):
        guard.check()
        validate_case(case)
        need(case['case_id'] not in seen, 'Duplicate codec case')
        seen.add(case['case_id'])
        count['cases'] += 1
        domains[case['primary_domain']] += 1
        for field in case['fields']:
            if not field['known']:
                count['unknown_fields_retained_no_forward'] += 1
                continue
            prefix, suffixes = field_tokens(tokenizer, field['state'], field['question'], field['options'])
            same(prefix, field['prefix_ids'], 'Actual codec prefix IDs differ')
            same(suffixes, field['suffix_ids'], 'Actual codec candidate/EOS IDs differ')
            same([len(prefix) + len(s) for s in suffixes], field['pair_lengths'], 'Actual codec rows differ')
            need(all(s[-1] == 151645 for s in suffixes), 'EOS-last codec contract differs')
            count['known_fields'] += 1
            count['candidates'] += len(suffixes)
            count['real_pair_tokens'] += sum(field['pair_lengths'])
    need(count['cases'] == 375 and count['known_fields'] == 615 and count['candidates'] == 5190 and
         count['unknown_fields_retained_no_forward'] == 90 and domains == Counter({d: 75 for d in DOMAINS}),
         'Full original D_select codec coverage differs')
    verify([source_pin])
    guard.check()
    need('torch' not in sys.modules, 'Codec must never import/load a model')
    commit(ROOT / 'CODEC_COMPLETE.json', {'complete': True, **dict(count), 'domains': dict(domains),
           'all_ids_exact': True, 'model_forward_calls': 0, 'frozen_candidates': guard.frozen_pin,
           'source_files': guard.cfg['source_files'], 'stock_directory': guard.cfg['stock_directory'],
           'renderer_source': guard.cfg['renderer_source'],
           'input_pins': [source_pin, guard.cfg['renderer_source']], 'confirm_opened': False,
           'public_payload_opened': False})


def average(values):
    return math.fsum(values) / len(values) if values else None


def field_item(raw, fi):
    return {'field_index': fi, 'coordinate': raw['coordinate'], 'domain': raw['domain'],
            'known': raw['known'], 'K': len(raw['options']) if raw['known'] else None}


def case_result(case, items):
    known = sum(f['known'] for f in case['fields'])
    supported = [f for f in items if f['status'] == 'supported_complete_field']
    mass = [f['argmax_target_mass'] for f in supported]
    exact_supported = bool(case['complete_case_exact_static_supported'])
    need(not exact_supported or (known == len(case['fields']) == len(supported) and
         all(f['onehot'] for f in case['fields'])), 'Illegal onehot exact denominator')
    return {'case_id': case['case_id'], 'component': case['component'],
            'primary_domain': case['primary_domain'], 'known_fields': known,
            'supported_known_fields': len(supported), 'unknown_fields': len(items) - known,
            'known_field_coverage': len(supported) / known,
            'coverage_adjusted_argmax_target_mass': math.fsum(mass) / known,
            'supported_known_field_argmax_target_mass': average(mass),
            'supported_known_field_NLL': average([f['NLL'] for f in supported]),
            'supported_known_field_normalized_NLL': average([f['normalized_NLL'] for f in supported]),
            'supported_known_field_Brier_sum': average([f['Brier_sum'] for f in supported]),
            'complete_onehot_case_exact_supported': exact_supported,
            'complete_onehot_case_exact': all(x == 1.0 for x in mass) if exact_supported else None,
            'fields': items}


def summarize(records):
    need(len(records) == 250 and len({r['case_id'] for r in records}) == 250 and
         len({r['component'] for r in records}) == 250, 'Confirm must preserve all 250 unique cases/components')
    coverage, domains, secondary = Counter(), {}, {}
    for domain in DOMAINS:
        cases = [r for r in records if r['primary_domain'] == domain]
        need(len(cases) == 50, 'Native confirm primary-domain quota must be 50')
        domains[domain] = {'cases': 50}
        for metric in METRICS:
            values = [r[metric] for r in cases if r[metric] is not None]
            domains[domain][metric] = average(values)
            domains[domain][metric + '_case_denominator'] = len(values)
        field_cases = []
        for case in records:
            fields = [f for f in case['fields'] if f['known'] and f['domain'] == domain]
            if fields:
                support = [f for f in fields if f['status'] == 'supported_complete_field']
                field_cases.append({'argmax_target_mass': average([f['argmax_target_mass'] for f in support]),
                                    'NLL': average([f['NLL'] for f in support]),
                                    'Brier_sum': average([f['Brier_sum'] for f in support]),
                                    'known_field_coverage': len(support) / len(fields)})
        secondary[domain] = {'cases_with_known_field_in_domain': len(field_cases)}
        for metric in ('argmax_target_mass', 'NLL', 'Brier_sum', 'known_field_coverage'):
            values = [x[metric] for x in field_cases if x[metric] is not None]
            secondary[domain][metric] = average(values)
            secondary[domain][metric + '_case_denominator'] = len(values)
    for case in records:
        coverage['cases'] += 1
        coverage['known_fields'] += case['known_fields']
        coverage['supported_known_fields'] += case['supported_known_fields']
        coverage['cases_with_any_supported_known_field'] += case['supported_known_fields'] > 0
        coverage['cases_with_all_known_fields_supported'] += case['supported_known_fields'] == case['known_fields']
        coverage['complete_onehot_case_exact_supported'] += case['complete_onehot_case_exact_supported']
        for field in case['fields']:
            coverage['unknown_fields_retained_no_forward'] += field['status'] == 'masked_unknown_no_forward'
            coverage['static_unsupported_known_fields'] += field['status'] == 'static_unsupported'
            if field['status'] == 'supported_complete_field':
                coverage['forwarded_complete_candidates'] += field['K']
                coverage['forwarded_real_pair_tokens'] += sum(field['original_pair_lengths'])
    exacts = [r['complete_onehot_case_exact'] for r in records if r['complete_onehot_case_exact_supported']]
    return {'roster': 'D_confirm', 'native_split': 'validation',
            'statistical_unit': 'complete native case; equal mean of five primary domains',
            'primary_metric': 'supported_known_field_argmax_target_mass',
            'coverage_sensitivity_metric': 'coverage_adjusted_argmax_target_mass',
            'five_domain_case_means': {m: average([domains[d][m] for d in DOMAINS])
                                      if all(domains[d][m] is not None for d in DOMAINS) else None
                                      for m in METRICS},
            'per_domain': domains, 'coverage': dict(coverage),
            'field_domain_case_means_secondary_not_selection_metric': secondary,
            'complete_onehot_case_exact': average(exacts),
            'complete_onehot_case_exact_denominator': len(exacts),
            'coverage_complete': coverage['known_fields'] == coverage['supported_known_fields'],
            'q_renormalized': False, 'unknown_forwarded': False, 'candidate_pruning': False,
            'static_missing_definition': 'all cases retained; conditional metrics plus zero-credit known denominator',
            'Brier_definition': 'sum over complete native candidates of (softmax(m)-native q)^2',
            'NLL_definition': '-sum(native q*log_softmax(m)); normalized by log(K), K1=0',
            'public_list_changed': False, 'winner_selected': False}


def weights_ok(value):
    need(value['complete'] is True and value['weights_unchanged'] is True and
         value['keys_compared'] == 393 and value['served_parameters'] == 616237056 and
         value['post_eval_reload'] is False, 'Pure trained393 after-proof required')


def confirm(guard, role):
    grant(guard, 'confirm', role)
    codec_pin = codec_receipt(guard)
    output = output_for('confirm', role)
    absent(output)
    directory(output.parent)
    output.mkdir(mode=0o700)
    from backend import configure_role, TerminalModel
    need('torch' not in sys.modules, 'Role configuration must precede first Torch')
    configure_role(guard, role)
    model = TerminalModel(guard, role)
    store, records, proof = None, [], None
    started = time.monotonic()
    try:
        import torch
        runtime_before = model.runtime.receipt()
        # The original native reader is called only AFTER actual trained loader admission.
        store = model.case_runtime.CaseStore(Path(guard.cfg['data_root']) / 'data' / 'confirm.jsonl', 'validation')
        need(store.domains == {d: 50 for d in DOMAINS} and len(store.offsets) == 250,
             'Full native validation confirm roster required, never select alias')
        source_pin = pin(store.path)
        with (output / 'case-metrics.jsonl').open('x', encoding='utf-8') as handle:
            for case in store:
                guard.check()
                validate_case(case)
                pairs, fields = model.case_runtime.slots([case], dev=True)
                if pairs:
                    logits, margins = model.score(pairs, phase='dev_confirm')
                    need(logits.dtype == margins.dtype == torch.float32 and logits.device.type ==
                         margins.device.type == 'cpu' and tuple(logits.shape) == (len(pairs), 2) and
                         tuple(margins.shape) == (len(pairs),) and torch.isfinite(logits).all().item() and
                         torch.isfinite(margins).all().item(), 'Invalid actual confirm output')
                    need(torch.equal(margins, logits[:, 1] - logits[:, 0]), 'Actual FP32 margin subtraction differs')
                else:
                    logits = margins = None
                by_index = {f['field_index']: f for f in fields}
                items = []
                for fi, raw in enumerate(case['fields']):
                    item = field_item(raw, fi)
                    if not raw['known']:
                        item.update(status='masked_unknown_no_forward', unknown_reason=raw['unknown_reason'],
                                    raw_target=raw['raw_target'], raw_options=raw['raw_options'])
                    elif fi not in by_index:
                        need(raw['static_unsupported_reason'] == 'pair_context>40960', 'No dynamic field deletion')
                        item.update(status='static_unsupported', reason=raw['static_unsupported_reason'],
                                    original_pair_lengths=raw['pair_lengths'])
                    else:
                        f = by_index[fi]
                        m = margins[f['start']:f['end']].double()
                        q = torch.tensor(raw['target'], dtype=torch.float64)
                        log_p = torch.log_softmax(m, dim=0)
                        p = log_p.exp()
                        best = int(torch.argmax(m).item())
                        nll = float(-(q * log_p).sum().item())
                        item.update(status='supported_complete_field', argmax_index=best,
                                    argmax_target_mass=float(q[best].item()), NLL=nll,
                                    normalized_NLL=nll / math.log(len(q)) if len(q) > 1 else 0.0,
                                    Brier_sum=float(((p - q) ** 2).sum().item()), margins=m.tolist(),
                                    logits=logits[f['start']:f['end']].tolist(),
                                    native_target_not_renormalized=raw['target'],
                                    original_pair_lengths=raw['pair_lengths'])
                    items.append(item)
                result = case_result(case, items)
                handle.write(ordered(result) + '\n')
                flush(handle)
                records.append(result)
            flush(handle, readonly=True)
        verify([source_pin])
        summary = summarize(records)
        runtime_after = model.runtime.receipt()
        before_counts, after_counts = runtime_before['forward'], runtime_after['forward']
        expected_tokens = summary['coverage'].get('forwarded_real_pair_tokens', 0)
        need(after_counts.get('dev_confirm.completed_real_tokens', 0) -
             before_counts.get('dev_confirm.completed_real_tokens', 0) == expected_tokens and
             after_counts.get('dev_confirm.completed_tokenwise_rows_per_operator', 0) -
             before_counts.get('dev_confirm.completed_tokenwise_rows_per_operator', 0) == expected_tokens,
             'Actual confirm native realT counters differ from all complete candidate rows')
        waves = after_counts.get('dev_confirm.completed_waves', 0) - before_counts.get('dev_confirm.completed_waves', 0)
        need(after_counts.get('dev_confirm.attempted_waves', 0) - before_counts.get('dev_confirm.attempted_waves', 0) == waves and
             after_counts.get('dev_confirm.completed_native_FA3_calls', 0) -
             before_counts.get('dev_confirm.completed_native_FA3_calls', 0) == 28 * waves,
             'Confirm must use exactly native causal full-row layers, no failed forward/retry')
        summary.update(role=role, model=guard.producer(role)['response_name'],
                       terminal=guard.producer(role)['terminal'], codec=codec_pin,
                       input=source_pin, source_files=guard.cfg['source_files'],
                       runtime_before=runtime_before, runtime_after=runtime_after,
                       wall_seconds=time.monotonic() - started)
    finally:
        if store is not None:
            store.close()
        # Even a failed attempt must not silently skip its pure after audit. Never reload.
        proof = model.finish()
        commit(output / 'WEIGHTS.json', proof)
    weights_ok(proof)
    guard.check()
    commit(output / 'SUMMARY.json', summary)
    files = durable_files([output / n for n in ('case-metrics.jsonl', 'SUMMARY.json', 'WEIGHTS.json')])
    commit(output / 'COMPLETE.json', {'complete': True, 'role': role, 'cases': 250,
           'domain_quota': {d: 50 for d in DOMAINS}, 'coverage_complete': summary['coverage_complete'],
           'source_files': guard.cfg['source_files'], 'frozen_candidates': guard.frozen_pin,
           'files': files, 'dependencies_fsynced_before_pins_and_complete': True,
           'public_list_changed': False, 'optimizer_steps': 0})


def f32(value):
    need(type(value) in (float, int) and math.isfinite(value), 'Nonfinite actual FP32 scalar')
    return struct.unpack('<f', struct.pack('<f', value))[0]


def close_number(actual, expected, name):
    if type(expected) is not float:
        same(actual, expected, name)
    else:
        need(type(actual) in (float, int) and math.isfinite(actual) and
             math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), name)


def compare_tree(actual, expected, name):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), name + ': keys')
        for k, value in expected.items():
            compare_tree(actual[k], value, name + '/' + k)
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), name + ': list length')
        for i, value in enumerate(expected):
            compare_tree(actual[i], value, name + '/' + str(i))
    else:
        close_number(actual, expected, name)


def independent_case(case, observed):
    validate_case(case)
    need(len(observed['fields']) == len(case['fields']), 'Independent field roster differs')
    items = []
    for fi, (raw, stored) in enumerate(zip(case['fields'], observed['fields'])):
        item = field_item(raw, fi)
        for name, value in item.items():
            same(stored[name], value, 'Exact independent native field coordinate/type differs')
        if not raw['known']:
            same(stored['raw_target'], raw['raw_target'], 'Unknown raw target changed')
            same(stored['raw_options'], raw['raw_options'], 'Unknown raw criteria changed')
            item.update(status='masked_unknown_no_forward', unknown_reason=raw['unknown_reason'],
                        raw_target=raw['raw_target'], raw_options=raw['raw_options'])
        elif raw['static_unsupported_reason']:
            item.update(status='static_unsupported', reason=raw['static_unsupported_reason'],
                        original_pair_lengths=raw['pair_lengths'])
        else:
            logits, margins = stored['logits'], stored['margins']
            q = raw['target']
            same(stored['native_target_not_renormalized'], q, 'Native q changed or was renormalized')
            need(len(logits) == len(margins) == len(q), 'Missing complete raw candidate outputs')
            derived = []
            for pair, margin in zip(logits, margins):
                need(isinstance(pair, list) and len(pair) == 2 and
                     all(f32(x) == x for x in pair) and f32(margin) == margin,
                     'Stored logits/margins must be actual FP32 values')
                expected = f32(pair[1] - pair[0])
                need(expected == margin, 'Independent FP32 logits subtraction differs')
                derived.append(expected)
            best = max(range(len(q)), key=lambda i: derived[i])
            shift = max(derived)
            denominator = math.fsum(math.exp(x - shift) for x in derived)
            log_z = math.log(denominator)
            log_p = [(x - shift) - log_z for x in derived]
            p = [math.exp(x) for x in log_p]
            nll = -math.fsum(t * lp for t, lp in zip(q, log_p))
            item.update(status='supported_complete_field', argmax_index=best,
                        argmax_target_mass=q[best], NLL=nll,
                        normalized_NLL=nll / math.log(len(q)) if len(q) > 1 else 0.0,
                        Brier_sum=math.fsum((v - t) ** 2 for v, t in zip(p, q)), margins=derived,
                        logits=logits, native_target_not_renormalized=q,
                        original_pair_lengths=raw['pair_lengths'])
        compare_tree(stored, item, 'Independent field recomputation')
        items.append(item)
    # Deliberately do not call the GPU loop's case_result/summarize aggregators.
    # A denominator/coverage bug there must not automatically reproduce here.
    supported = [item for item in items if item['status'] == 'supported_complete_field']
    known_count = len([raw for raw in case['fields'] if raw['known']])
    support_count = len(supported)
    conditional = {}
    for target, source in (('supported_known_field_argmax_target_mass', 'argmax_target_mass'),
                           ('supported_known_field_NLL', 'NLL'),
                           ('supported_known_field_normalized_NLL', 'normalized_NLL'),
                           ('supported_known_field_Brier_sum', 'Brier_sum')):
        conditional[target] = math.fsum(item[source] for item in supported) / support_count if support_count else None
    exact_supported = case['complete_case_exact_static_supported']
    if exact_supported:
        need(type(exact_supported) is bool and known_count == support_count == len(case['fields']) and
             all(raw['onehot'] and sum(t == 1.0 for t in raw['target']) == 1 and
                 all(t in (0.0, 1.0) for t in raw['target']) for raw in case['fields']),
             'Independent onehot exact eligibility differs')
    rebuilt = {'case_id': case['case_id'], 'component': case['component'],
               'primary_domain': case['primary_domain'], 'known_fields': known_count,
               'supported_known_fields': support_count, 'unknown_fields': len(items) - known_count,
               'known_field_coverage': support_count / known_count,
               'coverage_adjusted_argmax_target_mass': math.fsum(item['argmax_target_mass'] for item in supported) / known_count,
               **conditional, 'complete_onehot_case_exact_supported': bool(exact_supported),
               'complete_onehot_case_exact': (all(raw['target'][item['argmax_index']] == 1.0
                                                 for raw, item in zip(case['fields'], items))
                                                if exact_supported else None),
               'fields': items}
    compare_tree(observed, rebuilt, 'Independent complete case recomputation')
    return rebuilt


def independent_summary(records):
    """Second case/domain/coverage implementation; never calls GPU summarizers."""
    need(len(records) == 250 and len({r['case_id'] for r in records}) ==
         len({r['component'] for r in records}) == 250, 'Independent full case/component roster differs')
    per_domain, secondary = {}, {}
    for domain in DOMAINS:
        cases = [r for r in records if r['primary_domain'] == domain]
        need(len(cases) == 50, 'Independent primary-domain quota differs')
        metrics = {'cases': len(cases)}
        for metric in METRICS:
            observed = [r[metric] for r in cases if r[metric] is not None]
            metrics[metric] = math.fsum(observed) / len(observed) if observed else None
            metrics[metric + '_case_denominator'] = len(observed)
        per_domain[domain] = metrics
        secondary_rows = []
        for case in records:
            known = [f for f in case['fields'] if f['known'] and f['domain'] == domain]
            if not known:
                continue
            support = [f for f in known if f['status'] == 'supported_complete_field']
            row = {'known_field_coverage': len(support) / len(known)}
            for metric in ('argmax_target_mass', 'NLL', 'Brier_sum'):
                row[metric] = math.fsum(f[metric] for f in support) / len(support) if support else None
            secondary_rows.append(row)
        values = {'cases_with_known_field_in_domain': len(secondary_rows)}
        for metric in ('argmax_target_mass', 'NLL', 'Brier_sum', 'known_field_coverage'):
            observed = [r[metric] for r in secondary_rows if r[metric] is not None]
            values[metric] = math.fsum(observed) / len(observed) if observed else None
            values[metric + '_case_denominator'] = len(observed)
        secondary[domain] = values
    fields = [f for r in records for f in r['fields']]
    support = [f for f in fields if f['status'] == 'supported_complete_field']
    coverage = {'cases': len(records), 'known_fields': sum(f['known'] for f in fields),
                'supported_known_fields': len(support),
                'cases_with_any_supported_known_field': sum(r['supported_known_fields'] > 0 for r in records),
                'cases_with_all_known_fields_supported': sum(r['supported_known_fields'] == r['known_fields'] for r in records),
                'complete_onehot_case_exact_supported': sum(r['complete_onehot_case_exact_supported'] for r in records),
                'unknown_fields_retained_no_forward': sum(not f['known'] for f in fields),
                'static_unsupported_known_fields': sum(f['known'] and f['status'] == 'static_unsupported' for f in fields)}
    if support:
        coverage['forwarded_complete_candidates'] = sum(f['K'] for f in support)
        coverage['forwarded_real_pair_tokens'] = sum(sum(f['original_pair_lengths']) for f in support)
    means = {}
    for metric in METRICS:
        values = [per_domain[d][metric] for d in DOMAINS]
        means[metric] = math.fsum(values) / 5 if all(v is not None for v in values) else None
    exact_values = [r['complete_onehot_case_exact'] for r in records if r['complete_onehot_case_exact_supported']]
    return {'roster': 'D_confirm', 'native_split': 'validation',
            'statistical_unit': 'complete native case; equal mean of five primary domains',
            'primary_metric': 'supported_known_field_argmax_target_mass',
            'coverage_sensitivity_metric': 'coverage_adjusted_argmax_target_mass',
            'five_domain_case_means': means, 'per_domain': per_domain, 'coverage': coverage,
            'field_domain_case_means_secondary_not_selection_metric': secondary,
            'complete_onehot_case_exact': sum(exact_values) / len(exact_values) if exact_values else None,
            'complete_onehot_case_exact_denominator': len(exact_values),
            'coverage_complete': coverage['known_fields'] == coverage['supported_known_fields'],
            'q_renormalized': False, 'unknown_forwarded': False, 'candidate_pruning': False,
            'static_missing_definition': 'all cases retained; conditional metrics plus zero-credit known denominator',
            'Brier_definition': 'sum over complete native candidates of (softmax(m)-native q)^2',
            'NLL_definition': '-sum(native q*log_softmax(m)); normalized by log(K), K1=0',
            'public_list_changed': False, 'winner_selected': False}


def quantile(sorted_values, p):
    index = (len(sorted_values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def confirm_analysis(guard):
    grant(guard, 'confirm-analysis')
    codec_receipt(guard)
    inputs, summaries, records = [], {}, {}
    for role in ROLES:
        inputs.append(child_exit('confirm', role))
        output = output_for('confirm', role)
        done = read(output / 'COMPLETE.json')
        need(done['complete'] is True and done['role'] == role and done['cases'] == 250,
             'Complete own confirm receipt required')
        verify(done['files'])
        same(done['source_files'], guard.cfg['source_files'], 'Confirm source differs')
        same(done['frozen_candidates'], guard.frozen_pin, 'Confirm selected predictor changed')
        proof = read(output / 'WEIGHTS.json')
        weights_ok(proof)
        stored = list(jsonl(output / 'case-metrics.jsonl'))
        path = Path(guard.cfg['data_root']) / 'data' / 'confirm.jsonl'
        data_pin = pin(path)
        rebuilt = []
        iterator = iter(stored)
        for case in jsonl(path):
            row = next(iterator, None)
            need(row is not None and row['case_id'] == case['case_id'], 'Independent native roster order differs')
            rebuilt.append(independent_case(case, row))
        need(next(iterator, None) is None, 'Extra confirm case outputs')
        verify([data_pin])
        summary = independent_summary(rebuilt)
        observed_summary = read(output / 'SUMMARY.json')
        compare_tree({k: observed_summary[k] for k in summary}, summary, 'Independent domain summary')
        same(observed_summary['input'], data_pin, 'Confirm input identity differs')
        same(observed_summary['model'], guard.producer(role)['response_name'], 'Confirm model name differs')
        same(observed_summary['terminal'], guard.producer(role)['terminal'], 'Confirm terminal differs')
        same(observed_summary['runtime_after'], proof['runtime'], 'Post-proof counters imply extra/missing forwards')
        same(done['coverage_complete'], summary['coverage_complete'], 'Confirm completion coverage differs')
        summary.update(role=role, model=guard.producer(role)['response_name'],
                       terminal=guard.producer(role)['terminal'], input=data_pin,
                       pure_after_weights=pin(output / 'WEIGHTS.json'), runtime=proof['runtime'])
        records[role], summaries[role] = rebuilt, summary
        inputs.extend([pin(output / 'COMPLETE.json'), *done['files'], data_pin])
    baseline, challenger = (records[r] for r in ROLES)
    need([r['case_id'] for r in baseline] == [r['case_id'] for r in challenger], 'Paired case roster differs')
    deltas = {d: [] for d in DOMAINS}
    for left, right in zip(baseline, challenger):
        need(left['primary_domain'] == right['primary_domain'], 'Paired native domain differs')
        # Zero-credit known-denominator sensitivity remains defined if support is incomplete.
        deltas[left['primary_domain']].append(right['coverage_adjusted_argmax_target_mass'] -
                                             left['coverage_adjusted_argmax_target_mass'])
    need(all(len(v) == 50 for v in deltas.values()), 'Bootstrap must retain five native 50-case strata')
    point = average([average(deltas[d]) for d in DOMAINS])
    rng = random.Random(20261112)
    boot = []
    for _ in range(10000):
        boot.append(average([average([deltas[d][rng.randrange(50)] for _ in range(50)]) for d in DOMAINS]))
    boot.sort()
    interval = [quantile(boot, 0.025), quantile(boot, 0.975)]
    complete_coverage = all(summaries[r]['coverage_complete'] for r in ROLES)
    verify(inputs)
    guard.check()
    need('torch' not in sys.modules, 'Independent CPU analysis must not import Torch')
    commit(ROOT / 'CONFIRM_ANALYSIS.json', {'complete': True, 'roles': summaries,
           'paired_mass_delta': point, 'descriptive_95pct_case_interval': interval,
           'paired_mass_definition': 'five-domain mean case zero-credit known-denominator mass; equals primary when coverage=1',
           'coverage_complete': complete_coverage,
           'improvement_confirmed': bool(complete_coverage and point > 0 and interval[0] > 0),
           'bootstrap_replicates': 10000, 'bootstrap_seed': 20261112,
           'percentile_method': 'linear interpolation at (replicates-1)*p for p=.025,.975',
           'bootstrap_unit': 'paired whole native case, 50 resampled within each of five primary domains',
           'interval_scope': 'descriptive fixed contrast, not seed/generalization/pretraining-exposure proof',
           'input_pins': inputs, 'source_files': guard.cfg['source_files'],
           'frozen_candidates': guard.frozen_pin, 'public_roles': list(ROLES),
           'public_list_changed': False, 'winner_selected': False, 'model_forward_calls': 0})


def sdk(guard):
    guard.check()
    root = str(Path(guard.cfg['sdk_root']).resolve())
    need(root == guard.cfg['sdk_root'], 'Canonical SDK root required')
    if root not in sys.path:
        sys.path.insert(0, root)
    from decision_index.suite.io import Suite, dumps
    import decision_index.suite.io as io
    actual = pin(io.__file__)
    bound = [p for p in guard.cfg['sdk_files'] if p['resolved'] == actual['resolved']]
    need(bound and all(actual['bytes'] == p['bytes'] and actual['mtime_ns'] == p['mtime_ns'] for p in bound) and
         Path(actual['resolved']).is_relative_to(Path(root)), 'Wrong SDK Suite module')
    verify(bound)  # Keep alias→resolved identity, not a demand that __file__ use the alias.
    return Suite, dumps


def analysis_receipt(guard):
    value = read(ROOT / 'CONFIRM_ANALYSIS.json')
    need(value['complete'] is True and value['public_list_changed'] is False and
         value['public_roles'] == list(ROLES) and value['model_forward_calls'] == 0,
         'Independent completed confirm analysis required; significance is not a list selector')
    same(value['source_files'], guard.cfg['source_files'], 'Analysis source changed')
    same(value['frozen_candidates'], guard.frozen_pin, 'Analysis frozen candidates differ')
    verify(value['input_pins'])
    return pin(ROOT / 'CONFIRM_ANALYSIS.json')


def public_prepare(guard):
    grant(guard, 'public-prepare')
    analysis_pin = analysis_receipt(guard)
    child_exit('confirm-analysis')
    Suite, dumps = sdk(guard)
    suite = Suite(guard.cfg['suite_root'], '0.3')
    official = suite.verify(strict=True)
    need(official['match'] is True and official.get('exclusions_match') is True,
         'Strict official verification AND exclusions match required')
    exclusions = suite.excluded()
    need(len(exclusions) == 442, 'Official exclusion count differs')
    corpus = ROOT / 'public' / 'corpus'
    absent(corpus)
    corpus.mkdir(mode=0o700)
    seen, counts, excluded_seen = set(), Counter(), set()
    with ExitStack() as stack:
        handles = [stack.enter_context((corpus / ('shard-' + str(i) + '.jsonl')).open('x', encoding='utf-8'))
                   for i in range(4)]
        for i, row in enumerate(suite.rows(apply_exclusions=False)):
            rid = row['_evaluation']['run_id']
            need(rid not in seen, 'Duplicate official request ID')
            need('state' in row and isinstance(row['questions'], dict) and row['questions'], 'Invalid official payload')
            seen.add(rid)
            if rid in exclusions:
                excluded_seen.add(rid)
            rank = i % 4
            handles[rank].write(dumps(row) + '\n')
            counts[rank] += 1
        for handle in handles:
            flush(handle, readonly=True)
    need(len(seen) == 140620 and counts == Counter({i: 35155 for i in range(4)}) and
         excluded_seen == exclusions, 'Full official corpus/shards/exclusions required')
    guard.check()
    shard_pins = durable_files([corpus / ('shard-' + str(i) + '.jsonl') for i in range(4)])
    commit(ROOT / 'PUBLIC_READY.json', {'complete': True, 'executed_requests': 140620,
           'scored_requests': 140178, 'excluded_requests': 442, 'official_verification': official,
           'official_sha256': official['sha256'], 'confirm_analysis': analysis_pin,
           'shards': [{'rank': i, 'rows': shard_pins[i],
                       'requests': counts[i]} for i in range(4)],
           'source_files': guard.cfg['source_files'], 'frozen_candidates': guard.frozen_pin,
           'model_forward_calls': 0, 'token_precomputation': False,
           'dependencies_fsynced_before_pins_and_complete': True,
           'public_roles': list(ROLES), 'public_list_changed': False})


def ready_receipt(guard):
    value = read(ROOT / 'PUBLIC_READY.json')
    need(value['complete'] is True and value['executed_requests'] == 140620 and
         value['scored_requests'] == 140178 and value['public_roles'] == list(ROLES) and
         value['public_list_changed'] is False and value['official_verification']['match'] is True and
         value['official_verification']['exclusions_match'] is True, 'Full strict Suite preparation required')
    same(value['source_files'], guard.cfg['source_files'], 'Public source differs')
    same(value['frozen_candidates'], guard.frozen_pin, 'Public frozen candidates differ')
    need([s['rank'] for s in value['shards']] == list(range(4)) and
         all(s['requests'] == 35155 for s in value['shards']), 'Exact four official shards required')
    verify([s['rows'] for s in value['shards']])
    return value


def check_prediction(row, result, model_name, validate):
    evaluation = row['_evaluation']
    expected_keys = set(evaluation) | {'started_utc', 'engine', 'status', 'payload', 'response', 'raw_output',
                                       'completed_utc', 'total_wall_ms', 'model_request_wall_ms'}
    need(set(result) == expected_keys, 'Unexpected/missing SDK result metadata')
    for name, value in evaluation.items():
        need(name in result and ordered(result[name]) == ordered(value), 'Result evaluation metadata differs: ' + name)
    need(result['engine'] == ENGINE and result['status'] == 'ok', 'Every request must use own engine and be ok')
    need(ordered(result['payload']) == ordered({'state': row['state'], 'questions': row['questions']}),
         'Result actual payload/criteria order differs from original Suite')
    need(result['response']['model'] == model_name, 'Foreign model prediction')
    tokens = result['response']['usage']['input_tokens']
    need(type(tokens) is int and tokens > 0, 'Missing actual full-pair input token usage')
    validate(row['questions'], result['response'])
    need(result.get('raw_output') is None, 'Unexpected private backend raw output')
    for name in ('total_wall_ms', 'model_request_wall_ms'):
        need(type(result[name]) in (float, int) and math.isfinite(result[name]) and result[name] >= 0,
             'Missing actual request timing')


def public_worker(guard, role, rank):
    grant(guard, 'public-worker', role, rank)
    codec_receipt(guard)
    analysis_receipt(guard)
    ready = ready_receipt(guard)
    child_exit('public-prepare')
    shard = ready['shards'][rank]
    expected = ROOT / 'public' / 'corpus' / ('shard-' + str(rank) + '.jsonl')
    need(shard['rows']['path'] == str(expected), 'Official shard path redirected')
    output = output_for('public-worker', role, rank)
    absent(output)
    directory(output.parent)
    output.mkdir(mode=0o700)
    # SDK run imports Torch before engine construction. This order is mandatory.
    from backend import configure_role
    need('torch' not in sys.modules, 'Configure role BEFORE SDK.run Torch import')
    configure_role(guard, role)
    sdk(guard)
    from decision_index.runner import run
    from decision_index.engines import validate

    def log(message):
        # Frozen SDK ordinary progress is emitted every ten rows. A non-ok
        # request can therefore be followed by up to nine more requests before
        # this event-level abort (device errors/SDK failure budget may stop
        # sooner). This is NOT first-error-before-next-forward fail-fast and
        # never retries the failure; COMPLETE still requires every row ok.
        guard.check()
        record = json.loads(message)
        counts = record.get('counts', {})
        need(all(name == 'ok' or number == 0 for name, number in counts.items()) and
             record.get('event') != 'failed', 'Non-ok SDK event detected; partial preserved, no retry/fallback')
        print(message, flush=True)

    started = time.monotonic()
    absent(output / 'results.jsonl')  # SDK itself uses append even with resume=False.
    final = run(ENGINE, {'config': str(ROOT / 'config.json'), 'role': role, 'output': str(output)},
                str(expected), str(output), resume=False, warm=False, limit=None, keep=None,
                corpus_sha256=ready['official_sha256'], halt_on_device_error=True, log=log)
    need(final['event'] == 'complete' and final['completed'] == 35155 and
         final['counts'] == {'ok': 35155}, 'Full own SDK shard completion required')
    weights = read(output / 'WEIGHTS.json')
    weights_ok(weights)
    model_name = guard.producer(role)['response_name']
    environment = read(output / 'environment.json')
    need(environment['engine'] == ENGINE and environment['rows_path'] == str(expected) and
         environment['frozen_corpus_sha256'] == ready['official_sha256'], 'Wrong SDK engine/shard/corpus provenance')
    same(environment['engine_options'], {'config': str(ROOT / 'config.json'), 'role': role, 'output': str(output)},
         'SDK engine options changed')
    seen, latencies, pair_tokens = set(), [], 0
    results = iter(jsonl(output / 'results.jsonl'))
    for row in jsonl(expected):
        result = next(results, None)
        need(result is not None, 'Missing own SDK request')
        check_prediction(row, result, model_name, validate)
        rid = result['run_id']
        need(rid not in seen, 'Duplicate SDK request')
        seen.add(rid)
        latencies.append(result['model_request_wall_ms'])
        pair_tokens += result['response']['usage']['input_tokens']
    need(next(results, None) is None and len(seen) == 35155, 'Extra/incomplete SDK shard')
    guard.check()
    files = durable_files([output / n for n in ('results.jsonl', 'WEIGHTS.json', 'environment.json')])
    commit(output / 'COMPLETE.json', {'complete': True, 'role': role, 'rank': rank,
           'executed_requests': 35155, 'counts': {'ok': 35155}, 'engine': ENGINE,
           'model': model_name, 'shard': shard['rows'],
           'source_files': guard.cfg['source_files'], 'frozen_candidates': guard.frozen_pin,
           'files': files, 'dependencies_fsynced_before_pins_and_complete': True,
           'wall_seconds': time.monotonic() - started, 'mean_request_ms': average(latencies),
           'p80_request_ms': quantile(sorted(latencies), 0.8), 'runtime': weights['runtime'],
           'actual_response_input_pair_tokens': pair_tokens,
           'resume': False, 'warm': False, 'limit': None, 'keep': None,
           'first_use_cost_in_actual_request': True, 'automatic_retry': False,
           'non_ok_detection': NON_OK_DETECTION})


def public_score(guard, role):
    grant(guard, 'public-score', role)
    ready = ready_receipt(guard)
    analysis_receipt(guard)
    Suite, dumps = sdk(guard)
    from decision_index.engines import validate
    from decision_index.pipeline import score_run
    suite = Suite(guard.cfg['suite_root'], '0.3')
    output = output_for('public-score', role)
    absent(output / 'results.jsonl')
    absent(output / 'COMPLETE.json')
    workers, inputs, exits = [], [], []
    model_name = guard.producer(role)['response_name']
    for rank in range(4):
        exit_pin = child_exit('public-worker', role, rank)
        inputs.append(exit_pin)
        exits.append(read(exit_pin['path']))
        folder = output / 'workers' / str(rank)
        value = read(folder / 'COMPLETE.json')
        need(value['complete'] is True and value['role'] == role and value['rank'] == rank and
             value['executed_requests'] == 35155 and value['counts'] == {'ok': 35155} and
             value['engine'] == ENGINE and value['model'] == model_name,
             'Complete exact own role/rank required')
        same(value['source_files'], guard.cfg['source_files'], 'Worker source differs')
        same(value['frozen_candidates'], guard.frozen_pin, 'Worker predictor differs')
        same(value['shard'], ready['shards'][rank]['rows'], 'Worker original shard source differs')
        weights_ok(read(folder / 'WEIGHTS.json'))
        verify(value['files'])
        workers.append(value)
        inputs.extend([pin(folder / 'COMPLETE.json'), *value['files']])
    streams = [iter(jsonl(output / 'workers' / str(i) / 'results.jsonl')) for i in range(4)]
    shard_streams = [iter(jsonl(ROOT / 'public' / 'corpus' / ('shard-' + str(i) + '.jsonl'))) for i in range(4)]
    seen, latencies, counts, pair_tokens = set(), [], Counter(), 0
    with (output / 'results.jsonl').open('x', encoding='utf-8') as handle:
        for i, row in enumerate(suite.rows(apply_exclusions=False)):
            rank = i % 4
            shard_row = next(shard_streams[rank], None)
            result = next(streams[rank], None)
            need(shard_row is not None and ordered(shard_row) == ordered(row) and result is not None,
                 'Shard is not exact original official round-robin payload/metadata')
            check_prediction(row, result, model_name, validate)
            rid = result['run_id']
            need(rid not in seen, 'Duplicate merged ID')
            seen.add(rid)
            counts[rank] += 1
            latencies.append(result['model_request_wall_ms'])
            pair_tokens += result['response']['usage']['input_tokens']
            handle.write(dumps(result) + '\n')
        need(len(seen) == 140620 and counts == Counter({i: 35155 for i in range(4)}) and
             all(next(s, None) is None for s in streams) and all(next(s, None) is None for s in shard_streams),
             'Strict full merge count/shard EOF required')
        flush(handle, readonly=True)
    # SDK score outputs must also be new; no old prediction/reference cases are passed.
    for name in ('benchmark-summary.json', 'index.json', 'scores.json'):
        absent(output / name)
    score = score_run(suite, output / 'results.jsonl', ENGINE, output, reference_results=None)
    need(score['complete'] is True and score['completed'] == 140178 and
         score['counts'] == {'ok': 140178} and score['edition'] == '0.3',
         'Exactly 140178 all-ok official scored requests required')
    need(type(score['decision_index']) in (float, int) and math.isfinite(score['decision_index']),
         'Official complete score must be finite')
    need(len(suite.excluded()) == 442, 'Official exclusions changed')
    verify(inputs)
    guard.check()
    pipeline_seconds = max(e['wait_monotonic'] for e in exits) - min(e['spawn_monotonic'] for e in exits)
    need(math.isfinite(pipeline_seconds) and pipeline_seconds > 0, 'Missing actual manager pipeline timing')
    files = durable_files([output / n for n in ('results.jsonl', 'scores.json', 'benchmark-summary.json', 'index.json')])
    commit(output / 'COMPLETE.json', {'complete': True, 'role': role, 'engine': ENGINE,
           'model': model_name, 'executed_requests': 140620,
           'scored_requests': 140178, 'excluded_requests': 442, 'counts': {'ok': 140620},
           'source_files': guard.cfg['source_files'], 'frozen_candidates': guard.frozen_pin,
           'worker_inputs': inputs, 'workers': workers, 'decision_index': score['decision_index'],
           'official_scores': files[1], 'latency_ms': score['latency_ms'],
           'all_request_latency_ms': {'mean': average(latencies), 'p80': quantile(sorted(latencies), 0.8)},
           'actual_response_input_pair_tokens': pair_tokens,
           'pipeline': {'gpus': 4, 'max_worker_wall_seconds': max(w['wall_seconds'] for w in workers),
                        'manager_spawn_to_actual_all_wait_seconds': pipeline_seconds,
                        'executed_requests_per_second': 140620 / pipeline_seconds,
                        'scope': 'first actual spawn to last actual wait; includes model load/scheduling/postvalidation, not CPU official scoring',
                        'single_card_latency_admission': False},
           'files': files, 'dependencies_fsynced_before_pins_and_complete': True,
           'non_ok_detection': NON_OK_DETECTION,
           'public_list_changed': False, 'winner_selected': False, 'reference_results': None,
           'full_geometry_or_cache_serving_admission': False, 'published': False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=PHASES)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--rank', type=int, choices=range(4))
    args = parser.parse_args(argv)
    need(args.config == ROOT / 'config.json' and args.config.resolve() == args.config,
         'Only fixed deployed config permitted')
    need(Path(__file__).absolute() == ROOT / 'source' / 'evaluate.py' and
         Path(__file__).resolve() == ROOT / 'source' / 'evaluate.py', 'Only frozen deployed evaluator may execute')
    role_phase = args.phase in ('confirm', 'public-worker', 'public-score')
    need((args.role is not None) == role_phase and (args.rank is not None) == (args.phase == 'public-worker'),
         'Role/rank permitted only for required phase')
    os.umask(0o077)
    guard = Guard(args.config)
    functions = {'codec': codec, 'confirm': confirm, 'confirm-analysis': confirm_analysis,
                 'public-prepare': public_prepare, 'public-worker': public_worker, 'public-score': public_score}
    try:
        if args.phase == 'public-worker':
            functions[args.phase](guard, args.role, args.rank)
        elif role_phase:
            functions[args.phase](guard, args.role)
        else:
            functions[args.phase](guard)
    except BaseException as error:
        # Keep all raw partials remote. No retries or best-effort fabricated completion.
        target = ROOT / 'CODEC_BLOCKED.json' if args.phase == 'codec' else ROOT / (key(args.phase, args.role, args.rank) + '-BLOCKED.json')
        if not target.exists() and not target.is_symlink():
            commit(target, {'complete': False, 'phase': args.phase, 'role': args.role, 'rank': args.rank,
                            'error_type': type(error).__name__, 'error': str(error),
                            'partial_preserved': True, 'automatic_retry': False, 'public_list_changed': False})
        raise


if __name__ == '__main__':
    main()
