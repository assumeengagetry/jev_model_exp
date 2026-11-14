"""MAIN direct CPU saved-artifact readback, NOT the resume authorization writer."""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path('/tmp/opencode/jev-clean-trainer-20261112-followup-v1')
OLD = Path('/tmp/opencode/jev-clean-trainer-20261112-v2')
ARMS = ('STATE_NL_PB_P_CE', 'JPAIR_P_BCE_RANK')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def pin(path):
    path = Path(path)
    s = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def read(path):
    return json.loads(Path(path).read_text())


def verify(records):
    for record in records:
        need(pin(record['path']) == record, 'Changed input: ' + record['path'])


def ident(pid):
    try:
        v = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return None if v[0] == 'Z' else {'pid': pid, 'starttime': v[19],
                                      'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def canonical(name):
    return name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.').replace(
        '.score.modules_to_save.default.', '.score.')


def main():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only direct readback required')
    need(not (ROOT / 'MAIN_DIRECT_ACTUAL_REVIEW.json').exists(), 'Prior direct review preserved')
    need(not (ROOT / 'AUTHORIZE_RESUME.json').exists() and not list(ROOT.glob('*BLOCKED*')), 'Blocked or already authorized')
    source_pin = pin(Path(__file__))
    ready_pin = pin(ROOT / 'PRIMARY_READY.json')
    ready = read(ROOT / 'PRIMARY_READY.json')
    need(ready['complete'] and set(ready['arms']) == set(ARMS), 'Both actual primaries required')
    launch = read(ROOT / 'QUEUE_LAUNCH.json')
    need(ident(launch['owner']['pid']) == launch['owner'], 'Exact lease-holding queue must be live')
    resources = read(ROOT / 'RESOURCES_READY.json')
    need(resources['leases_held'] and set(resources['devices']) == {'2', '3'}, 'Both actual resource leases required')
    sys.path.insert(0, str(ROOT / 'source'))
    from bindings import Context, write_json
    contexts = {arm: Context(ROOT / 'seed42.json', arm) for arm in ARMS}
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(), 'No CUDA initialization allowed')

    def equal(a, b):
        return (a.shape == b.shape and a.dtype == b.dtype and
                torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))

    def cumulative(values):
        result = [0]
        for value in values:
            result.append(result[-1] + value)
        return result

    fresh = load_file(str(ROOT / 'init-seed42/adapter_model.safetensors'), device='cpu')
    need(len(fresh) == 393, 'Actual fresh393 required')
    all_pins = [source_pin, ready_pin, pin(ROOT / 'seed42.json'), pin(ROOT / 'QUEUE_LAUNCH.json'),
                pin(ROOT / 'RESOURCES_READY.json'), pin(ROOT / 'init-seed42/adapter_model.safetensors')]
    summaries = {}
    for arm in ARMS:
        binding = ready['arms'][arm]
        verify([binding[k] for k in ('report', 'complete', 'pin_file', 'ledger')])
        report = read(binding['report']['path'])
        need(not list((ROOT / 'primary' / arm).glob('BLOCKED*')), 'Primary blocked')
        need(report['complete'] and report['allow_limited_scientific_resume'] and not report['automatic_resume'], 'Primary scope differs')
        need(report['scope'] == binding['scope'] == contexts[arm].snapshot, 'Actual frozen scope differs')
        need(report['cursor'] == {'macro_updates': 1, 'cases': 16, 'logical_pair_tokens': 20947}, 'Cursor1 required')
        verify(report['evidence_pins'])
        folder = ROOT / (arm + '-seed42')
        artifact = folder / 'first-actual-macro'
        done = read(artifact / 'COMPLETE.json')
        verify(done['files'])
        slots = read(artifact / 'SLOTS.json')
        meta = read(artifact / 'ACTUAL_MACRO.json')
        dispatch = read(artifact / 'ACTUAL_NATIVE_DISPATCH.json')
        cache = torch.load(artifact / 'actual-cache-and-coefficients.pt', map_location='cpu', weights_only=True)
        replay = torch.load(artifact / 'actual-replay-logits.pt', map_location='cpu', weights_only=True)
        need(all(equal(cache[k], replay[k]) for k in ('logits', 'margins')), 'Actual cache/replay bytes differ')
        need(len(slots['exposures']) == 16 and len(slots['slots']) == len(cache['margins']) == 60, 'Full actual macro coverage differs')
        need(meta['logical_pair_tokens'] == 20947 and meta['actual_completed_forward_real_tokens_this_macro'] == 41894 and
             meta['actual_completed_backward_real_tokens_this_macro'] == 20947, 'Actual token accounting differs')
        m = cache['margins'].clone().requires_grad_(True)
        terms = []
        for field in slots['fields']:
            v = m[field['start']:field['end']]
            host_q = field['target']
            q = torch.tensor(host_q, dtype=torch.float32)
            if arm == 'JPAIR_P_BCE_RANK':
                positive = [i for i, x in enumerate(host_q) if x == 1]
                negative = [i for i, x in enumerate(host_q) if x == 0]
                rank = (torch.logsumexp(v[positive + negative], 0) - torch.logsumexp(v[positive], 0)
                        if positive and negative else (v * 0.0).sum())
                loss = F.binary_cross_entropy_with_logits(v, q, reduction='mean') + rank
            else:
                loss = -(q * torch.log_softmax(v, 0)).sum()
            terms.append(loss / field['known_fields_in_case'] / 16)
        objective = torch.stack(terms).sum()
        coef, = torch.autograd.grad(objective, m)
        need(equal(coef, cache['coefficients']) and float(objective.detach()) == cache['objective'] == report['independent_objective'],
             'MAIN independent actual loss/coef mismatch')
        if arm == 'JPAIR_P_BCE_RANK':
            control = OLD / 'C_P_BCE-seed42/first-actual-macro'
            old = torch.load(control / 'actual-cache-and-coefficients.pt', map_location='cpu', weights_only=True)
            need(slots == read(control / 'SLOTS.json') and all(equal(old[k], cache[k]) for k in ('logits', 'margins')),
                 'Rank actual input/logits differ from matched control')
        else:
            wanted = {x['case_id'] for x in slots['slots']}
            side = Path(contexts[arm].state_boundary['sidecars']['train']['path'])
            cuts = {}
            with side.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if row['case_id'] in wanted:
                        for field in row['fields']:
                            cuts[(row['case_id'], field['field_index'])] = (field['state_prefix_tokens'], field['coordinate'])
            for slot in slots['slots']:
                cut, coordinate = cuts[(slot['case_id'], slot['field_index'])]
                need(slot['prefix_tokens'] == cut and slot['coordinate'] == coordinate, 'State P is not exact prepared cut')
        expected_calls = 0
        for phase in ('train_first', 'train_replay'):
            for wave_index, micro in enumerate(slots['micros']):
                records = [slots['slots'][i] for i in micro]
                lengths = [x['pair_tokens'] for x in records]
                prefixes = [x['prefix_tokens'] for x in records]
                suffixes = [n - p for n, p in zip(lengths, prefixes)]
                cu, cp, cs = map(cumulative, (lengths, prefixes, suffixes))
                wave = dispatch['waves'][(0 if phase == 'train_first' else len(slots['micros'])) + wave_index]
                need(wave['positions'] == [i for n in lengths for i in range(n)] and wave['batch_cu'] == cu and
                     wave['eos_indices'] == [i - 1 for i in cu[1:]] and wave['eos_token_ids'] == [151645] * len(micro), 'Actual positions/EOS differ')
                partitions = ([(sum(lengths), sum(lengths), cu, cu, max(lengths), max(lengths), True)]
                              if arm == 'JPAIR_P_BCE_RANK' else
                              [(sum(prefixes), sum(prefixes), cp, cp, max(prefixes), max(prefixes), False),
                               (sum(suffixes), sum(lengths), cs, cu, max(suffixes), max(lengths), False)])
                for layer in range(28):
                    for tq, tk, cq, ck, mq, mk, causal in partitions:
                        call = dispatch['calls'][expected_calls]
                        need(call['completed'] and call['q_shape'] == [tq, 16, 128] and call['k_shape'] == call['v_shape'] == [tk, 8, 128] and
                             call['cu_q'] == cq and call['cu_k'] == ck and call['max_q'] == mq and call['max_k'] == mk and
                             call['kwargs'] == {'softmax_scale': 128**-0.5, 'causal': causal, 'deterministic': True}, 'MAIN actual native partition differs')
                        expected_calls += 1
        need(expected_calls == len(dispatch['calls']), 'Native call coverage differs')
        raw = torch.load(artifact / 'all393-gradients-unclipped.pt', map_location='cpu', weights_only=True)
        clipped = torch.load(artifact / 'all393-gradients-clipped.pt', map_location='cpu', weights_only=True)
        actual = torch.load(artifact / 'all393-actual-update-and-optimizer.pt', map_location='cpu', weights_only=True)
        need(len(raw) == len(clipped) == len(actual['before']) == len(actual['after']) == len(actual['delta']) == 393, '393 groups required')
        need(set(raw) == set(clipped) and set(actual['before']) == set(actual['after']) == set(actual['delta']) == set(fresh), 'Parameter key set differs')
        clip = (1.0 / (torch.tensor(meta['preclip_l2'], dtype=torch.float32) + 1e-6)).clamp(max=1.0)
        for name in raw:
            need(torch.isfinite(raw[name]).all() and torch.isfinite(clipped[name]).all() and equal(clipped[name], raw[name] * clip), 'Actual raw/clipped gradient mismatch')
            key = canonical(name)
            need(equal(actual['before'][key], fresh[key]) and equal(actual['after'][key] - actual['before'][key], actual['delta'][key]), 'Actual fresh/delta mismatch')
        need(len(actual['optimizer']['state']) == 393 and actual['scheduler']['completed'] == 1, 'Adam/scheduler scope differs')
        for state in actual['optimizer']['state'].values():
            need(float(state['step']) == 1 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all(), 'Actual Adam state differs')
        checkpoint = Path(binding['checkpoint'])
        saved = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
        need(set(saved) == set(fresh) and all(equal(saved[k], actual['after'][k]) for k in saved), 'Latest checkpoint weights differ')
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
        need(state['cursor'] == report['cursor'] and state['scheduler'] == actual['scheduler'] and state['scope'] == contexts[arm].snapshot,
             'Latest actual training state differs')
        events = [json.loads(line) for line in (folder / 'ledger.jsonl').read_text().splitlines()]
        need(sum(x['event'] == 'macro_started' for x in events) == sum(x['event'] == 'macro_updated' for x in events) == 1 and
             events[-1]['event'] == 'paused' and events[-1]['exit_code'] == 75, 'Formal macro must occur exactly once and remain paused')
        need([x for x in events if x['event'] == 'checkpoint_committed'][-1]['complete'] == binding['complete'], 'Not the latest checkpoint')
        verify(report['evidence_pins'])
        all_pins.extend(report['evidence_pins'])
        all_pins.extend(binding[k] for k in ('report', 'complete', 'pin_file', 'ledger'))
        summaries[arm] = {'actual_loss': cache['objective'], 'actual_native_calls': expected_calls,
            'actual_replay_bytes_equal': True, 'MAIN_independent_actual_coef_bytes_equal': True,
            '393_raw_clipped_before_after_delta_and_Adam_read_directly': True,
            'actual_fresh_and_latest_checkpoint_bytes_equal': True, 'macro0_consumed_once': True,
            'cursor': report['cursor'], 'preclip_l2': meta['preclip_l2'], 'used_lr': meta['used_lr'],
            'scalar_book': report['scalar_book'], 'new_prefix_numeric_backend_or_neighbor_equivalence_admitted': False}
        del raw, clipped, actual, saved, state, cache, replay
    for ctx in contexts.values():
        ctx.check()
    verify(all_pins)
    need(ident(launch['owner']['pid']) == launch['owner'] and not (ROOT / 'AUTHORIZE_RESUME.json').exists() and
         not list(ROOT.glob('*BLOCKED*')) and not torch.cuda.is_initialized(), 'Live unapproved scope changed')
    result = {'complete': True, 'version': 'main-direct-formal-macro-readback-v1', 'primary_ready': ready_pin,
        'analysis_source': source_pin, 'queue_owner': launch['owner'], 'arms': summaries,
        'evidence_pins': list({v['path']: v for v in all_pins}.values()),
        'next_macro_index': 1, 'repeat_macro0_allowed': False, 'authorization_created': False,
        'model_forward_calls': 0, 'optimizer_steps': 0, 'D_confirm_opened': False, 'public_run': False}
    write_json(ROOT / 'MAIN_DIRECT_ACTUAL_REVIEW.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence_pins'}, indent=2))


if __name__ == '__main__':
    main()
