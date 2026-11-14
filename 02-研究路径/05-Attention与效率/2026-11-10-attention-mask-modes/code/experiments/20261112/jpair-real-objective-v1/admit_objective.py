"""New objective on a complete actual saved macro, never a model/synthetic test."""
import base64
import json
import os
from pathlib import Path
import sys


ROOT = Path('/tmp/opencode/jev-jpair-objective-real-macro-20261112-v1')
OLD = Path('/tmp/opencode/jev-clean-trainer-20261112-v2/C_P_BCE-seed42/first-actual-macro')


def pin(path):
    path = Path(path)
    value = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve()), 'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU saved-only mechanism required')
    sys.path.insert(0, str(ROOT / 'source'))
    import torch
    import torch.nn.functional as F
    from objectives import coefficients
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    done = read(OLD / 'COMPLETE.json')
    assert done['complete'] and done['is_formal_exposure_macro0']
    for item in done['files']:
        assert pin(item['path']) == item
    slots = read(OLD / 'SLOTS.json')
    assert len(slots['exposures']) == 16 and slots['macro_index'] == 0
    actual = torch.load(OLD / 'actual-cache-and-coefficients.pt', map_location='cpu', weights_only=True)
    margins = actual['margins']
    coefficient, value, book = coefficients(margins, slots['fields'], 16, 'BCE_RANK_HARD')
    m = margins.clone().requires_grad_(True)
    total, bs, rs = [], [], []
    pos_count = neg_count = soft_count = active_fields = 0
    for field in slots['fields']:
        q_host = field['target']
        pos = [i for i, q in enumerate(q_host) if q == 1]
        neg = [i for i, q in enumerate(q_host) if q == 0]
        segment = m[field['start']:field['end']]
        target = torch.tensor(q_host, dtype=torch.float32)
        b = F.binary_cross_entropy_with_logits(segment, target, reduction='mean')
        r = (torch.logsumexp(segment[pos + neg], 0) - torch.logsumexp(segment[pos], 0)
             if pos and neg else (segment * 0.0).sum())
        divisor = field['known_fields_in_case']
        total.append((b + r) / divisor / 16)
        bs.append(b / divisor / 16)
        rs.append(r / divisor / 16)
        pos_count += len(pos)
        neg_count += len(neg)
        soft_count += len(q_host) - len(pos) - len(neg)
        active_fields += bool(pos and neg)
    loss = torch.stack(total).sum()
    independent_coef, = torch.autograd.grad(loss, m)
    assert torch.equal(independent_coef.view(torch.uint8), coefficient.view(torch.uint8))
    assert float(loss.detach()) == value == book['macro_total_loss']
    assert float(torch.stack(bs).sum().detach()) == book['macro_bce_loss']
    assert float(torch.stack(rs).sum().detach()) == book['macro_rank_loss']
    assert (pos_count, neg_count, soft_count, active_fields) == tuple(book[k] for k in
        ('hard_positive_candidates', 'hard_negative_candidates', 'soft_candidates', 'rank_active_fields'))
    assert len(margins) == len(slots['slots']) and sorted(s['slot'] for s in slots['slots']) == list(range(len(margins)))
    assert not torch.cuda.is_initialized()
    sources = [pin(p) for p in sorted((ROOT / 'source').iterdir())]
    for item in done['files']:
        assert pin(item['path']) == item
    result = {'complete': True, 'scope': 'new-objective-on-complete-actual-control-macro0-saved-margins-only',
        'source_files': sources, 'source_bytes': {p.name: base64.b64encode(p.read_bytes()).decode() for p in (ROOT / 'source').iterdir()},
        'actual_saved_control_macro': pin(OLD / 'COMPLETE.json'), 'cases': 16, 'known_fields': len(slots['fields']),
        'candidates': len(margins), 'independent_loss_and_total_coef_bitwise_equal': True, 'bookkeeping': book,
        'model_forward_calls': 0, 'optimizer_steps': 0, 'new_training_exposure_consumed': False,
        'formal_new_arm_macro0_admitted': False, 'D_confirm_opened': False, 'public_run': False,
        'remaining_runtime_gaps': ['new-arm fresh identity and actual formal macro0/replay/393gradient/actualupdate',
            'unobserved target hard/soft/zero-rank branches are not covered by this saved macro']}
    with (ROOT / 'COMPLETE.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_bytes', 'source_files')}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        with (ROOT / 'BLOCKED.json').open('x') as handle:
            json.dump({'complete': False, 'error': str(error), 'partial_preserved': True}, handle)
        raise
