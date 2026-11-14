"""Enumerated standalone CPU-analysis port; no trainer/evaluator/model import."""
import ast
from pathlib import Path

ORIGINAL = Path('/tmp/opencode/jev-clean-trainer-ops-20261112-local/verify_select_node.py')


def source():
    text = ORIGINAL.read_text()
    changes = [
        ('import statistics\n', 'import statistics\nimport struct\n'),
        ('whole case, paired across all four arms, stratified by five primary domains', 'whole case, paired across all three mask arms, stratified by five primary domains'),
        ("ARMS = ('C_D_BCE', 'C_P_BCE', 'C_D_CE', 'C_P_CE')", "ARMS = ('C_P_CE', 'BI_P_CE', 'PB_P_CE')"),
        ('out = ROOT / f"D_SELECT_PRIMARY-cap{args.cap}.json"', 'out = Path(\'/tmp/opencode/jev-mask-select-analysis-20261112-v1\') / f"MASK_SELECT_PRIMARY-cap{args.cap}.json"'),
        ('assert all(v == pair[1] - pair[0] for v, pair in zip(m, logits))', "assert all(v == struct.unpack('<f', struct.pack('<f', pair[1] - pair[0]))[0] for v, pair in zip(m, logits))"),
        ("assert before[phase + '.completed_native_FA3_calls'] == 28 * before[phase + '.completed_waves']", "assert before[phase + '.completed_native_FA3_calls'] == 28 * (2 if cfg['arms'][arm]['mode'] == 'prefix_block' else 1) * before[phase + '.completed_waves']"),
        ("'descriptive single-seed intermediate D_select uncertainty only; not multiplicity-adjusted and not training-seed uncertainty'", "'descriptive single-seed scheduled D_select uncertainty only; not multiplicity-adjusted and not training-seed uncertainty'"),
        ("'10M is intermediate; do not change frozen LR/exposure or choose winner yet'", "'10M/30M intermediate and100M final; no public checkpoint selection or protocol adaptation'"),
    ]
    start = text.index('    contrasts = {\n')
    end = text.index('    paired = {}\n', start)
    text = text[:start] + """    contrasts = {
        'bidirectional_minus_causal_CE': {'BI_P_CE': 1, 'C_P_CE': -1},
        'candidate_prefix_block_minus_causal_CE': {'PB_P_CE': 1, 'C_P_CE': -1},
        'candidate_prefix_block_minus_bidirectional_CE': {'PB_P_CE': 1, 'BI_P_CE': -1},
    }
""" + text[end:]
    for old, new in changes:
        if text.count(old) != 1:
            raise RuntimeError('Analysis delta not uniquely applicable: ' + repr(old))
        text = text.replace(old, new, 1)
    text = text.replace(
        '# Saved logits are BF16 values represented exactly as FP32; their\n                # subtraction is exact for this observed finite score range.',
        '# Saved BF16 logits are represented exactly as FP32. Reproduce\n                # actual FP32 subtraction rather than assume FP64 subtraction exactness.')
    ast.parse(text, filename='verify_mask_node.py')
    return text.encode()
