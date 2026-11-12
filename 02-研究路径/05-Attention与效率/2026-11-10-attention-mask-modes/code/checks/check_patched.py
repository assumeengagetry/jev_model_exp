"""CPU check of the patched collate() and the patched inference engine (tiny random Qwen3, SDPA)."""
import ast
import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

code = Path(__file__).resolve().parents[1]                         # attention_modes.py, baseline/, patched/
sys.path[:0] = [str(code), str(Path(__file__).resolve().parent / 'stubs')]

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, Qwen3Config, Qwen3ForSequenceClassification

import attention_modes as am

PAD, EOS, VOCAB = 0, 1, 64
RESULTS = []


def check(name, ok):
    RESULTS.append(bool(ok))
    print(('PASS  ' if ok else 'FAIL  ') + name)


# ---- collate ------------------------------------------------------------------------------------

def runtime_functions(path, pinned):
    source = path.read_text()
    if not pinned:
        source = source.replace(',pin_memory=True', '').replace('.pin_memory()', '')
    tree = ast.parse(source)
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('clip_tokens', 'assemble', 'collate')]
    assert len(wanted) == 3
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Store:
    def __init__(self, rng):
        self.rows = {'prefix': [rng.integers(2, VOCAB, n).astype(np.int32) for n in (5, 40, 3000, 9, 1500)],
                     'suffix': [rng.integers(2, VOCAB, n).astype(np.int32) for n in (3, 700, 1, 2500, 0)]}

    def get(self, kind, index):
        return self.rows[kind][index]


store = Store(np.random.default_rng(0))
batch = {'state': np.array([0, 1, 2, 3, 4, 2]), 'candidate': np.array([0, 1, 2, 3, 4, 3]),
         'max_length': np.array([64, 256, 512, 512, 2048, 96]), 'target': np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)}
device = torch.device('cpu')
pinned = True
try:
    torch.zeros(1, pin_memory=True)
except RuntimeError:
    pinned = False
print('pin_memory available here:', pinned)
before, after = (runtime_functions(code / folder / 'pair_training_runtime.py', pinned) for folder in ('baseline', 'patched'))

old = before['collate'](store, batch, PAD, EOS, device)
new = after['collate'](store, batch, PAD, EOS, device)
check('collate() default return is unchanged (3 values, same tensors)',
      len(new) == 3 and all(torch.equal(old[0][k], new[0][k]) for k in old[0]) and torch.equal(old[1], new[1]) and old[2] == new[2])
check('assemble() default return is unchanged (plain id array)',
      all(np.array_equal(before['assemble'](store, s, c, int(m), EOS), after['assemble'](store, s, c, int(m), EOS))
          for s, c, m in zip(batch['state'], batch['candidate'], batch['max_length'])))
inputs, targets, count, (prefix_len, row_len) = after['collate'](store, batch, PAD, EOS, device, layout=True)
check('collate(layout=True) keeps inputs/targets/count', all(torch.equal(old[0][k], inputs[k]) for k in old[0]) and torch.equal(old[1], targets) and old[2] == count)
check('layout tensors are int32, one entry per row', prefix_len.dtype == row_len.dtype == torch.int32 and len(prefix_len) == len(row_len) == len(batch['state']))
check('row_len equals the number of real tokens', torch.equal(row_len.long(), inputs['attention_mask'].sum(1)))
check('the last real token of every row is EOS', all(int(inputs['input_ids'][i, n - 1]) == EOS for i, n in enumerate(row_len.tolist())))
clipped = []
for i, (s, c, m) in enumerate(zip(batch['state'], batch['candidate'], batch['max_length'])):
    prefix, suffix = store.get('prefix', int(s)), store.get('suffix', int(c))
    reserve = min(len(prefix), min(1024, int(m) // 2))
    suffix = before['clip_tokens'](suffix, int(m) - 1 - reserve)
    prefix = before['clip_tokens'](prefix, int(m) - 1 - len(suffix))
    row = inputs['input_ids'][i].numpy()
    clipped.append(int(prefix_len[i]) == len(prefix) and np.array_equal(row[:len(prefix)], prefix)
                   and np.array_equal(row[len(prefix):len(prefix) + len(suffix)], suffix) and int(row_len[i]) == len(prefix) + len(suffix) + 1)
check('prefix_len is the boundary of the prefix after clipping (incl. clipped rows and an empty candidate)', all(clipped))
check('some test rows really are clipped', any(len(store.get('prefix', int(s))) > int(p) for s, p in zip(batch['state'], prefix_len)))

# ---- inference engine ---------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location('decision_engine_patched', code / 'patched' / 'decision_engine.py')
engine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_module)

torch.manual_seed(0)
config = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, max_position_embeddings=512, pad_token_id=PAD, eos_token_id=EOS, num_labels=2)
with tempfile.TemporaryDirectory() as folder:
    Qwen3ForSequenceClassification(config).save_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder, attn_implementation='sdpa').double().eval()


class Tokenizer:
    def __call__(self, value, add_special_tokens=False, truncation=False):
        return {'input_ids': [2 + ord(c) % (VOCAB - 2) for c in value]}


def engine_for(mode, max_pairs):
    engine = object.__new__(engine_module.PairClassifierEngine)
    engine.device, engine.max_pairs, engine.token_budget = torch.device('cpu'), max_pairs, 10 ** 6
    engine.model, engine.context, engine.pad, engine.eos = model, 512, PAD, EOS
    engine.tokenizer, engine.attention = Tokenizer(), mode
    return engine


@torch.inference_mode()
def reference(mode, rows, prefixes):
    """One unpadded row at a time, mask built directly for that row."""
    margins = []
    for row, prefix in zip(rows, prefixes):
        ids = torch.tensor([row])
        if mode == 'causal':
            mask = torch.ones_like(ids, dtype=torch.bool)
        else:
            mask = am.dense_mask(mode, torch.tensor([len(row)]), len(row), torch.tensor([prefix]))
        logits = model(input_ids=ids, attention_mask=mask).logits[0]
        margins.append(float(logits[1] - logits[0]))
    return np.asarray(margins)


generator = torch.Generator().manual_seed(1)
shapes = [(30, 9), (30, 2), (30, 17), (6, 4), (6, 25), (6, 1), (14, 11)]          # long rows first, so sorting reorders them
prefix_tokens = {n: torch.randint(2, VOCAB, (n,), generator=generator).tolist() for n in (30, 6, 14)}
rows = [tuple(prefix_tokens[p] + torch.randint(2, VOCAB, (c,), generator=generator).tolist() + [EOS]) for p, c in shapes]
prefixes = [p for p, _ in shapes]
lengths = [len(row) for row in rows]
check('test rows are reordered by the length sort', np.argsort(lengths, kind='stable').tolist() != list(range(len(rows))))

warnings.filterwarnings('ignore', message='.*CUDA is not available.*')
margins = {}
for mode in am.MODES:
    expected = reference(mode, rows, prefixes)
    for max_pairs in (3, 32):
        got, count = engine_for(mode, max_pairs).score(rows, prefixes)
        check(f'engine.score {mode}, batches of {max_pairs}: equals single-row reference', np.abs(got - expected).max() < 1e-6 and count == sum(lengths))    # the engine subtracts float32 logits
    margins[mode] = expected
gaps = [np.abs(margins[a] - margins[b]).max() for a, b in (('causal', 'bidirectional'), ('causal', 'prefix_block'), ('bidirectional', 'prefix_block'))]
check(f'the three modes give different margins (smallest gap {min(gaps):.3g})', min(gaps) > 1e-3)
shifted = prefixes[1:] + prefixes[:1]
gap = np.abs(reference('prefix_block', rows, shifted) - margins['prefix_block']).max()
check(f'prefix_block margins depend on which boundary goes with which row (gap {gap:.3g})', gap > 1e-3)

state = 'the scene text'
questions = {'first': {'type': 'choice', 'instructions': 'pick one', 'criteria': {'a': 'short', 'b': 'a much longer candidate text', 'c': 'mid length'}},
             'second': {'type': 'noul', 'instructions': 'a longer question than the first one'}}
for mode in am.MODES:
    engine = engine_for(mode, 2)
    answer, _ = engine(state, questions)
    ok = True
    for field, question in questions.items():
        options = question.get('criteria') or {'true': 'Yes.', 'false': 'No.'}
        prefix = 'Decide whether the candidate is a correct match.\nInput:\n' + state + '\nQuestion:\n' + question['instructions'] + '\nCandidate:\n'
        prefix_ids = engine.tokens(prefix)
        field_rows = [prefix_ids + engine.tokens(value) + (EOS,) for value in options.values()]
        scores = reference(mode, field_rows, [len(prefix_ids)] * len(field_rows))
        probabilities = np.exp(scores - scores.max()); probabilities /= probabilities.sum()
        got = answer['answers'][field]['probabilities']
        ok &= np.allclose([got[key] for key in options], probabilities, atol=1e-9)
    check(f'engine(state, questions) {mode}: field probabilities equal the single-row reference', ok)

print(f'{sum(RESULTS)}/{len(RESULTS)} checks passed; torch {torch.__version__}')
sys.exit(0 if all(RESULTS) else 1)
