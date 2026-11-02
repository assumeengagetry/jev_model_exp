#!/usr/bin/env python3
"""Prepare remaining native train sources; annotations are data, never executed tests."""
import ast
import collections
import csv
import gzip
import json
import math
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pair_data_common import criteria_context, text

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'datasets/benchmarks-train'
SUPP = HOME / 'datasets/overnight-supplement/train'
OUT = HOME / 'cache/pair-training/extra-pairs'
OUT.mkdir(parents=True, exist_ok=True)
SCHEMA = pa.schema([(key, pa.string()) for key in ['state', 'question', 'criteria', 'candidate']]
                   + [('label', pa.float32())]
                   + [(key, pa.string()) for key in ['domain', 'source', 'source_repo', 'group_id']])
REPORT = []


def rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048): yield from batch.to_pylist()
    elif path.suffix == '.csv':
        with path.open(encoding='utf-8-sig') as f: yield from csv.DictReader(f)
    elif path.suffix == '.json':
        data = json.loads(path.read_text()); yield from data if isinstance(data, list) else data.get('data', [])
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)


def labels(path, column):
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    feature = json.loads(meta.get(b'huggingface', b'{}')).get('info', {}).get('features', {}).get(column, {})
    return feature.get('names', [])


class Writer:
    def __init__(self, name, domain, provenance):
        self.name, self.domain, self.provenance = name, domain, provenance
        self.path = OUT / domain / (name + '.parquet'); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(self.path.with_suffix('.parquet.preparing'), SCHEMA, compression='zstd')
        self.buffer = []; self.count = collections.Counter()
    def pair(self, state, question, candidate, label, group, criteria=''):
        state, question, candidate = text(state), text(question), text(candidate)
        if not state or not candidate or not 0 <= float(label) <= 1: return
        self.buffer.append(dict(state=state, question=question, criteria=criteria, candidate=candidate,
                                label=float(label), domain=self.domain, source=self.name,
                                source_repo=self.provenance, group_id=str(group)))
        self.count['pairs'] += 1
        if len(self.buffer) >= 8192:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=SCHEMA)); self.buffer.clear()
    def choice(self, state, question, options, positive, group):
        options = [text(value) for value in options]
        context = criteria_context(question, options)
        for index, candidate in enumerate(options): self.pair(state, question, candidate, float(index in positive), group, context)
    def finish(self):
        if self.buffer: self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=SCHEMA))
        self.writer.close(); self.path.with_suffix('.parquet.preparing').replace(self.path)
        report = dict(name=self.name, domain=self.domain, path=str(self.path.relative_to(HOME)),
                      counts=dict(self.count), source=self.provenance, complete=True, tests_executed=False)
        self.path.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        REPORT.append(report); print(self.name, dict(self.count), flush=True)


def boxed(solution):
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', solution))
    if not matches: return None
    start = matches[-1].end(); level = 1
    for index in range(start, len(solution)):
        if solution[index] == '{': level += 1
        elif solution[index] == '}':
            level -= 1
            if not level: return solution[start:index].strip()
    return None


def answer_key(value):
    return re.sub(r'\s+', '', str(value).replace(',', '').replace('\\left', '').replace('\\right', ''))


def math_pairs():
    for name, paths, domain in [
        ('gsm8k', list((ROOT / 'knowledge/gsm8k/main').glob('train-*.parquet')), 'knowledge'),
        ('math', list((SUPP / 'knowledge/math').rglob('train-*.parquet')), 'knowledge')]:
        if not paths: continue
        data = []
        for path in paths:
            for row in rows(path):
                value = row['answer'].rsplit('####', 1)[-1].strip() if name == 'gsm8k' else boxed(row['solution'])
                if value is not None: data.append((row.get('question', row.get('problem')), value))
        bank = list(dict.fromkeys(value for _, value in data))
        w = Writer(name, domain, 'native official train; distractors from distinct other train answers')
        rng = random.Random(42)
        for i, (question, value) in enumerate(data):
            w.count['raw_rows'] += 1; wrong = []
            while len(wrong) < 3 and len(bank) > 3:
                candidate = bank[rng.randrange(len(bank))]
                if answer_key(candidate) != answer_key(value) and candidate not in wrong: wrong.append(candidate)
            w.choice(question, 'Which answer is correct?', [value] + wrong, {0}, f'{name}:{i}')
        w.finish()


def native_mc():
    jobs = [('mmlu', 'knowledge', list((ROOT / 'knowledge/mmlu/auxiliary_train').glob('train-*.parquet'))),
            ('arc', 'knowledge', list((ROOT / 'knowledge/arc').rglob('train-*.parquet'))),
            ('winogrande', 'knowledge', list((ROOT / 'knowledge/winogrande/winogrande_xl').glob('train-*.parquet'))),
            ('hellaswag', 'language', list((ROOT / 'language/hellaswag/data').glob('train-*.parquet'))),
            ('anli', 'language', list((ROOT / 'language/anli/plain_text').glob('train_*.parquet'))),
            ('contractnli', 'language', list((ROOT / 'language/contractnli/contractnli_a').glob('train.parquet'))),
            ('banking77', 'retrieval', list((ROOT / 'retrieval/banking77/data').glob('train-*.parquet'))),
            ('clinc150', 'retrieval', list((ROOT / 'retrieval/clinc150/plus').glob('train-*.parquet')))]
    for name, domain, paths in jobs:
        if not paths: continue
        w = Writer(name, domain, 'native official train; no evaluation split')
        for path in sorted(paths):
            names = labels(path, 'intent' if name == 'clinc150' else 'label')
            for i, row in enumerate(rows(path)):
                w.count['raw_rows'] += 1; group = f'{name}:{path.parent.name}:{i}'
                if name == 'mmlu': w.choice(row['question'], 'Which answer is correct?', row['choices'], {int(row['answer'])}, group)
                elif name == 'arc':
                    options = row['choices']; gold = options['label'].index(row['answerKey'])
                    w.choice(row['question'], 'Which answer is correct?', options['text'], {gold}, group)
                elif name == 'winogrande':
                    if row['answer'] not in ['1', '2']: continue
                    w.choice(row['sentence'], 'Which option correctly fills the blank?', [row['option1'], row['option2']], {int(row['answer']) - 1}, group)
                elif name == 'hellaswag':
                    value = int(row['label']); state = row.get('ctx', row.get('ctx_a', '') + ' ' + row.get('ctx_b', ''))
                    w.choice(state, 'Which continuation is most plausible?', row['endings'], {value}, group)
                elif name in ['anli', 'contractnli']:
                    value = int(row['label'])
                    options = names or ['entailment', 'neutral', 'contradiction']
                    relation = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}
                    options = [relation.get(label, label) for label in options]
                    if 0 <= value < len(options):
                        w.choice({'premise': row['premise'], 'hypothesis': row['hypothesis']}, 'What is the relation of the hypothesis to the premise?', options, {value}, group)
                else:
                    column = 'intent' if name == 'clinc150' else 'label'; value = int(row[column])
                    if not names: raise RuntimeError('Missing native intent label names for ' + name)
                    w.choice(row['text'], 'Which user intent is expressed?', [value.replace('_', ' ') for value in names], {value}, group)
        w.finish()


def mbpp():
    w = Writer('mbpp', 'knowledge', 'MBPP train assertions used as static annotations; never executed')
    for path in (SUPP / 'knowledge/mbpp').rglob('train-*.parquet'):
        for row in rows(path):
            w.count['raw_rows'] += 1
            for index, annotation in enumerate(row.get('test_list', [])):
                try:
                    node = ast.parse(annotation).body[0]
                    condition = node.test
                    if not isinstance(node, ast.Assert) or not isinstance(condition, ast.Compare) or len(condition.ops) != 1 or not isinstance(condition.ops[0], ast.Eq): continue
                    expected = ast.literal_eval(condition.comparators[0])
                    wrong = []
                    if isinstance(expected, bool): wrong = [not expected]
                    elif isinstance(expected, (int, float)): wrong = [expected + 1, expected - 1, expected + 2]
                    elif isinstance(expected, str): wrong = [expected + ' x', expected[:-1] if expected else 'x', expected + ' 0']
                    elif isinstance(expected, list): wrong = [[], expected + [None], [None]]
                    elif isinstance(expected, tuple): wrong = [(), expected + (None,), (None,)]
                    elif isinstance(expected, dict): wrong = [{}, dict(expected, __different__=None)]
                    options = [repr(expected)] + list(dict.fromkeys(repr(x) for x in wrong if x != expected))
                    if len(options) < 2: continue
                    state = {'program': row['code'], 'setup': row.get('test_setup_code', ''), 'expression': ast.unparse(condition.left)}
                    w.choice(state, 'What is the value of the expression after applying the given program?', options, {0}, f"mbpp:{row['task_id']}:{index}")
                except (SyntaxError, ValueError, TypeError, AttributeError): w.count['unsupported_annotations'] += 1
    w.finish()


def parse_tools(system):
    decoder = json.JSONDecoder(); result = []
    for match in re.finditer(r'[\[{]', system):
        try: value, _ = decoder.raw_decode(system[match.start():])
        except ValueError: continue
        for tool in value if isinstance(value, list) else [value]:
            if isinstance(tool, dict) and tool.get('name') and ('parameters' in tool or 'description' in tool): result.append(tool)
        if result: break
    return result


def call_names(content):
    match = re.search(r'<(?:functioncall|TOOLCALL)>\s*', content, re.I)
    if not match: return set()
    value, _ = json.JSONDecoder().raw_decode(content[match.end():].lstrip())
    calls = value if isinstance(value, list) else [value]
    return {call.get('name', call.get('function', {}).get('name')) for call in calls}


def tool_selection(w, state, tools, content, group):
    names = call_names(content); available = {tool['name'] for tool in tools if isinstance(tool, dict) and tool.get('name')}
    if not names.issubset(available): w.count['unmapped_calls'] += 1; return
    question = 'Which tool is needed for the next assistant action? Match each candidate independently.'
    for tool in tools:
        if isinstance(tool, dict) and tool.get('name'):
            w.pair(state, question, tool, float(tool['name'] in names), group)
    w.pair(state, question, 'No tool call is needed; respond directly or request more information.', float(not names), group)


def tools():
    path = SUPP / 'tools/glaive/glaive-function-calling-v2.json'
    if path.exists():
        w = Writer('glaive', 'tools', 'Glaive function-calling-v2 official train, every assistant decision')
        marker = re.compile(r'(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):', re.I)
        for i, row in enumerate(rows(path)):
            w.count['raw_rows'] += 1; catalog = parse_tools(row['system']); chat = row['chat']; messages = list(marker.finditer(chat))
            for j, message in enumerate(messages):
                if message.group(1).upper() != 'ASSISTANT': continue
                content = chat[message.end():messages[j + 1].start() if j + 1 < len(messages) else len(chat)].strip()
                state = {'available_tools': catalog, 'conversation': chat[:message.start()].strip()}
                try: tool_selection(w, state, catalog, content, f'glaive:{i}:{j}')
                except (ValueError, TypeError, KeyError): w.count['unparsed_calls'] += 1
        w.finish()
    w = Writer('when2call', 'tools', 'When2Call official SFT and preference train converted to candidate decisions')
    for path in sorted((ROOT / 'tools/when2call/train').glob('*.jsonl')):
        for i, row in enumerate(rows(path)):
            w.count['raw_rows'] += 1
            catalog = [json.loads(tool) if isinstance(tool, str) else tool for tool in row['tools']]
            messages = row['messages']
            if 'chosen_response' in row:
                chosen, rejected = row['chosen_response']['content'], row['rejected_response']['content']
                state = {'available_tools': catalog, 'conversation': messages}
                options = [chosen, rejected] if i % 2 == 0 else [rejected, chosen]
                w.choice(state, 'Which assistant response or action is more appropriate?', options, {0 if i % 2 == 0 else 1}, f'when2call-pref:{i}')
                content = chosen
            else:
                if not messages or messages[-1]['role'] != 'assistant': continue
                content = messages[-1]['content']; state = {'available_tools': catalog, 'conversation': messages[:-1]}
                w.pair(state, 'Is this an appropriate next assistant response or action?', content, 1, f'when2call-sft:{i}')
            try: tool_selection(w, state, catalog, content, f'when2call-select:{path.name}:{i}')
            except (ValueError, TypeError, KeyError): w.count['unparsed_calls'] += 1
    w.finish()


def language():
    path = ROOT / 'language/vast/train.csv'
    if path.exists():
        w = Writer('vast', 'language', 'VAST official train, native 0=against/1=favor/2=neutral')
        for i, row in enumerate(rows(path)):
            w.count['raw_rows'] += 1
            w.choice({'topic': row['topic_str'], 'post': row['post']}, 'What stance does the post take toward the topic?', ['against', 'favor', 'neutral'], {int(row['label'])}, f'vast:{i}')
        w.finish()
    w = Writer('isarcasm', 'language', 'iSarcasmEval official English and Arabic train annotations')
    for path in (ROOT / 'language/isarcasm/train').glob('*.csv'):
        for i, row in enumerate(rows(path)):
            w.count['raw_rows'] += 1; state = row.get('tweet', row.get('text', ''))
            for label in ['sarcastic', 'sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']:
                if str(row.get(label)) in ['0', '1']:
                    w.choice(state, 'Does this text exhibit ' + label.replace('_', ' ') + '?', ['No.', 'Yes.'], {int(row[label])}, f'isarcasm:{path.name}:{i}:{label}')
    w.finish()
    path = ROOT / 'language/nli4ct/train.json'
    if path.exists():
        w = Writer('nli4ct', 'language', 'NLI4CT official train with full indicated clinical-trial sections')
        for group, row in json.loads(path.read_text()).items():
            w.count['raw_rows'] += 1; state = {'statement': row['Statement']}
            for key in ['Primary_id', 'Secondary_id']:
                if row.get(key):
                    document = json.loads((path.parent / 'CT json' / (row[key] + '.json')).read_text())
                    state[key] = {'section': row['Section_id'], 'text': document[row['Section_id']]}
            w.choice(state, 'Classify the statement against the clinical-trial evidence.', ['Entailment', 'Contradiction'], {0 if row['Label'] == 'Entailment' else 1}, group)
        w.finish()
    path = ROOT / 'language/ragtruth/train_source_info.jsonl'
    if path.exists():
        sources = {str(row['source_id']): row for row in rows(path)}
        w = Writer('ragtruth', 'language', 'RAGTruth response-level hallucination, official train only')
        for row in rows(path.parent / 'train_responses.jsonl'):
            w.count['raw_rows'] += 1; source = sources[str(row['source_id'])]
            state = {'prompt': source['prompt'], 'response': row['response']}
            w.choice(state, 'Does the response contain any unsupported or hallucinated content?', ['No, it is grounded.', 'Yes, it contains unsupported content.'], {int(bool(row['labels']))}, row['id'])
        w.finish()
    for path in (ROOT / 'language/acos/data').rglob('*_train.tsv'):
        data = []
        for line in path.read_text().splitlines():
            pieces = line.split('\t'); gold = set()
            for annotation in pieces[1:]:
                fields = annotation.split()
                if len(fields) >= 4: gold.add((fields[1], int(fields[2])))
            data.append((pieces[0], gold))
        categories = sorted({category for _, gold in data for category, _ in gold})
        name = 'acos_' + path.parent.name.lower(); w = Writer(name, 'language', 'ACOS official train aspect-category/sentiment annotations')
        for i, (state, gold) in enumerate(data):
            w.count['raw_rows'] += 1
            for category in categories:
                for sentiment, label in enumerate(['negative', 'neutral', 'positive']):
                    w.pair(state, 'Which aspect and sentiment combinations are expressed in this review?', category.replace('#', ' / ') + ': ' + label, float((category, sentiment) in gold), f'{name}:{i}')
        w.finish()


def arts():
    base = ROOT / 'arts/humicroedit'
    w = Writer('humicroedit', 'arts', 'Humicroedit official train; human humor grades and paired edits')
    edit = lambda original, replacement: re.sub(r'<[^<>]*/>', lambda _: replacement, original)
    for path in base.rglob('train*.csv'):
        for i, row in enumerate(rows(path)):
            w.count['raw_rows'] += 1; group = f'humicroedit:{path.parent.name}:{path.name}:{i}'
            if 'original1' in row:
                options = [edit(row['original1'], row['edit1']), edit(row['original2'], row['edit2'])]
                label = int(row['label'])
                if label == 0: options.append('The two edits are equally funny.'); positive = 2
                else: positive = label - 1
                w.choice('Two edited news headlines.', 'Which edit is funnier?', options, {positive}, group)
            elif row.get('meanGrade') not in [None, '']:
                value = max(0.0, min(3.0, float(row['meanGrade'])))
                state = edit(row['original'], row['edit']); low, high = math.floor(value), math.ceil(value)
                for rating in range(4):
                    probability = 1.0 if low == high == rating else (high - value if rating == low else value - low if rating == high else 0.0)
                    w.pair(state, 'Rate humor on the human 0 (not funny) to 3 (very funny) scale.', str(rating), probability, group)
    w.finish()
    path = ROOT / 'arts/cfcolor/train.parquet'
    if path.exists():
        users = collections.defaultdict(list)
        for row in rows(path): users[int(row['user_id'])].append(row)
        w = Writer('cfcolor', 'arts', 'Original cfcolor train ratings only; target palettes excluded from visible history')
        rng = random.Random(42)
        for user, ratings in users.items():
            rng.shuffle(ratings)
            for i in range(0, len(ratings) - 1, 2):
                a, b = ratings[i:i + 2]
                if a['colors'] == b['colors']: continue
                history = [{'colors': r['colors'], 'rating': r['rating']} for r in ratings
                           if r['palette_id'] not in [a['palette_id'], b['palette_id']] and r['colors'] not in [a['colors'], b['colors']]][:8]
                if len(history) < 4: continue
                state = {'history': history, 'rating_scale': '1 lowest preference, 5 highest preference', 'candidate_palettes': [a['colors'], b['colors']]}
                options = [text(a['colors']), text(b['colors'])]
                positive = 0 if a['rating'] > b['rating'] else 1
                if a['rating'] == b['rating']: options.append('The palettes are equally preferred.'); positive = 2
                w.count['raw_rating_pairs'] += 1
                w.choice(state, 'Which palette would this user prefer?', options, {positive}, f'cfcolor:{user}:{i}')
        w.finish()


if __name__ == '__main__':
    math_pairs(); native_mc(); mbpp(); tools(); language(); arts()
    (OUT / 'MANIFEST.json').write_text(json.dumps({'complete': True, 'sources': REPORT,
        'all_available_train_rows_considered': True, 'pair_cap': None,
        'gaps': ['HoVer claims available but oracle Wikipedia documents not available; not used as full-context training.',
                 'ChessBench/ToolRet/NewYorker/Habermas native downloads unresolved; typed and independent train sources supplement capabilities.'],
        'tests_executed': False, 'extra_file_hash_checks': False}, ensure_ascii=False, indent=2))
