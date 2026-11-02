#!/usr/bin/env python3
"""Convert acquired train supplements into reusable candidate-matching pairs.

This is actual data preparation, not a test or a model run. Global deduplication
and final shared tokenization remain the full-pool training owner's next stage.
"""

import collections
import concurrent.futures
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from pair_data_common import criteria_context

HOME = Path('/home/dev/assumeengage')
ROOT = HOME / 'datasets/overnight-supplement'
FORMAT_VERSION = 2
SCHEMA = pa.schema([
    ('state', pa.string()), ('question', pa.string()), ('criteria', pa.string()), ('candidate', pa.string()),
    ('label', pa.float32()), ('domain', pa.string()), ('source', pa.string()),
    ('source_repo', pa.string()), ('group_id', pa.string()),
])
SUPPORTED = {
    'snli', 'multinli', 'sst2', 'tweet_sentiment', 'openbookqa', 'sciq',
    'esci', 'dbpedia', 'ag_news', 'massive', 'xlam',
    'hh_rlhf', 'ultrafeedback', 'helpsteer', 'helpsteer2',
}


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def raw_rows(path):
    if path.suffix == '.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()
    elif path.suffix == '.arrow':
        with pa.memory_map(str(path), 'r') as source:
            for batch in ipc.open_stream(source):
                yield from batch.to_pylist()
    elif path.suffix == '.json':
        data = json.loads(path.read_text())
        yield from data if isinstance(data, list) else data['data']
    else:
        open_file = gzip.open if path.suffix == '.gz' else open
        with open_file(path, 'rt', encoding='utf-8') as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def choices(state, question, candidates, positive):
    for key, candidate in candidates.items():
        yield text(state), question, text(candidate), float(key in positive)


def convert(name, row, intent_names):
    if 'split' in row and row['split'] not in ('train', 'training', None):
        return
    if name in ('snli', 'multinli'):
        label = row['label']
        if label not in (0, 1, 2):
            return
        candidates = {0: 'Entailment: the premise supports the hypothesis.',
                      1: 'Neutral: the premise does not establish or contradict the hypothesis.',
                      2: 'Contradiction: the premise contradicts the hypothesis.'}
        yield from choices({'premise': row['premise'], 'hypothesis': row['hypothesis']},
                           'What is the relation of the hypothesis to the premise?', candidates, {label})
    elif name in ('sst2', 'tweet_sentiment'):
        names = ['negative', 'positive'] if name == 'sst2' else ['negative', 'neutral', 'positive']
        label = int(row['label'])
        if not 0 <= label < len(names):
            return
        yield from choices(row.get('sentence', row.get('text', '')),
                           'What sentiment does this text express?', dict(enumerate(names)), {label})
    elif name == 'openbookqa':
        candidates = dict(zip(row['choices']['label'], row['choices']['text']))
        yield from choices(row['question_stem'], 'Which answer is correct?', candidates, {row['answerKey']})
    elif name == 'sciq':
        candidates = {0: row['correct_answer'], 1: row['distractor1'],
                      2: row['distractor2'], 3: row['distractor3']}
        yield from choices(row['question'], 'Which answer is correct?', candidates, {0})
    elif name == 'esci':
        candidates = {'E': 'Exact: the product satisfies the query.',
                      'S': 'Substitute: a reasonable alternative, but not an exact match.',
                      'C': 'Complement: a product useful alongside the requested product.',
                      'I': 'Irrelevant: not a useful match for the query.'}
        label = str(row['esci_label']).upper()
        if label not in candidates:
            return
        state = {k: row.get(k) for k in ['query', 'product_title', 'product_description',
                                        'product_bullet_point', 'product_brand', 'product_color']}
        yield from choices(state, 'How relevant is this product to the query?', candidates, {label})
    elif name in ('dbpedia', 'ag_news'):
        names = (['company', 'educational institution', 'artist', 'athlete', 'office holder',
                  'means of transportation', 'building', 'natural place', 'village', 'animal',
                  'plant', 'album', 'film', 'written work'] if name == 'dbpedia'
                 else ['world news', 'sports', 'business', 'science and technology'])
        label = row.get('label')
        if not isinstance(label, int) or not 0 <= label < len(names) or 'text' not in row:
            return
        yield from choices(row['text'], 'Which category describes this text?', dict(enumerate(names)), {label})
    elif name == 'massive':
        label = str(row['label'])
        if label not in intent_names:
            return
        yield from choices(row['text'], 'Which user intent is expressed?', intent_names, {label})
    elif name == 'xlam':
        tools = json.loads(row['tools']) if isinstance(row['tools'], str) else row['tools']
        answers = json.loads(row['answers']) if isinstance(row['answers'], str) else row['answers']
        positive = {answer['name'] for answer in answers}
        candidates = {tool['name']: tool for tool in tools}
        candidates['__no_tool__'] = 'No tool call is needed.'
        if not positive:
            positive = {'__no_tool__'}
        if not positive.issubset(candidates):
            return
        yield from choices({'request': row['query'], 'available_tools': tools},
                           'Which tool should be used for this request? Multiple tools may be appropriate.',
                           candidates, positive)
    elif name == 'hh_rlhf':
        chosen = row['chosen'].rsplit('\n\nAssistant:', 1)
        rejected = row['rejected'].rsplit('\n\nAssistant:', 1)
        if len(chosen) != 2 or len(rejected) != 2 or chosen[0] != rejected[0]:
            return
        if chosen[1].strip() == rejected[1].strip():
            return
        yield from choices(chosen[0], 'Which response is preferred for this conversation?',
                           {0: chosen[1].strip(), 1: rejected[1].strip()}, {0})
    elif name == 'ultrafeedback':
        if row['score_chosen'] <= row['score_rejected']:
            return
        chosen = row['chosen'][-1]['content']
        rejected = row['rejected'][-1]['content']
        if chosen == rejected:
            return
        yield from choices(row['prompt'], 'Which response is preferred for this request?',
                           {0: chosen, 1: rejected}, {0})
    elif name in ('helpsteer', 'helpsteer2'):
        state = {'request': row['prompt'], 'response': row['response']}
        for metric in ['helpfulness', 'correctness', 'coherence']:
            rating = row[metric]
            if rating not in range(5):
                continue
            candidates = {i: f'{i}/4 on {metric}, where 0 is lowest and 4 is highest.' for i in range(5)}
            yield from choices(state, f'Rate the response on {metric} using the human 0–4 scale.',
                               candidates, {rating})


def prepare(job):
    name, files = job
    domain = files[0]['area']
    output = ROOT / 'pairs' / domain / (name + '.parquet')
    report_path = output.with_suffix('.json')
    inputs = [{'path': r['path'], 'bytes': r['bytes'], 'repo': r['repo']} for r in files]
    if output.exists() and report_path.exists():
        previous = json.loads(report_path.read_text())
        if (previous.get('inputs') == inputs and previous.get('status') == 'converted'
                and previous.get('pair_format_version') == FORMAT_VERSION):
            return dict(previous, reused=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.parquet.preparing')
    counts = collections.Counter()
    errors = collections.Counter()
    intent_names = {}
    if name == 'massive':
        for file in files:
            for row in raw_rows(HOME / file['path']):
                value = row.get('label_text')
                if value is None and isinstance(row.get('label'), str):
                    value = row['label']
                if value is not None:
                    intent_names[str(row['label'])] = str(value).replace('_', ' ')
        if not intent_names:
            return {'name': name, 'domain': domain, 'status': 'needs_intent_label_mapping', 'inputs': inputs}
    batch = []
    with pq.ParquetWriter(temporary, SCHEMA, compression='zstd') as writer:
        for file_index, file in enumerate(files):
            for row_index, raw in enumerate(raw_rows(HOME / file['path'])):
                counts['raw_rows'] += 1
                used = False
                try:
                    for state, question, candidate, label in convert(name, raw, intent_names):
                        used = True
                        group_id = f'{name}:{file_index}:{row_index}'
                        context = ''
                        if name == 'hh_rlhf':
                            options = [raw[key].rsplit('\n\nAssistant:', 1)[1].strip()
                                       for key in ('chosen', 'rejected')]
                            if row_index % 2:
                                options.reverse()
                            context = criteria_context(question, options)
                        elif name == 'ultrafeedback':
                            options = [raw[key][-1]['content'] for key in ('chosen', 'rejected')]
                            if row_index % 2:
                                options.reverse()
                            context = criteria_context(question, options)
                        batch.append({'state': state, 'question': question, 'criteria': context, 'candidate': candidate,
                                      'label': label, 'domain': domain, 'source': name,
                                      'source_repo': file['repo'], 'group_id': group_id})
                        counts['pairs'] += 1
                        counts['positive_pairs' if label == 1 else 'negative_pairs'] += 1
                        if len(batch) >= 5000:
                            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                            batch.clear()
                except (KeyError, ValueError, TypeError, IndexError) as error:
                    errors[type(error).__name__] += 1
                counts['used_rows' if used else 'unconverted_rows'] += 1
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    temporary.replace(output)
    report = {'name': name, 'domain': domain, 'status': 'converted' if counts['pairs'] else 'needs_adapter_mapping',
              'pair_format_version': FORMAT_VERSION,
              'output': str(output.relative_to(HOME)), 'inputs': inputs, 'counts': dict(counts),
              'conversion_errors': dict(errors), 'global_deduplication': 'pending with complete training pool',
              'tokenization': 'pending shared tokenizer/input format', 'extra_tests': False, 'hash_checks': False}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    manifest = json.loads((ROOT / 'DOWNLOAD.json').read_text())
    sources = collections.defaultdict(list)
    for row in manifest['results']:
        if row['status'] in ('present', 'downloaded'):
            sources[row['name']].append(row)
    jobs = [(name, files) for name, files in sorted(sources.items()) if name in SUPPORTED]
    results = [{'name': name, 'status': 'raw_ready_adapter_pending'}
               for name in sources if name not in SUPPORTED]
    print('Converting', len(jobs), 'actual training sources without token/model tests.', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare, job): job[0] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {'name': futures[future], 'status': 'conversion_failed', 'error': str(error)[:400]}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            temporary = ROOT / 'PAIRS.json.tmp'
            temporary.write_text(json.dumps({'results': results, 'schema': str(SCHEMA),
                                             'note': 'Reusable conversion cache, not a claim of completed full-pool deduplication or training.'},
                                            ensure_ascii=False, indent=2))
            temporary.replace(ROOT / 'PAIRS.json')


if __name__ == '__main__':
    main()
