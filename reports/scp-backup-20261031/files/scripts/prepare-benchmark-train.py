#!/usr/bin/env python3
"""Extract only official train records. Does not run evaluations or compute hashes."""
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

HOME = Path('/home/dev/assumeengage')
SOURCES = HOME / 'datasets/benchmark-sources'
OUTPUT = HOME / 'datasets/benchmarks-train'
report = []


def save(relative, data):
    path = OUTPUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path.relative_to(HOME))


def extract_tar(source, destination, select):
    path = SOURCES / source
    if not path.exists():
        return
    paths = []
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = member.name.partition('/')[2]
            if select(relative):
                paths.append(save(destination + '/' + relative, archive.extractfile(member).read()))
    report.append({'source': source, 'files': paths, 'status': 'extracted'})


extract_tar('language/acos.tar.gz', 'language/acos',
            lambda n: n.startswith('data/') and n.endswith('_train.tsv'))
extract_tar('language/isarcasm.tar.gz', 'language/isarcasm',
            lambda n: n.startswith('train/') and n.endswith('.csv'))
extract_tar('retrieval/sgd.tar.gz', 'retrieval/sgd',
            lambda n: n.startswith('train/') and n.endswith('.json'))
extract_tar('games/boxoban.tar.gz', 'games/boxoban',
            lambda n: '/train/' in n and n.endswith('.txt'))

path = SOURCES / 'knowledge/mmlu-data.tar'
if path.exists():
    import csv
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isfile() and '/auxiliary_train/' in member.name and member.name.endswith('.csv'):
                subject = Path(member.name).stem.removesuffix('_train')
                for record in csv.reader(io.TextIOWrapper(archive.extractfile(member), encoding='utf-8')):
                    rows.append({'question': record[0], 'choices': record[1:5],
                                 'answer': ord(record[5].strip()) - ord('A'), 'subject': subject})
    if rows:
        target = OUTPUT / 'knowledge/mmlu/auxiliary_train/train-00000-of-00001.parquet'
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), target)
        report.append({'source': str(path.relative_to(HOME)), 'files': [str(target.relative_to(HOME))],
                       'rows': len(rows), 'status': 'extracted', 'split': 'original auxiliary_train only'})

path = SOURCES / 'language/hellaswag-train.jsonl'
if path.exists():
    import pyarrow.json as paj
    import pyarrow.parquet as pq
    target = OUTPUT / 'language/hellaswag/data/train-00000-of-00001.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    table = paj.read_json(path)
    pq.write_table(table, target)
    report.append({'source': str(path.relative_to(HOME)), 'files': [str(target.relative_to(HOME))],
                   'rows': len(table), 'status': 'extracted', 'split': 'official HellaSwag train'})

for source, dest in [('language/vast_train.csv', 'language/vast/train.csv'),
                     ('retrieval/hover_train.json', 'retrieval/hover/train.json'),
                     ('games/chessbench-first-train-shard.bag', 'games/chessbench/train/action_value-00000-of-02148_data.bag')]:
    path = SOURCES / source
    if path.exists():
        target = OUTPUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                target.hardlink_to(path)
            except OSError:
                shutil.copyfile(path, target)
        report.append({'source': source, 'files': [str(target.relative_to(HOME))],
                       'status': 'partial' if source.endswith('.bag') else 'extracted'})

path = SOURCES / 'arts/humicroedit.zip'
if path.exists():
    paths = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if Path(name).name in {'train.csv', 'train_funlines.csv'}:
                relative = name.partition('/')[2]
                paths.append(save('arts/humicroedit/' + relative, archive.read(name)))
    report.append({'source': str(path.relative_to(HOME)), 'files': paths, 'status': 'extracted'})

path = SOURCES / 'language/nli4ct.tar.gz'
if path.exists():
    paths = []
    with tarfile.open(path) as archive:
        member = next(m for m in archive if m.name.endswith('/training_data.zip'))
        data = archive.extractfile(member).read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.endswith('/'): continue
            if Path(name).name.lower() in {'train.json', 'training.json'} or name.startswith('CT json/'):
                paths.append(save('language/nli4ct/' + name, archive.read(name)))
    report.append({'source': str(path.relative_to(HOME)), 'files': paths, 'status': 'extracted'})

path = SOURCES / 'language/ragtruth.tar.gz'
if path.exists():
    with tarfile.open(path) as archive:
        responses = next(m for m in archive if m.name.endswith('/dataset/response.jsonl'))
        source_info = next(m for m in archive if m.name.endswith('/dataset/source_info.jsonl'))
        rows = [json.loads(line) for line in archive.extractfile(responses) if line.strip()]
        train = [r for r in rows if str(r.get('split', '')).lower() == 'train']
        source_ids = {str(r['source_id']) for r in train}
        contexts = [json.loads(line) for line in archive.extractfile(source_info) if line.strip()]
        contexts = [r for r in contexts if str(r.get('source_id', r.get('id', ''))) in source_ids]
    if train:
        files = [save('language/ragtruth/train_responses.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train).encode()),
                 save('language/ragtruth/train_source_info.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in contexts).encode())]
        report.append({'source': str(path.relative_to(HOME)), 'files': files, 'rows': len(train),
                       'contexts': len(contexts), 'status': 'extracted'})
    else:
        report.append({'source': str(path.relative_to(HOME)), 'status': 'needs_split_mapping',
                       'note': 'No rows with split=train; evaluation records were not substituted.'})

path = SOURCES / 'arts/cfcolor.zip'
if path.exists():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.io import loadmat
    with zipfile.ZipFile(path) as archive:
        ratings_name = next(n for n in archive.namelist() if n.endswith('allMTurkRatings.mat'))
        themes_name = next(n for n in archive.namelist() if n.endswith('themeData.mat'))
        ratings = loadmat(io.BytesIO(archive.read(ratings_name)), squeeze_me=True)
        themes = loadmat(io.BytesIO(archive.read(themes_name)), squeeze_me=True, struct_as_record=False)
    train = np.asarray(ratings['train_vec'])
    rgb = np.asarray(themes['datapoints'].rgb)
    colors = [['#' + ''.join(f'{int(round(float(v) * 255)):02x}' for v in row[i:i + 3])
               for i in range(0, 15, 3)] for row in rgb]
    target = OUTPUT / 'arts/cfcolor/train.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({'user_id': train[:, 0].astype('int64'),
                      'palette_id': train[:, 1].astype('int64'), 'rating': train[:, 2],
                      'colors': [colors[int(i) - 1] for i in train[:, 1]]})
    pq.write_table(table, target)
    report.append({'source': str(path.relative_to(HOME)), 'files': [str(target.relative_to(HOME))],
                   'rows': len(table), 'status': 'extracted', 'split': 'original train_vec only'})

path = SOURCES / 'arts/habermas.parquet'
if path.exists():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    splits = table['question.split']
    train = table.filter(pc.is_in(pc.utf8_upper(splits), value_set=pa.array(['TRAIN', 'TRAINING'])))
    if len(train):
        target = OUTPUT / 'arts/habermas/train.parquet'
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(train, target)
        report.append({'source': str(path.relative_to(HOME)), 'files': [str(target.relative_to(HOME))],
                       'rows': len(train), 'status': 'extracted', 'split': 'question.split=TRAIN/TRAINING'})
    else:
        report.append({'source': str(path.relative_to(HOME)), 'status': 'needs_split_mapping',
                       'available_split_labels': splits.unique().to_pylist()})

(OUTPUT / 'EXTRACTED_TRAIN.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
for entry in report:
    print(entry['source'], entry['status'], len(entry.get('files', [])), flush=True)
