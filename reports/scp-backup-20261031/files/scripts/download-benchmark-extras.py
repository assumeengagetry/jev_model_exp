#!/usr/bin/env python3
import concurrent.futures
import json
import argparse
from download_utils import HOME, download

parser = argparse.ArgumentParser()
parser.add_argument('--skip', nargs='*', default=[])
args = parser.parse_args()
items = json.loads((HOME / 'config/benchmark-extra-sources.json').read_text())
results = []


def run(item):
    name, url = item
    try:
        return download(url, HOME / 'datasets/benchmark-sources' / name)
    except Exception as error:
        return {'file': name, 'error': str(error)}


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    for result in pool.map(run, [x for x in items.items() if x[0] not in args.skip]):
        results.append(result)
        print(result, flush=True)
        (HOME / 'datasets/benchmark-sources/DOWNLOADS.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
raise SystemExit(any('error' in entry for entry in results))
