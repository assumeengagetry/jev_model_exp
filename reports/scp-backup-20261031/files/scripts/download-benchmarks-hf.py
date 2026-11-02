#!/usr/bin/env python3
import concurrent.futures
import fnmatch
import json
import threading
import argparse

from download_utils import HOME, download, hf_metadata, hf_url

sources = json.loads((HOME / 'config/benchmark-hf-sources.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--only', nargs='*', default=[])
parser.add_argument('--skip', nargs='*', default=[])
args = parser.parse_args()
sources = [s for s in sources if (not args.only or s['name'] in args.only) and s['name'] not in args.skip]
root = HOME / 'datasets/benchmarks-train'
root.mkdir(parents=True, exist_ok=True)
results = []
lock = threading.Lock()


def run(source):
    entry = dict(source)
    dest = root / source['area'] / source['name']
    try:
        metadata = hf_metadata(source['repo'])
        entry['revision'] = metadata.get('sha')
        names = [x['rfilename'] for x in metadata['siblings']]
        selected = [n for n in names if any(fnmatch.fnmatch(n, p) for p in source['patterns'])]
        if not selected:
            raise RuntimeError('No matching official train files found; no evaluation split substituted.')
        if 'README.md' in names:
            selected.append('README.md')
        files = []
        for name in selected:
            files.append(download(hf_url(source['repo'], name), dest / name))
        entry.update(status='downloaded', files=files)
        (dest / 'DOWNLOAD.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2))
    except Exception as error:
        entry.update(status='failed', error=str(error))
        print(f"FAILED {source['name']}: {error}", flush=True)
    with lock:
        results.append(entry)
        (root / 'HF_DOWNLOADS.json').write_text(json.dumps({
            'hash_verification': False, 'train_only': True, 'sources': results,
            'complete': len(results) == len(sources) and all(x['status'] == 'downloaded' for x in results),
        }, ensure_ascii=False, indent=2))
    return entry


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    list(pool.map(run, sources))
failures = sum(x['status'] == 'failed' for x in results)
print(f'Finished {len(results)} sources, {failures} failures', flush=True)
raise SystemExit(bool(failures))
