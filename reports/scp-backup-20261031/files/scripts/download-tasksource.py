#!/usr/bin/env python3
import concurrent.futures
import json
import shutil

from download_utils import HOME, download, hf_url

repo = 'tasksource/tasksource-jev-typed-decisions'
metadata = json.loads((HOME / 'config/tasksource_meta.json').read_text())
destination = HOME / 'datasets/tasksource-jev-typed-decisions'
destination.mkdir(parents=True, exist_ok=True)
selected = [entry['rfilename'] for entry in metadata['siblings']
            if entry['rfilename'].startswith(('data/', 'full/', 'filtered-full/'))
            and entry['rfilename'].endswith(('.parquet', '.md', '.json'))]
selected += ['README.md', 'sources.yaml', 'build-summary.json']
sizes = {x['path']: x['size'] for x in json.loads((HOME / 'config/tasksource_tree.json').read_text())}
results = []
errors = []
print(f'Downloading {len(selected)} text files; vision (140GB) is not needed by Qwen3-0.6B.', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    pending = {pool.submit(download, hf_url(repo, name), destination / name, sizes.get(name)): name
               for name in selected}
    for future in concurrent.futures.as_completed(pending):
        try:
            results.append(future.result())
        except Exception as error:
            errors.append({'file': pending[future], 'error': str(error)})
            print(f'FAILED {pending[future]}: {error}', flush=True)
        (destination / 'DOWNLOAD.json').write_text(json.dumps({
            'repo': repo, 'revision': metadata['sha'], 'configs': ['default', 'full', 'filtered-full'],
            'hash_verification': False, 'files': results, 'errors': errors,
            'complete': len(results) == len(selected) and not errors,
            'omitted': {'vision': 'Image data, not supported by the requested text-only Qwen3-0.6B',
                        'extraction': 'Separate span-extraction task, not the typed-decision text training mix'},
        }, ensure_ascii=False, indent=2))
print(f'Finished: {len(results)} files, {len(errors)} errors', flush=True)
raise SystemExit(bool(errors))
