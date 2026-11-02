#!/usr/bin/env python3
"""Prepare the real KnowLine teacher, on CPU, without touching the Student env.

No model construction, inference, smoke tests or application hash checks. Native
sources are downloaded as data only. HTTP/Xet credentials stay in memory and
subprocess diagnostics are deliberately not copied into the public receipts.
"""

import argparse
import concurrent.futures
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote


ROOT = Path('/home/dev/assumeengage')
REPO = 'PelaAI/KnowLine-31B-Gen1'
MODEL = ROOT / 'models/KnowLine-31B-Gen1'
BACKING = Path('/tmp/assumeengage-distillation-20261101/KnowLine-31B-Gen1')
RUN = ROOT / 'runs/distillation-after-five-specialists-20261031'
ASSETS = RUN / 'teacher-assets'
ENV = ROOT / 'envs/knowline'
BASE_PYTHON = ROOT / 'envs/jev/bin/python'
NATIVE = ['knowline_server.py', 'knowline_layout.py', 'knowline_engine.py']
SMALL_REQUIRED = [
    'config.json', 'generation_config.json', 'processor_config.json',
    'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
    'temperatures.json', 'LICENSE', 'README.md', 'INFERENCE.md',
    'model.safetensors.index.json', *NATIVE,
]
MANIFEST = ASSETS / 'ASSET_MANIFEST.json'
PROGRESS = ASSETS / 'ASSET_PROGRESS.json'
ENV_READY = ASSETS / 'ENVIRONMENT_READY.json'
READY = RUN / 'TEACHER_READY.json'
GUARD = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(message):
    print(now(), message, flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text())


def configure():
    # These paths all use the project's deliberate logical model entry point.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_HOME'] = str(MODEL / '.cache/huggingface')
    os.environ['HF_XET_CACHE'] = str(MODEL / '.cache/xet')
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    os.environ['HF_XET_HIGH_PERFORMANCE'] = '0'
    os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS'] = '4'
    os.environ['HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['RUST_LOG'] = 'off'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(ROOT / 'scripts'))


def request(url, **kwargs):
    from download_utils import direct_hf_request
    return direct_hf_request(url, **kwargs)


def setup_paths():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.is_symlink():
        if MODEL.resolve() != BACKING:
            raise RuntimeError('Existing logical teacher path targets a different directory')
    elif MODEL.exists():
        raise RuntimeError('Existing logical teacher path is not the declared backing symlink')
    BACKING.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_symlink():
        MODEL.symlink_to(BACKING, target_is_directory=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / 'file-progress').mkdir(exist_ok=True)
    (RUN / 'logs').mkdir(exist_ok=True)
    for directory in [MODEL, ASSETS, ENV.parent]:
        instructions = directory / 'AGENTS.md'
        if instructions.exists():
            # Nested instructions must be read by the operator before proceeding.
            raise RuntimeError('Nested AGENTS.md requires operator review: ' + str(instructions))


def file_status(row):
    path = MODEL / row['file']
    receipt = ASSETS / 'file-progress' / (row['file'] + '.json')
    item = dict(row)
    if path.exists():
        if path.stat().st_size != row['bytes']:
            raise RuntimeError('Existing final asset has an unexpected byte size: ' + row['file'])
        item.update(status='ready', downloaded_bytes=row['bytes'])
    else:
        detail = load_json(receipt) if receipt.exists() else {}
        partial = path.with_name(path.name + '.http.part')
        received = partial.stat().st_size if partial.exists() else detail.get('downloaded_bytes', 0)
        status = detail.get('status', 'pending')
        item.update(status='pending' if status == 'ready' else status,
                    downloaded_bytes=min(received, row['bytes']))
        for key in ['transport', 'attempt', 'error_type', 'updated_at']:
            if key in detail:
                item[key] = detail[key]
    return item


def progress(manifest, stage, **extra):
    with GUARD:
        files = [file_status(row) for row in manifest.get('files', [])]
        payload = {
            'complete': False, 'pid': os.getpid(), 'updated_at': now(),
            'stage': stage, 'repo': REPO, 'model_path': str(MODEL),
            'actual_model_path': str(MODEL.resolve()), 'files': files,
            'ready_files': sum(x['status'] == 'ready' for x in files),
            'total_files': len(files),
            'downloaded_bytes': sum(x['downloaded_bytes'] for x in files),
            'total_bytes': sum(x['bytes'] for x in files),
            'environment_ready': ENV_READY.exists(),
            'extra_file_hash_checks': False, **extra,
        }
        write_json(PROGRESS, payload)


def pinned_manifest():
    if MANIFEST.exists():
        data = load_json(MANIFEST)
        if data['repo'] != REPO or data['model_path'] != str(MODEL):
            raise RuntimeError('Existing asset manifest belongs to another teacher')
        return data
    response, connection = request('https://huggingface.co/api/models/' + REPO + '?blobs=true')
    try:
        if response.status != 200:
            raise RuntimeError('Model metadata HTTP ' + str(response.status))
        metadata = json.load(response)
    finally:
        response.close()
        connection.close()
    entries = {x['rfilename']: x for x in metadata['siblings']}
    for name in SMALL_REQUIRED:
        if name not in entries:
            raise RuntimeError('Required native teacher asset missing: ' + name)
    shards = sorted(name for name in entries if name.endswith('.safetensors'))
    if len(shards) != 13:
        raise RuntimeError('Expected the actual thirteen-shard BF16 teacher')
    selected = SMALL_REQUIRED + [x for x in ['README.zh.md', 'serve_knowline.sh'] if x in entries] + shards
    files = []
    for name in selected:
        if Path(name).name != name:
            raise RuntimeError('Unexpected non-root asset path')
        size = entries[name].get('size')
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError('Missing public asset size: ' + name)
        files.append({'file': name, 'bytes': size,
                      'kind': 'weights' if name.endswith('.safetensors') else 'metadata_or_native_source'})
    data = {
        'complete': False, 'repo': REPO, 'revision': metadata['sha'],
        'created_at': now(), 'model_path': str(MODEL),
        'actual_model_path': str(MODEL.resolve()), 'files': files,
        'extra_file_hash_checks': False, 'excluded': ['SHA256SUMS', '.gitattributes'],
        'native_sources_executed': False,
        'transport': 'official hf_xet with verified-TLS DNS adapter; resumable HTTP fallback',
    }
    write_json(MANIFEST, data)
    return data


def asset_url(manifest, name):
    return 'https://huggingface.co/' + REPO + '/resolve/' + manifest['revision'] + '/' + quote(name)


def file_receipt(row, **values):
    write_json(ASSETS / 'file-progress' / (row['file'] + '.json'),
               {'file': row['file'], 'updated_at': now(), **values})


def download_http(manifest, row):
    from download_utils import direct_hf_download
    final = MODEL / row['file']
    partial = final.with_name(final.name + '.http.part')
    # Adapter retries/ranges are retained; an outer retry renews its 900s deadline.
    for attempt in range(1, 25):
        if partial.exists() and partial.stat().st_size > row['bytes']:
            raise RuntimeError('Oversized dedicated HTTP partial: ' + row['file'])
        file_receipt(row, status='downloading', transport='http-range', attempt=attempt)
        try:
            direct_hf_download(asset_url(manifest, row['file']), partial, row['bytes'])
            if partial.stat().st_size != row['bytes']:
                raise RuntimeError('HTTP asset byte-size mismatch')
            partial.replace(final)
            file_receipt(row, status='ready', transport='http-range', downloaded_bytes=row['bytes'])
            return
        except Exception as error:
            file_receipt(row, status='retrying', transport='http-range', attempt=attempt,
                         error_type=type(error).__name__,
                         downloaded_bytes=partial.stat().st_size if partial.exists() else 0)
            if attempt == 24:
                raise
            time.sleep(min(30, attempt * 2))


def download_xet(manifest, row):
    import requests
    from requests.structures import CaseInsensitiveDict
    from hf_xet import PyXetDownloadInfo, download_files
    from huggingface_hub.utils._xet import (
        parse_xet_file_data_from_response, parse_xet_connection_info_from_headers,
    )
    url = asset_url(manifest, row['file'])
    response, connection = request(url, method='HEAD', follow_redirects=False)
    try:
        if response.status not in [200, 302, 307]:
            raise RuntimeError('Xet metadata HTTP ' + str(response.status))
        wrapped = requests.Response()
        wrapped.status_code = response.status
        wrapped.url = url
        wrapped.headers = CaseInsensitiveDict(response.getheaders())
        data = parse_xet_file_data_from_response(wrapped, endpoint='https://huggingface.co')
        if data is None or int(wrapped.headers['X-Linked-Size']) != row['bytes']:
            raise RuntimeError('Native Xet metadata or byte size unavailable')
    finally:
        response.close()
        connection.close()

    def authorization():
        response, connection = request(data.refresh_route)
        try:
            if response.status != 200:
                raise RuntimeError('Xet authorization HTTP ' + str(response.status))
            info = parse_xet_connection_info_from_headers(CaseInsensitiveDict(response.getheaders()))
            if info is None:
                raise RuntimeError('Xet transfer metadata unavailable')
            return info
        finally:
            response.close()
            connection.close()

    initial = authorization()

    def refresh():
        info = authorization()
        return info.access_token, info.expiration_unix_epoch

    last = [0.0]
    received = [0]
    callback_lock = threading.Lock()

    def transferred(value):
        # Official huggingface_hub.xet_get passes each callback to tqdm.update:
        # values are incremental bytes, not a cumulative position.
        with callback_lock:
            received[0] += int(value)
            if time.monotonic() - last[0] >= 5:
                last[0] = time.monotonic()
                file_receipt(row, status='downloading', transport='hf_xet',
                             downloaded_bytes=min(received[0], row['bytes']))

    final = MODEL / row['file']
    partial = final.with_name(final.name + '.xet.part')
    file_receipt(row, status='downloading', transport='hf_xet', downloaded_bytes=0)
    # file_hash is an opaque CAS address required by the official transfer API,
    # not an additional application hash computation or post-download hash check.
    info = PyXetDownloadInfo(destination_path=str(partial), hash=data.file_hash, file_size=row['bytes'])
    download_files([info], endpoint=initial.endpoint,
                   token_info=(initial.access_token, initial.expiration_unix_epoch),
                   token_refresher=refresh, progress_updater=[transferred])
    if partial.stat().st_size != row['bytes']:
        raise RuntimeError('Xet asset byte-size mismatch')
    partial.replace(final)
    file_receipt(row, status='ready', transport='hf_xet', downloaded_bytes=row['bytes'])


def download_weight(manifest, row):
    if file_status(row)['status'] == 'ready':
        return
    final = MODEL / row['file']
    http_partial = final.with_name(final.name + '.http.part')
    # An existing HTTP range partial must resume, not start a second full copy.
    if not http_partial.exists():
        try:
            process = subprocess.Popen(
                [str(BASE_PYTHON), str(Path(__file__).absolute()), '--download-one', row['file']],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            last_bytes, last_progress = -1, time.monotonic()
            while True:
                try:
                    process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    transferred_bytes = file_status(row)['downloaded_bytes']
                    if transferred_bytes > last_bytes:
                        last_bytes, last_progress = transferred_bytes, time.monotonic()
                    if time.monotonic() - last_progress > 600:
                        process.kill()
                        process.wait()
                        break
            if process.returncode == 0 and file_status(row)['status'] == 'ready':
                emit('Ready ' + row['file'] + ' via official Xet')
                return
        except subprocess.TimeoutExpired:
            pass
        # Only our failed transfer's dedicated partial is removed. It is never
        # copied to another disk or mistaken for a complete preallocated file.
        partial = final.with_name(final.name + '.xet.part')
        if partial.exists():
            partial.unlink()
        emit('Using resumable verified-TLS HTTP ranges for ' + row['file'])
    download_http(manifest, row)
    emit('Ready ' + row['file'] + ' via HTTP ranges')


def prepare_environment():
    marker = ENV / 'KNOWLINE_ENV_OWNER.json'
    if ENV.exists() and not marker.exists():
        raise RuntimeError('Existing teacher environment lacks this preparation ownership marker')
    ENV.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(marker, {'repo': REPO, 'created_at': now(), 'student_environment_modified': False})
    elif load_json(marker)['repo'] != REPO:
        raise RuntimeError('Teacher environment ownership mismatch')
    py = ENV / 'bin/python'
    if not (ENV / 'pyvenv.cfg').exists():
        result = subprocess.run([str(BASE_PYTHON), '-m', 'venv', '--system-site-packages', str(ENV)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Teacher venv creation failed; exit ' + str(result.returncode))
    site = ENV / ('lib/python' + str(sys.version_info.major) + '.' + str(sys.version_info.minor)) / 'site-packages'
    # Conda-hosted venvs already inherit this, but the explicit .pth also makes
    # reuse robust when jev itself is a venv. Local teacher wheels take precedence.
    torch_site = str(importlib.metadata.distribution('torch').locate_file(''))
    if Path(torch_site).resolve() == site.resolve():
        raise RuntimeError('Preparation must be launched with the unchanged Student Python')
    (site / 'knowline-reuse-jev.pth').write_text(torch_site + '\n')
    constraints = ENV / 'preserve-student-runtime.txt'
    preserved = []
    for dist in importlib.metadata.distributions(path=[torch_site]):
        name = dist.metadata.get('Name', '')
        if name.lower() in ['torch', 'torchvision', 'torchaudio', 'triton'] or name.lower().startswith('nvidia-'):
            preserved.append(name + '==' + dist.version)
    constraints.write_text('\n'.join(sorted(set(preserved))) + '\n')
    emit('Installing Transformers 5.12.1 into envs/knowline; reusing Student Torch/CUDA read-only')
    command = [str(py), '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
               '--no-cache-dir', '--only-binary=:all:', '--upgrade-strategy', 'only-if-needed',
               '--constraint', str(constraints), '--index-url', 'https://mirrors.aliyun.com/pypi/simple',
               'transformers==5.12.1']
    # No raw dependency resolver/HTTP errors are persisted: they can contain
    # proxy details. Successful package/version facts are recorded below instead.
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
    if result.returncode:
        command[command.index('https://mirrors.aliyun.com/pypi/simple')] = 'https://pypi.org/simple'
        emit('Teacher package installation retry using public PyPI')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
    if result.returncode:
        raise RuntimeError('Teacher package installation failed; exit ' + str(result.returncode))
    # Required installation-time runtime imports only. No model/config/tokenizer
    # loading, model code execution from the repo, GPU allocation or forward pass.
    code = '''import json, importlib.metadata as md
from pathlib import Path
import torch, transformers
from transformers import Gemma4ForConditionalGeneration, AutoModelForImageTextToText
assert transformers.__version__ == "5.12.1"
print(json.dumps({"python": __import__("sys").executable,
 "torch": torch.__version__, "torch_file": torch.__file__, "torch_cuda": torch.version.cuda,
 "transformers": transformers.__version__, "transformers_file": transformers.__file__,
 "gemma4_class": Gemma4ForConditionalGeneration.__name__,
 "auto_class": AutoModelForImageTextToText.__name__,
 "packages": {n: md.version(n) for n in ["transformers", "huggingface_hub", "tokenizers", "accelerate", "safetensors"]}}))
'''
    result = subprocess.run([str(py), '-c', code], cwd=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=180)
    if result.returncode:
        # A local-only dependency diagnostic gives the operator actionable package
        # names without leaking traceback strings or network/authentication data.
        lines = result.stderr.splitlines()
        last = lines[-1] if lines else 'No diagnostic'
        safe = last if last.startswith(('ModuleNotFoundError:', 'ImportError:', 'RuntimeError:', 'ValueError:', 'AttributeError:')) and 'http' not in last.lower() and 'token' not in last.lower() else 'Runtime import failed'
        emit('Teacher installation import: ' + safe[:400])
        raise RuntimeError('Required built-in Gemma4 runtime import failed')
    runtime = json.loads(result.stdout.strip().splitlines()[-1])
    if not Path(runtime['torch_file']).is_relative_to(ROOT / 'envs/jev'):
        raise RuntimeError('Teacher must reuse the unchanged Student Torch installation')
    if not Path(runtime['transformers_file']).is_relative_to(ENV):
        raise RuntimeError('Teacher Transformers must be isolated from Student')
    runtime.update(complete=True, prepared_at=now(), system_site_packages=True,
                   native_sources_executed=False, model_loaded=False, gpu_process_started=False,
                   student_environment_modified=False)
    write_json(ENV_READY, runtime)
    emit('Teacher runtime ready: Transformers ' + runtime['transformers'] + ', shared Torch ' + runtime['torch'])
    return runtime


def complete_receipt(manifest, runtime):
    config = load_json(MODEL / 'config.json')
    index = load_json(MODEL / 'model.safetensors.index.json')
    shards = {x['file'] for x in manifest['files'] if x['kind'] == 'weights'}
    if set(index['weight_map'].values()) != shards:
        raise RuntimeError('Native weight index and downloaded shard set differ')
    if config['architectures'] != ['Gemma4ForConditionalGeneration'] or config['model_type'] != 'gemma4':
        raise RuntimeError('Unexpected native teacher model architecture')
    if config['text_config']['final_logit_softcapping'] != 20.833333333333336:
        raise RuntimeError('Native folded-temperature softcap must remain unchanged')
    for row in manifest['files']:
        if file_status(row)['status'] != 'ready':
            raise RuntimeError('Teacher readiness requires every complete native asset')
    manifest.update(complete=True, completed_at=now(), files=[file_status(row) for row in manifest['files']])
    write_json(MANIFEST, manifest)
    receipt = {
        'complete': True, 'prepared_at': now(), 'repo': REPO,
        'revision': manifest['revision'], 'model_path': str(MODEL),
        'actual_model_path': str(MODEL.resolve()), 'python': str(ENV / 'bin/python'),
        'teacher_python': str(ENV / 'bin/python'), 'dtype': 'bfloat16',
        'architecture': config['architectures'][0], 'backend': 'hf',
        'runtime': runtime, 'asset_manifest': str(MANIFEST),
        'environment_receipt': str(ENV_READY),
        'model_config': str(MODEL / 'config.json'),
        'native_sources': [str(MODEL / x) for x in NATIVE],
        'artifacts': [str(MANIFEST), str(ENV_READY), str(MODEL / 'config.json'),
                      str(MODEL / 'model.safetensors.index.json'),
                      *[str(MODEL / x) for x in NATIVE], str(MODEL / 'INFERENCE.md'),
                      str(MODEL / 'temperatures.json')],
        'semantics': {
            'read_twice': True, 'twice_max_tokens': 3000,
            'layout': 'Native knowline_layout.py, chat style; un-repeated request <=3000 tokens',
            'probability': 'Independent per-pair noul true/false label-token softmax; P(true)=P(match)',
            'folded_readout_temperature': 1.44, 'additional_noul_temperature': 1.0,
            'do_not_apply_temperature_twice': True,
            'score_only_additional_temperature': load_json(MODEL / 'temperatures.json')['score'],
            'final_logit_softcapping': config['text_config']['final_logit_softcapping'],
            'reference': str(MODEL / 'INFERENCE.md'),
        },
        'weight_shards': len(shards),
        'weight_bytes': sum(x['bytes'] for x in manifest['files'] if x['kind'] == 'weights'),
        'extra_tests': False, 'extra_file_hash_checks': False,
        'native_sources_executed': False, 'model_loaded': False,
        'public_weights_auto_scp': False,
    }
    write_json(READY, receipt)
    progress(manifest, 'ready', complete=True)
    emit('TEACHER_READY.json published; all assets and isolated runtime ready')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=[3, 4], default=3)
    parser.add_argument('--download-one', help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure()
    if args.download_one:
        manifest = load_json(MANIFEST)
        rows = [x for x in manifest['files'] if x['file'] == args.download_one and x['kind'] == 'weights']
        if len(rows) != 1:
            raise RuntimeError('Unknown native teacher shard')
        download_xet(manifest, rows[0])
        return
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'TEACHER_ASSETS.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        emit('Another teacher asset preparer holds the separate asset lock; leaving it unchanged')
        return
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + '\n')
    lock.flush()
    manifest = {'files': []}
    try:
        setup_paths()
        manifest = pinned_manifest()
        progress(manifest, 'preparing')
        missing = sum(row['bytes'] - file_status(row)['downloaded_bytes'] for row in manifest['files'])
        if shutil.disk_usage(MODEL).free < missing + 5 * 1024 ** 3:
            raise RuntimeError('Dedicated teacher backing has insufficient free capacity')
        emit('Preparing actual thirteen-shard BF16 teacher; ' + str(args.workers) + ' bounded download workers')
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            environment = pool.submit(prepare_environment)
            for row in manifest['files']:
                if row['kind'] != 'weights' and row['bytes'] <= 1024 ** 2 and file_status(row)['status'] != 'ready':
                    download_http(manifest, row)
                    emit('Ready native asset ' + row['file'])
                    progress(manifest, 'small-assets')
            # A slow LFS tokenizer transfer must not serialize all 62.55 GB of
            # weights behind it. Preserve its existing range partial, in parallel.
            jobs = [pool.submit(download_http, manifest, row) for row in manifest['files']
                    if row['kind'] != 'weights' and row['bytes'] > 1024 ** 2
                    and file_status(row)['status'] != 'ready']
            jobs += [pool.submit(download_weight, manifest, row) for row in manifest['files'] if row['kind'] == 'weights']
            pending = set(jobs + [environment])
            failures = []
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=10,
                                                       return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(type(error).__name__)
                        if future is environment:
                            write_json(ASSETS / 'ENVIRONMENT_BLOCKED.json', {
                                'complete': False, 'updated_at': now(),
                                'error_type': type(error).__name__, 'python': str(ENV / 'bin/python'),
                            })
                        emit('Preparation component blocked: ' + type(error).__name__)
                progress(manifest, 'weights-and-environment', component_errors=failures)
            if failures:
                raise RuntimeError('Teacher preparation component failure')
            runtime = environment.result()
        complete_receipt(manifest, runtime)
    except BaseException as error:
        # This new task owns readiness, but never touches EXPERIMENT.lock or
        # another unit's pipeline state. No secret-bearing exception reprs.
        progress(manifest, 'blocked', error_type=type(error).__name__)
        emit('Teacher preparation blocked: ' + type(error).__name__ + '; inspect asset progress and safe diagnostics')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
