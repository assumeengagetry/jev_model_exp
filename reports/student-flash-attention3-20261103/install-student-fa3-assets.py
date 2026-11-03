#!/usr/bin/env python3
"""Fetch a matching, isolated native FA3 build without changing Python packages."""
from datetime import datetime, timezone
import importlib.util
import importlib
import json
from pathlib import Path
import sys
import re

HOME = Path('/home/dev/assumeengage')
DEST = HOME / 'cache/attention-kernels/flash-attn3-torch29-cu128'
REPO = 'kernels-community/flash-attn3'
PREFIX = 'build/torch29-cxx11-cu128-x86_64-linux/'
sys.path.insert(0, str(HOME / 'scripts'))
from download_utils import direct_hf_request, direct_hf_download


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def main():
    import torch
    if torch.__version__.split('+')[0] != '2.9.1' or torch.version.cuda != '12.8' or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError('This prebuilt FA3 requires the unchanged Torch2.9/CUDA12.8/CXX11 runtime')
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = DEST / 'ASSETS.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data['repo'] != REPO or data['build'] != PREFIX.rstrip('/'):
            raise RuntimeError('Existing kernel manifest differs; preserved')
    else:
        response, connection = direct_hf_request('https://huggingface.co/api/models/' + REPO + '?blobs=true')
        try:
            if response.status != 200: raise RuntimeError('Kernel metadata unavailable, HTTP ' + str(response.status))
            metadata = json.loads(response.read())
        finally: response.close(); connection.close()
        files = []
        for item in metadata['siblings']:
            name = item['rfilename']
            if name.startswith(PREFIX) and '__pycache__' not in name:
                relative = name[len(PREFIX):]
                if '..' in Path(relative).parts: raise RuntimeError('Unexpected kernel asset path')
                files.append({'name': relative, 'bytes': item['size']})
        if not files or len([f for f in files if f['name'].endswith('.so')]) != 1:
            raise RuntimeError('Expected exactly one native FA3 extension')
        data = {'repo': REPO, 'revision': metadata['sha'], 'build': PREFIX.rstrip('/'),
                'files': files, 'extra_hash_checks': False, 'complete': False}
        atomic(manifest, data)
    for row in data['files']:
        target = DEST / row['name']; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != row['bytes']: raise RuntimeError('Existing final kernel asset has wrong size')
            continue
        partial = target.with_name(target.name + '.part')
        direct_hf_download('https://huggingface.co/' + REPO + '/resolve/' + data['revision'] + '/' + PREFIX + row['name'], partial, row['bytes'])
        if partial.stat().st_size != row['bytes']: raise RuntimeError('Incomplete native kernel transfer')
        partial.replace(target)
        print(json.dumps({'asset_ready': row['name'], 'bytes': row['bytes']}), flush=True)
    # Installation check only: load native symbols; no CUDA tensor or forward.
    name = '_jev_student_fa3_installation'
    spec = importlib.util.spec_from_file_location(name, DEST / '__init__.py', submodule_search_locations=[str(DEST)])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    interface = sys.modules[name + '.flash_attn_interface']
    ops = sys.modules[name + '._ops'].ops
    flags = importlib.import_module(name + '.flash_attn_config').CONFIG.get('build_flags', {})
    if any(bool(value) for key, value in flags.items()
           if 'DISABLE' in key and any(part in key for part in ('BACKWARD', 'BWD', 'HDIM128', 'SM90'))):
        raise RuntimeError('The selected FA3 build disables required Hopper/head128/backward support')
    if not all(callable(getattr(module, key, None)) for key in ('flash_attn_func', 'flash_attn_varlen_func')):
        raise RuntimeError('FA3 forward/varlen exports are missing')
    if not hasattr(interface, 'FlashAttnFunc') or not hasattr(interface, 'FlashAttnVarlenFunc') or not hasattr(ops, 'bwd'):
        raise RuntimeError('A real training-capable FA3 backward implementation is required')
    data.update(complete=True, installed_utc=datetime.now(timezone.utc).isoformat())
    atomic(manifest, data)
    ready = {'complete': True, 'backend': 'flash_attention_3', 'implementation': 'flash_attn3_hopper',
             'provider': REPO, 'build': data['build'], 'kernel_directory': str(DEST),
             'torch': torch.__version__, 'cuda': torch.version.cuda, 'cxx11_abi': True,
             'forward_export': module.flash_attn_func.__module__,
             'varlen_export': module.flash_attn_varlen_func.__module__, 'native_backward_export': True,
             'gpu_forward_or_backward_executed': False, 'global_environment_modified': False,
             'sdpa_fallback': False, 'installed_utc': data['installed_utc']}
    atomic(DEST / 'READY.json', ready); print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        message = re.sub(r'https?://\S+', '<url>', str(error))[:600]
        print(json.dumps({'installation_failed': type(error).__name__, 'message': message, 'sdpa_fallback': False}), flush=True)
        raise SystemExit(1)
