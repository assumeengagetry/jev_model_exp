"""Resumable HTTP downloads. No content hashes or model/data tests."""
import json
import subprocess
import time
import http.client
import socket
import ssl
import warnings
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

HOME = Path('/home/dev/assumeengage')
HF_ENDPOINT = 'https://hf-cdn.sufy.com'
_ADDRESS_CACHE = {}
_TLS_CONTEXT = ssl.create_default_context()
_TLS_CONTEXT.check_hostname = False
_HF_TOKEN_FILE = HOME / '.secrets/hf_token'
_HF_TOKEN = _HF_TOKEN_FILE.read_text().strip() if _HF_TOKEN_FILE.exists() else None


def connect_ipv4(address):
    last_error = None
    for family, kind, protocol, _, sockaddr in socket.getaddrinfo(address, 443, socket.AF_INET, socket.SOCK_STREAM):
        sock = socket.socket(family, kind, protocol)
        sock.settimeout(12)
        # Conservative MSS avoids oversized return packets through pod overlays.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1024)
        try:
            sock.connect(sockaddr)
            return sock
        except Exception as error:
            last_error = error
            sock.close()
    raise last_error


def direct_hf_request(url, headers=None, method='GET', body=None, follow_redirects=True):
    """Reach HF with a fully verified TLS connection and DNS-safe server name."""
    url = url.replace(HF_ENDPOINT + '/', 'https://huggingface.co/', 1)
    for _ in range(8):
        parsed = urlsplit(url)
        host = parsed.hostname
        addresses = ['3.168.73.111', '3.168.178.122'] if host == 'huggingface.co' else [host + '.']
        if host in _ADDRESS_CACHE:
            addresses = [_ADDRESS_CACHE[host]] + addresses
        last_error = None
        for address in addresses:
            try:
                raw = connect_ipv4(address)
                # Keep certificate-chain and original-hostname verification.
                sock = _TLS_CONTEXT.wrap_socket(raw, server_hostname=host + '.')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    ssl.match_hostname(sock.getpeercert(), host)
                sock.settimeout(30)
                _ADDRESS_CACHE[host] = sock.getpeername()[0]
                connection = http.client.HTTPConnection(host, timeout=90)
                connection.sock = sock
                request_headers = {'Host': host, 'User-Agent': 'huggingface_hub/0.36.2',
                                   'Connection': 'close', 'Accept-Encoding': 'identity'}
                if host == 'huggingface.co' and _HF_TOKEN:
                    request_headers['Authorization'] = 'Bearer ' + _HF_TOKEN
                request_headers.update(headers or {})
                target = parsed.path + ('?' + parsed.query if parsed.query else '')
                connection.request(method, target, body=body, headers=request_headers)
                response = connection.getresponse()
                break
            except Exception as error:
                last_error = error
        else:
            raise last_error
        if follow_redirects and response.status in {301, 302, 303, 307, 308}:
            url = urljoin(url, response.getheader('location'))
            if response.status == 303:
                method, body = 'GET', None
            response.close()
            connection.close()
            continue
        response.resolved_url = url
        return response, connection
    raise RuntimeError('Too many HF redirects')


def direct_hf_download(url, partial, size=None):
    deadline = time.monotonic() + 900
    control = partial.with_name(partial.name + '.aria2')
    if control.exists():
        # Segmented aria2 files are not contiguous prefixes; never resume them as one.
        if partial.exists(): partial.unlink()
        control.unlink()
    for attempt in range(5):
        try:
            request_url = url
            total = size
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Download time limit reached; partial data kept for resume')
                offset = partial.stat().st_size if partial.exists() else 0
                if total is not None and offset == total:
                    return
                end = offset + 8 * 1024 * 1024 - 1
                if total is not None: end = min(end, total - 1)
                response, connection = direct_hf_request(request_url, {'Range': f'bytes={offset}-{end}'})
                try:
                    if response.status == 416:
                        actual = response.getheader('content-range', '').rsplit('/', 1)[-1]
                        if actual.isdigit() and int(actual) == offset: return
                    if response.status not in {200, 206}:
                        raise RuntimeError(f'HF HTTP {response.status}: {response.read(200)!r}')
                    request_url = response.resolved_url
                    content_range = response.getheader('content-range', '')
                    if content_range:
                        actual = content_range.rsplit('/', 1)[-1]
                        if actual.isdigit(): total = int(actual)
                    mode = 'ab' if response.status == 206 and offset else 'wb'
                    expected = response.getheader('content-length')
                    received = 0
                    with partial.open(mode) as output:
                        while True:
                            if time.monotonic() > deadline:
                                raise TimeoutError('Download time limit reached; partial data kept for resume')
                            chunk = response.read1(64 * 1024)
                            if not chunk: break
                            output.write(chunk)
                            received += len(chunk)
                    if expected is not None and received != int(expected):
                        raise RuntimeError('Incomplete HTTP response')
                    if response.status == 200:
                        if size is not None and partial.stat().st_size != size:
                            raise RuntimeError('Incomplete file byte count')
                        return
                    if total is not None and partial.stat().st_size == total:
                        return
                finally:
                    response.close()
                    connection.close()
        except Exception:
            if attempt == 4 or time.monotonic() > deadline: raise
            time.sleep(1)


def get_json(url, cache=None):
    if cache and Path(cache).exists():
        try:
            return json.loads(Path(cache).read_text())
        except (ValueError, OSError):
            pass
    if url.startswith(HF_ENDPOINT):
        response, connection = direct_hf_request(url)
        try:
            if response.status != 200: raise RuntimeError(f'HF HTTP {response.status}')
            data = json.loads(response.read())
        finally:
            response.close(); connection.close()
    else:
        command = ['curl', '-4', '-fsSL', '--connect-timeout', '12', '--max-time', '90',
                   '--retry', '3', '--retry-delay', '1', url]
        result = subprocess.run(command, capture_output=True, text=True, timeout=400)
        if result.returncode:
            raise RuntimeError(result.stderr[-800:])
        data = json.loads(result.stdout)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(data, ensure_ascii=False))
    return data


def download(url, path, size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        return {'path': str(path.relative_to(HOME)), 'bytes': path.stat().st_size, 'status': 'present'}
    partial = path.with_name(path.name + '.part')
    if url.startswith(HF_ENDPOINT + '/datasets/'):
        direct_hf_download(url, partial, size)
        partial.replace(path)
        print(f'Downloaded {path.relative_to(HOME)} ({path.stat().st_size:,} bytes)', flush=True)
        return {'path': str(path.relative_to(HOME)), 'bytes': path.stat().st_size, 'status': 'downloaded'}
    command = ['curl', '-4', '-fsSL', '--connect-timeout', '15', '--speed-limit', '1024',
               '--speed-time', '60', '--max-time', '600', '--retry', '2', '--retry-max-time', '900',
               '--retry-all-errors', '--retry-delay', '2', '-C', '-', '-o', str(partial), url]
    result = subprocess.run(command, capture_output=True, text=True, timeout=930)
    if result.returncode == 33:
        # Some archive servers do not support HTTP Range; restart this partial only.
        index = command.index('-C')
        del command[index:index + 2]
        result = subprocess.run(command, capture_output=True, text=True, timeout=930)
    if result.returncode:
        raise RuntimeError(f'{path.name}: exit {result.returncode}: {(result.stdout + result.stderr)[-1000:]}')
    if size is not None and partial.stat().st_size != size:
        raise RuntimeError(f'{path.name}: incomplete byte count')
    partial.replace(path)
    print(f'Downloaded {path.relative_to(HOME)} ({path.stat().st_size:,} bytes)', flush=True)
    return {'path': str(path.relative_to(HOME)), 'bytes': path.stat().st_size, 'status': 'downloaded'}


def hf_metadata(repo):
    url = f'{HF_ENDPOINT}/api/datasets/{repo}?expand%5B%5D=siblings&expand%5B%5D=sha'
    return get_json(url, HOME / 'config/hub-metadata' / (repo.replace('/', '__') + '.json'))


def hf_url(repo, filename, revision='main'):
    return f'{HF_ENDPOINT}/datasets/{repo}/resolve/{revision}/{quote(filename, safe="/")}?download=true'
