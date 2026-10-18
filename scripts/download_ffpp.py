"""Download the complete standard FF++ c23 video set from approved official sources.

Requires the approved tools/download-FaceForensics.py. Existing files are verified
against the previous download receipt; interrupted .part files are downloaded again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--limit-per-method', type=int, default=None)
    parser.add_argument('--accept-terms', action='store_true', required=True,
                        help='Confirm the already-approved FF++ terms of use.')
    args = parser.parse_args()
    if args.limit_per_method is not None and args.limit_per_method < 1:
        parser.error('--limit-per-method must be positive')
    spec = importlib.util.spec_from_file_location(
        'ffpp_official', ROOT / 'tools/download-FaceForensics.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    previous_argv = sys.argv
    sys.argv = ['download', str(ROOT / 'data/raw/ffpp'), '--server', 'EU2', '-c', 'c23']
    try:
        config = official.parse_args()
    finally:
        sys.argv = previous_argv
    with urllib.request.urlopen(config.base_url + official.FILELIST_URL, timeout=45) as r:
        pairs = json.load(r)
    if not pairs or not all(len(p) == 2 and all(re.fullmatch(r'\d{3}', i) for i in p)
                            for p in pairs):
        raise ValueError('Unexpected official file list')
    save_json(ROOT / 'data/manifests/ffpp-official-pairs.json', pairs)
    receipt_path = ROOT / 'data/manifests/ffpp-download-receipts.json'
    receipts = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
    tasks = []
    for method in METHODS:
        names = ([i for p in pairs for i in p] if method == 'original' else
                 ['_'.join(q) for p in pairs for q in (p, p[::-1])])
        names = list(dict.fromkeys(names))
        if args.limit_per_method:
            names = names[:args.limit_per_method]
        for name in names:
            relative = official.DATASETS[method] + '/c23/videos/' + name + '.mp4'
            path = ROOT / 'data/raw/ffpp' / relative
            tasks.append((method, path, config.base_url + relative))
    status_path = ROOT / 'outputs/ffpp-download-status.json'
    status = {'state': 'running', 'scope': 'standard FF++ original + four methods, c23',
              'expected': len(tasks), 'completed': 0, 'bytes': 0, 'failed': [],
              'limit_per_method': args.limit_per_method}

    def update():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        save_json(receipt_path, receipts)
        save_json(status_path, status)

    def download(task):
        method, path, url = task
        key = path.relative_to(ROOT).as_posix()
        old = receipts.get(key)
        if old and path.is_file() and path.stat().st_size == old['bytes']:
            if digest(path) == old['sha256']:
                return key, old
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix('.mp4.part')
        for attempt in range(3):
            try:
                if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                    raise OSError('Less than 20 GiB free; free disk space before resuming')
                with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as out:
                    length = response.headers.get('Content-Length')
                    written = 0
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        out.write(block)
                        written += len(block)
                        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                            raise OSError('Less than 20 GiB free')
                if length and written != int(length):
                    raise ValueError('Incomplete response body')
                with part.open('rb') as stream:
                    if b'ftyp' not in stream.read(32):
                        raise ValueError('Response is not an MP4 file')
                checksum = digest(part)
                part.replace(path)
                return key, {'method': method, 'bytes': written, 'sha256': checksum}
            except Exception:
                if part.exists():
                    part.unlink()
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    update()
    print('Downloading', len(tasks), 'videos with', args.workers, 'workers', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                key, receipt = future.result()
                receipts[key] = receipt
                status['completed'] += 1
                status['bytes'] += receipt['bytes']
            except Exception as exc:
                status['failed'].append({'path': task[1].relative_to(ROOT).as_posix(),
                                         'error': str(exc)})
            update()
            if (status['completed'] + len(status['failed'])) % 25 == 0:
                print(status['completed'], '/', len(tasks), 'completed;',
                      len(status['failed']), 'failed', flush=True)
    status['state'] = 'complete' if not status['failed'] else 'incomplete'
    update()
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
