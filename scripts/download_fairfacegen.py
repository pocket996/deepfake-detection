"""Download only the two official FairFaceGen seed archives and README.

Retains partial downloads for resumption; checks official MD5 and ZIP CRCs.
Does not extract archives or assign training splits.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://zenodo.org/records/17062365/files/'
FILES = {
    'README.md': 'dcb4e2be522d17a657f36965a9730f32',
    'flux-seed.zip': '87fd189e875d11f6f09269ad1d140afc',
    'sd35-seed.zip': '4e9b2531444ec20b7394d0f194111f7d',
}
DEST = ROOT / 'data/raw/fairfacegen'
STATUS = ROOT / 'outputs/fairfacegen-download-status.json'


def save(value):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(STATUS)


def md5(path):
    result = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fetch(name, expected, state):
    target = DEST / name
    part = DEST / (name + '.part')
    if not target.exists() or md5(target) != expected:
        for attempt in range(4):
            try:
                offset = part.stat().st_size if part.exists() else 0
                headers = {'User-Agent': 'FaceGuard-research-data/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(BASE + name + '?download=1', headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    append = response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected Content-Range')
                    if not append:
                        offset = 0
                    length = response.headers.get('Content-Length')
                    total = offset + int(length) if length else None
                    last = 0
                    with part.open('ab' if append else 'wb') as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            offset += len(block)
                            if time.monotonic() - last > 10:
                                if shutil.disk_usage(DEST).free < 5 * 1024**3:
                                    raise OSError('Less than 5 GiB free disk space')
                                state['current'] = {'file': name, 'bytes': offset, 'total': total}
                                save(state)
                                print(f'{name}: {offset / 1e6:.1f} / {total / 1e6 if total else 0:.1f} MB', flush=True)
                                last = time.monotonic()
                    if total is not None and offset != total:
                        raise ValueError('Incomplete response')
                if md5(part) != expected:
                    part.unlink()
                    raise ValueError('Official MD5 mismatch; discarded partial archive')
                part.replace(target)
                break
            except Exception as error:
                print(f'{name}, attempt {attempt + 1}: {error}', flush=True)
                if attempt == 3:
                    raise
                time.sleep(2)
    receipt = {'bytes': target.stat().st_size, 'md5': expected, 'url': BASE + name, 'verified': True}
    if name.endswith('.zip'):
        with zipfile.ZipFile(target) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f'ZIP CRC failure: {bad}')
            members = [entry for entry in archive.infolist() if not entry.is_dir()]
            receipt['files'] = len(members)
            receipt['images'] = sum(Path(entry.filename).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} for entry in members)
            receipt['uncompressed_bytes'] = sum(entry.file_size for entry in members)
            receipt['sample_paths'] = [entry.filename for entry in members[:5]]
    state['files'][name] = receipt
    save(state)
    print(f'Verified {name}: {receipt}', flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    state = {'state': 'running', 'source': 'https://zenodo.org/records/17062365', 'files': {}}
    save(state)
    try:
        for name, expected in FILES.items():
            fetch(name, expected, state)
        state['state'] = 'complete'
        state.pop('current', None)
    except Exception as error:
        state['state'] = 'incomplete'
        state['error'] = str(error)
        raise
    finally:
        save(state)


if __name__ == '__main__':
    main()
