"""Pin and hash-check two public FFHQ-256 mirror shards, not the full 7.6 GB."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = 'bitmind/ffhq-256'
REVISION = '05bc8040d1bfd541e836d09d347a943c292658ff'
SHARDS = [
    ('train-00000-of-00016.parquet', 475591078, 'db2e83ffafac0d8c31b6abc30816538a4e436a5214a7b778200ba240b267d8e0'),
    ('train-00001-of-00016.parquet', 478023608, 'a218636f9647dc9261d3e772b0037d18923f31530311845c60b67e25e4014968'),
]


def download(spec):
    name, size, expected = spec
    folder = HERE / 'data/raw/ffhq256'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}'
    if not target.exists() or target.stat().st_size != size:
        part = target.with_suffix('.parquet.part')
        for attempt in range(3):
            try:
                request = urllib.request.Request(url + f'?download=true&request={int(time.time())}', headers={'User-Agent': 'FaceGuard-research/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as stream:
                    done, last = 0, 0
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last > 10:
                            print(f'{name}: {done / 1e6:.1f}/{size / 1e6:.1f} MB', flush=True)
                            last = time.monotonic()
                if done != size:
                    raise ValueError('Shard byte count mismatch')
                part.replace(target)
                break
            except Exception as error:
                print(name, attempt + 1, str(error), flush=True)
                if attempt == 2:
                    raise
    h = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError(f'{name}: mirror LFS SHA256 mismatch')
    return dict(file=name, bytes=size, sha256=expected, url=url)


def main():
    receipts = list(ThreadPoolExecutor(2).map(download, SHARDS))
    (HERE / 'data/raw/ffhq256/receipts.json').write_text(json.dumps(dict(
        repo=REPO, revision=REVISION, files=receipts,
        provenance='Publisher-reported FFHQ-256 mirror, not an official NVIDIA distribution',
        upstream='https://github.com/NVlabs/ffhq-dataset',
        limitations='No identity or per-image original author/license metadata in this mirror; no biometric identity audit'), indent=2), encoding='utf-8')
    print('Both FFHQ mirror shards SHA256 verified', flush=True)


if __name__ == '__main__':
    main()
