"""Download scikit-learn's hash-pinned LFW archive into the isolated experiment."""
import hashlib
from pathlib import Path
import time
import urllib.request
import json

HERE = Path(__file__).resolve().parent
URL = 'https://ndownloader.figshare.com/files/5976015'
SHA256 = 'b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a'


def main():
    folder = HERE / 'data/raw/lfw'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'lfw-funneled.tgz'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print('Already verified LFW archive', flush=True)
        return
    part = target.with_suffix('.tgz.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={'User-Agent': 'FaceGuard-research/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as stream:
                total, done, last = int(response.headers.get('Content-Length', 0)), 0, 0
                h = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 5:
                        print(f'LFW: {done / 1e6:.1f}/{total / 1e6:.1f} MB', flush=True)
                        last = time.monotonic()
            if h.hexdigest() != SHA256:
                raise ValueError('Official scikit-learn SHA256 mismatch')
            part.replace(target)
            (folder / 'receipt.json').write_text(json.dumps(dict(url=URL, sha256=SHA256,
                bytes=done, checksum_source='scikit-learn 1.6.1 sklearn/datasets/_lfw.py',
                original_source='Labeled Faces in the Wild, UMass Amherst'), indent=2), encoding='utf-8')
            print('LFW SHA256 verified', flush=True)
            return
        except Exception as error:
            print(f'Attempt {attempt + 1}: {error}', flush=True)
            if attempt == 2:
                raise


if __name__ == '__main__':
    main()
