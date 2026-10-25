"""Inspect public source manifests without downloading a whole dataset."""
import concurrent.futures
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent


def fetch(url):
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'FaceGuard-research/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return url, response.read().decode('utf-8')
    except Exception as error:
        return url, str(error)


def main():
    urls = [
        'https://huggingface.co/api/datasets/bitmind/ffhq-256/tree/main/data?limit=100',
        'https://huggingface.co/datasets/bitmind/ffhq-256/raw/main/README.md',
        'https://huggingface.co/api/datasets/bitmind/ffhq-256',
    ]
    results = dict(concurrent.futures.ThreadPoolExecutor(3).map(fetch, urls))
    for archive in ('flux-seed.zip', 'sd35-seed.zip'):
        with zipfile.ZipFile(ROOT.parent / 'data/raw/fairfacegen' / archive) as stream:
            member = next(n for n in stream.namelist() if n.endswith('.jpg'))
            with Image.open(io.BytesIO(stream.read(member))) as image:
                results[archive] = dict(member=member, size=image.size)
    target = ROOT / 'artifacts/generation_source_inspection.json'
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for url, data in results.items():
        print(url, str(data)[:6000], flush=True)


if __name__ == '__main__':
    main()
