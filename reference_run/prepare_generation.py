"""Create source-auditable, identity-grouped real/FLUX splits; reserve SD3.5."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import threading
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps
import pyarrow.parquet as pq

from core import HERE, PROJECT, crop_face, digest, save_json
from generation_common import encode_canonical

LOCAL = threading.local()
DATA = HERE / 'data'


def unpack_real():
    folder = DATA / 'raw/lfw'
    with tarfile.open(folder / 'lfw-funneled.tgz', 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.jpg'):
                continue
            path = (folder / member.name).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError('Unsafe archive member')
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    path.write_bytes(source.read())
    image_folder = DATA / 'raw/ffhq256/images'
    image_folder.mkdir(parents=True, exist_ok=True)
    for shard in sorted((DATA / 'raw/ffhq256').glob('*.parquet')):
        index = 0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=['image']):
            for row in batch.to_pylist():
                # Preserve a stable source shard/row index; upstream IDs aren't guaranteed.
                path = image_folder / f'{shard.stem}_{index:05d}.png'
                if not path.exists():
                    path.write_bytes(row['image']['bytes'])
                index += 1
        print('Unpacked', shard.name, index, 'real images', flush=True)


def jobs():
    for path in sorted((DATA / 'raw/lfw/lfw_funneled').glob('*/*.jpg')):
        yield dict(source='LFW', method='real', label=0, raw_path=str(path),
                   group='lfw:' + path.parent.name, sample_id='lfw_' + path.stem)
    for path in sorted((DATA / 'raw/ffhq256/images').glob('*.png')):
        yield dict(source='FFHQ-256-mirror', method='real', label=0, raw_path=str(path),
                   group='ffhq:' + path.stem, sample_id='ffhq_' + path.stem)
    for method, filename in [('FLUX', 'flux-seed.zip'), ('SD35', 'sd35-seed.zip')]:
        archive = PROJECT / 'data/raw/fairfacegen' / filename
        with zipfile.ZipFile(archive) as stream:
            for name in sorted(stream.namelist()):
                if not name.endswith('.jpg'):
                    continue
                identity = Path(name).stem.split('-')[0]
                yield dict(source='FairFaceGen', method=method, label=1,
                    raw_path=str(archive), archive_member=name,
                    group=f'fairfacegen:{method}:{identity}', sample_id=f'{method}_{Path(name).stem}')


def process(job):
    if not hasattr(LOCAL, 'detector'):
        LOCAL.detector = cv2.FaceDetectorYN.create(str(HERE / 'weights/face_detection_yunet_2023mar.onnx'), '', (320, 320), 0.8, 0.3, 5000)
        LOCAL.archives = {}
    try:
        if job.get('archive_member'):
            if job['raw_path'] not in LOCAL.archives:
                LOCAL.archives[job['raw_path']] = zipfile.ZipFile(job['raw_path'])
            stream = io.BytesIO(LOCAL.archives[job['raw_path']].read(job['archive_member']))
        else:
            stream = job['raw_path']
        with Image.open(stream) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            raw_size = image.size
            face = crop_face(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), LOCAL.detector)
        if face is None:
            return None, dict(sample_id=job['sample_id'], reason='no_face', source=job['source'])
        encoded = encode_canonical(face)
        path = DATA / 'processed/generation128' / job['method'] / (job['sample_id'] + '.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        with Image.open(io.BytesIO(encoded)) as decoded:
            pixel_hash = hashlib.sha256(decoded.convert('RGB').tobytes()).hexdigest()
        return dict(job, path=path.relative_to(HERE).as_posix(), raw_width=raw_size[0],
                    raw_height=raw_size[1], pixel_sha256=pixel_hash), None
    except Exception as error:
        return None, dict(sample_id=job['sample_id'], reason=str(error), source=job['source'])


def group_split(group):
    number = int(hashlib.sha256(('generation-v1:' + group).encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if number < 0.7 else 'val' if number < 0.85 else 'test'


def main():
    start = time.monotonic()
    cv2.setNumThreads(1)
    unpack_real()
    tasks = list(jobs())
    rows, errors = [], []
    with ThreadPoolExecutor(4) as pool:
        for index, (row, error) in enumerate(pool.map(process, tasks), 1):
            if row:
                rows.append(row)
            if error:
                errors.append(error)
            if index % 1000 == 0 or index == len(tasks):
                progress = dict(stage='crop', done=index, total=len(tasks), accepted=len(rows), rejected=len(errors), seconds=time.monotonic() - start)
                save_json(HERE / 'artifacts/generation_preparation_progress.json', progress)
                print(json.dumps(progress), flush=True)
    # Quarantine ALL exact duplicate connected source groups across datasets/identities.
    # Do not let selecting a representative move a held-out generator into training.
    hashes = defaultdict(list)
    for row in rows:
        hashes[row['pixel_sha256']].append(row)
    excluded_groups = set()
    for values in hashes.values():
        if len({v['group'] for v in values}) > 1:
            excluded_groups.update(v['group'] for v in values)
    seen = set()
    clean = []
    for row in rows:
        if row['group'] in excluded_groups or row['pixel_sha256'] in seen:
            continue
        seen.add(row['pixel_sha256'])
        row['split'] = 'heldout_generator' if row['method'] == 'SD35' else group_split(row['group'])
        clean.append(row)
    fields = sorted(set().union(*(row.keys() for row in clean)))
    manifest = DATA / 'generation_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean)
    counts = Counter((row['split'], row['method'], row['source']) for row in clean)
    report = dict(raw_candidates=len(tasks), accepted=len(clean), rejected=errors,
        counts=[dict(split=s, method=m, source=d, images=n) for (s, m, d), n in sorted(counts.items())],
        exact_duplicate_groups_quarantined=len(excluded_groups), manifest_sha256=digest(manifest),
        preprocessing='YuNet 0.8 largest face, 1.3 margin ->256 AREA ->128 AREA ->JPEG90 4:4:4',
        split='LFW known identity; FFHQ mirror per image; generator per seed; SD35 excluded from train and val',
        identity_limitations='FFHQ mirror has no identity metadata; no biometric identity or near-duplicate audit; cross-dataset identities may overlap',
        license_limitations='FairFaceGen and mirror cards do not state full applicable image terms; no redistribution or commercial clearance established',
        external_user_image_in_training=False, seconds=time.monotonic() - start)
    save_json(HERE / 'artifacts/generation_data_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rejected'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
