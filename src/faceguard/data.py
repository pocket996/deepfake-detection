"""Manifest schema shared by extracted video frames and future image datasets."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
BASE_FIELDS = ['sample_id', 'path', 'label', 'dataset', 'task', 'method',
               'compression', 'target_id', 'source_id', 'group_id', 'split',
               'media_type', 'provenance_status']
FRAME_FIELDS = BASE_FIELDS + ['parent_video_id', 'source_video', 'frame_index',
                              'timestamp', 'crop_box', 'face_score']


def read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_manifest(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def validate_manifest(rows):
    if not rows:
        raise ValueError('Manifest has no samples')
    seen_ids, seen_paths = set(), set()
    group_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for row in rows:
        if any(not row.get(key) for key in ('sample_id', 'path', 'group_id', 'dataset')):
            raise ValueError('Missing sample_id/path/group_id/dataset')
        if row.get('split') not in ('train', 'val', 'test'):
            raise ValueError('Each sample requires an explicit train/val/test split')
        if str(row.get('label')) not in ('0', '1'):
            raise ValueError('label must be 0 (real) or 1 (fake)')
        if row['sample_id'] in seen_ids or row['path'] in seen_paths:
            raise ValueError('Duplicate sample ID or path')
        seen_ids.add(row['sample_id'])
        seen_paths.add(row['path'])
        # group_id is global: derived datasets must retain canonical source groups.
        group_splits[row['group_id']].add(row['split'])
        for key in ('target_id', 'source_id'):
            if row.get(key):
                source_splits[(row['dataset'], row[key])].add(row['split'])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError('Related samples cross splits (group_id leakage)')
    if any(len(splits) > 1 for splits in source_splits.values()):
        raise ValueError('Related source videos cross splits')


class FaceImageDataset(Dataset):
    """Read verified image rows; retain provenance for video-level evaluation.

    Default images are RGB float32 CHW tensors in [0, 1], resized to image_size.
    A custom transform receives an RGB PIL image and must return a tensor.
    Video files must first be converted to frame manifests; no random split occurs.
    """
    def __init__(self, manifest, split, image_size=256, transform=None, root=ROOT):
        rows = read_manifest(manifest)
        validate_manifest(rows)
        if split not in ('train', 'val', 'test') or image_size < 1:
            raise ValueError('Invalid split or image_size')
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No samples in split {split}')
        if any(r.get('media_type') != 'image' for r in self.rows):
            raise ValueError('FaceImageDataset requires images; preprocess videos first')
        if any(r.get('provenance_status') != 'verified' for r in self.rows):
            raise ValueError('Unverified provenance cannot enter this training loader')
        self.root = Path(root)
        self.image_size, self.transform = image_size, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['path'])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert('RGB')
            if self.transform:
                tensor = self.transform(image)
            else:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return {'image': tensor, 'label': torch.tensor(int(row['label']), dtype=torch.long),
                'sample_id': row['sample_id'], 'group_id': row['group_id'],
                'dataset': row['dataset'], 'method': row['method'],
                'parent_video_id': row.get('parent_video_id', '')}
