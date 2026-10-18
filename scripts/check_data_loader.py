"""Read an actual batch from each split and verify the shared image loader."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceguard.data import FaceImageDataset
from torch.utils.data import DataLoader, Subset
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    results = {}
    for split in ('train', 'val', 'test'):
        data = FaceImageDataset(args.manifest, split)
        # Include both labels rather than inspecting only the alphabetically
        # first fake method in a sorted frame manifest.
        selected = []
        for label in ('0', '1'):
            selected.extend([i for i, r in enumerate(data.rows) if r['label'] == label][:4])
        if len(selected) != 8:
            raise ValueError(f'{split}: need at least four real and fake frames')
        batch = next(iter(DataLoader(Subset(data, selected), batch_size=8,
                                     shuffle=False, num_workers=0)))
        pixels, labels = batch['image'], batch['label']
        if (pixels.ndim != 4 or pixels.shape[1:] != (3, 256, 256)
                or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1
                or set(labels.tolist()) != {0, 1}):
            raise ValueError('Invalid image batch or labels')
        results[split] = dict(samples=len(data), shape=list(pixels.shape),
                              labels=labels.tolist(), dtype=str(pixels.dtype))
    results['manifest'] = str(args.manifest)
    (ROOT / 'outputs/data-loader-check.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
