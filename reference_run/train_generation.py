"""Train a separate generation detector; never tune with SD3.5 or the user's GPT image."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from core import HERE, backbone, digest, image_transform, save_json
from train import metrics, select_threshold, write_csv

OUT = HERE / 'artifacts/generation'


class GenerationImages(Dataset):
    def __init__(self, rows, augment=False, stress=False):
        self.rows, self.augment, self.stress = rows, augment, stress
        self.base = image_transform()
        self.color = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(HERE / row['path']) as source:
            image = source.convert('RGB')
        if self.augment:
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = self.color(image)
            size = random.choice((96, 112, 128))
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            if random.random() < .25:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, .7)))
        if self.stress or (self.augment and random.random() < .7):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=60 if self.stress else random.randint(55, 98))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert('RGB')
        return self.base(image), float(row['label']), float(row.get('weight', 1))


@torch.inference_mode()
def score(network, rows, args, stress=False):
    network.eval()
    output = []
    loader = DataLoader(GenerationImages(rows, stress=stress), batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.device == 'cuda')
    for images, _, _ in loader:
        output.extend(network(images.to(args.device, non_blocking=True)).sigmoid().flatten().cpu().tolist())
    return np.asarray(output)


def audit(rows, allow_sd35_training=False):
    groups, hashes = defaultdict(set), defaultdict(set)
    seen = set()
    for row in rows:
        if row['sample_id'] in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(row['sample_id'])
        if not allow_sd35_training and row['method'] == 'SD35' and row['split'] != 'heldout_generator':
            raise ValueError('Held-out generator leaked into tuning')
        if row['method'] not in ('real', 'FLUX', 'SD35'):
            raise ValueError('Unknown method')
        groups[row['group']].add(row['split'])
        hashes[row['pixel_sha256']].add(row['split'])
        if not (HERE / row['path']).is_file():
            raise FileNotFoundError(row['path'])
    if any(len(values) != 1 for mapping in (groups, hashes) for values in mapping.values()):
        raise ValueError('Group or exact pixel leakage')
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {'0', '1'}:
            raise ValueError('Missing class in ' + split)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--include-sd35', action='store_true', help='Separate joint experiment; SD3.5 is no longer held out')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error('Invalid training arguments')
    start = time.monotonic()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    if args.include_sd35:
        OUT = HERE / 'artifacts/generation/joint'
    OUT.mkdir(parents=True, exist_ok=True)
    if args.include_sd35:
        (OUT / 'READY.json').unlink(missing_ok=True)
    path = HERE / 'data/generation_manifest.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.include_sd35:
        from prepare_generation import group_split
        for row in rows:
            if row['method'] == 'SD35':
                row['split'] = group_split(row['group'])
        path = OUT / 'manifest.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit(rows, allow_sd35_training=args.include_sd35)
    train_rows = [r.copy() for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    # Balance labels, real-photo sources, then known identity groups within a source.
    group_counts = Counter(r['group'] for r in train_rows)
    source_groups = defaultdict(set)
    class_sources = defaultdict(set)
    for row in train_rows:
        bucket = row['source'] + ':' + row['method']
        source_groups[(row['label'], bucket)].add(row['group'])
        class_sources[row['label']].add(bucket)
    for row in train_rows:
        row['weight'] = 1 / (2 * len(class_sources[row['label']]) *
            len(source_groups[(row['label'], row['source'] + ':' + row['method'])]) * group_counts[row['group']])
    scale = len(train_rows) / sum(r['weight'] for r in train_rows)
    for row in train_rows:
        row['weight'] *= scale
    encoder = backbone()
    imagenet = torch.load(HERE / 'weights/torch/checkpoints/resnet18-f37072fd.pth', map_location='cpu', weights_only=True)
    imagenet.pop('fc.weight')
    imagenet.pop('fc.bias')
    encoder.load_state_dict(imagenet)
    head = nn.Linear(512, 1)
    nn.init.normal_(head.weight, std=.01)
    nn.init.zeros_(head.bias)
    network = nn.Sequential(encoder, head).to(args.device)
    optimizer = torch.optim.AdamW([dict(params=encoder.parameters(), lr=1e-4),
                                  dict(params=head.parameters(), lr=3e-4)], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    best_auc, history = -1, []
    for epoch in range(args.epochs):
        loader = DataLoader(GenerationImages(train_rows, augment=True), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=args.device == 'cuda',
            generator=torch.Generator().manual_seed(2026 + epoch))
        network.train()
        count, total_loss = 0, 0.
        for step, (images, labels, weights) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)
            labels, weights = labels.to(args.device, dtype=torch.float32), weights.to(args.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=args.device, enabled=args.device == 'cuda'):
                logits = network(images).flatten()
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count += len(labels)
            total_loss += float(loss.detach()) * len(labels)
            if step % 75 == 0:
                progress = dict(stage='train', epoch=epoch + 1, epochs=args.epochs,
                    done=count, total=len(train_rows), seconds=time.monotonic() - start)
                save_json(OUT / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        scheduler.step()
        p = score(network, val_rows, args)
        y = [int(r['label']) for r in val_rows]
        auc = float(roc_auc_score(y, p))
        item = dict(epoch=epoch + 1, weighted_train_loss=total_loss / count,
                    validation_auc=auc, seconds=time.monotonic() - start)
        history.append(item)
        save_json(OUT / 'history.json', history)
        print('Validation:', json.dumps(item), flush=True)
        if auc > best_auc:
            best_auc = auc
            threshold = select_threshold(y, p)
            state = dict(backbone={k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                head={k: v.cpu().clone() for k, v in head.state_dict().items()},
                model_name='resnet18_generation', selected_epoch=epoch + 1,
                frame_threshold=threshold, video_threshold=threshold, seed=2026,
                manifest_sha256=digest(path),
                yunet_sha256=digest(HERE / 'weights/face_detection_yunet_2023mar.onnx'),
                preprocessing='generation128_jpeg90', training_generators=['FLUX.1-dev', 'Stable Diffusion 3.5'] if args.include_sd35 else ['FLUX.1-dev'],
                real_sources=['LFW', 'FFHQ-256 mirror subset'],
                training_scope='Experimental whole-face generation detector; GPT not represented in training',
                task='whole_face_generation')
            torch.save(state, OUT / 'model.pt')
    # Lock all choices before touching test or the unseen generator.
    state = torch.load(OUT / 'model.pt', map_location='cpu', weights_only=True)
    encoder.load_state_dict(state['backbone'])
    head.load_state_dict(state['head'])
    threshold = state['frame_threshold']
    report = dict(history=history, selected_epoch=state['selected_epoch'],
        selection='Epoch by validation AUC; threshold by validation balanced accuracy',
        training_generators=state['training_generators'], heldout_generators=[] if args.include_sd35 else ['Stable Diffusion 3.5'],
        user_gpt_image_used_for_training_or_selection=False,
        threshold=threshold, splits={})
    test_real = [r for r in rows if r['split'] == 'test' and r['label'] == '0']
    for split in ('val', 'test', 'heldout_generator'):
        selected = [r for r in rows if r['split'] == split]
        if not selected:
            continue
        if split == 'heldout_generator':
            selected = test_real + selected
        p = score(network, selected, args)
        y = [int(r['label']) for r in selected]
        summary = metrics(y, p, threshold)
        summary['per_source_real_false_positive_rate'] = {}
        for source in ('LFW', 'FFHQ-256-mirror'):
            indices = [i for i, r in enumerate(selected) if r['source'] == source and r['label'] == '0']
            summary['per_source_real_false_positive_rate'][source] = dict(
                n=len(indices), false_positive_rate=float(np.mean(p[indices] >= threshold)))
        report['splits'][split] = summary
        write_csv(OUT / f'{split}_predictions.csv', [dict(sample_id=r['sample_id'], path=r['path'],
            source=r['source'], method=r['method'], group=r['group'], label=int(r['label']),
            score=float(value), prediction=int(value >= threshold)) for r, value in zip(selected, p)])
        if split == 'test':
            degraded = score(network, selected, args, stress=True)
            report['test_jpeg60'] = metrics(y, degraded, threshold)
    report.update(seconds=time.monotonic() - start,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
        train_images=len(train_rows), manifest_sha256=digest(path),
        limitations=['Generator/source differences can still inflate results',
            'FFHQ mirror identities not audited; only LFW identity names grouped',
            'GPT generalization requires independent evidence; one user image is not a benchmark'])
    save_json(OUT / 'metrics.json', report)
    save_json(OUT / 'progress.json', dict(stage='complete', seconds=time.monotonic() - start))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
