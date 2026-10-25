"""Fine-tune all ResNet18 layers; choose checkpoints using validation video AUC only."""
import argparse
from collections import defaultdict
import json
import random
import time

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from core import HERE, PROJECT, backbone, digest, image_transform, load_rows, save_json
from train import aggregate, metrics, select_threshold, write_csv


class Images(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), image_transform()]) if augment else image_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(PROJECT / row['path']) as image:
            return self.transform(image.convert('RGB')), float(row['label'])


@torch.inference_mode()
def predict(network, rows, batch, workers, device):
    network.eval()
    output = []
    loader = DataLoader(Images(rows), batch_size=batch, num_workers=workers, pin_memory=device == 'cuda')
    for images, _ in loader:
        # Evaluate in FP32, as the exported model/inference do.
        output.extend(network(images.to(device, non_blocking=True)).sigmoid().flatten().cpu().tolist())
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('Invalid epochs/batch/workers')
    start = time.monotonic()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    __import__('cv2').setNumThreads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = HERE / 'artifacts'
    manifest = PROJECT / 'data/manifests/ffpp-frames-full-93e2b370610e.csv'
    rows = load_rows(manifest)
    # Require completed decoded-image audit from the frozen baseline.
    meta = json.loads((out / 'features.json').read_text(encoding='utf-8'))
    if meta['manifest_sha256'] != digest(manifest) or meta['exact_decoded_pixel_cross_split_duplicates'] != 0:
        raise ValueError('Run full data audit via train.py first')
    initial = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    encoder = backbone()
    encoder.load_state_dict(initial['backbone'])  # still the original, frozen ImageNet backbone
    head = nn.Linear(512, 1)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    network = nn.Sequential(encoder, head).to(device)
    optimizer = torch.optim.AdamW([dict(params=encoder.parameters(), lr=1e-4),
                                   dict(params=head.parameters(), lr=3e-4)], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    video_groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            video_groups[row['parent_video_id']].append(row)
    for group in video_groups.values():
        group.sort(key=lambda r: int(r['frame_index']))
    val_rows = [r for r in rows if r['split'] == 'val']
    history, best_auc, selected_epoch = [], -1., 0
    for epoch in range(args.epochs):
        # Four interleaved frames per video per epoch; four epochs see all 16 frames.
        train_rows = [row for group in video_groups.values() for i, row in enumerate(group)
                      if i % 4 == epoch % 4]
        loader = DataLoader(Images(train_rows, True), batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=device == 'cuda', generator=torch.Generator().manual_seed(42 + epoch))
        network.train()
        total_loss, count = 0., 0
        for step, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=device == 'cuda'):
                logits = network(images).flatten()
                # Equal total real/fake weight: 1 real video for every 4 fake videos.
                weights = torch.where(labels == 0, 2.5, 0.625)
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
            if step % 50 == 0:
                progress = dict(stage='finetune', epoch=epoch + 1, epochs=args.epochs,
                    frames_done=count, epoch_frames=len(train_rows), seconds=time.monotonic() - start)
                save_json(out / 'finetuned_progress.json', progress)
                print(json.dumps(progress), flush=True)
        scheduler.step()
        probabilities = predict(network, val_rows, args.batch_size, args.workers, device)
        videos = aggregate(val_rows, probabilities)
        auc = float(roc_auc_score([v['label'] for v in videos], [v['score'] for v in videos]))
        item = dict(epoch=epoch + 1, train_frames=count, train_weighted_loss=total_loss / count,
                    validation_video_auc=auc, seconds=time.monotonic() - start)
        history.append(item)
        save_json(out / 'finetuned_history.json', history)
        print('Epoch:', json.dumps(item), flush=True)
        if auc > best_auc:
            best_auc, selected_epoch = auc, epoch + 1
            video_threshold = select_threshold([v['label'] for v in videos], [v['score'] for v in videos])
            frame_threshold = select_threshold([int(r['label']) for r in val_rows], probabilities)
            checkpoint = dict(backbone={k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                head={k: v.cpu().clone() for k, v in head.state_dict().items()},
                model_name='resnet18_full_finetune', seed=42, selected_epoch=selected_epoch,
                video_threshold=video_threshold, frame_threshold=frame_threshold,
                manifest_sha256=digest(manifest), yunet_sha256=initial['yunet_sha256'],
                input_size=224, feature_normalization='ImageNet',
                training_scope='FF++ c23 four known face manipulation methods')
            torch.save(checkpoint, out / 'model_finetuned.pt')
    # Test is first scored after checkpoint selection has finished.
    state = torch.load(out / 'model_finetuned.pt', map_location='cpu', weights_only=True)
    encoder.load_state_dict(state['backbone'])
    head.load_state_dict(state['head'])
    result = dict(selection='Epoch by validation video AUC; thresholds by validation balanced accuracy',
                  history=history, selected_epoch=selected_epoch, splits={})
    for split in ('val', 'test'):
        selected = [r for r in rows if r['split'] == split]
        p = predict(network, selected, args.batch_size, args.workers, device)
        videos = aggregate(selected, p)
        y = [v['label'] for v in videos]
        vp = [v['score'] for v in videos]
        summary = dict(frame=metrics([int(r['label']) for r in selected], p, state['frame_threshold']),
                       video=metrics(y, vp, state['video_threshold']), by_method={})
        for method in ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'):
            subset = [v for v in videos if v['method'] in ('original', method)]
            summary['by_method'][method] = metrics([v['label'] for v in subset], [v['score'] for v in subset], state['video_threshold'])
        for video in videos:
            video['prediction'] = int(video['score'] >= state['video_threshold'])
        write_csv(out / f'finetuned_{split}_videos.csv', videos)
        write_csv(out / f'finetuned_{split}_frames.csv', [dict(sample_id=r['sample_id'],
            video_id=r['parent_video_id'], label=int(r['label']), score=float(score)) for r, score in zip(selected, p)])
        result['splits'][split] = summary
    result['timing_seconds'] = time.monotonic() - start
    result['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
    save_json(out / 'finetuned_metrics.json', result)
    save_json(out / 'finetuned_progress.json', dict(stage='complete', seconds=time.monotonic() - start))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
