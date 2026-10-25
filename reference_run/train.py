"""Frozen ImageNet ResNet18 + trained logistic head. No test-driven tuning."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score,
                             confusion_matrix, precision_score, recall_score, f1_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from core import HERE, PROJECT, digest, image_transform, load_rows, save_json


class Frames(Dataset):
    def __init__(self, rows):
        self.rows, self.transform = rows, image_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(PROJECT / row['path']) as image:
            image = image.convert('RGB')
            pixel_hash = __import__('hashlib').sha256(image.tobytes()).hexdigest()
            tensor = self.transform(image)
        return tensor, pixel_hash


def aggregate(rows, probabilities):
    groups = defaultdict(list)
    for row, p in zip(rows, probabilities):
        groups[row['parent_video_id']].append((row, float(p)))
    return [dict(video_id=k, label=int(v[0][0]['label']), method=v[0][0]['method'],
                 group_id=v[0][0]['group_id'], split=v[0][0]['split'],
                 score=float(np.mean([p for _, p in v])), frames=len(v)) for k, v in sorted(groups.items())]


def select_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    indices = np.flatnonzero(valid)
    return float(thresholds[indices[np.argmax((tpr - fpr)[indices])]])


def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    return dict(n=len(y), auc=float(roc_auc_score(y, p)),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                threshold=float(threshold))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=PROJECT / 'data/manifests/ffpp-frames-full-93e2b370610e.csv')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Invalid batch size/workers')
    start = time.monotonic()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv = __import__('cv2')
    cv.setNumThreads(1)
    torch.hub.set_dir(str(HERE / 'weights/torch'))
    out = HERE / 'artifacts'
    out.mkdir(exist_ok=True)
    rows = load_rows(args.manifest)
    fingerprint = digest(args.manifest)
    save_json(out / 'data_audit.json', dict(manifest_sha256=fingerprint, frames=len(rows),
        videos=len({r['parent_video_id'] for r in rows}),
        groups=len({r['group_id'] for r in rows}),
        frames_by_split=dict(Counter(r['split'] for r in rows)),
        frames_by_method=dict(Counter(r['method'] for r in rows)),
        source_group_split_overlap=False, all_files_present=True,
        identity_audit=False, note='Source IDs are not biometric identity verification'))
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    features_path, metadata_path = out / 'features.npy', out / 'features.json'
    if features_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text())['manifest_sha256'] == fingerprint:
        features = np.load(features_path)
        if features.shape != (len(rows), 512):
            raise ValueError('Feature cache shape mismatch')
        print('Reusing matching feature cache', flush=True)
    else:
        features = np.empty((len(rows), 512), dtype=np.float32)
        loader = DataLoader(Frames(rows), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=args.device.startswith('cuda'))
        hash_splits, offset = {}, 0
        with torch.inference_mode():
            for batch_index, (images, hashes) in enumerate(loader):
                for index, image_hash in enumerate(hashes):
                    split = rows[offset + index]['split']
                    if image_hash in hash_splits and hash_splits[image_hash] != split:
                        raise ValueError('Exact decoded-image duplicate crosses splits; resolve before training')
                    hash_splits[image_hash] = split
                z = model(images.to(args.device, non_blocking=True)).cpu().numpy()
                features[offset:offset + len(z)] = z
                offset += len(z)
                if batch_index % 20 == 0 or offset == len(rows):
                    elapsed = time.monotonic() - start
                    progress = dict(stage='features', done=offset, total=len(rows), seconds=elapsed,
                                    frames_per_second=offset / elapsed)
                    save_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        np.save(features_path, features)
        save_json(metadata_path, dict(manifest_sha256=fingerprint, shape=list(features.shape),
                  preprocessing='RGB square face crop -> 224x224 bilinear -> ImageNet normalize',
                  exact_decoded_pixel_cross_split_duplicates=0,
                  backbone='resnet18 IMAGENET1K_V1 frozen FP32'))
    feature_seconds = time.monotonic() - start
    labels = np.asarray([int(r['label']) for r in rows])
    masks = {s: np.asarray([r['split'] == s for r in rows]) for s in ('train', 'val', 'test')}
    scaler = StandardScaler().fit(features[masks['train']])
    # Equal weight per video within a class, then equal total class weight.
    train_rows = [r for r in rows if r['split'] == 'train']
    counts = Counter(r['parent_video_id'] for r in train_rows)
    class_videos = Counter(int(r['label']) for r in {r['parent_video_id']: r for r in train_rows}.values())
    sample_weight = np.array([1 / counts[r['parent_video_id']] / class_videos[int(r['label'])] for r in train_rows])
    sample_weight *= len(sample_weight) / sample_weight.sum()
    xtrain = scaler.transform(features[masks['train']])
    xval = scaler.transform(features[masks['val']])
    val_rows = [r for r in rows if r['split'] == 'val']
    candidates, best_auc, best = [], -1., None
    for regularization in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=regularization, max_iter=2000, random_state=42, solver='lbfgs')
        clf.fit(xtrain, labels[masks['train']], sample_weight=sample_weight)
        if clf.n_iter_[0] >= clf.max_iter:
            raise RuntimeError('Head did not converge')
        p = clf.predict_proba(xval)[:, 1]
        videos = aggregate(val_rows, p)
        auc = float(roc_auc_score([v['label'] for v in videos], [v['score'] for v in videos]))
        candidates.append(dict(C=regularization, val_video_auc=auc, iterations=int(clf.n_iter_[0])))
        print('Validation candidate:', candidates[-1], flush=True)
        if auc > best_auc:
            best_auc, best = auc, clf
    # Fold StandardScaler into a torch head: (x - mean) / scale @ w + b.
    weight = best.coef_[0] / scaler.scale_
    bias = best.intercept_[0] - np.dot(weight, scaler.mean_)
    head = torch.nn.Linear(512, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight, dtype=torch.float32)[None])
        head.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    with torch.no_grad():
        probabilities = head(torch.from_numpy(features)).sigmoid().flatten().numpy()
    reference = best.predict_proba(scaler.transform(features[masks['val']]))[:, 1]
    max_error = float(np.max(np.abs(probabilities[masks['val']] - reference)))
    if max_error > 1e-4:
        raise RuntimeError(f'Export parity error {max_error}')
    val_videos = aggregate(val_rows, probabilities[masks['val']])
    video_threshold = select_threshold([v['label'] for v in val_videos], [v['score'] for v in val_videos])
    frame_threshold = select_threshold(labels[masks['val']], probabilities[masks['val']])
    checkpoint = dict(backbone={k: v.cpu() for k, v in model.state_dict().items()},
        head=head.state_dict(), video_threshold=video_threshold, frame_threshold=frame_threshold,
        model_name='resnet18_frozen_logistic', seed=42, manifest_sha256=fingerprint,
        yunet_sha256=digest(HERE / 'weights/face_detection_yunet_2023mar.onnx'),
        training_scope='FF++ c23 four known manipulation methods',
        selected_C=float(best.C), feature_normalization='ImageNet', input_size=224)
    torch.save(checkpoint, out / 'model.pt')
    evaluation = dict(selection='C by validation video AUC; thresholds by validation balanced accuracy',
                      candidates=candidates, selected_C=float(best.C),
                      head_export_max_probability_error=max_error, splits={})
    for split in ('val', 'test'):
        selected_rows = [r for r in rows if r['split'] == split]
        ps = probabilities[masks[split]]
        videos = aggregate(selected_rows, ps)
        y, p = [v['label'] for v in videos], [v['score'] for v in videos]
        summary = dict(frame=metrics(labels[masks[split]], ps, frame_threshold),
                       video=metrics(y, p, video_threshold), by_method={})
        for method in ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'):
            subset = [v for v in videos if v['method'] in ('original', method)]
            summary['by_method'][method] = metrics([v['label'] for v in subset], [v['score'] for v in subset], video_threshold)
        evaluation['splits'][split] = summary
        for v in videos:
            v['prediction'] = int(v['score'] >= video_threshold)
        write_csv(out / f'{split}_videos.csv', videos)
        write_csv(out / f'{split}_frames.csv', [dict(sample_id=r['sample_id'], video_id=r['parent_video_id'],
            label=int(r['label']), score=float(p)) for r, p in zip(selected_rows, ps)])
        if split == 'test':
            rng = np.random.default_rng(42)
            source_groups = sorted({v['group_id'] for v in videos})
            by_group = {g: [i for i, v in enumerate(videos) if v['group_id'] == g] for g in source_groups}
            boot = []
            for _ in range(500):
                indices = [i for g in rng.choice(source_groups, len(source_groups), replace=True) for i in by_group[g]]
                boot.append(roc_auc_score(np.asarray(y)[indices], np.asarray(p)[indices]))
            summary['video_auc_95pct_group_bootstrap'] = np.quantile(boot, [0.025, 0.975]).tolist()
    evaluation['timing_seconds'] = dict(feature_stage=feature_seconds, total=time.monotonic() - start)
    evaluation['environment'] = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
        device=args.device, gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
    save_json(out / 'metrics.json', evaluation)
    save_json(out / 'progress.json', dict(stage='complete', seconds=time.monotonic() - start))
    print(json.dumps(evaluation, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
