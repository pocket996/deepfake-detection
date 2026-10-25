"""Unseen-method experiment: exclude NeuralTextures from training AND validation."""
from collections import Counter
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

from core import HERE, PROJECT, digest, load_rows, save_json
from train import aggregate, metrics, select_threshold, write_csv


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    artifact = HERE / 'artifacts'
    path = PROJECT / 'data/manifests/ffpp-frames-full-93e2b370610e.csv'
    rows = load_rows(path)
    meta = json.loads((artifact / 'features.json').read_text(encoding='utf-8'))
    if digest(path) != meta['manifest_sha256']:
        raise ValueError('Feature cache does not match manifest')
    features = np.load(artifact / 'features.npy')
    heldout = 'NeuralTextures'
    masks = {s: np.array([r['split'] == s and r['method'] != heldout for r in rows]) for s in ('train', 'val')}
    scaler = StandardScaler().fit(features[masks['train']])
    selected = {s: [r for r, use in zip(rows, masks[s]) if use] for s in masks}
    y = np.array([int(r['label']) for r in selected['train']])
    frame_counts = Counter(r['parent_video_id'] for r in selected['train'])
    class_counts = Counter(int(r['label']) for r in {r['parent_video_id']: r for r in selected['train']}.values())
    w = np.array([1 / frame_counts[r['parent_video_id']] / class_counts[int(r['label'])] for r in selected['train']])
    w *= len(w) / w.sum()
    xtrain, xval = scaler.transform(features[masks['train']]), scaler.transform(features[masks['val']])
    candidates, best_auc, best = [], -1, None
    for c in (0.001, 0.01, 0.1, 1.):
        clf = LogisticRegression(C=c, max_iter=2000, random_state=42)
        clf.fit(xtrain, y, sample_weight=w)
        if clf.n_iter_[0] >= clf.max_iter:
            raise RuntimeError('Classifier did not converge')
        p = clf.predict_proba(xval)[:, 1]
        videos = aggregate(selected['val'], p)
        auc = float(roc_auc_score([v['label'] for v in videos], [v['score'] for v in videos]))
        candidates.append(dict(C=c, seen_validation_auc=auc))
        if auc > best_auc:
            best_auc, best = auc, clf
    pv = best.predict_proba(xval)[:, 1]
    vv = aggregate(selected['val'], pv)
    threshold = select_threshold([v['label'] for v in vv], [v['score'] for v in vv])
    frame_threshold = select_threshold([int(r['label']) for r in selected['val']], pv)
    # Lock all choices using seen-method validation; test only the held-out method + real.
    mask = np.array([r['split'] == 'test' and r['method'] in ('original', heldout) for r in rows])
    test_rows = [r for r, use in zip(rows, mask) if use]
    pt = best.predict_proba(scaler.transform(features[mask]))[:, 1]
    vt = aggregate(test_rows, pt)
    result = dict(heldout_method=heldout, heldout_in_training=False, heldout_in_validation=False,
        selected_C=float(best.C), candidates=candidates,
        test_video=metrics([v['label'] for v in vt], [v['score'] for v in vt], threshold),
        validation_seen_video=metrics([v['label'] for v in vv], [v['score'] for v in vv], threshold),
        note='Method holdout inside FF++; not an independent dataset or modern generator evaluation',
        seconds=time.monotonic() - start)
    checkpoint = torch.load(artifact / 'model.pt', map_location='cpu', weights_only=True)
    weight = best.coef_[0] / scaler.scale_
    bias = best.intercept_[0] - np.dot(weight, scaler.mean_)
    checkpoint.update(head={'weight': torch.tensor(weight, dtype=torch.float32)[None],
                            'bias': torch.tensor([bias], dtype=torch.float32)},
                      selected_C=float(best.C), video_threshold=threshold, frame_threshold=frame_threshold,
                      training_scope='FF++ c23 excluding NeuralTextures in both train and validation')
    torch.save(checkpoint, artifact / 'model_holdout.pt')
    save_json(artifact / 'holdout_metrics.json', result)
    write_csv(artifact / 'holdout_test_videos.csv', vt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
