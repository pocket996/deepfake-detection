"""Use the midpoint of the optimal validation decision interval, without test tuning."""
import argparse
import csv
import json
import shutil

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import torch

from core import HERE, load_detector, save_json
from train import metrics, select_threshold, write_csv
from train_generation import GenerationImages, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['single', 'joint'], default='single')
    options = parser.parse_args()
    out = HERE / ('artifacts/generation/joint' if options.experiment == 'joint' else 'artifacts/generation')
    state = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    if state.get('threshold_strategy') == 'Midpoint of optimal validation balanced-accuracy interval':
        print('This checkpoint already has its validation-only final threshold', flush=True)
        return
    # Read only validation before fixing the new threshold.
    with (out / 'val_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        validation = list(csv.DictReader(stream))
    y = np.array([int(r['label']) for r in validation])
    p = np.array([float(r['score']) for r in validation])
    boundary = select_threshold(y, p)
    below, above = p[p < boundary], p[p >= boundary]
    threshold = float((below.max() + above.min()) / 2) if len(below) and len(above) else boundary
    assert balanced_accuracy_score(y, p >= threshold) == balanced_accuracy_score(y, p >= boundary)
    old_threshold = state['frame_threshold']
    state['frame_threshold'] = state['video_threshold'] = threshold
    state['threshold_strategy'] = 'Midpoint of optimal validation balanced-accuracy interval'
    torch.save(state, out / 'model.pt')
    save_json(out / 'threshold_selection.json', dict(
        data='validation only', optimal_observed_boundary=boundary,
        lower_validation_score=float(below.max()) if len(below) else None,
        upper_validation_score=float(above.min()) if len(above) else None,
        old_threshold=old_threshold, final_threshold=threshold,
        validation_balanced_accuracy=float(balanced_accuracy_score(y, p >= threshold)),
        test_or_user_image_used=False))
    # Only after saving the threshold, re-score metrics at that locked threshold.
    shutil.copyfile(out / 'metrics.json', out / 'pre_calibration_metrics.json')
    report = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    report['threshold'] = threshold
    report['selection'] += '; threshold placed at midpoint of same optimal validation decision interval'
    for split in ('val', 'test', 'heldout_generator'):
        path = out / f'{split}_predictions.csv'
        if not path.is_file():
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        probabilities = np.array([float(r['score']) for r in rows])
        summary = metrics([int(r['label']) for r in rows], probabilities, threshold)
        summary['per_source_real_false_positive_rate'] = {}
        for source in ('LFW', 'FFHQ-256-mirror'):
            indices = [i for i, r in enumerate(rows) if r['source'] == source and r['label'] == '0']
            summary['per_source_real_false_positive_rate'][source] = dict(n=len(indices),
                false_positive_rate=float(np.mean(probabilities[indices] >= threshold)))
        report['splits'][split] = summary
        for row, value in zip(rows, probabilities):
            row['prediction'] = int(value >= threshold)
        write_csv(path, rows)
    manifest = out / 'manifest.csv' if options.experiment == 'joint' else HERE / 'data/generation_manifest.csv'
    with manifest.open(encoding='utf-8', newline='') as stream:
        test = [r for r in csv.DictReader(stream) if r['split'] == 'test']
    args = argparse.Namespace(batch_size=48, workers=2, device='cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    encoder, head, _ = load_detector(out / 'model.pt', args.device)
    network = torch.nn.Sequential(encoder, head)
    ps = score(network, test, args, stress=True)
    report['test_jpeg60'] = metrics([int(r['label']) for r in test], ps, threshold)
    save_json(out / 'metrics.json', report)
    print('Validation-only final threshold:', threshold, flush=True)


if __name__ == '__main__':
    main()
