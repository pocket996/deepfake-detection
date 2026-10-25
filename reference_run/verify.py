"""Check saved metrics, CPU/GPU model parity, raw video preprocessing, and no-face behavior."""
import csv
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch

from core import HERE, PROJECT, Predictor, image_transform, load_detector, load_rows, save_json
from train import select_threshold


def main(variant='frozen'):
    start = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    artifact = HERE / 'artifacts'
    prefix = 'finetuned_' if variant == 'finetuned' else ''
    checkpoint = artifact / ('model_finetuned.pt' if variant == 'finetuned' else 'model.pt')
    report = json.loads((artifact / f'{prefix}metrics.json').read_text(encoding='utf-8'))
    manifest = PROJECT / 'data/manifests/ffpp-frames-full-93e2b370610e.csv'
    rows = load_rows(manifest)
    checks = {}
    # CUDA convolution kernels/TF32 can differ between batch=1 and batch=96.
    # Compare probabilities within 0.001, recording actual deviations.
    probability_tolerance = 1e-3
    checks['probability_tolerance'] = probability_tolerance
    with (artifact / f'{prefix}test_videos.csv').open(encoding='utf-8-sig', newline='') as stream:
        videos = list(csv.DictReader(stream))
    y = [int(v['label']) for v in videos]
    p = [float(v['score']) for v in videos]
    t = report['splits']['test']['video']['threshold']
    assert abs(roc_auc_score(y, p) - report['splits']['test']['video']['auc']) < 1e-12
    assert abs(balanced_accuracy_score(y, np.asarray(p) >= t) - report['splits']['test']['video']['balanced_accuracy']) < 1e-12
    checks['independent_video_metrics'] = 'passed'
    # A perfect ordering must have an attainable threshold with perfect balanced accuracy.
    threshold = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert balanced_accuracy_score([0, 0, 1, 1], np.asarray([0.1, 0.2, 0.8, 0.9]) >= threshold) == 1
    checks['threshold_rule'] = 'passed'
    selected = [next(r for r in rows if r['split'] == 'test' and r['label'] == label) for label in ('0', '1')]
    cpu_model, cpu_head, state = load_detector(checkpoint, 'cpu')
    with Image.open(PROJECT / selected[0]['path']) as image:
        x = image_transform()(image.convert('RGB'))[None]
    with torch.inference_mode():
        cpu_score = float(cpu_head(cpu_model(x)).sigmoid().item())
    if torch.cuda.is_available():
        gpu_model, gpu_head, _ = load_detector(checkpoint, 'cuda')
        with torch.inference_mode():
            gpu_score = float(gpu_head(gpu_model(x.cuda())).sigmoid().item())
        assert abs(cpu_score - gpu_score) < 2e-4, (cpu_score, gpu_score)
        checks['cpu_gpu_probability_error'] = abs(cpu_score - gpu_score)
        del gpu_model, gpu_head
    del cpu_model, cpu_head
    predictor = Predictor(checkpoint)
    with (artifact / f'{prefix}test_frames.csv').open(encoding='utf-8-sig', newline='') as stream:
        frame_predictions = {r['sample_id']: float(r['score']) for r in csv.DictReader(stream)}
    for row in selected:
        image_result = predictor.predict(PROJECT / row['path'], face_crop=True)
        image_error = abs(image_result['score'] - frame_predictions[row['sample_id']])
        assert image_error < probability_tolerance, (row['sample_id'], image_result['score'], frame_predictions[row['sample_id']])
        video_result = predictor.predict(PROJECT / row['source_video'])
        expected = next(v for v in videos if v['video_id'] == row['parent_video_id'])
        error = abs(video_result['score'] - float(expected['score']))
        assert error < probability_tolerance, (row['source_video'], error)
        checks[row['parent_video_id']] = dict(raw_video_vs_manifest_score_error=error,
                                             image_vs_batch_score_error=image_error,
                                             frames=video_result['faces'])
        save_json(artifact / f"{prefix}example_{row['label']}.json", video_result)
    with tempfile.TemporaryDirectory(dir=artifact) as folder:
        blank = Path(folder) / 'blank.png'
        Image.new('RGB', (256, 256), 'gray').save(blank)
        result = predictor.predict(blank)
        assert result['decision'] == 'no_face' and result['score'] is None
        checks['no_face_is_not_real_label'] = 'passed'
    save_json(artifact / f'{prefix}verification.json', dict(checks=checks, seconds=time.monotonic() - start))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['frozen', 'finetuned'], default='frozen')
    main(parser.parse_args().variant)
