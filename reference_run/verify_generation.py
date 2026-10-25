"""Independent raw-image parity, routing, no-face, and held-out user GPT regression."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch

from core import HERE, Predictor, digest, image_transform, load_detector, save_json
from face_analysis import FaceAnalyzer, GenerationPredictor
from generation_common import canonical_face
from train_generation import audit


def materialize_raw(row, target):
    if row.get('archive_member'):
        with zipfile.ZipFile(row['raw_path']) as archive:
            target.write_bytes(archive.read(row['archive_member']))
    else:
        target.write_bytes(Path(row['raw_path']).read_bytes())
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['single', 'joint'], default='single')
    parser.add_argument('--user-image', type=Path, help='Optional local regression photo; never used for training or threshold selection')
    options = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(4)
    __import__('cv2').setNumThreads(1)
    out = HERE / ('artifacts/generation/joint' if options.experiment == 'joint' else 'artifacts/generation')
    official_archives = {'flux-seed.zip': '87fd189e875d11f6f09269ad1d140afc',
                         'sd35-seed.zip': '4e9b2531444ec20b7394d0f194111f7d'}
    for filename, expected in official_archives.items():
        with (HERE.parent / 'data/raw/fairfacegen' / filename).open('rb') as stream:
            assert hashlib.file_digest(stream, 'md5').hexdigest() == expected
    manifest = out / 'manifest.csv' if options.experiment == 'joint' else HERE / 'data/generation_manifest.csv'
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    audit(rows, allow_sd35_training=options.experiment == 'joint')
    with (out / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        tests = list(csv.DictReader(stream))
    report = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    assert abs(roc_auc_score([int(r['label']) for r in tests], [float(r['score']) for r in tests]) - report['splits']['test']['auc']) < 1e-12
    checks = dict(independent_test_auc='passed', group_and_pixel_split_audit='passed',
                  sd35_protocol='grouped train/val/test; NOT unseen' if options.experiment == 'joint' else 'excluded from train and val')
    analyzer = GenerationPredictor(checkpoint=out / 'model.pt')
    predictor = analyzer
    examples = HERE / ('data/examples/joint' if options.experiment == 'joint' else 'data/examples')
    examples.mkdir(parents=True, exist_ok=True)
    for label in ('0', '1'):
        row = next(r for r in rows if r['split'] == 'test' and r['label'] == label
                   and (label == '0' or options.experiment != 'joint' or r['method'] == 'SD35'))
        raw = materialize_raw(row, examples / f'generation_{label}.jpg')
        result = analyzer.predict(raw)
        assert result['task'] == 'whole_face_generation'
        expected = next(r for r in tests if r['sample_id'] == row['sample_id'])
        error = abs(result['score'] - float(expected['score']))
        assert error < 1e-3, (row['sample_id'], error)
        checks[f'raw_photo_parity_{label}'] = dict(sample_id=row['sample_id'], error=error)
        save_json(out / f'example_{label}.json', dict(result, known_label=int(label),
            source=row['source'], method=row['method'], sample_id=row['sample_id']))
    with tempfile.TemporaryDirectory(dir=out) as directory:
        blank = Path(directory) / 'no_face.png'
        Image.new('RGB', (256, 256), 'gray').save(blank)
        result = analyzer.predict(blank)
        assert result['decision'] == 'no_face' and result['score'] is None
        checks['no_face_is_not_real_label'] = 'passed'
    # Compare exported inference on CPU vs GPU on the same raw face/photo.
    photo = examples / 'generation_1.jpg'
    cpu_result = GenerationPredictor(checkpoint=out / 'model.pt', device='cpu').predict(photo)
    gpu_result = predictor.predict(photo)
    delta = abs(cpu_result['score'] - gpu_result['score'])
    assert delta < 1e-3, delta
    checks['cpu_gpu_probability_error'] = delta
    # Independent real video-frame source; report FPR only, not a fake/real AUC.
    with (HERE.parent / 'data/manifests/ffpp-frames-full-93e2b370610e.csv').open(encoding='utf-8', newline='') as stream:
        ffpp = list(csv.DictReader(stream))
    originals = {}
    for row in ffpp:
        if row['split'] == 'test' and row['label'] == '0':
            originals.setdefault(row['parent_video_id'], row)
    tensors = []
    for row in originals.values():
        with Image.open(HERE.parent / row['path']) as source:
            tensors.append(image_transform()(canonical_face(source.convert('RGB'))))
    values = []
    with torch.inference_mode():
        for offset in range(0, len(tensors), 48):
            x = torch.stack(tensors[offset:offset + 48]).to(predictor.device)
            values.extend(predictor.head(predictor.model(x)).sigmoid().flatten().cpu().tolist())
    checks['independent_ffpp_real_frames'] = dict(n=len(values),
        false_positive_rate=float(np.mean(np.array(values) >= predictor.state['frame_threshold'])),
        note='Real-only cross-source stress test, not a cross-dataset AUC')
    user_path = options.user_image
    if user_path is not None and user_path.is_file():
        result = analyzer.predict(user_path)
        assert result['faces'] == 1
        from core import crop_face
        import cv2
        with Image.open(user_path) as source:
            face = crop_face(cv2.cvtColor(np.asarray(source.convert('RGB')), cv2.COLOR_RGB2BGR), predictor.detector)
        pixel_hash = hashlib.sha256(np.asarray(canonical_face(face)).tobytes()).hexdigest()
        assert pixel_hash not in {r['pixel_sha256'] for r in rows}, 'User image overlaps experiment dataset'
        old = Predictor().predict(user_path)
        regression = dict(result, image_sha256=digest(user_path), model_sha256=digest(out / 'model.pt'),
            expected_origin='GPT-generated, supplied by user; not independently attributed',
            in_training_or_validation=False,
            detected_generation=result['decision'] == 'suspected_generation',
            previous_manipulation_model=old,
            note='One user regression image cannot establish GPT-wide accuracy; joint iteration follows the single-model observation')
        save_json(out / 'user_gpt_regression.json', regression)
        checks['user_gpt_image'] = dict(detected_generation=regression['detected_generation'],
            score=result['score'], threshold=result['threshold'], excluded_from_training=True)
    else:
        checks['user_gpt_image'] = dict(status='not_requested' if user_path is None else 'not_found',
                                      path=str(user_path) if user_path is not None else None)
    save_json(out / 'verification.json', dict(checks=checks, model_sha256=digest(out / 'model.pt'),
        manifest_sha256=digest(manifest), seconds=time.monotonic() - start))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
