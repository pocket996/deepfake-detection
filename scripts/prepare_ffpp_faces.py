"""Sample FF++ frames and detect/crop faces with the same settings in each video."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceguard.data import FRAME_FIELDS, read_manifest, validate_manifest, write_manifest


def sampled_frames(path, samples):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open video')
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total < 1 or fps <= 0:
            raise ValueError('Invalid frame count or FPS')
        targets = set(np.linspace(0, total - 1, min(samples, total), dtype=int).tolist())
        result = {}
        grabbed = 0
        while capture.grab():
            index = grabbed
            grabbed += 1
            if index in targets:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise ValueError(f'Cannot retrieve sampled frame {index}')
                result[index] = frame
        if grabbed != total or set(result) != targets:
            raise ValueError(f'Video frame traversal mismatch: expected {total}, got {grabbed}')
        return result, total, fps
    finally:
        capture.release()


def square_box(frame, detector, margin):
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    small = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return None
    # FF++ tracks one main face. Use the largest detected face consistently.
    face = max(faces, key=lambda row: row[2] * row[3])
    x, y, w, h = face[:4] / scale
    size = min(max(w, h) * margin, width, height)
    left = min(max(x + w / 2 - size / 2, 0), width - size)
    top = min(max(y + h / 2 - size / 2, 0), height - size)
    return ([left / width, top / height, (left + size) / width,
             (top + size) / height], float(face[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--pairs-per-split', type=int, default=1)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--margin', type=float, default=1.3)
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--model', type=Path, default=ROOT / 'weights/face_detection_yunet_2023mar.onnx')
    args = parser.parse_args()
    if min(args.frames, args.size, args.pairs_per_split) < 1 or args.margin < 1:
        parser.error('Positive frame/size/pair counts and margin >= 1 required')
    cv2.setNumThreads(1)
    if not args.model.is_file():
        parser.error('YuNet model is missing; see docs/ffpp-preparation.md')
    rows = read_manifest(ROOT / 'data/manifests/ffpp-videos.csv')
    validate_manifest(rows)
    receipts = json.loads((ROOT / 'data/manifests/ffpp-download-receipts.json').read_text())
    config = dict(frames=args.frames, size=args.size, margin=args.margin,
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), version=2)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / 'data/processed/ffpp' / key
    cache = output / '_records'
    cache.mkdir(parents=True, exist_ok=True)
    selected_groups = set()
    for split in ('train', 'val', 'test'):
        groups = sorted({r['group_id'] for r in rows if r['split'] == split},
                        key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        selected_groups.update(groups if args.all else groups[:args.pairs_per_split])
    selected = [row for row in rows if row['group_id'] in selected_groups]
    targets = {}
    for row in selected:
        targets.setdefault(row['target_id'], []).append(row)

    def process(target_rows):
        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
            raise OSError('Less than 10 GiB free; preprocessing stopped')
        original = next(row for row in target_rows if row['label'] == '0')
        record_path = cache / (original['target_id'] + '.json')
        fingerprint = hashlib.sha256(json.dumps(
            [(r['sample_id'], receipts[r['path']]['sha256']) for r in target_rows]
        ).encode()).hexdigest()
        if record_path.is_file():
            prior = json.loads(record_path.read_text())
            if (prior['fingerprint'] == fingerprint and not prior['errors']
                    and all((ROOT / r['path']).is_file() for r in prior['rows'])):
                return prior
        record = {'fingerprint': fingerprint, 'rows': [], 'errors': [], 'checks': [], 'no_face': []}
        detector = cv2.FaceDetectorYN.create(str(args.model), '', (320, 320), 0.8, 0.3, 5000)
        for video in target_rows:
            try:
                images, video_count, video_fps = sampled_frames(ROOT / video['path'], args.frames)
                record['checks'].append({'sample_id': video['sample_id'], 'frames': video_count,
                                         'fps': video_fps, 'sampled_frames': len(images)})
                for index, frame in images.items():
                    found = square_box(frame, detector, args.margin)
                    if found is None:
                        record['no_face'].append({'sample_id': video['sample_id'], 'frame_index': index})
                        continue
                    box, score = found
                    h, w = frame.shape[:2]
                    x1, y1 = int(box[0] * w), int(box[1] * h)
                    x2, y2 = min(w, int(np.ceil(box[2] * w))), min(h, int(np.ceil(box[3] * h)))
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError('Empty crop')
                    crop = cv2.resize(frame[y1:y2, x1:x2], (args.size, args.size), interpolation=cv2.INTER_AREA)
                    path = output / video['split'] / video['method'] / video['sample_id'] / f'{index:06d}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), crop):
                        raise OSError(f'Cannot save crop: {path}')
                    row = dict(video)
                    row.update(sample_id=video['sample_id'] + f'_frame_{index:06d}',
                        path=path.relative_to(ROOT).as_posix(), media_type='image',
                        parent_video_id=video['sample_id'], source_video=video['path'],
                        frame_index=index, timestamp=f'{index / video_fps:.6f}',
                        crop_box=json.dumps([x1, y1, x2, y2]), face_score=f'{score:.6f}')
                    record['rows'].append(row)
            except Exception as exc:
                record['errors'].append({'sample_id': video['sample_id'], 'error': str(exc)})
        temp = record_path.with_suffix('.tmp')
        temp.write_text(json.dumps(record), encoding='utf-8')
        temp.replace(record_path)
        return record

    name = 'full' if args.all else f'preview-{args.pairs_per_split}'
    manifest = ROOT / f'data/manifests/ffpp-frames-{name}-{key}.csv'
    report_path = ROOT / f'outputs/ffpp-preprocess-{name}-{key}.json'
    collected, errors, checks, misses = [], [], [], []
    done = 0
    def report(state):
        value = dict(state=state, config=config, config_id=key,
            target_groups_done=done, target_groups_total=len(targets),
            videos_selected=len(selected), videos_checked=len(checks),
            crops=len(collected), by_split=dict(Counter(r['split'] for r in collected)),
            by_method=dict(Counter(r['method'] for r in collected)),
            errors=errors, undetected_frames=misses,
            manifest=manifest.relative_to(ROOT).as_posix(),
            validation='sequential frame traversal + sampled image decoding; source-video IDs disjoint, not a biometric identity audit')
        temp = report_path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temp.replace(report_path)
    report('running')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, group): group for group in targets.values()}
        for future in as_completed(futures):
            try:
                result = future.result()
                collected.extend(result['rows']); errors.extend(result['errors'])
                checks.extend(result['checks']); misses.extend(result['no_face'])
            except Exception as exc:
                errors.append({'target_id': futures[future][0]['target_id'], 'error': str(exc)})
            done += 1
            report('running')
            print(f'{done}/{len(targets)} targets; {len(collected)} crops; {len(errors)} errors', flush=True)
    collected.sort(key=lambda r: r['sample_id'])
    if collected:
        validate_manifest(collected)
        write_manifest(manifest, collected, FRAME_FIELDS)
    report('complete' if collected and not errors else 'incomplete')
    print('Manifest:', manifest, flush=True)
    return 0 if collected and not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
