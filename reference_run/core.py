"""Standalone FF++ reference model, data checks, and inference preprocessing."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(manifest):
    with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty manifest')
    samples, paths = set(), set()
    groups, sources, videos = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        if row['sample_id'] in samples or row['path'] in paths:
            raise ValueError('Duplicate sample ID or path')
        samples.add(row['sample_id'])
        paths.add(row['path'])
        if row['split'] not in ('train', 'val', 'test') or row['label'] not in ('0', '1'):
            raise ValueError('Invalid split/label')
        if row['provenance_status'] != 'verified' or row['media_type'] != 'image':
            raise ValueError('Only verified image manifests are supported')
        if not row['group_id'] or not row['parent_video_id']:
            raise ValueError('Source grouping/video ID required')
        groups[row['group_id']].add(row['split'])
        for key in ('target_id', 'source_id'):
            if row.get(key):
                sources[(row['dataset'], row[key])].add(row['split'])
        videos[row['parent_video_id']].add((row['split'], row['label'], row['group_id'], row['method']))
        if not (PROJECT / row['path']).is_file():
            raise FileNotFoundError(row['path'])
    if any(len(v) != 1 for mapping in (groups, sources, videos) for v in mapping.values()):
        raise ValueError('Source/group/video metadata crosses splits or is inconsistent')
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {'0', '1'}:
            raise ValueError('Each split needs both classes')
    return rows


def image_transform():
    # Input already is a square YuNet face crop. Match this in training/inference.
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def backbone():
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    return model


def load_detector(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = backbone()
    model.load_state_dict(state['backbone'])
    model = model.to(device).eval()
    head = nn.Linear(512, 1).to(device)
    head.load_state_dict(state['head'])
    head.eval()
    return model, head, state


def crop_face(frame, detector):
    height, width = frame.shape[:2]
    scale = min(1., 640 / max(height, width))
    small = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return None
    face = max(faces, key=lambda row: row[2] * row[3])
    x, y, w, h = face[:4] / scale
    size = min(max(w, h) * 1.3, width, height)
    left = min(max(x + w / 2 - size / 2, 0), width - size)
    top = min(max(y + h / 2 - size / 2, 0), height - size)
    # Same integer rounding as the original project's prepared PNGs.
    box = [left / width, top / height, (left + size) / width, (top + size) / height]
    x1, y1 = int(box[0] * width), int(box[1] * height)
    x2, y2 = min(width, int(np.ceil(box[2] * width))), min(height, int(np.ceil(box[3] * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    face_image = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))


class Predictor:
    def __init__(self, checkpoint=None, device=None):
        if checkpoint is None:
            fine = HERE / 'artifacts/model_finetuned.pt'
            checkpoint = fine if fine.is_file() else HERE / 'artifacts/model.pt'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.head, self.state = load_detector(checkpoint, self.device)
        weight = HERE / 'weights/face_detection_yunet_2023mar.onnx'
        if digest(weight) != self.state['yunet_sha256']:
            raise ValueError('YuNet weights differ from training preprocessing')
        self.detector = cv2.FaceDetectorYN.create(str(weight), '', (320, 320), 0.8, 0.3, 5000)
        self.transform = image_transform()

    @torch.inference_mode()
    def predict(self, path, face_crop=False, progress=None):
        path = Path(path).resolve()
        images, skipped, attempted = [], 0, 0
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'):
            if progress:
                progress('reading_image', 0, 1)
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                if face_crop:
                    images.append(image)
                else:
                    attempted = 1
                    face = crop_face(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), self.detector)
                    if face is not None:
                        images.append(face)
                    else:
                        skipped += 1
            kind = 'image'
        else:
            if face_crop:
                raise ValueError('--face-crop is only supported for images')
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise ValueError('Cannot open video')
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if count < 1:
                    raise ValueError('Invalid video frame count')
                targets = set(np.linspace(0, count - 1, min(16, count), dtype=int).tolist())
                index = 0
                while cap.grab():
                    if progress and index % max(1, count // 100) == 0:
                        progress('reading_video', index, count)
                    if index in targets:
                        attempted += 1
                        ok, frame = cap.retrieve()
                        if not ok:
                            raise ValueError('Sampled video frame could not decode')
                        face = crop_face(frame, self.detector)
                        if face is None:
                            skipped += 1
                        else:
                            images.append(face)
                    index += 1
                if index != count or attempted != len(targets):
                    raise ValueError('Video is truncated or frame metadata is inconsistent')
            finally:
                cap.release()
            kind = 'video'
        result = dict(path=str(path), media_type=kind, faces=len(images),
                      sampled_frames=attempted, no_face_frames=skipped,
                      scope='FF++ c23 face manipulation baseline; uncalibrated score')
        if not images:
            return dict(result, decision='no_face', score=None)
        if progress:
            progress('inference', 0, len(images))
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        scores = self.head(self.model(batch)).sigmoid().flatten().cpu().tolist()
        score = float(np.mean(scores))
        threshold = self.state['video_threshold'] if kind == 'video' else self.state['frame_threshold']
        return dict(result, score=score, threshold=threshold,
                    decision='suspected_manipulation' if score >= threshold else 'no_manipulation_detected',
                    frame_scores=scores)
