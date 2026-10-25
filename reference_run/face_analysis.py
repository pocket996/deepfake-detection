"""Route photos to whole-face generation detection, videos to manipulation detection."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from core import HERE, Predictor, crop_face, digest, image_transform, load_detector
from generation_common import canonical_face

PHOTO_SUFFIXES = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')


def generation_directory():
    base = HERE / 'artifacts/generation'
    joint = base / 'joint'
    return joint if (joint / 'READY.json').is_file() and (joint / 'model.pt').is_file() else base


class GenerationPredictor:
    def __init__(self, checkpoint=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        path = Path(checkpoint) if checkpoint else generation_directory() / 'model.pt'
        if not path.is_file():
            raise FileNotFoundError('整脸生成模型尚未准备好，请先运行 train_generation.py。')
        self.model, self.head, self.state = load_detector(path, self.device)
        if self.state.get('task') != 'whole_face_generation':
            raise ValueError('此权重不属于整脸生成模型')
        weight = HERE / 'weights/face_detection_yunet_2023mar.onnx'
        if digest(weight) != self.state['yunet_sha256']:
            raise ValueError('YuNet 权重与训练预处理不匹配')
        self.detector = cv2.FaceDetectorYN.create(str(weight), '', (320, 320), .8, .3, 5000)
        self.transform = image_transform()

    @torch.inference_mode()
    def predict(self, path, face_crop=False, progress=None):
        path = Path(path).resolve()
        if path.suffix.lower() not in PHOTO_SUFFIXES:
            raise ValueError('整脸生成检测目前支持照片；视频请使用换脸/篡改检测。')
        if progress:
            progress('reading_image', 0, 1)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            face = image if face_crop else crop_face(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), self.detector)
        result = dict(path=str(path), media_type='image', task='whole_face_generation',
            faces=int(face is not None), sampled_frames=1, no_face_frames=int(face is None),
            scope='Experimental whole-face generation detector; uncalibrated; see experiment report for evaluation scope',
            training_generators=self.state['training_generators'])
        if face is None:
            return dict(result, decision='no_face', score=None)
        if progress:
            progress('inference', 0, 1)
        tensor = self.transform(canonical_face(face))[None].to(self.device)
        value = float(self.head(self.model(tensor)).sigmoid().item())
        threshold = self.state['frame_threshold']
        return dict(result, decision='suspected_generation' if value >= threshold else 'no_generation_detected',
                    score=value, threshold=threshold, frame_scores=[value])


class FaceAnalyzer:
    def __init__(self, device=None):
        self.device = device
        self.generation, self.manipulation = None, None

    def predict(self, path, face_crop=False, progress=None, task='auto'):
        is_photo = Path(path).suffix.lower() in PHOTO_SUFFIXES
        if task == 'auto':
            task = 'generation' if is_photo else 'manipulation'
        if task == 'generation':
            if not is_photo:
                raise ValueError('此项目用于照片。请将上方「检测项目」切换为「自动」或「换脸 / 篡改」。')
            if self.generation is None:
                self.generation = GenerationPredictor(device=self.device)
            return self.generation.predict(path, face_crop=face_crop, progress=progress)
        if task != 'manipulation':
            raise ValueError('Unknown detection task')
        if self.manipulation is None:
            self.manipulation = Predictor(device=self.device)
        result = self.manipulation.predict(path, face_crop=face_crop, progress=progress)
        result['task'] = 'face_manipulation'
        return result
