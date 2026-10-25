"""Identical canonical preprocessing for generation training and raw photo inference."""
import io
import cv2
import numpy as np
from PIL import Image


def canonical_face(image):
    array = cv2.resize(np.asarray(image.convert('RGB')), (128, 128), interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='JPEG', quality=90, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert('RGB')


def encode_canonical(image):
    array = cv2.resize(np.asarray(image.convert('RGB')), (128, 128), interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='JPEG', quality=90, subsampling=0)
    return buffer.getvalue()
