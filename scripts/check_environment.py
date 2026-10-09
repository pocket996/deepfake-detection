"""Verify actual GPU forward/backward execution and image/video I/O."""
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "checks": {},
    }
    checks = report["checks"]
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["gpu_capability"] = torch.cuda.get_device_capability(0)
        layer = torch.nn.Conv2d(3, 8, 3).cuda()
        inputs = torch.randn(2, 3, 64, 64, device="cuda", requires_grad=True)
        loss = layer(inputs).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(inputs.grad).all():
            raise RuntimeError("Non-finite GPU result")
        checks["gpu_forward_backward"] = "passed"
    except Exception as exc:
        checks["gpu_forward_backward"] = str(exc)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        folder = Path(folder)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 128
        try:
            image_path = folder / "sample.png"
            Image.fromarray(frame).save(image_path)
            decoded = cv2.imread(str(image_path))
            if decoded is None or decoded.shape != frame.shape:
                raise RuntimeError("Image decode failed")
            checks["image_io"] = "passed"
        except Exception as exc:
            checks["image_io"] = str(exc)
        try:
            video_path = folder / "sample.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
            try:
                if not writer.isOpened():
                    raise RuntimeError("Video writer failed")
                for _ in range(10):
                    writer.write(frame)
            finally:
                writer.release()
            capture = cv2.VideoCapture(str(video_path))
            try:
                count = 0
                while capture.read()[0]:
                    count += 1
            finally:
                capture.release()
            if count != 10:
                raise RuntimeError(f"Expected 10 frames, decoded {count}")
            checks["video_io"] = "passed"
        except Exception as exc:
            checks["video_io"] = str(exc)
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        report["ffmpeg"] = result.stdout.splitlines()[0]
        checks["ffmpeg"] = "passed"
    except Exception as exc:
        checks["ffmpeg"] = str(exc)
    destination = root / "outputs" / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if any(value != "passed" for value in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
