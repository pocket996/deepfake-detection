"""Check CPU/GPU dependencies and I/O; optionally train and reload a tiny model."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import pandas as pd
from PIL import Image


def training_smoke(root, folder, device, manifest):
    """Two optimizer steps, not a benchmark or a trained detector."""
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(root / 'reference_run'))
    from faceguard.data import FaceImageDataset
    from core import backbone, image_transform, load_detector
    from torch.utils.data import DataLoader, Subset

    dataset = FaceImageDataset(manifest, 'train', transform=image_transform(), root=root)
    selected = []
    for label in ('0', '1'):
        indices = [i for i, row in enumerate(dataset.rows) if row['label'] == label][:2]
        if len(indices) != 2:
            raise ValueError('Training smoke check needs two training images per class')
        selected.extend(indices)
    batch = next(iter(DataLoader(Subset(dataset, selected), batch_size=4, num_workers=0)))
    torch.manual_seed(20261027)
    encoder = backbone().to(device).train()
    head = torch.nn.Linear(512, 1).to(device).train()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    images, labels = batch['image'].to(device), batch['label'].float().to(device)
    before = head.weight.detach().cpu().clone()
    losses = []
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=device == 'cuda'):
            logits = head(encoder(images)).flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise RuntimeError('Non-finite training gradient')
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    if torch.equal(before, head.weight.detach().cpu()):
        raise RuntimeError('Optimizer did not update classifier weights')
    training_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    encoder, head = encoder.cpu().eval(), head.cpu().eval()
    checkpoint = folder / 'smoke-model.pt'
    torch.save(dict(backbone=encoder.state_dict(), head=head.state_dict(), architecture='resnet18'), checkpoint)
    restored, restored_head, _ = load_detector(checkpoint, 'cpu')
    with torch.inference_mode():
        expected = head(encoder(batch['image']))
        actual = restored_head(restored(batch['image']))
    torch.testing.assert_close(actual, expected)
    return dict(manifest=str(manifest.resolve()), sample_ids=list(batch['sample_id']),
                steps=2, batch_size=4, losses=losses, optimizer_updated=True,
                checkpoint_reload='passed', training_seconds=training_seconds,
                peak_allocated_bytes=peak, scope='Environment smoke check, not model quality evaluation')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--training-smoke', action='store_true')
    parser.add_argument('--manifest', type=Path, default=root / 'data/manifests/ffpp-frames-full-93e2b370610e.csv')
    parser.add_argument('--output', type=Path, default=root / 'outputs/environment.json')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "device": device,
        "packages": {name: metadata.version(name) for name in (
            'torch', 'torchvision', 'numpy', 'opencv-python', 'Pillow',
            'pandas', 'scikit-learn', 'tqdm', 'pyarrow')},
        "checks": {},
    }
    checks = report["checks"]
    try:
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if device == 'cuda':
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            report["gpu_capability"] = torch.cuda.get_device_capability(0)
        layer = torch.nn.Conv2d(3, 8, 3).to(device)
        inputs = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
        loss = layer(inputs).square().mean()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(inputs.grad).all():
            raise RuntimeError("Non-finite GPU result")
        checks[f"{device}_forward_backward"] = "passed"
    except Exception as exc:
        checks[f"{device}_forward_backward"] = str(exc)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        folder = Path(folder)
        try:
            table = pd.DataFrame({'label': [0, 1], 'sample_id': ['real', 'fake']})
            parquet = folder / 'samples.parquet'
            table.to_parquet(parquet, engine='pyarrow', index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(parquet, engine='pyarrow'), table)
            checks['parquet_io'] = 'passed'
        except Exception as exc:
            checks['parquet_io'] = str(exc)
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
        if args.training_smoke:
            try:
                report['training_smoke'] = training_smoke(root, folder, device, args.manifest)
                checks['training_smoke'] = 'passed'
            except Exception as exc:
                checks['training_smoke'] = str(exc)
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr).strip())
        checks['pip_check'] = 'passed'
    except Exception as exc:
        checks['pip_check'] = str(exc)
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True, timeout=15)
        report["ffmpeg"] = result.stdout.splitlines()[0]
        checks["ffmpeg"] = "passed"
    except Exception as exc:
        checks["ffmpeg"] = str(exc)
    report['status'] = 'passed' if all(value == 'passed' for value in checks.values()) else 'failed'
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if any(value != "passed" for value in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
