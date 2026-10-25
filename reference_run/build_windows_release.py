"""Build a CPU-only portable application; publish only an explicit resource allowlist."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from app_paths import APP_VERSION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME_MODULES = ['app_paths.py', 'core.py', 'generation_common.py', 'face_analysis.py', 'gui.py', 'desktop_entry.py']
RESOURCES = ['artifacts/model_finetuned.pt', 'artifacts/generation/joint/model.pt',
             'artifacts/generation/joint/READY.json', 'weights/face_detection_yunet_2023mar.onnx',
             'weights/YuNet-LICENSE.txt']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze():
    import torch
    if sys.platform != 'win32' or torch.version.cuda is not None:
        raise RuntimeError('Build on Windows with CPU-only PyTorch in a separate environment')
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
        '--noupx', '--name', 'FaceGuard', '--paths', str(HERE), '--distpath', str(HERE / 'dist'),
        '--workpath', str(HERE / 'build'), '--specpath', str(HERE / 'build'), '--collect-all', 'torchvision']
    for name in ('pytest', 'matplotlib', 'pandas', 'scipy', 'sklearn', 'pyarrow'):
        command += ['--exclude-module', name]
    subprocess.run(command + [str(HERE / 'desktop_entry.py')], cwd=ROOT, check=True)


def collect_licenses(target):
    target.mkdir(parents=True, exist_ok=True)
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name', 'unknown').replace('/', '_')
        for file in distribution.files or []:
            if ('license' in str(file).lower() or 'copying' in str(file).lower()) and not str(file).endswith('.py'):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    relative = Path(str(file))
                    if relative.is_absolute() or '..' in relative.parts:
                        continue
                    destination = target / name / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license, target / 'Python-LICENSE.txt')
    # Python runtime ships Tcl/Tk; keep their copyright/license notices too.
    for source in (Path(sys.base_prefix) / 'tcl').rglob('*license*'):
        if source.is_file():
            destination = target / 'Tcl-Tk' / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def assemble():
    folder = HERE / 'dist/FaceGuard'
    if not (folder / 'FaceGuard.exe').is_file():
        raise FileNotFoundError('Build FaceGuard.exe first')
    entries = []
    for name in RESOURCES:
        source, destination = HERE / name, folder / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append(dict(path=name, sha256=sha256(source), bytes=source.stat().st_size))
    (folder / 'runtime_manifest.json').write_text(json.dumps(dict(version=APP_VERSION,
        platform='Windows x64', compute='CPU', files=entries), indent=2), encoding='utf-8')
    shutil.copy2(ROOT / 'LICENSE', folder / 'CODE_LICENSE.txt')
    shutil.copy2(ROOT / 'docs/TEAM_QUICKSTART.md', folder / 'START_HERE.md')
    shutil.copy2(ROOT / 'docs/PROJECT_LIMITATIONS_PLAIN.md', folder / 'PROJECT_LIMITATIONS.md')
    shutil.copy2(ROOT / 'docs/MODEL_CARD.md', folder / 'MODEL_CARD.md')
    (folder / 'START_HERE.txt').write_text(
        'FaceGuard Windows 64-bit portable CPU edition\n\n'
        '1. Extract the WHOLE ZIP to a normal folder. Do not run from inside the ZIP.\n'
        '2. Double-click FaceGuard.exe. Python and a GPU are NOT required.\n'
        '3. Choose your own photo/video, then click the blue Start button.\n\n'
        'Photos: face generation detection. Videos: face manipulation detection.\n'
        'No training images, personal photos, or datasets are included. Files stay local.\n'
        'Scores are not probabilities. Read MODEL_CARD.md for scope and limits.\n', encoding='utf-8')
    collect_licenses(folder / 'THIRD_PARTY_LICENSES')
    versions = {d.metadata.get('Name', 'unknown'): d.version for d in metadata.distributions()}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    info = dict(version=APP_VERSION, source_commit=commit,
        runtime_source_sha256={name: sha256(HERE / name) for name in RUNTIME_MODULES},
        dependencies=versions, training_data_included=False, personal_images_included=False,
        python=sys.version, note='Built on Windows; CPU inference; no system Python required')
    (folder / 'BUILD_INFO.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return folder


def archive(folder):
    destination = HERE / 'release-output'
    destination.mkdir(exist_ok=True)
    target = destination / f'FaceGuard-v{APP_VERSION}-windows-x64-cpu.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                stream.write(path, Path('FaceGuard') / path.relative_to(folder))
    checksum = sha256(target)
    (destination / 'SHA256SUMS.txt').write_text(f'{checksum}  {target.name}\n', encoding='ascii')
    print(json.dumps(dict(asset=str(target), bytes=target.stat().st_size, sha256=checksum), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-freeze', action='store_true', help='Reassemble resources/documents around an already built executable')
    args = parser.parse_args()
    if not args.skip_freeze:
        freeze()
    archive(assemble())


if __name__ == '__main__':
    main()
