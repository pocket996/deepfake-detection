"""Exercise the frozen executable outside the repository with no Python on PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import zipfile

HERE = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--generation-photo', type=Path, required=True)
    parser.add_argument('--real-photo', type=Path, required=True)
    parser.add_argument('--fake-video', type=Path, required=True)
    args = parser.parse_args()
    parent = HERE / 'release-output'
    folder = (parent / ('clean install ' + uuid.uuid4().hex[:8])).resolve()
    assert folder.is_relative_to(parent.resolve())
    folder.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as stream:
        for entry in stream.infolist():
            assert (folder / entry.filename).resolve().is_relative_to(folder)
        stream.extractall(folder)
    app = folder / 'FaceGuard'
    executable = app / 'FaceGuard.exe'
    env = os.environ.copy()
    windows_root = os.environ.get('SystemRoot', os.environ.get('WINDIR', 'C:/Windows'))
    env['SystemRoot'] = windows_root
    env['PATH'] = str(Path(windows_root) / 'System32') + os.pathsep + windows_root
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env.update(HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    checks = []
    def run(name, flags, expected=0):
        start = time.monotonic()
        result = subprocess.run([str(executable), *map(str, flags)], cwd=folder,
            env=env, capture_output=True, timeout=240)
        assert result.returncode == expected, (name, result.returncode, result.stderr[-1000:])
        checks.append(dict(name=name, status='passed', seconds=time.monotonic() - start))
        print(name, 'passed', flush=True)
    run('startup_without_system_python', ['--smoke-test'])
    for name, task, file, decision in [
        ('generated_photo', 'generation', args.generation_photo, 'suspected_generation'),
        ('real_photo', 'generation', args.real_photo, 'no_generation_detected'),
        ('manipulated_video', 'manipulation', args.fake_video, 'suspected_manipulation'),
    ]:
        output = folder / (name + '.json')
        run(name, ['--self-test', '--self-test-task', task, '--test-input', file.resolve(),
            '--test-output', output, '--expected-decision', decision])
        value = json.loads(output.read_text(encoding='utf-8'))
        assert value['preview_loaded'] and value['decision'] == decision
    from PIL import Image
    blank = folder / 'no_face.png'
    Image.new('RGB', (256, 256), 'gray').save(blank)
    run('no_face_is_not_real', ['--self-test', '--test-input', blank, '--test-output', folder / 'no_face_check.json',
        '--expected-decision', 'no_face'])
    manifest = app / 'runtime_manifest.json'
    original = manifest.read_bytes()
    try:
        corrupted = json.loads(original)
        corrupted['files'][0]['sha256'] = '0' * 64
        manifest.write_text(json.dumps(corrupted), encoding='utf-8')
        run('corrupted_resource_is_rejected', ['--smoke-test'], expected=1)
    finally:
        manifest.write_bytes(original)
    report = dict(archive_sha256=sha256(args.zip), executable_sha256=sha256(executable),
        checks=checks, installed_python_on_path=False, compute='CPU',
        network_proxies_disabled_for_test=True,
        limitations='Relocated-folder test on the build Windows host; not yet independently verified on teammate hardware')
    (parent / 'portable_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('All portable checks passed', flush=True)


if __name__ == '__main__':
    main()
