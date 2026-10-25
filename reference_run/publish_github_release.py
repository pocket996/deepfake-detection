"""Publish reviewed release files via GitHub API, using Git's configured credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request

REPO = 'pocket996/deepfake-detection'


def credentials():
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
        text=True, capture_output=True, check=True)
    record = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not record.get('password'):
        raise RuntimeError('GitHub credentials unavailable; no credentials were printed')
    return record['password']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--tag', default='v0.1.0')
    parser.add_argument('--commit')
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--asset', type=Path, action='append', default=[])
    parser.add_argument('--result', type=Path, default=Path('reference_run/release-output/github_release.json'))
    args = parser.parse_args()
    token = credentials()
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
               'User-Agent': 'FaceGuard-release', 'X-GitHub-Api-Version': '2022-11-28'}
    def api(method, path, value=None):
        payload = json.dumps(value).encode('utf-8') if value is not None else None
        request = urllib.request.Request('https://api.github.com' + path, data=payload,
            headers={**headers, 'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(request, timeout=60) as response:
            return None if response.status == 204 else json.load(response)
    releases = api('GET', f'/repos/{REPO}/releases?per_page=30')
    if args.inspect:
        print(json.dumps([dict(tag=r['tag_name'], draft=r['draft'], url=r['html_url']) for r in releases], indent=2))
        return
    if not args.commit or not args.notes or not args.asset:
        parser.error('--commit, --notes and --asset are required to publish')
    for path in args.asset:
        if not path.is_file() or path.stat().st_size >= 2 * 1024**3:
            raise ValueError('Missing or oversized asset: ' + path.name)
    release = next((r for r in releases if r['tag_name'] == args.tag), None)
    if release is not None:
        previous = json.loads(args.result.read_text(encoding='utf-8')) if args.result.exists() else {}
        if not release['draft'] or previous.get('id') != release['id'] or previous.get('commit') != args.commit:
            raise RuntimeError('Tag already exists and is not this publication draft; inspect before changing it')
    else:
        release = api('POST', f'/repos/{REPO}/releases', dict(tag_name=args.tag,
            target_commitish=args.commit, name='FaceGuard ' + args.tag + ' · Windows 免安装 CPU 预览版',
            body=args.notes.read_text(encoding='utf-8'), draft=True, prerelease=True))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(dict(id=release['id'], tag=args.tag, commit=args.commit, draft=True), indent=2), encoding='utf-8')
    upload_url = urllib.parse.urlsplit(release['upload_url'].split('{')[0])
    assert upload_url.hostname == 'uploads.github.com'
    assets = []
    for file in args.asset:
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        existing = next((a for a in release.get('assets', []) if a['name'] == file.name), None)
        if existing:
            if existing.get('digest') == 'sha256:' + digest and existing['size'] == file.stat().st_size:
                assets.append(dict(name=file.name, bytes=existing['size'], sha256=digest, url=existing['browser_download_url']))
                continue
            api('DELETE', f"/repos/{REPO}/releases/assets/{existing['id']}")
        class UploadReader:
            def __init__(self, stream):
                self.stream, self.total, self.last = stream, 0, 0
            def read(self, size=-1):
                block = self.stream.read(size)
                self.total += len(block)
                if self.total - self.last >= 16 * 1024**2:
                    print('Uploading', file.name, f'{self.total / 1024**2:.0f} MiB', flush=True)
                    self.last = self.total
                return block
        with file.open('rb') as stream:
            request = urllib.request.Request(release['upload_url'].split('{')[0] + '?name=' + urllib.parse.quote(file.name),
                data=UploadReader(stream), headers={**headers, 'Content-Type': 'application/octet-stream',
                    'Content-Length': str(file.stat().st_size)}, method='POST')
            with urllib.request.urlopen(request, timeout=300) as response:
                assert response.status == 201
                asset = json.load(response)
        assert asset['size'] == file.stat().st_size
        if asset.get('digest'):
            assert asset['digest'] == 'sha256:' + digest
        assets.append(dict(name=asset['name'], bytes=asset['size'], sha256=digest, url=asset['browser_download_url']))
        print('Uploaded', file.name, asset['size'], flush=True)
    release = api('PATCH', f"/repos/{REPO}/releases/{release['id']}", dict(draft=False))
    result = dict(id=release['id'], tag=args.tag, commit=args.commit, url=release['html_url'], draft=release['draft'],
                  prerelease=release['prerelease'], assets=assets)
    args.result.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
