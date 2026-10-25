"""Windowed release entry point with integrity checks and readable startup failures."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from app_paths import APP_VERSION, application_directory


def check_resources():
    root = application_directory()
    manifest = root / 'runtime_manifest.json'
    if not manifest.is_file():
        raise RuntimeError('缺少发布包资源清单。请完整解压 ZIP，不要只复制 FaceGuard.exe。')
    files = json.loads(manifest.read_text(encoding='utf-8'))['files']
    for item in files:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"缺少模型或检测文件：{item['path']}。请重新完整解压发布包。")
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise RuntimeError(f"文件校验失败：{item['path']}。请重新下载并完整解压发布包。")


def main():
    folder = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'FaceGuard'
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / 'startup.log'
    log = log_path.open('a', encoding='utf-8', buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    try:
        check_resources()
        from gui import main as gui_main
        gui_main()
    except Exception as error:
        traceback.print_exc(file=log)
        if '--self-test' not in sys.argv and '--smoke-test' not in sys.argv:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(f'FaceGuard {APP_VERSION} 启动失败',
                f'{error}\n\n请确认 ZIP 已完整解压。\n诊断日志：{log_path}', parent=root)
            root.destroy()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
