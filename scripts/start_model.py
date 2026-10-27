#!/usr/bin/env python3
"""Start a local llama-server using the recorded Ward-v4 GGUF deployment profile."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    metadata = json.loads((ROOT / 'models/model-manifest.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-server', default=os.getenv('LLAMA_SERVER', 'llama-server'))
    parser.add_argument('--model-path', type=Path, default=Path(os.getenv('WARD_MODEL_PATH', str(ROOT / 'models' / metadata['filename']))))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--gpu-layers', type=int, default=20)
    parser.add_argument('--ctx-size', type=int, default=4096)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    server = shutil.which(args.llama_server) or args.llama_server
    command = [server, '--model', str(args.model_path), '--alias', metadata['model_id'],
               '--host', args.host, '--port', str(args.port), '--n-gpu-layers', str(0 if args.cpu else args.gpu_layers),
               '--ctx-size', str(args.ctx_size), '--parallel', '1', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0']
    key = os.getenv('QWEN_API_KEY')
    if key:
        command += ['--api-key', key]
    displayed = command[:-1] + ['<set via QWEN_API_KEY>'] if key else command
    print(json.dumps(displayed, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not args.model_path.is_file():
        print('缺少模型权重。请先阅读 models/README.md 并取得GGUF文件。', file=sys.stderr)
        return 1
    if not Path(server).is_file():
        print('找不到llama-server，请用 --llama-server 指定可执行文件路径。', file=sys.stderr)
        return 1
    digest = hashlib.sha256()
    with args.model_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != metadata['sha256']:
        print('模型SHA-256与最终版记录不符，未启动。请检查权重版本。', file=sys.stderr)
        return 1
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f'启动失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
