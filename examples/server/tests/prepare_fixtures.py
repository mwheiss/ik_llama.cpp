"""Download and verify the small, pinned GGUFs used by the server regressions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import requests


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(directory):
    manifest = json.loads(Path(__file__).with_name('fixtures.json').read_text())
    directory.mkdir(parents=True, exist_ok=True)
    for source, entries in manifest.items():
        repo, revision = source.rsplit('@', 1)
        for remote, expected, names in entries:
            target = directory / names[0]
            if not target.exists():
                url = f'https://huggingface.co/{repo}/resolve/{revision}/{remote}'
                partial = target.with_suffix('.gguf.partial')
                print(f'Downloading {source}/{remote}', flush=True)
                with requests.get(url, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                if sha256(partial) != expected:
                    raise RuntimeError(f'Fixture checksum mismatch: {partial}')
                partial.rename(target)
            if sha256(target) != expected:
                raise RuntimeError(f'Existing fixture differs from fixtures.json: {target}')
            for alias in names[1:]:
                path = directory / alias
                if not path.exists():
                    try:
                        os.link(target, path)
                    except OSError:
                        shutil.copy2(target, path)
                if sha256(path) != expected:
                    raise RuntimeError(f'Existing fixture alias differs: {path}')
            print(f'Verified {target.name}: {expected}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path.cwd())
    prepare(parser.parse_args().directory)
