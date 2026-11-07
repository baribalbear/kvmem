#!/usr/bin/env python3
"""Re-append the fork's EXL3 patch to upstream's NInfer patch series.

Upstream regenerates its NInfer series whenever it moves the base or adds a patch, and its
backends/versions.json then has no exl3 entry. This script takes the series in the working tree's
versions.json, replays it, applies the EXL3 patch from --previous on top with a three-way merge, and
records the regenerated patch as the last entry. All replays run in temporary indexes, so neither
backends/ninfer nor the developer's index changes. A conflicting EXL3 patch fails; regenerate it by
hand then (docs/upstream-sync.md).
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXL3 = 'exl3'
spec = importlib.util.spec_from_file_location('prepare_backends', ROOT / 'scripts/prepare-backends.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
git, temporary_index = prepare.git, prepare.temporary_index


def show(revision, path):
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{revision}:{path}'], check=True,
                          capture_output=True).stdout


def ensure_commit(directory, commit):
    if git(directory, 'cat-file', '-e', f'{commit}^{{commit}}', check=False).returncode != 0:
        git(directory, 'fetch', '--depth', '1', 'origin', commit)


def replay(directory, base, patches, env, stop=None):
    """Apply (path, item) pairs onto base in the index of env, checking each recorded tree."""
    git(directory, 'read-tree', base, env=env)
    for path, item in patches:
        if item['id'] == stop:
            return
        git(directory, 'apply', '--cached', '--whitespace=nowarn', str(path), env=env)
        tree = git(directory, 'write-tree', env=env).stdout.strip()
        if tree != item['after_tree']:
            raise RuntimeError(f'{item["id"]}: patch produces {tree}, expected {item["after_tree"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--previous', required=True,
                        help='revision whose versions.json carries the current EXL3 patch, e.g. master')
    args = parser.parse_args()

    versions_path = ROOT / 'backends/versions.json'
    versions = json.loads(versions_path.read_text(encoding='utf-8'))
    entry = versions['ninfer']
    old_entry = json.loads(show(args.previous, 'backends/versions.json'))['ninfer']
    old_exl3 = next(item for item in old_entry['patches'] if item['id'] == EXL3)
    # A clean textual merge may keep the old exl3 entry; rebuild it either way.
    entry['patches'] = [item for item in entry['patches'] if item['id'] != EXL3]
    if not entry['patches']:
        raise RuntimeError('upstream NInfer series is empty')
    last = entry['patches'][-1]
    entry['patched_tree'] = last['after_tree']

    directory = ROOT / entry['path']
    if not (directory / '.git').exists():
        git(ROOT, 'submodule', 'update', '--init', '--depth', '1', '--', entry['path'])
    ensure_commit(directory, entry['base_commit'])
    ensure_commit(directory, old_entry['base_commit'])

    with tempfile.TemporaryDirectory(prefix='kvmem-exl3-') as scratch:
        scratch = Path(scratch)
        # The previous series, read from --previous because upstream may have deleted its files.
        # Replaying it stores the blobs the EXL3 patch was cut against, which --3way needs.
        old_patches = []
        for index, item in enumerate(old_entry['patches']):
            path = scratch / f'old-{index:03}.patch'
            path.write_bytes(show(args.previous, item['path']))
            old_patches.append((path, item))
        with temporary_index() as env:
            replay(directory, old_entry['base_commit'], old_patches, env, stop=EXL3)
        exl3_source = next(path for path, item in old_patches if item['id'] == EXL3)

        new_patches = [(ROOT / item['path'], item) for item in entry['patches']]
        with temporary_index() as env:
            replay(directory, entry['base_commit'], new_patches, env)
            applied = git(directory, 'apply', '--cached', '--3way', '--whitespace=nowarn',
                          str(exl3_source), env=env, check=False)
            if applied.returncode != 0:
                sys.exit(f'The EXL3 patch does not apply to upstream\'s NInfer series:\n{applied.stderr}')
            tree = git(directory, 'write-tree', env=env).stdout.strip()

    patch = subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(directory), 'diff',
                            '--binary', '--full-index', last['after_tree'], tree],
                           check=True, capture_output=True).stdout
    digest = hashlib.sha256(patch).hexdigest()
    (ROOT / old_exl3['path']).write_bytes(patch)
    entry['patches'].append({**old_exl3, 'sha256': digest, 'requires': [last['id']], 'after_tree': tree})
    entry['patched_tree'] = tree
    versions_path.write_bytes((json.dumps(versions, indent=2, ensure_ascii=False) + '\n').encode())

    provenance_path = ROOT / 'scripts/model_health_ninfer/PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for item in provenance['local_extensions']:
        if item['id'] == EXL3:
            item['sha256'] = digest
    provenance_path.write_bytes((json.dumps(provenance, indent=2) + '\n').encode())
    print(f'exl3: NInfer patch {len(entry["patches"])}, after {last["id"]}, tree {tree[:12]}, '
          f'sha256 {digest[:12]}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        sys.exit(f'EXL3 rebase failed: {detail}')
