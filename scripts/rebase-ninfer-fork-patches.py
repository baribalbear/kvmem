#!/usr/bin/env python3
"""Re-append this fork's NInfer patches to upstream's NInfer patch series.

Upstream regenerates its NInfer series whenever it moves the base or adds a patch, and its
backends/versions.json then has none of the fork's entries (FORK_PATCHES). This script takes the
series in the working tree's versions.json, replays it, applies each fork patch from --previous on
top in order with a three-way merge, and records the regenerated patches as the last entries. All
replays run in temporary indexes, so neither backends/ninfer nor the developer's index changes. A
conflicting fork patch fails; regenerate it by hand then (docs/upstream-sync.md).
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
# The fork's NInfer patches, in series order, after all of upstream's.
FORK_PATCHES = ('exl3', 'gguf-ragged-rows')
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


def replay(directory, base, patches, env):
    """Apply (path, item) pairs onto base in the index of env, checking each recorded tree."""
    git(directory, 'read-tree', base, env=env)
    for path, item in patches:
        git(directory, 'apply', '--cached', '--whitespace=nowarn', str(path), env=env)
        tree = git(directory, 'write-tree', env=env).stdout.strip()
        if tree != item['after_tree']:
            raise RuntimeError(f'{item["id"]}: patch produces {tree}, expected {item["after_tree"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--previous', required=True,
                        help='revision whose versions.json carries the current fork patches, e.g. master')
    args = parser.parse_args()

    versions_path = ROOT / 'backends/versions.json'
    versions = json.loads(versions_path.read_text(encoding='utf-8'))
    entry = versions['ninfer']
    old_entry = json.loads(show(args.previous, 'backends/versions.json'))['ninfer']
    old_fork = [item for item in old_entry['patches'] if item['id'] in FORK_PATCHES]
    if not old_fork:
        raise RuntimeError('--previous has no fork patches')
    if [item['id'] for item in old_fork] != [i for i in FORK_PATCHES if any(p['id'] == i for p in old_fork)]:
        raise RuntimeError('fork patches in --previous are out of order')
    if [item['id'] for item in old_entry['patches'][-len(old_fork):]] != [p['id'] for p in old_fork]:
        raise RuntimeError('fork patches in --previous must follow all of upstream\'s')
    # A clean textual merge may keep the old fork entries; rebuild them either way.
    entry['patches'] = [item for item in entry['patches'] if item['id'] not in FORK_PATCHES]
    if not entry['patches']:
        raise RuntimeError('upstream NInfer series is empty')
    upstream_last = entry['patches'][-1]

    directory = ROOT / entry['path']
    if not (directory / '.git').exists():
        git(ROOT, 'submodule', 'update', '--init', '--depth', '1', '--', entry['path'])
    ensure_commit(directory, entry['base_commit'])
    ensure_commit(directory, old_entry['base_commit'])

    rebuilt = []
    with tempfile.TemporaryDirectory(prefix='kvmem-ninfer-fork-') as scratch:
        scratch = Path(scratch)
        # The previous series, read from --previous because upstream may have deleted its files.
        # Replaying it stores the blobs each fork patch was cut against, which --3way needs.
        old_patches = []
        for index, item in enumerate(old_entry['patches']):
            path = scratch / f'old-{index:03}.patch'
            path.write_bytes(show(args.previous, item['path']))
            old_patches.append((path, item))
        with temporary_index() as env:
            replay(directory, old_entry['base_commit'], old_patches, env)
        sources = {item['id']: path for path, item in old_patches}

        new_patches = [(ROOT / item['path'], item) for item in entry['patches']]
        with temporary_index() as env:
            replay(directory, entry['base_commit'], new_patches, env)
            previous_tree, previous_id = upstream_last['after_tree'], upstream_last['id']
            for item in old_fork:
                applied = git(directory, 'apply', '--cached', '--3way', '--whitespace=nowarn',
                              str(sources[item['id']]), env=env, check=False)
                if applied.returncode != 0:
                    sys.exit(f'The {item["id"]} patch does not apply after {previous_id}:\n'
                             f'{applied.stderr}')
                tree = git(directory, 'write-tree', env=env).stdout.strip()
                patch = subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(directory),
                                        'diff', '--binary', '--full-index', previous_tree, tree],
                                       check=True, capture_output=True).stdout
                rebuilt.append((item, previous_id, tree, patch))
                previous_tree, previous_id = tree, item['id']

    for item, requires, tree, patch in rebuilt:
        digest = hashlib.sha256(patch).hexdigest()
        (ROOT / item['path']).write_bytes(patch)
        entry['patches'].append({**item, 'sha256': digest, 'requires': [requires], 'after_tree': tree})
        print(f'{item["id"]}: NInfer patch {len(entry["patches"])}, after {requires}, '
              f'tree {tree[:12]}, sha256 {digest[:12]}')
    entry['patched_tree'] = entry['patches'][-1]['after_tree']
    versions_path.write_bytes((json.dumps(versions, indent=2, ensure_ascii=False) + '\n').encode())

    # scripts/model_health_ninfer records the hashes of the fork patches it carries (EXL3's reader).
    provenance_path = ROOT / 'scripts/model_health_ninfer/PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    hashes = {item['id']: item['sha256'] for item in entry['patches'] if item['id'] in FORK_PATCHES}
    for extension in provenance['local_extensions']:
        if extension['id'] in hashes:
            extension['sha256'] = hashes[extension['id']]
    provenance_path.write_bytes((json.dumps(provenance, indent=2) + '\n').encode())


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        sys.exit(f'NInfer fork patch rebase failed: {detail}')
