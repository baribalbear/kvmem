#!/usr/bin/env python3
"""Print the release version of HEAD: the newer of VERSION and upstream's latest release in HEAD.

Upstream tags releases (v0.19.1) without always bumping VERSION, and a fork receives none of its
tags. This lists kvmem/kvmem's stable vX.Y.Z tags and takes the highest whose commit HEAD contains,
which needs HEAD's commit history (actions/checkout with fetch-depth 0; filter tree:0 keeps it
small). Without that history, or without network access, it falls back to VERSION.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'https://github.com/kvmem/kvmem.git'
STABLE = re.compile(r'refs/tags/v(\d+)\.(\d+)\.(\d+)(\^\{\})?$')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def parse(version):
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', version)
    if not match:
        raise SystemExit(f'VERSION is not X.Y.Z: {version!r}')
    return tuple(int(part) for part in match.groups())


def main():
    version = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
    best = parse(version)
    if git('rev-parse', '--is-shallow-repository').stdout.strip() != 'false':
        print('release-version: shallow checkout, using VERSION', file=sys.stderr)
        print(version)
        return
    listed = git('ls-remote', '--tags', UPSTREAM, 'v*')
    if listed.returncode != 0:
        print(f'release-version: cannot list upstream tags, using VERSION\n{listed.stderr}',
              file=sys.stderr)
        print(version)
        return
    # An annotated tag lists its own object and, as ^{}, the commit it points at.
    commits = {}
    for line in listed.stdout.splitlines():
        sha, ref = line.split('\t')
        match = STABLE.fullmatch(ref)
        if match and (match.group(4) or tuple(map(int, match.group(1, 2, 3))) not in commits):
            commits[tuple(map(int, match.group(1, 2, 3)))] = sha
    for tag in sorted(commits, reverse=True):
        if tag <= best:
            break
        if git('merge-base', '--is-ancestor', commits[tag], 'HEAD').returncode == 0:
            best = tag
            break
    print('.'.join(map(str, best)))


if __name__ == '__main__':
    main()
