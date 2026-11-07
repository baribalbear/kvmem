# Syncing this fork with kvmem/kvmem

`master` is kvmem/kvmem plus this fork's additions: the EXL3 NInfer patch, the pruned-vocabulary
tools and the Windows release workflow. `.github/workflows/upstream-sync.yml` keeps it current.
Every Monday, or from **Actions → Upstream sync → Run workflow**, it does the following:

1. Merges kvmem/kvmem `master` into this fork's `master` on the `sync/upstream` branch.
2. If upstream changed its NInfer series, re-appends the EXL3 patch to it (below).
3. Applies every backend patch series, checks that each reproduces its recorded trees, and runs the
   Python tests.
4. Opens a pull request into `master`, or updates the open one. Its description says whether the
   merge was clean, whether the EXL3 patch was re-appended, and whether the checks passed.

A green run's pull request can be merged as it is. A run fails when the merge conflicts or the
checks fail. It still opens the pull request, which lists the conflicting files or the check log.
Do not use GitHub's **Sync fork** button: it merges into `master` without these checks.

Before merging, run **KVMem Windows release** on `sync/upstream`; the sync checks no C++.

## The EXL3 patch after an upstream NInfer change

`backends/patches/ninfer-exl3.patch` is the last patch in the NInfer series. It must apply to the
tree that upstream's last patch produces, so it breaks whenever upstream changes the NInfer pin or
its patches. Upstream's `backends/versions.json` then has a new series without `exl3`, and the
merge conflicts in that file alone.

`scripts/rebase-ninfer-exl3.py --previous master` handles this. It takes the working tree's
`versions.json` (upstream's), replays its NInfer series, applies the EXL3 patch from `master` on top
with `git apply --3way`, and records the regenerated patch as the last entry, updating
`PROVENANCE.json` too. It replays in temporary indexes and leaves `backends/ninfer` alone. The sync
runs it whenever `versions.json` is the only conflict, or a clean merge changed `versions.json`.

When the three-way apply fails, upstream changed code the EXL3 patch edits. The run then offers
upstream itself with the script's log, and the patch needs regenerating by hand. On `sync/upstream`:

1. Resolve `backends/versions.json` to upstream's NInfer entry, without the `exl3` patch. Its
   `patched_tree` is then upstream's last `after_tree`.
2. Apply upstream's series and the old EXL3 patch on top:

   ```bash
   python scripts/prepare-backends.py --backend ninfer
   cd backends/ninfer
   git apply --3way ../patches/ninfer-exl3.patch    # resolve any conflicts in the sources
   git add -A
   git diff --cached --full-index <upstream's last after_tree> > ../patches/ninfer-exl3.patch
   git write-tree                                   # the new after_tree
   ```

3. Add the `exl3` entry back as the last patch, with the new `sha256` (of the patch file),
   `requires` (upstream's last patch id) and `after_tree`. Set the backend's `patched_tree` to the
   same tree.
4. Run `python scripts/prepare-backends.py --check` and `python tests/test_prune_ninfer_vocab.py`.

If upstream changed NInfer's `tools/artifact` reader, also refresh `scripts/model_health_ninfer`
and its `PROVENANCE.json`.

## Tokens

The workflow uses `GITHUB_TOKEN` unless a `SYNC_TOKEN` secret exists. `GITHUB_TOKEN` has two
limits:

- It cannot push commits that change `.github/workflows`, which upstream syncs sometimes do.
- Pull requests it opens do not start the repository's other workflows.

To lift both, create a fine-grained token for this repository with Contents, Pull requests and
Workflows read/write. Add it under **Settings → Secrets and variables → Actions** as `SYNC_TOKEN`.
