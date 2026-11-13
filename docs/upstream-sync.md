# Syncing this fork with kvmem/kvmem

`master` is kvmem/kvmem plus this fork's additions: two NInfer patches (`exl3`, then
`gguf-ragged-rows`), the pruned-vocabulary tools, the Windows release workflow and the console's
`--reasoning-loop` and `--request-log-jsonl` native options (`scripts/launcher/config.py`,
`NATIVE_CHOICES` and `NATIVE_OUTPUT_FILES`). An upstream change to the console's `WorkerArgs`
checks can conflict with them.
`.github/workflows/upstream-sync.yml` keeps it current. Every Monday, or from
**Actions → Upstream sync → Run workflow**, it does the following:

1. Merges kvmem/kvmem `master` into this fork's `master` on the `sync/upstream` branch.
2. If upstream changed its NInfer series, re-appends the fork's NInfer patches to it (below).
3. Applies every backend patch series, checks that each reproduces its recorded trees, and runs the
   Python tests.
4. Opens a pull request into `master`, or updates the open one. Its description says whether the
   merge was clean, whether the fork's patches were re-appended, and whether the checks passed.

A green run's pull request can be merged as it is. A run fails when the merge conflicts or the
checks fail. It still opens the pull request, which lists the conflicting files or the check log.
Do not use GitHub's **Sync fork** button: it merges into `master` without these checks.

Before merging, run **KVMem Windows release** on `sync/upstream`; the sync checks no C++.

## The fork's NInfer patches after an upstream NInfer change

The fork's patches come last in the NInfer series, in the order of `FORK_PATCHES` in
`scripts/rebase-ninfer-fork-patches.py`:

| Patch | What it does |
|---|---|
| `exl3` | EXL3 weights ([ninfer-exl3.md](ninfer-exl3.md)) |
| `gguf-ragged-rows` | GGUF matrices whose rows are no multiple of 128, such as a pruned output head ([ninfer-pruned-vocab.md](ninfer-pruned-vocab.md)) |

They must apply to the tree that upstream's last patch produces, so they break whenever upstream
changes the NInfer pin or its patches. Upstream's `backends/versions.json` then has a new series
without them, and the merge conflicts in that file alone.

`scripts/rebase-ninfer-fork-patches.py --previous master` handles this. It takes the working tree's
`versions.json` (upstream's), replays its NInfer series, applies each fork patch from `master` on top
in order with `git apply --3way`, and records the regenerated patches as the last entries, updating
`PROVENANCE.json` too. It replays in temporary indexes and leaves `backends/ninfer` alone. The sync
runs it whenever `versions.json` is the only conflict, or a clean merge changed `versions.json`.

When a three-way apply fails, upstream changed code that patch edits. The run then offers upstream
itself with the script's log, and the patches need regenerating by hand. On `sync/upstream`:

1. Resolve `backends/versions.json` to upstream's NInfer entry, without the fork's patches. Its
   `patched_tree` is then upstream's last `after_tree`.
2. Apply upstream's series, then each old fork patch in order. For each one:

   ```bash
   python scripts/prepare-backends.py --backend ninfer      # once, for upstream's series
   cd backends/ninfer
   git apply --3way ../patches/ninfer-exl3.patch    # resolve any conflicts in the sources
   git add -A
   git diff --cached --full-index <previous after_tree> > ../patches/ninfer-exl3.patch
   git write-tree                                   # this patch's new after_tree
   ```

3. Add each entry back in order, with its new `sha256` (of the patch file), `requires` (the
   previous patch id) and `after_tree`. Set the backend's `patched_tree` to the last tree.
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
