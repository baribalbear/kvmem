# EXL3 weights on the NInfer backend

`backends/patches/ninfer-exl3.patch` ports the EXL3 runtime from
[giveen/ninfer-ext](https://github.com/giveen/ninfer-ext) (`b6886f67`, retired 2026-10-22) onto the
pinned NInfer tree. EXL3 here is ninfer-ext's own format: `exl3_mul1` trellis codes over 16x16 tiles
(`trellis_t16_v1`) with folded 128-point Hadamard rotations. It is not exllamav3's checkpoint format.

## Use

Download a published artifact, e.g. `qwen3_8_27b_exl3_4bpw.ninfer` from
[jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models), and start it
with the NInfer backend like any other `.ninfer`. Its non-EXL3 tensors (BF16 embedding and vision,
Q4/Q5/Q6/Q8 proposal head and small matrices) already load on this tree.

## Pruned (ASCII-condensed) vocabulary

exllamav3 checkpoints, such as `turboderp/Qwen3.8-27B-exl3`, do not load: there is no importer, and
exllamav3's codebook rounding and per-member input scales differ from this format. Instead, give the
published EXL3 artifact the vocabulary of a vocab-pruned `.ninfer` of the same model (see
[ninfer-pruned-vocab.md](ninfer-pruned-vocab.md)) with `scripts/prune-ninfer-vocab.py`:

```bash
python scripts/prune-ninfer-vocab.py --source qwen3_8_27b_exl3_4bpw.ninfer \
  --donor qwen3_8_27b_ascii_iq4xs.ninfer --out qwen3_8_27b_exl3_4bpw_ascii.ninfer
```

It copies every layer, the MTP layer and Vision byte for byte, with no calibration or GPU. Only the
vocabulary changes:

* The BF16 token embedding keeps the source's rows of the donor's tokens, in the donor's order.
* The output head is the donor's. An EXL3 head cannot drop rows: `trellis_t16_v1` needs both
  dimensions divisible by 128, and the output rotation mixes 128-row blocks.
* `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and `vocab_size` come from
  the donor. `--chat-template donor` also takes the donor's template.
* The indexed proposal head is dropped. Its fused top-k kernel takes exactly 131072 rows, more
  than a pruned vocabulary has, so MTP drafts through the output head, as in the donor.

For the 4.0 bpw artifact and troed's 129006-row vocabulary, the output is 13.8 GiB instead of
15.7 GiB. The head keeps the donor's format and quality, not the source's 6 bpw EXL3.

`tests/test_prune_ninfer_vocab.py` checks the tool on synthetic artifacts. Its planning was also
checked against the published 4.0 bpw directory. The output has not yet been loaded on a GPU.

## What the patch adds

| Area | Files |
|---|---|
| Format and geometry | `core/weight.h` (`QType::EXL3_MUL1 = 35`, `QuantLayout::TrellisT16 = 7`, `Weight::input_scales`, `bitrate_half_bits`), `core/weight_view.*` |
| Artifact | `artifact/schema.*` (`bitrate_half_bits` tensor member), `formats.cpp`, `layouts.cpp`, new `exl3_trellis.*` |
| Kernels | new `ops/linear/exl3/` (A16 decode, GEMV and MMA routes) |
| Consumers | `linear`, `attn_input_proj`, `gdn_input_proj` (plain, conv snapshot, conv record), `linear_swiglu`, `linear_add`, `weight_input.cpp` |
| Model | `qwen3_5/execution/parameters.cpp`: an EXL3 MTP attention bank projects through its packed parent |
| Python | `tools/artifact/{formats,layouts,schema}.py`, `codecs/exl3.py` |
| Tests | `tests/artifact/test_exl3_{trellis.cpp,layout.py}`, `tests/ops/linear/test_exl3_{a16,consumers}.cpp` |

`scripts/model_health_ninfer` carries the same reader changes, so the console's model inspection
accepts EXL3 artifacts.

Changes from ninfer-ext's code:

* Enum numbers differ (ninfer-ext's 9 and 4 are taken here). Artifacts store format names, so files
  are unaffected.
* The kernels' bf16 `mma.sync` goes through `ops/common/mma.cuh`, which emulates it on the sm_70 and
  sm_75 tiers. ninfer-ext targets sm_120 only.
* kvmem's tree builds per-row MTP attention weights for every format. A row slice of an EXL3 parent is
  not a valid operand, so EXL3 banks skip them.
* Tensor parallelism (`heads-tp`) splits projections into row and column pieces, which an EXL3 matrix
  cannot be cut into. EXL3 runs on one GPU; a tensor-parallel launch rejects it as an unsupported
  format.

Not ported: the native quantizer (`ninfer-quantize`, `src/quantize/exl3`, `tools/exl3`), the EXL3
artifact writer (`tensor_output.py`) and Flash-Next expert offload (`ops/offload_moe` EXL3 banks).

## Verification status

Done without a GPU:

* `prepare-backends.py`'s `verify_patch_series` replays every NInfer patch from the base commit and
  reproduces every recorded tree.
* `tests/artifact/test_exl3_layout.py`: 26 passed. The existing CPU-only artifact tests still pass.
* `tests/test_model_health.py`: 24 passed.
* The published `qwen3_8_27b_exl3_4bpw.ninfer` directory (1,054 tensors, 262 of them EXL3 at 8, 10
  and 12 half-bits) parses, and every tensor's byte size matches its geometry. The unmodified reader
  rejects the file.
* Every EXL3 shape in that file has a consumer in this patch: `[14336,5120]` attention,
  `[16384,5120]` GDN, `[34816,5120]` gate/up, `[5120,6144]` and `[5120,17408]` residual projections,
  `[5120,10240]` MTP input projection and the `[248320,5120]` output head.

**Not done yet:** a CUDA build, the C++ tests, loading the model and a perplexity check against
ninfer-ext's published 4.2939. Run these on an sm_80+ GPU before relying on the patch:

```bash
python scripts/prepare-backends.py --check
ctest -R exl3 --output-on-failure
```

## Windows build in CI

[`windows-release.yml`](windows-release-ci.md) builds the complete kvmem Windows release, including
this NInfer worker.

## Regenerating the patch

Apply the series in `backends/ninfer`, edit, then diff against the `after_tree` of the patch before
`exl3` in `backends/versions.json` (the one `exl3` requires), not against `HEAD`, which is the
upstream base:

```bash
git add -A
git diff --cached --full-index <previous patch's after_tree> > ../patches/ninfer-exl3.patch
git write-tree
```

Update the `exl3` entry's `sha256` and `after_tree` and the backend's `patched_tree` in
`backends/versions.json`.

After an upstream NInfer change, `scripts/rebase-ninfer-fork-patches.py` re-appends the patch; see
[upstream-sync.md](upstream-sync.md).
