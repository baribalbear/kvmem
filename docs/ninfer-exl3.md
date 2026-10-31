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

## What the patch adds

| Area | Files |
|---|---|
| Format and geometry | `core/weight.h` (`QType::EXL3_MUL1 = 32`, `QuantLayout::TrellisT16 = 7`, `Weight::input_scales`, `bitrate_half_bits`), `core/weight_view.*` |
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

Not ported: the native quantizer (`ninfer-quantize`, `src/quantize/exl3`, `tools/exl3`), the EXL3
artifact writer (`tensor_output.py`) and Flash-Next expert offload (`ops/offload_moe` EXL3 banks).

## Verification status

Done without a GPU:

* `prepare-backends.py`'s `verify_patch_series` replays all 25 NInfer patches from the base commit
  and reproduces every recorded tree.
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

`.github/workflows/ninfer-windows.yml` compiles and packages the patched NInfer for Windows x64 the
way NInfer's own native release does. It runs on pushes that change the NInfer patches, or manually
with a CUDA version and architecture list (default CUDA 13.4.3, `120a` for RTX 50-series cards). One job builds the
vcpkg FFmpeg and curl packages and caches them; four jobs fill a ccache with object shards; the last
job links `ninfer-serve` and the other release products and runs NInfer's `package-release.ps1`.
The `ninfer-windows-cuda<version>` artifact is the release archive. The runners have no GPU, so the
run proves compilation and packaging only; `BUILD_TESTING` is off, as in the upstream release.

## Regenerating the patch

Apply the series in `backends/ninfer`, edit, then diff against the tree the previous patch
(`cross-owner-fork`) produces, not against `HEAD`, which is the upstream base:

```bash
git add -A
git diff --cached --full-index bb98e6d4d106096627df42219d73807c3caba48e > ../patches/ninfer-exl3.patch
git write-tree
```

Update the `exl3` entry's `sha256` and `after_tree` and the backend's `patched_tree` in
`backends/versions.json`.
