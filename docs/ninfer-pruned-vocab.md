# Vocab-pruned GGUF → `.ninfer` (kvmem)

Converts a GGUF pruned with
[ASCII-Condensed-prune-tools](https://github.com/bsaleh03/ASCII-Condensed-prune-tools) — e.g.
[troed/Qwen3.8-27B-ASCII-Condensed](https://huggingface.co/troed/Qwen3.8-27B-ASCII-Condensed) — into
a `.ninfer` artifact for kvmem's NInfer backend.

## Why only a resource script

* **Converter: no patch needed in kvmem's tree.** kvmem's NInfer patches replace upstream's
  `qwen3_8_27b_gguf` recipe (hardcoded `VOCABULARY = 248320`) with `qwen3_5_dense_gguf`. That recipe
  takes the vocabulary size from `config.json` (`tools/convert/dense_geometry.py`). Unpatched upstream
  `ninfer-all` still rejects a pruned GGUF, so use kvmem's `backends/ninfer`.
* **Tokenizer files must be renumbered.** `prune_vocab.py` renumbers the surviving tokens, but the
  converter takes `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and
  `config.json` from `--model`. `scripts/build-pruned-vocab-resources.py` rewrites them to match the GGUF.

## Steps

1. Prepare kvmem's backends so `backends/ninfer` is the patched tree:

   ```bash
   python scripts/prepare-backends.py
   ```

2. Download only the small files of the base checkpoint (no weights):

   ```bash
   hf download Qwen/Qwen3.8-27B config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja preprocessor_config.json video_preprocessor_config.json --local-dir Qwen3.8-27B
   ```

3. Build the renumbered `--model` directory (standard library only, reads just the GGUF header):

   ```bash
   python scripts/build-pruned-vocab-resources.py --gguf Qwen3.8-27B-ASCII-Condensed-IQ4_XS-3.84bpw.gguf --base Qwen3.8-27B --out Qwen3.8-27B-ascii
   ```

4. Convert from `backends/ninfer` (Python 3.11 with PyTorch and NumPy; `--device cpu` if you have
   no CUDA PyTorch):

   ```bash
   python -m tools.convert \
     --model /path/to/Qwen3.8-27B-ascii \
     --recipe qwen3_5_dense_gguf \
     --source gguf=/path/to/Qwen3.8-27B-ASCII-Condensed-IQ4_XS-3.84bpw.gguf \
     --components text,mtp \
     --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
     --name qwen3.8-27b-ascii \
     --out models/qwen3_8_27b_ascii_iq4xs.ninfer
   ```

5. Run it in kvmem with the NInfer backend. Start with plain generation, then try `--spec mtp`.

## Leave out

| Option | Why |
|---|---|
| `--proposal` | Built from a ranking of the original token ids, with 131,072 rows (more than the pruned vocab). |
| `dflash2` component / `--spec dflash2` | DFlash2's sparse-draft acceptance requires a 248,077-token domain (`src/ops/wrapper/speculative_round.cpp:232`). |
| `vision` | The troed release has no mmproj. To add one, the remapped `image_token_id` etc. are already in `config.json`. |

MTP uses the coupled acceptance path, which takes the token count from the tokenizer, and the
troed GGUF includes the MTP block (`blk.64.nextn.*`).

The pruned output head has 129,006 rows, no multiple of the GGUF matrix kernel's 128-row tiles.
Upstream NInfer sends a product wider than 8 columns to that kernel, so a verify batch with n-gram
drafts (`-NgramDrafts 31` with MTP 3) stopped startup with `gguf matrix product: unsupported
geometry`. The fork's `gguf-ragged-rows` NInfer patch sends such a matrix to the vector kernel up to
64 columns and to dequantize-and-GEMM beyond, so n-gram drafts work on pruned models.

## What `scripts/build-pruned-vocab-resources.py` does

* Maps every GGUF token string to its id in the base `tokenizer.json`. It fails if a token is
  unknown, if the original order is broken, or if anything follows llama.cpp's `[PADn]` filler rows.
* Rewrites the vocab and added tokens. Merges come from the GGUF, and each one is checked against
  the kept vocab.
* Renumbers every `*_token_id` field in `config.json`, `generation_config.json` and
  `tokenizer_config.json`, plus `added_tokens_decoder`. Sets `vocab_size` to the GGUF's embedding
  rows.
* Copies every other non-weight file unchanged. Refuses to write into an existing directory.
* Checks: the converter's contiguous token-domain rule, matching embedding/output row counts, and
  that the GGUF's own EOS id is among the remapped EOS ids.

## Verified (2026-10-31, against the troed GGUF header and Qwen/Qwen3.8-27B files)

* 129,006 embedding rows: 128,763 public tokens and 243 padding rows. 128,449 merges, none missing
  compared with the base merges filtered to the kept vocab. EOS `[248046, 248044]` → `[128732, 128730]`
  (`<|im_end|>`, `<|endoftext|>`).
* NInfer's own `tools/convert/resources.load_resources` and `DenseGeometry.from_config` accept the
  output.
* HF `tokenizers`: ASCII prompts (code, chat markup, SQL) give the same token sequence as the base
  tokenizer after the id remap. Non-ASCII text round-trips through byte fallback.

**Not verified yet:** the full conversion (needs the 11.4 GiB GGUF and PyTorch), the recipe's
`validate()` on the full file, and loading or generating in kvmem.
