#!/usr/bin/env python3
"""Build `--model` resources for converting a vocab-pruned GGUF to `.ninfer`.

`prune_vocab.py` (bsaleh03/ASCII-Condensed-prune-tools) drops tokens from a byte-level BPE GGUF
and renumbers the survivors in their original order. NInfer's converter takes the tokenizer and
config from `--model`, so the base checkpoint's files must be renumbered the same way:

* tokenizer.json         vocab and added_tokens renumbered, merges taken from the pruned GGUF
* tokenizer_config.json  added_tokens_decoder renumbered
* generation_config.json bos/eos/pad ids renumbered
* config.json            vocab_size set to the pruned embedding rows, every *_token_id renumbered

Every other small file (chat template, processor configs) is copied unchanged. The mapping is
derived by token string, so only the pruned GGUF and the base checkpoint's files are needed. Only
the GGUF header is read; tensor data is never touched.

Usage:
    python scripts/build-pruned-vocab-resources.py --gguf PRUNED.gguf --base Qwen3.8-27B --out OUT_DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import struct
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ninfer")
REWRITTEN = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json")
PAD_TOKEN = re.compile(r"\[PAD\d+\]")  # llama.cpp's filler for embedding rows past the tokenizer

# GGUF metadata value types.
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class _Reader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("GGUF header is truncated")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        n = self.read("<Q")
        data = self.stream.read(n)
        if len(data) != n:
            raise ValueError("GGUF header is truncated")
        return data.decode("utf-8")

    def value(self, kind: int):
        if kind in _SCALARS:
            return self.read(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item, count = self.read("<I"), self.read("<Q")
            return [self.value(item) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")


def read_gguf_header(path: Path) -> tuple[dict, dict[str, tuple[int, ...]]]:
    """Metadata and tensor shapes (ggml order: ne[0] first) of a GGUF v2/v3 file."""

    with path.open("rb") as stream:
        r = _Reader(stream)
        if stream.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.read("<I")
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        tensors, kvs = r.read("<Q"), r.read("<Q")
        kv = {}
        for _ in range(kvs):
            key = r.string()
            kv[key] = r.value(r.read("<I"))
        shapes = {}
        for _ in range(tensors):
            name = r.string()
            dims = r.read("<I")
            shapes[name] = tuple(r.read("<Q") for _ in range(dims))
            r.read("<I")  # type
            r.read("<Q")  # offset
    return kv, shapes


def _merge_pair(entry) -> tuple[str, str]:
    if isinstance(entry, str):
        left, sep, right = entry.partition(" ")
        if not sep:
            raise ValueError(f"malformed merge {entry!r}")
        return left, right
    if isinstance(entry, list) and len(entry) == 2:
        return entry[0], entry[1]
    raise ValueError(f"malformed merge {entry!r}")


def build_mapping(tokens: list[str], tokenizer: dict) -> tuple[dict[int, int], int]:
    """Old id -> new id for every surviving token, and the public token count."""

    original: dict[str, int] = dict(tokenizer["model"]["vocab"])
    for token in tokenizer.get("added_tokens", []):
        if original.get(token["content"], token["id"]) != token["id"]:
            raise ValueError(f"base tokenizer maps {token['content']!r} to two ids")
        original[token["content"]] = token["id"]
    old_to_new: dict[int, int] = {}
    previous, padding_from = -1, None
    for new, token in enumerate(tokens):
        old = original.get(token)
        if old is None:
            if not PAD_TOKEN.fullmatch(token):
                raise ValueError(
                    f"GGUF token {new} {token!r} is not in the base tokenizer: wrong --base?"
                )
            padding_from = new if padding_from is None else padding_from
            continue
        if padding_from is not None:
            raise ValueError(f"GGUF token {new} {token!r} follows padding rows")
        if old <= previous:
            raise ValueError(f"GGUF token {new} {token!r} breaks the original order")
        old_to_new[old] = new
        previous = old
    return old_to_new, len(old_to_new)


def remap_id(value, old_to_new: dict[int, int], label: str):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value not in old_to_new:
            raise ValueError(f"{label} = {value} was pruned")
        return old_to_new[value]
    if isinstance(value, list):
        return [remap_id(v, old_to_new, label) for v in value]
    return value


def remap_token_ids(obj, old_to_new: dict[int, int], path: str = "") -> None:
    """Renumber every `*_token_id` / `*_token_ids` field of a config, recursively."""

    if isinstance(obj, dict):
        for key, value in obj.items():
            label = f"{path}.{key}" if path else key
            if key.endswith(("_token_id", "_token_ids")):
                obj[key] = remap_id(value, old_to_new, label)
            else:
                remap_token_ids(value, old_to_new, label)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            remap_token_ids(value, old_to_new, f"{path}[{index}]")


def build_tokenizer(tokenizer: dict, old_to_new: dict[int, int], merges: list[str]) -> dict:
    out = copy.deepcopy(tokenizer)
    vocab = {token: old_to_new[old] for token, old in tokenizer["model"]["vocab"].items()
             if old in old_to_new}
    out["model"]["vocab"] = dict(sorted(vocab.items(), key=lambda item: item[1]))
    for token in out.get("added_tokens", []):
        token["id"] = remap_id(token["id"], old_to_new, f"added token {token['content']!r}")
    as_pairs = bool(tokenizer["model"].get("merges")) and isinstance(
        tokenizer["model"]["merges"][0], list)
    kept = []
    for entry in merges:
        left, right = _merge_pair(entry)
        for symbol in (left, right, left + right):
            if symbol not in vocab:
                raise ValueError(f"GGUF merge {entry!r} references pruned symbol {symbol!r}")
        kept.append([left, right] if as_pairs else f"{left} {right}")
    out["model"]["merges"] = kept
    return out


def build_tokenizer_config(config: dict, old_to_new: dict[int, int]) -> dict:
    out = copy.deepcopy(config)
    decoder = config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        out["added_tokens_decoder"] = {
            str(remap_id(int(raw), old_to_new, f"added_tokens_decoder[{raw}]")): token
            for raw, token in decoder.items()
        }
    remap_token_ids(out, old_to_new)
    return out


def build_model_config(config: dict, old_to_new: dict[int, int], rows: int) -> dict:
    out = copy.deepcopy(config)
    remap_token_ids(out, old_to_new)
    text = out["text_config"] if isinstance(out.get("text_config"), dict) else out
    text["vocab_size"] = rows
    if text is not out and "vocab_size" in out:
        out["vocab_size"] = rows
    return out


def check_domain(tokenizer: dict, tokenizer_config: dict, rows: int) -> int:
    """The converter's token_domain check (tools/convert/resources.py)."""

    ids: dict[int, str] = {}
    entries = [(i, t) for t, i in tokenizer["model"]["vocab"].items()]
    entries += [(t["id"], t["content"]) for t in tokenizer.get("added_tokens", [])]
    entries += [(int(i), t["content"])
                for i, t in tokenizer_config.get("added_tokens_decoder", {}).items()]
    for index, content in entries:
        if not 0 <= index < rows or ids.setdefault(index, content) != content:
            raise ValueError(f"token {index} {content!r} is out of range or inconsistent")
    if set(ids) != set(range(len(ids))):
        raise ValueError("token ids are not contiguous")
    return len(ids)


def compare_merges(base: dict, vocab: dict, gguf_merges: list[str]) -> int:
    """Count base merges that survive the pruned vocab but are missing from the GGUF."""

    gguf = {_merge_pair(m) for m in gguf_merges}
    missing = 0
    for entry in base["model"].get("merges", []):
        left, right = _merge_pair(entry)
        if left in vocab and right in vocab and left + right in vocab and (left, right) not in gguf:
            missing += 1
    return missing


def write_json(path: Path, value: dict, indent: int | None) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", type=Path, required=True, help="the pruned GGUF")
    parser.add_argument("--base", type=Path, required=True,
                        help="base checkpoint directory (config and tokenizer files)")
    parser.add_argument("--out", type=Path, required=True, help="new directory for --model")
    args = parser.parse_args(argv)

    kv, shapes = read_gguf_header(args.gguf)
    if kv.get("tokenizer.ggml.model") not in ("gpt2", "bpe"):
        raise SystemExit("only byte-level BPE vocabularies are supported")
    tokens, merges = kv["tokenizer.ggml.tokens"], kv.get("tokenizer.ggml.merges", [])
    rows = shapes["token_embd.weight"][1]
    if len(tokens) != rows:
        raise SystemExit(f"GGUF has {len(tokens)} tokens but {rows} embedding rows")
    if "output.weight" in shapes and shapes["output.weight"][1] != rows:
        raise SystemExit("token_embd.weight and output.weight row counts differ")

    load = lambda name: json.loads((args.base / name).read_text(encoding="utf-8"))
    tokenizer, tokenizer_config = load("tokenizer.json"), load("tokenizer_config.json")
    generation, model_config = load("generation_config.json"), load("config.json")
    if tokenizer.get("model", {}).get("type") != "BPE":
        raise SystemExit("base tokenizer.json is not BPE")

    old_to_new, public = build_mapping(tokens, tokenizer)
    new_tokenizer = build_tokenizer(tokenizer, old_to_new, merges)
    new_tokenizer_config = build_tokenizer_config(tokenizer_config, old_to_new)
    new_generation = copy.deepcopy(generation)
    remap_token_ids(new_generation, old_to_new)
    new_config = build_model_config(model_config, old_to_new, rows)

    count = check_domain(new_tokenizer, new_tokenizer_config, rows)
    if count != public:
        raise SystemExit(f"public domain {count} differs from mapped tokens {public}")
    eos = kv.get("tokenizer.ggml.eos_token_id")
    stops = new_generation.get("eos_token_id")
    if eos is not None and eos not in (stops if isinstance(stops, list) else [stops]):
        raise SystemExit(f"GGUF eos {eos} is not among the remapped eos ids {stops}")
    missing = compare_merges(tokenizer, new_tokenizer["model"]["vocab"], merges)

    args.out.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.base.iterdir()):
        if path.is_file() and path.name not in REWRITTEN and not path.name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(path, args.out / path.name)
    write_json(args.out / "tokenizer.json", new_tokenizer, None)
    write_json(args.out / "tokenizer_config.json", new_tokenizer_config, 2)
    write_json(args.out / "generation_config.json", new_generation, 2)
    write_json(args.out / "config.json", new_config, 2)

    print(f"embedding rows   {rows} (base {model_config.get('text_config', model_config)['vocab_size']})")
    print(f"public tokens    {public} ({rows - public} padding rows)")
    print(f"merges           {len(merges)} from GGUF; {missing} base merges absent from GGUF")
    print(f"eos ids          {generation.get('eos_token_id')} -> {stops}")
    print(f"wrote            {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
