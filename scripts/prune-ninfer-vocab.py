#!/usr/bin/env python3
"""Give a `.ninfer` artifact the vocabulary of a vocab-pruned one, without requantizing it.

`--source` is the artifact whose layers you want, e.g. ninfer-ext's EXL3
`qwen3_8_27b_exl3_4bpw.ninfer`. `--donor` is a `.ninfer` converted from a vocab-pruned GGUF of the
same model (see docs/ninfer-pruned-vocab.md). The output keeps the source byte for byte except
for what depends on the vocabulary:

* text/token_embedding  the source's rows, gathered into the donor's token order. Rows past the
                        donor tokenizer (embedding padding) are zero.
* text/output_head      the donor's head. A quantized head cannot drop rows: EXL3 rotates whole
                        128-row blocks of outputs together.
* tokenizer.json, tokenizer_config.json, generation_config.json
                        the donor's, which carry the renumbered special token IDs.
* text vocab_size       the donor's.
* proposal head         dropped. Its fused top-k kernel takes exactly 131072 indexed rows, more
                        than a pruned vocabulary has. MTP drafts through the output head instead.

The mapping is by token string, checked in both directions: every donor token must exist in the
source tokenizer. Run `prepare-backends.py` first; this imports NInfer's artifact tools from
backends/ninfer, including the EXL3 formats of the `exl3` patch.

Usage:
    python scripts/prune-ninfer-vocab.py --source qwen3_8_27b_exl3_4bpw.ninfer \
        --donor qwen3_8_27b_ascii_iq4xs.ninfer --out qwen3_8_27b_exl3_4bpw_ascii.ninfer
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING, HEAD = "text/token_embedding", "text/output_head"
# Resources that carry token IDs; the donor's replace the source's.
TOKEN_RESOURCES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
COPY_CHUNK = 8 << 20


def load_tools(ninfer: Path):
    if not (ninfer / "tools/artifact/writer.py").is_file():
        raise SystemExit(f"{ninfer}: NInfer artifact tools not found; run scripts/prepare-backends.py")
    sys.path.insert(0, str(ninfer))
    from tools.artifact.formats import NUMERIC_FORMATS
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import ResourceObject, ResourceSpec, TensorSpec
    from tools.artifact.writer import ArtifactWriter
    if "exl3_mul1" not in NUMERIC_FORMATS:
        raise SystemExit(f"{ninfer}: these artifact tools lack EXL3; apply the kvmem patch series")
    return Artifact, ArtifactWriter, ResourceObject, ResourceSpec, TensorSpec


def bound_object(artifact, binding: str):
    value = artifact.directory.bindings.get(binding)
    if not isinstance(value, dict) or set(value) != {"object"}:
        raise SystemExit(f"{artifact.path.name}: {binding} must bind one whole object, got {value!r}")
    return artifact.object(value["object"])


def resource(artifact, name: str) -> tuple[str, bytes] | None:
    object_id = artifact.directory.components["text"].get("resources", {}).get(name)
    return None if object_id is None else (object_id, artifact.read_object(object_id))


def token_ids(tokenizer: dict, label: str) -> dict[str, int]:
    ids = dict(tokenizer["model"]["vocab"])
    for token in tokenizer.get("added_tokens", []):
        if ids.setdefault(token["content"], token["id"]) != token["id"]:
            raise SystemExit(f"{label}: {token['content']!r} has two IDs")
    if sorted(ids.values()) != list(range(len(ids))):
        raise SystemExit(f"{label}: token IDs are not contiguous from 0")
    return ids


def build_rows(source_tokens: dict[str, int], donor_tokens: dict[str, int]) -> list[int]:
    """Source row of each donor token, in donor ID order."""
    missing = [token for token in donor_tokens if token not in source_tokens]
    if missing:
        raise SystemExit(f"{len(missing)} donor tokens are not in the source tokenizer, "
                         f"e.g. {missing[:5]!r}; the artifacts are not the same model")
    rows = [0] * len(donor_tokens)
    for token, new in donor_tokens.items():
        rows[new] = source_tokens[token]
    return rows


def runs(rows: list[int]):
    """(destination row, source row, count) for consecutive source rows."""
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i] != rows[i - 1] + 1:
            yield start, rows[start], i - start
            start = i


class Plan:
    """Everything the output needs, derived from the two directories and their small resources."""

    def __init__(self, source, donor, tools, *, chat_template="source", name=None) -> None:
        _, _, ResourceObject, ResourceSpec, TensorSpec = tools
        src_dir, don_dir = source.directory, donor.directory
        src_text, don_text = src_dir.components["text"], don_dir.components["text"]
        self.embedding, self.head = bound_object(source, EMBEDDING), bound_object(source, HEAD)
        don_embedding, self.donor_head = bound_object(donor, EMBEDDING), bound_object(donor, HEAD)
        self.hidden = hidden = src_text["config"]["hidden_size"]
        self.vocab = vocab = don_text["config"]["vocab_size"]
        if don_text["config"]["hidden_size"] != hidden:
            raise SystemExit(f"hidden size differs: source {hidden}, donor {don_text['config']['hidden_size']}")
        if (self.embedding.format, self.embedding.layout) != ("bf16", "contiguous_le_v1"):
            raise SystemExit(f"source embedding is {self.embedding.format}/{self.embedding.layout}; "
                             "row gathering needs bf16/contiguous_le_v1")
        if (self.embedding.shape[1] != hidden or self.donor_head.shape != (vocab, hidden)
                or don_embedding.shape[0] != vocab):
            raise SystemExit("embedding or head shapes do not match the configured vocabulary and hidden size")

        tokenizers = {}
        for label, artifact in (("source", source), ("donor", donor)):
            found = resource(artifact, "tokenizer.json")
            if found is None:
                raise SystemExit(f"{label} has no tokenizer.json resource")
            tokenizers[label] = token_ids(json.loads(found[1]), f"{label} tokenizer.json")
        self.rows = build_rows(tokenizers["source"], tokenizers["donor"])
        if len(self.rows) > vocab:
            raise SystemExit(f"donor tokenizer has {len(self.rows)} tokens for {vocab} embedding rows")

        self.replaced = {}  # source resource object -> donor bytes
        for item in TOKEN_RESOURCES + (("chat_template.jinja",) if chat_template == "donor" else ()):
            ours, theirs = resource(source, item), resource(donor, item)
            if theirs is None:
                if ours is not None:
                    raise SystemExit(f"donor lacks {item}; the source's would keep unpruned token IDs")
                continue
            if ours is None:
                raise SystemExit(f"source lacks {item}; add it to the source artifact first")
            self.replaced[ours[0]] = theirs[1]

        self.proposal = {value["object"] for key, value in src_dir.bindings.items()
                         if key.startswith("proposal/")}
        self.specs = []
        for obj in src_dir.objects:
            if obj.id in self.proposal:
                continue
            if isinstance(obj, ResourceObject):
                size = len(self.replaced[obj.id]) if obj.id in self.replaced else obj.bytes
                self.specs.append(ResourceSpec(obj.id, size, obj.encoding))
            elif obj.id == self.embedding.id:
                self.specs.append(TensorSpec(obj.id, (vocab, hidden), obj.format, obj.layout))
            elif obj.id == self.head.id:
                head = self.donor_head
                self.specs.append(TensorSpec(obj.id, head.shape, head.format, head.layout,
                                             head.divisors, head.bitrate_half_bits))
            else:
                self.specs.append(TensorSpec(obj.id, obj.shape, obj.format, obj.layout,
                                             obj.divisors, obj.bitrate_half_bits))

        self.components = copy.deepcopy(src_dir.components)
        self.components["text"]["config"]["vocab_size"] = vocab
        self.components["text"].pop("proposal", None)
        self.bindings = {k: v for k, v in src_dir.bindings.items() if not k.startswith("proposal/")}
        head_uses = [u for u in don_dir.uses if u["parameter"] == HEAD]
        if {u["input"] for u in head_uses} != {u["input"] for u in src_dir.uses if u["parameter"] == HEAD}:
            raise SystemExit("donor head uses differ from the source's (is MTP missing from one of them?)")
        self.uses = [u for u in src_dir.uses
                     if u["parameter"] != HEAD and not u["parameter"].startswith("proposal/")] + head_uses
        self.metadata = copy.deepcopy(src_dir.metadata)
        self.metadata["name"] = name or f"{self.metadata.get('name', 'model')}-pruned"
        self.provenance = {"converter": "kvmem prune-ninfer-vocab",
                           "source": {"path": source.path.name, "provenance": src_dir.provenance},
                           "donor": {"path": donor.path.name, "provenance": don_dir.provenance},
                           "vocab_size": vocab, "tokens": len(self.rows)}

    def write(self, writer, source, donor) -> None:
        row_bytes = self.hidden * 2
        step = COPY_CHUNK // row_bytes
        for spec in self.specs:
            if spec.id in self.replaced:
                writer.write_object(spec.id, self.replaced[spec.id])
            elif spec.id == self.embedding.id:
                for dst, src, count in runs(self.rows):
                    for offset in range(0, count, step):
                        n = min(step, count - offset)
                        data = source.read_range(self.embedding.offset + (src + offset) * row_bytes,
                                                 n * row_bytes)
                        writer.write_region(spec.id, (dst + offset) * row_bytes, data)
                if self.vocab > len(self.rows):
                    writer.write_zeros(spec.id, len(self.rows) * row_bytes,
                                       (self.vocab - len(self.rows)) * row_bytes)
            elif spec.id == self.head.id:
                writer.write_object(spec.id, donor.iter_object(self.donor_head.id))
            else:
                writer.write_object(spec.id, source.iter_object(spec.id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="artifact whose layers are kept")
    parser.add_argument("--donor", type=Path, required=True, help="vocab-pruned artifact of the same model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", help="metadata name (default: the source's with -pruned)")
    parser.add_argument("--chat-template", choices=("source", "donor"), default="source",
                        help="which artifact's chat template to keep (default: source)")
    parser.add_argument("--ninfer", type=Path, default=ROOT / "backends/ninfer",
                        help="NInfer tree with the artifact tools (default: backends/ninfer)")
    args = parser.parse_args(argv)
    tools = load_tools(args.ninfer)
    Artifact, ArtifactWriter = tools[:2]
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    with Artifact(args.source) as source, Artifact(args.donor) as donor:
        plan = Plan(source, donor, tools, chat_template=args.chat_template, name=args.name)
        with ArtifactWriter(args.out, plan.specs, components=plan.components, bindings=plan.bindings,
                            uses=plan.uses, metadata=plan.metadata, provenance=plan.provenance) as writer:
            plan.write(writer, source, donor)

    with Artifact(args.out) as out:
        for obj in out.objects:
            out.object(obj.id)  # validates every encoding against its declared size
        print(f"{args.out}: {len(out.objects)} objects, {out.file_bytes / 2**30:.2f} GiB, "
              f"vocabulary {plan.vocab} rows ({len(plan.rows)} tokens), proposal head "
              f"{'dropped' if plan.proposal else 'absent'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
