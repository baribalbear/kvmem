"""Synthetic EXL3 and pruned artifacts check what prune-ninfer-vocab.py keeps, gathers and drops.

Needs NInfer's artifact tools with the EXL3 patch: backends/ninfer after prepare-backends.py, or
another tree named by NINFER_TOOLS.
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NINFER = Path(os.environ.get('NINFER_TOOLS', ROOT / 'backends/ninfer'))
HAVE_TOOLS = (NINFER / 'tools/artifact/writer.py').is_file()
if HAVE_TOOLS:
    sys.path.insert(0, str(NINFER))
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import ResourceSpec, TensorSpec
    from tools.artifact.writer import ArtifactWriter
    from tools.artifact.layouts import encoded_size

spec = importlib.util.spec_from_file_location('prune', ROOT / 'scripts/prune-ninfer-vocab.py')
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)

HIDDEN = 128
BASE = [f't{i}' for i in range(250)]
SPECIAL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']


def tokenizer(tokens, special, start):
    return {'model': {'type': 'BPE', 'vocab': {t: i for i, t in enumerate(tokens)}, 'merges': []},
            'added_tokens': [{'id': start + i, 'content': t, 'special': True} for i, t in enumerate(special)]}


def write(path, tensors, resources, *, vocab, uses, proposal=None):
    """tensors: (id, binding, shape, format, layout, half_bits); payload is deterministic noise."""
    specs, bindings, payloads = [], {}, {}
    rng = random.Random(path.name)
    for object_id, binding, shape, fmt, layout, half_bits in tensors:
        specs.append(TensorSpec(object_id, shape, fmt, layout, 1, half_bits))
        size = encoded_size(layout, fmt, shape, 1, half_bits)
        payloads[object_id] = rng.randbytes(size)
        bindings[binding] = {'object': object_id}
    names = {}
    for name, data in resources.items():
        object_id = f'resource/text/{name}'
        specs.append(ResourceSpec(object_id, len(data)))
        payloads[object_id] = data
        names[name] = object_id
    text = {'config': {'model_type': 'qwen3_5_text', 'hidden_size': HIDDEN, 'vocab_size': vocab},
            'resources': names}
    if proposal:
        text['proposal'] = {'domain': 'indexed', 'rows': proposal}
    components = {'text': text, 'mtp': {'config': {}, 'target': 'text'}}
    with ArtifactWriter(path, specs, components=components, bindings=bindings, uses=uses,
                        metadata={'name': path.stem}, provenance={'test': True}) as writer:
        for object_id, data in payloads.items():
            writer.write_object(object_id, data)
    return payloads


def head_uses(policy):
    return [{'parameter': 'text/output_head', 'input': i, 'activation_policy': policy}
            for i in ('text/final_hidden', 'mtp/final_hidden')]


@unittest.skipUnless(HAVE_TOOLS, 'NInfer artifact tools are not prepared')
class PruneVocab(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        base_vocab = 384  # 253 tokens plus embedding padding; EXL3 needs multiples of 128
        source_tok = tokenizer(BASE, SPECIAL, len(BASE))
        self.source = self.dir / 'source.ninfer'
        self.source_payloads = write(self.source, [
            ('weight/0', 'text/token_embedding', (base_vocab, HIDDEN), 'bf16', 'contiguous_le_v1', None),
            ('weight/1', 'text/output_head', (base_vocab, HIDDEN), 'exl3_mul1', 'trellis_t16_v1', 12),
            ('weight/2', 'text/layers/0/proj', (HIDDEN, HIDDEN), 'exl3_mul1', 'trellis_t16_v1', 8),
            ('weight/3', 'text/final_norm', (HIDDEN,), 'bf16', 'contiguous_le_v1', None),
            ('weight/4', 'proposal/head', (64, HIDDEN), 'q4_g64_fp16', 'row_split_k128_v1', None),
            ('weight/5', 'proposal/token_ids', (64,), 'int32', 'contiguous_le_v1', None),
        ], {'tokenizer.json': json.dumps(source_tok).encode(),
            'tokenizer_config.json': b'{"source": true}',
            'generation_config.json': b'{"eos_token_id": 252}',
            'chat_template.jinja': b'source template'},
            vocab=base_vocab, proposal=64,
            uses=head_uses('A16Only') + [{'parameter': 'proposal/head', 'input': 'mtp/final_hidden',
                                          'activation_policy': 'A16Only'}])
        # Keep every third base token, in order, then the special tokens: IDs are renumbered.
        self.kept = BASE[::3]
        donor_tok = tokenizer(self.kept, SPECIAL, len(self.kept))
        self.donor_vocab = 96
        self.donor = self.dir / 'donor.ninfer'
        self.donor_payloads = write(self.donor, [
            ('w/embd', 'text/token_embedding', (self.donor_vocab, HIDDEN), 'q8_g32_fp16', 'row_split_k128_v1', None),
            ('w/head', 'text/output_head', (self.donor_vocab, HIDDEN), 'q6_g64_fp16', 'row_split_k128_v1', None),
        ], {'tokenizer.json': json.dumps(donor_tok).encode(),
            'tokenizer_config.json': b'{"donor": true}',
            'generation_config.json': b'{"eos_token_id": 86}',
            'chat_template.jinja': b'donor template'},
            vocab=self.donor_vocab, uses=head_uses('AllowA8'))
        self.out = self.dir / 'out.ninfer'

    def run_tool(self, *extra):
        return prune.main(['--source', str(self.source), '--donor', str(self.donor),
                           '--out', str(self.out), '--ninfer', str(NINFER), *extra])

    def test_layers_kept_vocabulary_replaced(self):
        self.assertEqual(self.run_tool(), 0)
        with Artifact(self.out) as out:
            d = out.directory
            self.assertEqual(d.components['text']['config']['vocab_size'], self.donor_vocab)
            self.assertNotIn('proposal', d.components['text'])
            self.assertFalse([b for b in d.bindings if b.startswith('proposal/')])
            self.assertFalse({'weight/4', 'weight/5'} & {o.id for o in d.objects})
            # Untouched layers are byte-identical.
            for object_id in ('weight/2', 'weight/3'):
                self.assertEqual(out.read_object(object_id), self.source_payloads[object_id])
            # The head is the donor's, in its format, and its uses come with it.
            head = out.object('weight/1')
            self.assertEqual((head.shape, head.format), ((self.donor_vocab, HIDDEN), 'q6_g64_fp16'))
            self.assertEqual(out.read_object('weight/1'), self.donor_payloads['w/head'])
            self.assertEqual([u for u in d.uses if u['parameter'] == 'text/output_head'], head_uses('AllowA8'))
            self.assertFalse([u for u in d.uses if u['parameter'].startswith('proposal/')])
            # Embedding rows are the source's rows of the same tokens; padding rows are zero.
            embedding = out.read_object('weight/0')
            source = self.source_payloads['weight/0']
            row = HIDDEN * 2
            tokens = self.kept + SPECIAL
            source_ids = {t: i for i, t in enumerate(BASE + SPECIAL)}
            for new, token in enumerate(tokens):
                old = source_ids[token]
                self.assertEqual(embedding[new * row:(new + 1) * row], source[old * row:(old + 1) * row], token)
            self.assertEqual(embedding[len(tokens) * row:], bytes((self.donor_vocab - len(tokens)) * row))
            # Token-bearing resources are the donor's; the chat template stays the source's.
            resources = d.components['text']['resources']
            self.assertEqual(out.read_object(resources['generation_config.json']), b'{"eos_token_id": 86}')
            self.assertEqual(out.read_object(resources['tokenizer_config.json']), b'{"donor": true}')
            self.assertEqual(out.read_object(resources['chat_template.jinja']), b'source template')

    def test_donor_chat_template(self):
        self.assertEqual(self.run_tool('--chat-template', 'donor'), 0)
        with Artifact(self.out) as out:
            resources = out.directory.components['text']['resources']
            self.assertEqual(out.read_object(resources['chat_template.jinja']), b'donor template')

    def test_rejects_tokens_missing_from_source(self):
        self.kept = self.kept + ['not-in-source']
        tok = tokenizer(self.kept, SPECIAL, len(self.kept))
        self.donor.unlink()
        self.donor_payloads = write(self.donor, [
            ('w/embd', 'text/token_embedding', (self.donor_vocab, HIDDEN), 'q8_g32_fp16', 'row_split_k128_v1', None),
            ('w/head', 'text/output_head', (self.donor_vocab, HIDDEN), 'q6_g64_fp16', 'row_split_k128_v1', None),
        ], {'tokenizer.json': json.dumps(tok).encode(), 'tokenizer_config.json': b'{}',
            'generation_config.json': b'{}'}, vocab=self.donor_vocab, uses=head_uses('AllowA8'))
        with self.assertRaisesRegex(SystemExit, 'not in the source tokenizer'):
            self.run_tool()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
