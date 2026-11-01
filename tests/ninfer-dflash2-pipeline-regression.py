"""Real-GPU DFlash2 layer-pipeline or heads-TP qualification; no dependency download.

Run with a matching built worker and DLL PATH, a dense artifact containing DFlash2,
and two idle GPU UUIDs in stage order. Saves every request/response and native trace.
The ordinary decoder is an independent speculative-boundary reference for small
fully resident prompts. Token/output differences are reported, not assumed defects.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpus', required=True, help='Two GPU UUIDs, first owns the drafter')
    p.add_argument('--stage-layers', default='40,24')
    p.add_argument('--tensor-parallel', action='store_true')
    p.add_argument('--tensor-split', default='3,1', help='TP rank weights, or auto')
    p.add_argument('--kv', choices=['int8', 'rk8v4', 'nvfp4'], default='int8')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gpus = a.gpus.split(',')
    assert len(gpus) == 2 and len(set(gpus)) == 2 and all(x.startswith('GPU-') for x in gpus)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'gpu_uuids': gpus, 'model': str(a.model), 'profiles': [],
              'binary_sha256': hashlib.sha256(a.exe.read_bytes()).hexdigest()}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    prompts = ['Explain how a key-value cache helps a language model generate text. Use three short paragraphs.',
               'Write a Python function that returns the first n Fibonacci numbers. Include one example.']
    reference = None
    mode = 'tensor' if a.tensor_parallel else 'staged'
    for name, stage, graphs, lanes, draft in [('ordinary', True, True, 1, False),
            ('single-dflash2', False, True, 1, True), (mode+'-dflash2', True, True, 1, True),
            (mode+'-eager', True, False, 1, True), (mode+'-concurrent', True, True, 2, True)]:
        out = a.output / name
        out.mkdir()
        selected = gpus if stage else gpus[:1]
        occupancy = subprocess.check_output(['nvidia-smi', '-i', ','.join(selected),
            '--query-gpu=uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        for row in occupancy.strip().splitlines():
            uuid, model, used, busy = [x.strip() for x in row.split(',')]
            assert uuid in selected and int(used) < 512 and int(busy) < 5, row
        apps = subprocess.check_output(['nvidia-smi', '-i', ','.join(selected),
            '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert not apps.strip(), apps
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        argv = [str(a.exe.resolve()), str(a.model.resolve()), '--port', str(port),
            '--max-context', '8192', '--prefill-chunk', '128', '--max-concurrency', str(lanes),
            '--device-profile', 'off', '--no-webui', '--no-thinking', '--kv-dtype', a.kv,
            '--kvmem-budget', '768', '--kvmem-gen-reserve', '1024', '--kvmem-host-mib', '4096',
            '--kvmem-sessions', '4', '--kvmem-sink-tokens', '64', '--kvmem-recent-tokens', '64',
            '--kvmem-verify-transfers', '--ngram-draft-tokens', '0', '--spec', 'dflash2' if draft else 'none']
        if draft:
            argv += ['--draft-tokens', '5']
        if stage:
            argv += ['--devices', '0,1']
            if a.tensor_parallel:
                argv += ['--tensor-parallel', '--tensor-attention', 'heads']
                if a.tensor_split != 'auto':
                    argv += ['--tensor-split', a.tensor_split]
            else:
                argv += ['--stage-layers', a.stage_layers]
        else:
            argv += ['--device', '0']
        if not graphs:
            argv += ['--no-cuda-graph']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(selected), CUDA_DEVICE_ORDER='PCI_BUS_ID', KVMEM_TRACE='1')
        env['PATH'] = str(a.exe.resolve().parent) + os.pathsep + env['PATH']
        record = {'name': name, 'argv': argv, 'occupancy': occupancy}
        (out / 'manifest.json').write_text(json.dumps(record, indent=2))

        def call(path, body=None):
            req = urllib.request.Request(f'http://127.0.0.1:{port}'+path,
                None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with opener.open(req, timeout=180) as response:
                return json.load(response)

        def ask(index):
            body = {'messages': [{'role': 'user', 'content': prompts[index]}], 'temperature': 0,
                'seed': 42, 'max_tokens': 96, 'enable_thinking': False,
                'prompt_cache_key': f'pipeline-{index}', 'kvmem': {'conversation_id': f'pipeline-{index}'}}
            (out / f'{index}.request.json').write_text(json.dumps(body, indent=2))
            response = call('/v1/chat/completions', body)
            (out / f'{index}.response.json').write_text(json.dumps(response, indent=2))
            assert 0 < response['usage']['completion_tokens'] <= 96
            assert response['choices'][0]['message'].get('content')
            return response

        with (out / 'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'{name}: server exited; inspect {out}')
                    try:
                        call('/health')
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() > deadline:
                            raise TimeoutError(name)
                        time.sleep(.25)
                props = call('/props')
                (out / 'props-before.json').write_text(json.dumps(props, indent=2))
                assert props['execution']['mode'] == (
                    ('tensor' if a.tensor_parallel else 'layer') if stage else 'single')
                if stage and a.tensor_parallel:
                    assert props['execution']['tensor_attention'] == 'heads'
                    assert len(props['execution']['tensor_shares']) == len(gpus)
                if draft:
                    assert props['speculative']['algorithm'] == 'dflash2'
                    cap = next(c for c in props['speculative']['capabilities'] if c['algorithm'] == 'dflash2')
                    assert cap['max_devices'] == 8
                if lanes == 2:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        responses = list(pool.map(ask, range(len(prompts))))
                else:
                    responses = [ask(i) for i in range(len(prompts))]
                contents = [r['choices'][0]['message']['content'] for r in responses]
                if reference is None:
                    reference = contents
                record['matches_ordinary'] = [x == y for x, y in zip(contents, reference)]
                after = call('/props')
                (out / 'props-after.json').write_text(json.dumps(after, indent=2))
                if draft:
                    stats = after['speculative']['statistics']
                    assert stats['rounds'] > 0 and stats['drafted_tokens'] > 0
                    assert 0 <= stats['accepted_tokens'] <= stats['drafted_tokens']
                    record['statistics'] = stats
                record['status'] = 'passed'
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
        report['profiles'].append(record)
        (a.output / 'results.json').write_text(json.dumps(report, indent=2))
        print('PASS', name, 'matches ordinary:', record['matches_ordinary'], flush=True)
        # Let device utilization sampling settle after this profile's worker exits.
        time.sleep(2)
    report['status'] = 'passed'
    (a.output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
