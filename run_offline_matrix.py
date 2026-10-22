import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/opt/vllm-dev')
EVIDENCE = Path('/workspace/evidence')
METHODS = ROOT / 'vllm/model_executor/layers/pooler/seqwise/methods.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--label', default='offline-final')
    p.add_argument('--lengths', nargs='+', type=int, default=[512, 32])
    args = p.parse_args()
    lock = (EVIDENCE / 'experiment.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = EVIDENCE / args.label
    directory.mkdir(exist_ok=True)
    try:
        for round_id in range(1, args.rounds+1):
            variants = ['baseline', 'patch'] if round_id % 2 else ['patch', 'baseline']
            for variant in variants:
                source = 'methods-base.py' if variant == 'baseline' else 'methods-patched.py'
                METHODS.write_bytes((EVIDENCE / source).read_bytes())
                for length in args.lengths:
                    name = f'{variant}-r{round_id}-l{length}'
                    command = [str(ROOT / '.venv/bin/python'), str(EVIDENCE / 'offline_ab.py'),
                               '--label', name, '--lengths', str(length), '--num-prompts',
                               '8192' if length == 512 else '65536', '--repeats', '5',
                               '--output', str(directory / f'{name}.json')]
                    if round_id == 1 and length == 512:
                        command += ['--profile-dir', str(directory / f'{name}-traces')]
                    manifest = {'command': command,
                                'methods_sha256': hashlib.sha256(METHODS.read_bytes()).hexdigest(),
                                'kernel_sha256': hashlib.sha256((METHODS.parent / 'mean_pool.py').read_bytes()).hexdigest()}
                    (directory / f'{name}-manifest.json').write_text(json.dumps(manifest, indent=2))
                    with (directory / f'{name}.log').open('w') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=1200)
                    result = json.loads((directory / f'{name}.json').read_text())
                    print(json.dumps({'run': name, 'embeddings_per_second': result['median_embeddings_per_second'],
                                      'seconds': [x['elapsed_seconds'] for x in result['runs']]}), flush=True)
    finally:
        METHODS.write_bytes((EVIDENCE / 'methods-patched.py').read_bytes())


if __name__ == '__main__':
    main()
