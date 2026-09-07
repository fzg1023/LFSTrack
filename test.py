"""
LFSTrack — Evaluation entry point (single-stream).
"""
import os
import sys
import subprocess
import argparse

_PRJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)


def main():
    parser = argparse.ArgumentParser(description='LFSTrack Evaluation')
    parser.add_argument('--script', type=str, default='single_stream_gated')
    parser.add_argument('--config', type=str, default='rgbt')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='lasher')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    cmd = [
        'python', 'RGBT_workspace/test_uatrack.py',
        '--script', args.script,
        '--config', args.config,
        '--checkpoint', args.checkpoint,
        '--dataset', args.dataset,
        '--workers', str(args.workers),
    ]
    print(f'[LFSTrack] Running: {" ".join(cmd)}')
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
