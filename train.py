"""
LFSTrack — Training entry point (single-stream).
"""
import os
import sys
import argparse
import random

_PRJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='LFSTrack Training')
    parser.add_argument('--script', type=str, default='single_stream_gated')
    parser.add_argument('--config', type=str, default='rgbt')
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'multiple'])
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--use_lmdb', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'single':
        train_cmd = (
            f'python lib/train/run_training.py '
            f'--script {args.script} --config {args.config} '
            f'--save_dir {args.save_dir} --use_lmdb {args.use_lmdb} '
            f'--use_wandb {args.use_wandb}'
        )
    elif args.mode == 'multiple':
        train_cmd = (
            f'python -m torch.distributed.launch '
            f'--nproc_per_node {args.nproc_per_node} '
            f'--master_port {random.randint(10000, 50000)} '
            f'lib/train/run_training.py '
            f'--script {args.script} --config {args.config} '
            f'--save_dir {args.save_dir} --use_lmdb {args.use_lmdb} '
            f'--use_wandb {args.use_wandb}'
        )
    else:
        raise ValueError("mode should be 'single' or 'multiple'")

    print(f'[LFSTrack] Running: {train_cmd}')
    os.system(train_cmd)


if __name__ == '__main__':
    main()
