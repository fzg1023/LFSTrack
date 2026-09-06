#!/bin/bash
# LFSTrack - VTUAV training from scratch (4ch: RGB 3ch + TIR 1ch)
#
# Usage:
#   bash train_lfstrack_vtuav_4ch.sh
#   CUDA_VISIBLE_DEVICES=1 bash train_lfstrack_vtuav_4ch.sh
#
# Data path is set by the VTUAV_HOME environment variable
# (default /home/fzg/data/vtuav), see lib/train/admin/local.py.

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train.py \
    --script single_stream \
    --config rgbt_lfn_multilayer_gate_scratch_vtuav \
    --save_dir ./output \
    --mode single
