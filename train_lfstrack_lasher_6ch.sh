#!/bin/bash
# LFSTrack - LasHeR training from scratch (6ch: RGB 3ch + TIR 3ch)
#
# Usage:
#   bash train_lfstrack_lasher_6ch.sh
#   CUDA_VISIBLE_DEVICES=1 bash train_lfstrack_lasher_6ch.sh

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train.py \
    --script single_stream \
    --config rgbt_lfn_multilayer_gate_scratch_6ch_lasher \
    --save_dir ./output \
    --mode single
