#!/bin/bash
# LFSTrack - LasHeR training from scratch (4ch: RGB 3ch + TIR 1ch)
#
# Usage:
#   bash train_lfstrack_lasher_4ch.sh
#   GPU_IDS=2 bash train_lfstrack_lasher_4ch.sh            # pin to the only allowed GPU
#   CUDA_VISIBLE_DEVICES=1 bash train_lfstrack_lasher_4ch.sh

set -e
cd "$(dirname "$0")"

# GPU selection (priority: CUDA_VISIBLE_DEVICES > GPU_IDS > auto-select freest GPU)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${GPU_IDS:-}" ]; then
        CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    else
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | sort -t',' -k2 -n | head -1 | cut -d',' -f1)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    fi
fi
export CUDA_VISIBLE_DEVICES
echo "Training on GPU: ${CUDA_VISIBLE_DEVICES}"

python train.py \
    --script single_stream \
    --config rgbt_lfn_multilayer_gate_scratch_lasher \
    --save_dir ./output \
    --mode single
