#!/bin/bash
# LFSTrack - VTUAV 6ch test (default vtuav_st)
#
# Usage:
#   bash test_lfstrack_vtuav_6ch.sh                      # vtuav_st, latest checkpoint
#   bash test_lfstrack_vtuav_6ch.sh vtuav_lt             # vtuav_lt
#   bash test_lfstrack_vtuav_6ch.sh vtuav_st 30          # specify epoch
#   bash test_lfstrack_vtuav_6ch.sh vtuav_st /path/ckpt  # specify checkpoint path

set -e
cd "$(dirname "$0")"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="rgbt_lfn_multilayer_gate_scratch_6ch_vtuav"
CKPT_DIR="${PROJECT_DIR}/output/checkpoints/train/single_stream/${CONFIG}"

DATASET="${1:-vtuav_st}"
ARG="${2:-latest}"
if [ -f "$ARG" ]; then
    CHECKPOINT="$ARG"
elif [ "$ARG" = "latest" ]; then
    CHECKPOINT=$(ls -t "$CKPT_DIR"/BATrack_ep*.pth.tar 2>/dev/null | head -1)
else
    CHECKPOINT="${CKPT_DIR}/BATrack_ep$(printf '%04d' "${ARG}").pth.tar"
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found in $CKPT_DIR"
    exit 1
fi

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
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

echo "============================================"
echo "  LFSTrack VTUAV 6ch Evaluation"
echo "  Config:     ${CONFIG}"
echo "  Dataset:    ${DATASET}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "============================================"

python test.py \
    --script single_stream \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --workers "${3:-1}"
