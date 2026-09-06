#!/bin/bash
# LFSTrack - VTUAV 4ch test (default vtuav_st)
#
# Usage:
#   bash test_lfstrack_vtuav_4ch.sh                      # vtuav_st, latest checkpoint
#   bash test_lfstrack_vtuav_4ch.sh vtuav_lt             # vtuav_lt
#   bash test_lfstrack_vtuav_4ch.sh vtuav_st 30          # specify epoch
#   bash test_lfstrack_vtuav_4ch.sh vtuav_st /path/ckpt  # specify checkpoint path

set -e
cd "$(dirname "$0")"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="rgbt_lfn_multilayer_gate_scratch_vtuav"
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo "  LFSTrack VTUAV 4ch Evaluation"
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
