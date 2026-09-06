#!/bin/bash
# LFSTrack - GTOT test (6ch config)
#
# Usage:
#   bash test_lfstrack_gtot_6ch.sh                  # automatically pick the latest checkpoint
#   bash test_lfstrack_gtot_6ch.sh 30               # specify epoch
#   bash test_lfstrack_gtot_6ch.sh /path/to/ckpt    # specify checkpoint path
#   bash test_lfstrack_gtot_6ch.sh latest 2         # second argument: number of workers
#
# Data path: env var RGBT_DATA_ROOT (default /root/RGBTData, contains GTOT/)

set -e
cd "$(dirname "$0")"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="rgbt_lfn_multilayer_gate_scratch_6ch_lasher"
DATASET="gtot"
CKPT_DIR="${PROJECT_DIR}/output/checkpoints/train/single_stream/${CONFIG}"

ARG="${1:-latest}"
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
echo "  LFSTrack GTOT (6ch) Evaluation"
echo "  Config:     ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Data root:  ${RGBT_DATA_ROOT:-/root/RGBTData}"
echo "============================================"

python test.py \
    --script single_stream \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --workers "${2:-1}"
