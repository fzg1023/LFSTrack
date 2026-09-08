# LFSTrack: Four-Channel Single-Stream RGB-T Tracking with Template-Conditioned Modulation

LFSTrack is a single-stream RGB-T tracking framework: RGB (3ch) and TIR (1ch or 3ch)
are concatenated into 4/6 channels at the input layer, encoded by a unified ViT, and
enhanced by the **Layer-wise Fusion Network (LFN)** (dual branches at layers 8/12 with
template-conditioned channel-spatial gated modulation) before the Center Head regresses
the target box. The last three backbone layers use GatedSSBlock to suppress unstable
deep responses.

This repository is fully self-contained: it contains all training/testing code, 4
experiment configs, train/test scripts, the pretrained weights (DropTrack), and the
dependency list.

[Pretrained weights,modes and results](https://pan.baidu.com/s/1H7u0AHwlGXKvciUuxZcVZA?pwd=LFST)

## Directory Structure

```
LFSTrack/
├── README.md
├── requirements.txt
├── DropTrack_k700_800E_alldata.pth.tar      # pretrained weights (required)
├── train.py / test.py                        # training / testing entry points
├── train_lfstrack_*.sh / test_lfstrack_*.sh  # train/test scripts per config
├── experiments/single_stream/                # experiment configs (YAML)
│   ├── rgbt_lfn_multilayer_gate_scratch_lasher.yaml          # LasHeR 4ch
│   ├── rgbt_lfn_multilayer_gate_scratch_6ch_lasher.yaml      # LasHeR 6ch
│   ├── rgbt_lfn_multilayer_gate_scratch_vtuav.yaml           # VTUAV 4ch
│   └── rgbt_lfn_multilayer_gate_scratch_6ch_vtuav.yaml       # VTUAV 6ch
├── RGBT_workspace/test_uatrack.py            # evaluation runner (tracking + metrics)
├── lib/                                      # model / train / test / config code
└── docs/LFSTrack_ARCHITECTURE.md             # architecture docs
```

## Installation

```bash
conda create -n LFSTrack python=3.9 -y
conda activate LFSTrack
pip install -r requirements.txt
```

## Data Preparation

Data paths are configured via environment variables (defaults shown in
parentheses); you can also edit `lib/train/admin/local.py` (training side) and
`lib/test/evaluation/local.py` (testing side) directly:

| Environment variable | Description | Default |
|---|---|---|
| `LASHER_TRAIN_HOME` | LasHeR training set root | `/root/RGBTData/LasHeR/train` |
| `LASHER_HOME` | LasHeR test set root | `/root/RGBTData/LasHeR/test` |
| `VTUAV_HOME` | VTUAV root (contains train / test_ST / test_LT) | `/root/RGBTData/VTUAV` |
| `RGBT_DATA_ROOT` | RGBT evaluation data root (contains GTOT / RGBT210 / RGBT234 / VTUAV) | `/root/RGBTData` |
| `LFSTRACK_HOME` | Project root (auto-detected, usually not needed) | auto |

```bash
# Example
export LASHER_TRAIN_HOME=/root/RGBTData/LasHeR/train
export LASHER_HOME=/root/RGBTData/LasHeR/test
export VTUAV_HOME=/root/RGBTData/VTUAV
export RGBT_DATA_ROOT=/root/RGBTData
```

## Training

All four configs train from the DropTrack pretrained weights (checkpoint saved
every epoch, validation and automatic testing every epoch):

| Script | Config | Input | Dataset |
|---|---|---|---|
| `bash train_lfstrack_lasher_4ch.sh` | `rgbt_lfn_multilayer_gate_scratch_lasher` | RGB3+TIR1 | LasHeR (50 epochs) |
| `bash train_lfstrack_lasher_6ch.sh` | `rgbt_lfn_multilayer_gate_scratch_6ch_lasher` | RGB3+TIR3 | LasHeR (50 epochs) |
| `bash train_lfstrack_vtuav_4ch.sh` | `rgbt_lfn_multilayer_gate_scratch_vtuav` | RGB3+TIR1 | VTUAV (30 epochs) |
| `bash train_lfstrack_vtuav_6ch.sh` | `rgbt_lfn_multilayer_gate_scratch_6ch_vtuav` | RGB3+TIR3 | VTUAV (30 epochs) |

Checkpoints are saved to
`output/checkpoints/train/single_stream/<config>/BATrack_epXXXX.pth.tar`.

## Testing

`RGBT_workspace/test_uatrack.py` supports 6 datasets
(lasher / vtuav_st / vtuav_lt / gtot / rgbt210 / rgbt234) with multi-process
parallel tracking and metric computation. Dedicated scripts (all support
`[epoch|checkpoint_path] [workers]` arguments, defaulting to the latest
checkpoint):

| Script | Dataset | Config (4ch / 6ch) |
|---|---|---|
| `bash test_lfstrack_lasher_4ch.sh` / `test_lfstrack_lasher_6ch.sh` | LasHeR | `..._scratch_lasher` / `..._6ch_lasher` |
| `bash test_lfstrack_vtuav_4ch.sh` / `test_lfstrack_vtuav_6ch.sh` | vtuav_st / vtuav_lt | `..._scratch_vtuav` / `..._6ch_vtuav` |
| `bash test_lfstrack_gtot_4ch.sh` / `test_lfstrack_gtot_6ch.sh` | GTOT | `..._scratch_lasher` / `..._6ch_lasher` |
| `bash test_lfstrack_rgbt210_4ch.sh` / `test_lfstrack_rgbt210_6ch.sh` | RGBT210 | `..._scratch_lasher` / `..._6ch_lasher` |
| `bash test_lfstrack_rgbt234_4ch.sh` / `test_lfstrack_rgbt234_6ch.sh` | RGBT234 | `..._scratch_lasher` / `..._6ch_lasher` |

GTOT / RGBT210 / RGBT234 data live under `RGBT_DATA_ROOT` (default
`/root/RGBTData`) in the `GTOT/`, `RGBT210/`, `RGBT234/` directories.

Evaluation output (raw 4-column ltwh results + metrics):

```
RGBT_workspace/results/<dataset>/<script>_<config>_<ckpt_tag>/
├── <seq>.txt               # per-sequence 4-column ltwh raw results (compatible with the official toolkit)
├── per_seq_metrics.csv     # per-sequence metrics
├── summary.json            # dataset-level summary metrics (sequence mean / frame-weighted)
RGBT_workspace/results/<dataset>/eval_history.csv   # evaluation history
```

### Reproduce with the Official Toolkit

Raw results produced by testing can be re-evaluated with the official
`rgbt-1.0.1` toolkit:

```bash
python RGBT_workspace/eval_toolkit.py \
    --dataset gtot \
    --result_path RGBT_workspace/results/gtot/single_stream_rgbt_..._BATrack_ep0030.pth.tar \
    --name LFSTrack
# GTOT -> MPR/MSR; RGBT210 -> PR/SR; RGBT234 -> MPR/MSR

# Or recompute offline from existing txt files (without re-running the model):
python RGBT_workspace/eval_results_txt.py --results_dir <dir> --dataset rgbt234
```

## Metric Protocol

Aligned with the official VTUAV (calcPlotErr_TPR.m / plot_ST.m) and rgbttoolkit:

- For datasets with dual-modality GT (GTOT / RGBT234 / VTUAV), the better one is
  taken per frame: IoU takes `max(IoU_v, IoU_i)`, center error takes
  `min(CLE_v, CLE_i)`.
- Output metrics: AO, SR/MSR (Success curve AUC from simple averaging over 21
  thresholds), SR50, SR75, PS (CLE <= 20px), NPS (normalized <= 0.5),
  MPR (<5px for GTOT, <=20px otherwise).
- Under the official protocol, PR@20 == MPR and SR AUC == MSR; results are
  aggregated with equal sequence weights.
- Datasets without second-modality GT (LasHeR / RGBT210) degrade to the RGB
  single-modality protocol.

## Key Hyperparameters

| Config | SEARCH/TEMPLATE | BATCH | LR | EPOCH | DROP_PATH | WEIGHT_DECAY |
|---|---|---|---|---|---|---|
| LasHeR 4ch/6ch | 384 / 192 | 8 | 1e-4 (bb x0.1) | 50 | 0.2 | 5e-4 |
| VTUAV 4ch | 256 / 128 | 8 | 1e-4 (bb x0.1) | 30 | 0.2 | 5e-4 |
| VTUAV 6ch | 384 / 192 | 8 | 1e-4 (bb x0.1) | 30 | 0.2 | 5e-4 |

LR schedule: step (x0.1 every 10 epochs); backbone LR multiplier 0.1; FIX_BN on.

## FAQ

- **GPU selection**: train/test scripts auto-select the GPU with the most free
  memory. On shared servers that only allow a specific GPU, pin it with
  `GPU_IDS=<id>` (e.g. `GPU_IDS=2 bash train_lfstrack_lasher_4ch.sh`), or set
  `CUDA_VISIBLE_DEVICES=<id>` directly (takes precedence).
- **Test workers**: evaluation is single-process by default; the second argument
  of `test_lfstrack_*.sh` sets the number of workers (each worker loads its own
  model copy; workers are assigned to GPUs automatically when multiple GPUs are
  available).
- **Checkpoint loading**: the model loader is compatible with old weights using
  early naming (`liquid_fusion* -> lfn*`), mapped automatically by
  `lib/models/bat/ostrack_adapter.py::remap_legacy_keys`.
