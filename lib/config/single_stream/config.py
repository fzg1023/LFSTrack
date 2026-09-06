from easydict import EasyDict as edict
import yaml

"""
Config for Single-Stream BAT (4-channel RGBT input).
"""

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = ""
cfg.MODEL.RESUME_CKPT = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []
cfg.MODEL.IN_CHANS = 4  # 输入通道数: 4 = RGB(3) + TIR(1); 6ch 实验在 yaml 显式写 IN_CHANS: 6

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_single_stream"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'
cfg.MODEL.BACKBONE.GATE_LAYERS = []   # e.g. [9, 10, 11]; empty = all standard SSBlocks
# ViPT/AdaptFormer-style parallel bottleneck adapters, for frozen-backbone tuning.
cfg.MODEL.BACKBONE.ADAPTER_LAYERS = []   # e.g. list(range(12)); empty = no adapters
cfg.MODEL.BACKBONE.ADAPTER_DIM = 64      # bottleneck dimension of each Adapter

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.HEAD.COND_INIT_DECAY = 2.0   # Conditional Center Head: initial decay of ConditionalGate
# LFN configuration
cfg.MODEL.USE_LFN = True
cfg.MODEL.LFN_NUM_ITERATIONS = 2
cfg.MODEL.LFN_USE_ATTN_POOL = True
cfg.MODEL.LFN_USE_CROSS_ATTN = False
cfg.MODEL.LFN_USE_MULTISCALE_GATE = False
# Multi-Layer LFN: additional LFN branch reading an intermediate backbone layer,
# fused into the final (layer-12) LFN output via a zero-init gate. Off by default —
# does not affect any existing single_stream experiment.
cfg.MODEL.LFN_MULTILAYER = False
cfg.MODEL.LFN_MID_LAYER = 7   # 0-indexed block index; 7 == output of the 8th transformer layer
cfg.MODEL.LFN_3LAYER = False  # enable 3-layer LFN (early + mid + final)
cfg.MODEL.LFN_EARLY_LAYER = 3 # 0-indexed block index; 3 == output of the 4th transformer layer
                               # Only used when 3-layer LFN is configured in the YAML.
# Modality Prompt (ViPT-style): learnable prompt tokens prepended to the
# single-stream token sequence, giving the network an explicit modality prior
# while keeping the one-stream backbone. Off by default.
cfg.MODEL.USE_MODALITY_PROMPT = False
cfg.MODEL.MODALITY_PROMPT_NUM = 2   # prompt token count (e.g. 2: RGB + TIR)

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = 'single_stream'
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.NESTEROV = False
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.LFN_MULTIPLIER = 1.0   # 1.0=full LR, 0.5=half LR for LFN only
cfg.TRAIN.ADAPTER_MULTIPLIER = 1.0   # LR multiplier for Adapter params (see ADAPTER_LAYERS)
cfg.TRAIN.ADAPTER_TUNING = False     # True: requires_grad=False on everything except adapters+box_head
cfg.TRAIN.HEAD_ONLY_TUNING = False   # True: requires_grad=False on everything except box_head
cfg.TRAIN.PROMPT_TUNING = False      # True: requires_grad=False on everything except box_head + modality prompts
cfg.TRAIN.PROMPT_MULTIPLIER = 1.0    # LR multiplier for modality prompt params
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.FIX_BN = True
cfg.TRAIN.TEST_EVERY_EPOCH = False
cfg.TRAIN.MODAL_DROPOUT_RATE = 0.0   # 0.0=off, 0.2=CADTrack-style modality dropout
cfg.TRAIN.SAVE_EPOCH_INTERVAL = 1
cfg.TRAIN.SAVE_LAST_N_EPOCH = 1

cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80
cfg.TRAIN.DROP_PATH_RATE = 0.1
cfg.TRAIN.USE_EMA = False
cfg.TRAIN.EMA_DECAY = 0.9998
cfg.TRAIN.MAMBA_TEMPORAL_LAYERS = 2
cfg.TRAIN.MAMBA_TEMPORAL_KERNEL = 4
cfg.TRAIN.MAMBA_TEMPORAL_STATE = 16
cfg.TRAIN.MAMBA_TEMPORAL_EXPAND = 2

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.TEST_DATASET = "lasher"  # 每轮自动测试的数据集 (TEST_EVERY_EPOCH 时使用)
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = []
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000

# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 30


def update_config_from_file(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in cfg[k]:
                            cfg[k][vk] = vv
                        else:
                            raise ValueError("Key {} not exist in config.py".format(vk))
                else:
                    cfg[k] = v
            else:
                raise ValueError("Key {} not exist in config.py".format(k))
