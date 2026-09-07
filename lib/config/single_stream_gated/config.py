from easydict import EasyDict as edict
import yaml

"""
Config for Single-Stream + Uncertainty Gating (4-channel RGBT input).
Layers 9-11 use GatedSSBlock with per-channel uncertainty gating.
"""

cfg = edict()

cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_single_stream_gated"
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
cfg.MODEL.BACKBONE.GATE_LAYERS = [9, 10, 11]

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

cfg.TRAIN = edict()
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = 'single_stream_gated'
cfg.TRAIN.LR = 0.00005
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.NESTEROV = False
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.FIX_BN = True
cfg.TRAIN.TEST_EVERY_EPOCH = False
cfg.TRAIN.SAVE_EPOCH_INTERVAL = 1
cfg.TRAIN.SAVE_LAST_N_EPOCH = 1
cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80
cfg.TRAIN.DROP_PATH_RATE = 0.2
cfg.TRAIN.USE_EMA = False
cfg.TRAIN.EMA_DECAY = 0.9998

cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = []
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

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
