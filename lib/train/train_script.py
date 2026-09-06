import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related — single-stream variants only
from lib.models.bat.ostrack_adapter import build_single_stream_track, build_single_stream_gated_track, build_single_stream_3frame_track, build_single_stream_mamba_temporal_track
# forward propagation related
from lib.train.actors import BATActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'LFSTrack training'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network — detect temporal/3frame variants
    prompt_type = getattr(cfg.TRAIN.PROMPT, 'TYPE', '')
    if prompt_type == 'single_stream_3frame_temporal':
        net = build_single_stream_3frame_track(cfg)
    elif prompt_type == 'single_stream_mamba_temporal':
        pretrained_ckpt = getattr(cfg.MODEL, 'RESUME_CKPT', None)
        net = build_single_stream_mamba_temporal_track(cfg, training=True, pretrained_ckpt=pretrained_ckpt)
    elif settings.script_name == "single_stream":
        net = build_single_stream_track(cfg)
    elif settings.script_name == "single_stream_gated":
        net = build_single_stream_gated_track(cfg)
    else:
        raise ValueError("Unsupported script name: %s. Use single_stream or single_stream_gated." % settings.script_name)

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)

    # Loss functions and Actor — use BATActor3Frame if multi-frame (including temporal)
    num_search = getattr(cfg.DATA.SEARCH, 'NUMBER', 1)
    if num_search > 1:
        from lib.train.actors.bat_3frame import BATActor3Frame
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = BATActor3Frame(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = BATActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)
    settings.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)
    settings.use_ema = getattr(cfg.TRAIN, "USE_EMA", False)
    settings.ema_decay = getattr(cfg.TRAIN, "EMA_DECAY", 0.9998)

    # ── Optional: run full test at specified interval ──
    test_every = getattr(cfg.TRAIN, "TEST_EVERY_EPOCH", False)
    if test_every:
        settings.test_cmd = True
        settings.test_interval = test_every if isinstance(test_every, int) else 1
        settings.test_dataset = getattr(cfg.DATA, "TEST_DATASET", "lasher")

    if loader_val is None:
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    else:
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
