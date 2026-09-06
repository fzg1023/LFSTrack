import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import VisEvent, LasHeR, DepthTrack,RGBT234,LasHeR_testingSet,LasHeR_trainingSet
from lib.train.dataset import VTUAV
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val", "COCO17", "VID", "TRACKINGNET",
                        "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train", "LasHeR_val", "VisEvent","RGBT210",
                        "VTUAV", "VTUAVST", "VTUAVLT"]
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='train'))
        if name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='val'))
        if name == "LasHeR_all":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='all'))
        if name == "LasHeR_train":
            datasets.append(LasHeR_trainingSet(settings.env.lasher_dir))
        if name == "LasHeR_val":
            test_dir = getattr(settings.env, 'lasher_test_dir', settings.env.lasher_dir)
            datasets.append(LasHeR_testingSet(test_dir))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, dtype='rgbrgb', split='train'))
        if name == "VTUAV":
            datasets.append(VTUAV(settings.env.vtuav_dir, split='train', modality='RGBT'))
        if name == "VTUAVST":
            datasets.append(VTUAV(settings.env.vtuav_dir, split='val_st', modality='RGBT'))
        if name == "VTUAVLT":
            datasets.append(VTUAV(settings.env.vtuav_dir, split='val_lt', modality='RGBT'))
        """Following is rgb dataset"""
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.BATProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.BATProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")
    if 'single_stream' in train_type:
        # Single-stream: adapter/LFN at their own LR, box_head at full LR, backbone at lower LR
        lfn_mult = getattr(cfg.TRAIN, 'LFN_MULTIPLIER', 1.0)
        adapter_mult = getattr(cfg.TRAIN, 'ADAPTER_MULTIPLIER', 1.0)
        prompt_mult = getattr(cfg.TRAIN, 'PROMPT_MULTIPLIER', 1.0)

        def is_head(n):
            return "box_head" in n

        def is_adapter(n):
            return "adapter_attn" in n or "adapter_mlp" in n

        def is_prompt(n):
            return "modality_prompt" in n

        def is_lfn(n):
            return ("lfn" in n or ".gate." in n or "gated_ffn" in n or "norm_g" in n
                    or "temporal_fusion" in n or "temporal_gate" in n)

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if is_head(n) and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters()
                          if is_prompt(n) and not is_head(n) and p.requires_grad],
                "lr": cfg.TRAIN.LR * prompt_mult,
            },
            {
                "params": [p for n, p in net.named_parameters()
                          if is_adapter(n) and not is_head(n) and not is_prompt(n) and p.requires_grad],
                "lr": cfg.TRAIN.LR * adapter_mult,
            },
            {
                "params": [p for n, p in net.named_parameters()
                          if is_lfn(n) and not is_head(n) and not is_adapter(n) and not is_prompt(n) and p.requires_grad],
                "lr": cfg.TRAIN.LR * lfn_mult,
            },
            {
                "params": [p for n, p in net.named_parameters()
                          if not is_head(n) and not is_lfn(n) and not is_adapter(n) and not is_prompt(n) and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        # AdamW/SGD raise on an empty param group (e.g. no adapters in this experiment) — drop those.
        param_dicts = [pd for pd in param_dicts if len(pd["params"]) > 0]
        if is_main_process():
            total = sum(p.numel() for p in net.parameters())
            trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
            lfn_params = sum(p.numel() for n, p in net.named_parameters() if is_lfn(n) and not is_head(n))
            adapter_params = sum(p.numel() for n, p in net.named_parameters() if is_adapter(n))
            prompt_params = sum(p.numel() for n, p in net.named_parameters() if is_prompt(n))
            print(f'[LFSTrack] Total: {total:,}  Trainable: {trainable:,}  LFN: {lfn_params:,} (lr={cfg.TRAIN.LR * lfn_mult:.2e})  '
                  f'Adapter: {adapter_params:,} (lr={cfg.TRAIN.LR * adapter_mult:.2e})  '
                  f'Prompt: {prompt_params:,} (lr={cfg.TRAIN.LR * prompt_mult:.2e})')
    else:
        # Default: head at full LR, backbone at lower LR
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if ("head" in n or "gated_ffn" in n or "norm_g" in n) and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "head" not in n and "gated_ffn" not in n and "norm_g" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "SGD":
        momentum = getattr(cfg.TRAIN, 'MOMENTUM', 0.9)
        nesterov = getattr(cfg.TRAIN, 'NESTEROV', False)
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR,
                                    momentum=momentum,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                    nesterov=nesterov)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == 'cosine':
        t_max = getattr(cfg.TRAIN.SCHEDULER, 'T_MAX', cfg.TRAIN.EPOCH)
        eta_min = getattr(cfg.TRAIN.SCHEDULER, 'ETA_MIN', 0.0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
