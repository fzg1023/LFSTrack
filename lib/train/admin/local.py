import os

# 项目根目录: 自动从本文件位置推导 (local.py 位于 <根>/lib/train/admin/)
# 也可通过环境变量 LFSTRACK_HOME 覆盖
_PRJ_ROOT = os.environ.get('LFSTRACK_HOME') or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = _PRJ_ROOT
        self.tensorboard_dir = os.path.join(_PRJ_ROOT, 'tensorboard')
        self.pretrained_networks = os.path.join(_PRJ_ROOT, 'pretrained_models')
        self.got10k_val_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_lmdb_dir = ''
        self.coco_dir = ''
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.depthtrack_dir = ''
        self.lasher_dir = os.environ.get('LASHER_TRAIN_HOME', '/root/RGBTData/LasHeR/train')
        self.lasher_test_dir = os.environ.get('LASHER_HOME', '/root/RGBTData/LasHeR/test')
        self.visevent_dir = ''
        self.vtuav_dir = os.environ.get('VTUAV_HOME', '/root/RGBTData/VTUAV')
