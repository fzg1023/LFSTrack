from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.single_stream.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/single_stream/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path (epoch=None 时取配置中的 TEST.EPOCH)
    if epoch is None:
        epoch = int(getattr(cfg.TEST, 'EPOCH', 50))
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/single_stream/%s/BATrack_ep%04d.pth.tar" % (yaml_name, epoch))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
