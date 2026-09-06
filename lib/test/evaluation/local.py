import os
from lib.test.evaluation.environment import EnvSettings

# 项目根目录: 自动从本文件位置推导 (local.py 位于 <根>/lib/test/evaluation/)
# 也可通过环境变量 LFSTRACK_HOME 覆盖
_PRJ_ROOT = os.environ.get('LFSTRACK_HOME') or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def local_env_settings():
    settings = EnvSettings()

    settings.prj_dir = _PRJ_ROOT
    settings.save_dir = os.path.join(_PRJ_ROOT, 'output')
    settings.network_path = os.path.join(_PRJ_ROOT, 'output/test/networks')
    settings.results_path = os.path.join(_PRJ_ROOT, 'output/test/tracking_results')
    settings.result_plot_path = os.path.join(_PRJ_ROOT, 'output/test/result_plots')
    settings.segmentation_path = os.path.join(_PRJ_ROOT, 'output/test/segmentation_results')
    settings.lasher_path = os.environ.get('LASHER_HOME', '/home/fzg/data/lasher/testingset')
    settings.vtuav_path = os.environ.get('VTUAV_HOME', '/home/fzg/data/vtuav')
    settings.rgbt234_path = os.environ.get('RGBT234_HOME', '')
    settings.otb_path = ''
    settings.nfs_path = ''
    settings.uav_path = ''
    settings.tpl_path = ''
    settings.vot_path = ''
    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.trackingnet_path = ''
    settings.davis_dir = ''
    settings.youtubevos_dir = ''

    return settings
