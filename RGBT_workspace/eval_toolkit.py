#!/usr/bin/env python3
"""
用官方 rgbttoolkit (rgbt-1.0.1) 评测 test_uatrack.py 产生的预测结果。

结果目录要求: <result_path>/<seq_name>.txt，每行 4 列 x,y,w,h（ltwh）。
test_uatrack.py 的 _save_pred 输出即满足此格式:
    RGBT_workspace/results/<dataset>/<save_name>/<seq_name>.txt

用法:
  python RGBT_workspace/eval_toolkit.py \
      --dataset gtot \
      --result_path RGBT_workspace/results/gtot/single_stream_rgbt_xxx_BATrack_ep0030.pth \
      --name my_tracker

  python RGBT_workspace/eval_toolkit.py --dataset rgbt210 --result_path <dir> --name my_tracker
  python RGBT_workspace/eval_toolkit.py --dataset rgbt234 --result_path <dir> --name my_tracker

GTOT 输出: MPR / MSR (5px / max IoU AUC, 帧加权)
RGBT210 输出: PR / SR (20px / SR AUC, 序列均值)
RGBT234 输出: MPR / MSR (20px / max IoU AUC, 序列均值)
"""
import argparse
import sys

_TOOLKIT_SRC = '/home/fzg/rgbt-1.0.1/rgbt-1.0.1/src'
if _TOOLKIT_SRC not in sys.path:
    sys.path.insert(0, _TOOLKIT_SRC)

from rgbt import GTOT, RGBT210, RGBT234
from rgbt.utils import RGBT_start, RGBT_end


def main():
    p = argparse.ArgumentParser('rgbt toolkit 评测')
    p.add_argument('--dataset', required=True, choices=['gtot', 'rgbt210', 'rgbt234'])
    p.add_argument('--result_path', required=True, help='每序列一个 <seq>.txt 的目录')
    p.add_argument('--name', default='LFSTrack', help='tracker 显示名')
    p.add_argument('--prefix', default='', help='结果文件名前缀 (默认无)')
    p.add_argument('--bbox_type', default='ltwh', help='结果框格式 (默认 ltwh)')
    p.add_argument('--plot', action='store_true', help='绘制曲线图')
    args = p.parse_args()

    if args.dataset == 'gtot':
        RGBT_start()
        ds = GTOT()
        pr_name, sr_name = 'MPR', 'MSR'
    elif args.dataset == 'rgbt210':
        ds = RGBT210()
        pr_name, sr_name = 'PR', 'SR'
    else:
        ds = RGBT234()
        pr_name, sr_name = 'MPR', 'MSR'

    ds(tracker_name=args.name, result_path=args.result_path,
       bbox_type=args.bbox_type, prefix=args.prefix)

    pr_val, _ = getattr(ds, pr_name)(args.name)
    sr_val, _ = getattr(ds, sr_name)(args.name)
    print(f'\n===== {args.dataset} 官方指标 ({args.name}) =====')
    print(f'{pr_name}: {pr_val:.4f}  ({pr_val*100:.2f}%)')
    print(f'{sr_name}: {sr_val:.4f}  ({sr_val*100:.2f}%)')

    if args.plot:
        getattr(ds, 'draw_plot')(metric_fun=getattr(ds, pr_name))
        getattr(ds, 'draw_plot')(metric_fun=getattr(ds, sr_name))

    if args.dataset == 'gtot':
        RGBT_end()


if __name__ == '__main__':
    main()
