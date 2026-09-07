#!/usr/bin/env python3
"""
用官方 RGBT_toolkit (rgbttoolkit v1.0.1, mmic-lcl) 评测本项目已保存的原始结果 txt。

输入: 测试已保存的结果目录（每序列一个 <seq_name>.txt，4 列 x,y,w,h，
      逗号/空格分隔 —— 官方 load_text 均支持）。
输出: 官方口径最终指标:
      lasher  : PR@20 / NPR / SR
      gtot    : MPR(<5px, 帧加权) / MSR(帧加权 AUC)
      rgbt234 : MPR(≤20px) / MSR
      rgbt210 : PR@20 / SR

用法:
  # 单个结果目录
  python RGBT_workspace/eval_official_rgbt_toolkit.py \
      --dataset gtot \
      --results_dir RGBT_workspace/results/gtot/single_stream_xxx_BATrack_ep0050.pth

  # 一次评测某数据集下的所有结果目录（按目录名注册 tracker）
  python RGBT_workspace/eval_official_rgbt_toolkit.py --dataset lasher

  # 出图（PR/SR 曲线，保存到 results/<dataset>/official_plots/）
  python RGBT_workspace/eval_official_rgbt_toolkit.py --dataset gtot --plot
"""
from __future__ import annotations

import argparse
import os
import sys

_TOOLKIT_SRC_DEFAULT = '/home/fzg/rgbt-1.0.1/rgbt-1.0.1/src'

# 无头环境出图：必须在 import rgbt (其内部 import matplotlib) 之前设置
import matplotlib
matplotlib.use('Agg')


def main():
    p = argparse.ArgumentParser('官方 RGBT_toolkit 评测项目结果')
    p.add_argument('--dataset', required=True,
                   choices=['lasher', 'gtot', 'rgbt234', 'rgbt210'])
    p.add_argument('--results_dir', default=None, nargs='+',
                   help='单个或多个结果目录；不传则扫描 --results_root 下所有子目录')
    p.add_argument('--results_root', default=None,
                   help='默认 RGBT_workspace/results/<dataset>')
    p.add_argument('--toolkit_path', default=_TOOLKIT_SRC_DEFAULT,
                   help='官方评价包 src 目录')
    p.add_argument('--plot', action='store_true', help='保存 PR/SR 曲线图')
    args = p.parse_args()

    sys.path.insert(0, args.toolkit_path)
    from rgbt import GTOT, LasHeR, RGBT210, RGBT234

    prj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_root = args.results_root or os.path.join(
        prj, 'RGBT_workspace', 'results', args.dataset)

    if args.results_dir:
        dirs = {os.path.basename(os.path.abspath(d)): os.path.abspath(d)
                for d in args.results_dir}
    else:
        if not os.path.isdir(results_root):
            raise FileNotFoundError(f'结果根目录不存在: {results_root}')
        dirs = {
            d: os.path.join(results_root, d)
            for d in sorted(os.listdir(results_root))
            if os.path.isdir(os.path.join(results_root, d))
        }

    # ── 构造官方数据集对象（GT 已内置于评价包，无需数据帧）──
    if args.dataset == 'lasher':
        ds = LasHeR()
    elif args.dataset == 'gtot':
        ds = GTOT()
    elif args.dataset == 'rgbt234':
        ds = RGBT234()
    else:
        ds = RGBT210()

    # ── 结果行对齐官方 GT 长度（官方 LasHeR PR 会按 GT 帧数直接索引）──
    import shutil
    import tempfile

    def _read_rows(path):
        rows = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip().replace(',', ' ').replace('\t', ' ')
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    rows.append([float(x) for x in parts[:4]])
                except ValueError:
                    continue
        return rows

    def _pad_to_gt(src_dir, dst_dir):
        """每个序列结果与官方 GT 长度对齐，不足时重复最后一帧；返回是否全部序列齐备。"""
        os.makedirs(dst_dir, exist_ok=True)
        for seq in ds.seqs_name:
            gt = ds[seq]
            if isinstance(gt, dict):
                gt = gt['visible']
            gtn = len(gt)
            rows = _read_rows(os.path.join(src_dir, f'{seq}.txt'))
            if not rows:
                return False
            if len(rows) < gtn:
                rows = rows + [rows[-1]] * (gtn - len(rows))
            else:
                rows = rows[:gtn]
            with open(os.path.join(dst_dir, f'{seq}.txt'), 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(','.join(f'{v:.4f}' for v in r) + '\n')
        return True

    tmp_root = tempfile.mkdtemp(prefix='official_eval_')

    # ── 注册每个结果目录为一个 tracker ──
    for name, d in dirs.items():
        if not any(f.endswith('.txt') for f in os.listdir(d)):
            print(f'[SKIP] {name}: 目录下无结果 txt')
            continue
        pad_dir = os.path.join(tmp_root, name)
        try:
            ok = _pad_to_gt(d, pad_dir)
            if not ok:
                print(f'[SKIP] {name}: 缺少部分序列结果 (共需 {len(ds.seqs_name)} 条)')
                continue
            ds(tracker_name=name, result_path=pad_dir, bbox_type='ltwh')
        except Exception as e:  # noqa: BLE001
            print(f'[SKIP] {name}: {e}')
    shutil.rmtree(tmp_root, ignore_errors=True)

    if not ds.trackers:
        print('[ERROR] 没有成功注册任何结果目录')
        sys.exit(1)

    # ── 官方指标 ──
    if args.dataset == 'lasher':
        pr = ds.PR(); npr = ds.NPR(); sr = ds.SR()
        rows = {k: [pr[k][0], npr[k][0], sr[k][0]] for k in pr}
        header = f"{'tracker':60s} {'PR@20':>8} {'NPR':>8} {'SR':>8}"
        def fmt(v): return [f'{x * 100:8.2f}' for x in v]
        sort_key = lambda kv: kv[1][0]
    elif args.dataset == 'gtot':
        mpr = ds.MPR(); msr = ds.MSR()
        rows = {k: [mpr[k][0], msr[k][0]] for k in mpr}
        header = f"{'tracker':60s} {'MPR(<5px)':>10} {'MSR':>8}"
        def fmt(v): return [f'{v[0] * 100:10.2f}', f'{v[1] * 100:8.2f}']
        sort_key = lambda kv: kv[1][0]
    elif args.dataset == 'rgbt234':
        mpr = ds.MPR(); msr = ds.MSR()
        rows = {k: [mpr[k][0], msr[k][0]] for k in mpr}
        header = f"{'tracker':60s} {'MPR(≤20px)':>11} {'MSR':>8}"
        def fmt(v): return [f'{v[0] * 100:11.2f}', f'{v[1] * 100:8.2f}']
        sort_key = lambda kv: kv[1][0]
    else:  # rgbt210
        pr = ds.PR(); sr = ds.SR()
        rows = {k: [pr[k][0], sr[k][0]] for k in pr}
        header = f"{'tracker':60s} {'PR@20':>8} {'SR':>8}"
        def fmt(v): return [f'{x * 100:8.2f}' for x in v]
        sort_key = lambda kv: kv[1][0]

    print('=' * 84)
    print(f'官方 RGBT_toolkit v1.0.1 评测 | dataset={args.dataset} | 结果目录: {results_root}')
    print('=' * 84)
    print(header)
    for k, v in sorted(rows.items(), key=sort_key, reverse=True):
        print(f'{k:60s} ' + ' '.join(fmt(v)))
    print('=' * 84)
    print(f'[RESULT] ' + ' | '.join(
        f'{k}: {",".join(f"{x:.4f}" for x in v)}' for k, v in sorted(rows.items(), key=sort_key, reverse=True)))

    # ── 出图 ──
    if args.plot:
        plot_dir = os.path.join(results_root, 'official_plots')
        os.makedirs(plot_dir, exist_ok=True)
        os.chdir(plot_dir)
        if args.dataset == 'lasher':
            ds.draw_plot(metric_fun=ds.PR); ds.draw_plot(metric_fun=ds.NPR); ds.draw_plot(metric_fun=ds.SR)
        elif args.dataset in ('gtot', 'rgbt234'):
            ds.draw_plot(metric_fun=ds.MPR); ds.draw_plot(metric_fun=ds.MSR)
        else:
            ds.draw_plot(metric_fun=ds.PR); ds.draw_plot(metric_fun=ds.SR)
        print(f'[INFO] 曲线图已保存到 {plot_dir}')


if __name__ == '__main__':
    main()
