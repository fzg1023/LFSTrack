#!/usr/bin/env python3
"""
离线指标计算：直接读取本项目测试已保存的原始结果 txt，计算最终指标。
无需重新加载模型/跑追踪，只需结果文件与数据集 GT。

输入:
  --results_dir : 结果目录（内含每个序列一个 <seq_name>.txt，
                  每行 4 列 x,y,w,h，逗号或空格分隔），
                  即 test_uatrack.py 的 save_dir/<dataset>/<save_name>/ 目录
  --dataset     : lasher | vtuav_st | vtuav_lt | gtot | rgbt234 | rgbt210
  --data_root   : 可选，覆盖数据集根目录（默认用 DATASET_CFG / RGBT_DATA_ROOT）

输出:
  控制台逐序列表 + 汇总（序列均值 / 帧加权）
  结果目录下 per_seq_metrics_offline.csv 与 summary_offline.json

口径:
  与 RGBT_workspace/test_uatrack.py 的 _compute_metrics 完全一致
  （已对齐官方 VTUAV: PS=min(CLE_v,CLE_i)<=20px, SR/MSR=max(IoU_v,IoU_i) 曲线均值，
    双模态 GT 时 SR==MSR、PS==MPR；官方汇总为序列等权，即 seq_* 列）。

用法示例:
  python RGBT_workspace/eval_results_txt.py \
      --results_dir RGBT_workspace/results/vtuav_st/single_stream_rgbt_xxx_BATrack_ep0004 \
      --dataset vtuav_st
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)

from RGBT_workspace.test_uatrack import DATASET_CFG, _METRICS, _compute_metrics, _read_gt


def _read_pred(path: str):
    """读取项目结果 txt：每行 4 列 x,y,w,h（逗号/空格分隔）。"""
    preds = []
    with open(path, 'rb') as fb:
        raw = fb.read()
    if b'\x00' in raw:
        return []
    for line in raw.decode('utf-8', errors='replace').splitlines():
        line = line.strip().replace(',', ' ').replace('\t', ' ')
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            preds.append([float(v) for v in parts[:4]])
        except ValueError:
            continue
    return preds


def main():
    p = argparse.ArgumentParser('从结果 txt 离线计算指标')
    p.add_argument('--results_dir', required=True,
                   help='结果目录（内含 <seq_name>.txt）')
    p.add_argument('--dataset', default='lasher', choices=list(DATASET_CFG))
    p.add_argument('--data_root', default=None,
                   help='覆盖数据集根目录（默认使用 DATASET_CFG 中的路径）')
    p.add_argument('--out_prefix', default='offline',
                   help='输出文件名前缀（默认 offline，避免覆盖测试时生成的 CSV）')
    args = p.parse_args()

    dc = dict(DATASET_CFG[args.dataset])
    if args.data_root:
        dc['root'] = args.data_root
    root = dc['root']
    if not os.path.isdir(root):
        raise FileNotFoundError(f'数据集目录不存在: {root}')

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f'结果目录不存在: {results_dir}')

    seq_dirs = sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    recs = []
    n_missing = 0
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        pred_path = os.path.join(results_dir, f'{seq_name}.txt')
        if not os.path.isfile(pred_path):
            print(f'[SKIP] {seq_name:<32s} 结果文件缺失: {pred_path}')
            n_missing += 1
            continue
        try:
            preds = _read_pred(pred_path)
            gt_all = _read_gt(os.path.join(seq_dir, dc.get('gt_all', dc['gt'])), args.dataset)
            if not gt_all:
                gt_all = _read_gt(os.path.join(seq_dir, dc['gt']), args.dataset)
            gt_i_all = []
            if dc.get('gt_i'):
                gt_i_all = _read_gt(os.path.join(seq_dir, dc['gt_i']), args.dataset)

            n = min(len(preds), len(gt_all))
            if gt_i_all:
                n = min(n, len(gt_i_all))
            if n < 1:
                raise ValueError('no valid frames')

            m = _compute_metrics(
                preds[:n], gt_all[:n],
                gt_i_all[:n] if gt_i_all else None, args.dataset)
            m.update(seq_name=seq_name)
            recs.append(m)
        except Exception as e:  # noqa: BLE001
            print(f'[ERROR] {seq_name:<32s} {e}')

    if not recs:
        print('[ERROR] 没有计算出任何序列指标，请检查 --results_dir 与 --dataset')
        sys.exit(1)

    # ── 汇总（与 test_uatrack.py 相同的序列均值 / 帧加权口径）──
    total_frames = sum(r['n_valid'] for r in recs)
    seq_means = {k: float(np.mean([r[k] for r in recs])) for k in _METRICS}
    frm_means = {
        k: float(sum(r[k] * r['n_valid'] for r in recs) / total_frames)
        if total_frames > 0 else -1.
        for k in _METRICS
    }

    W = 9
    def _fmt(v):
        return '---' if v is None or v < 0 else f'{v * 100:>{W}.2f}'
    print('=' * 78)
    print(f'[汇总] dataset={args.dataset}  序列 {len(recs)} 条'
          f'（缺失 {n_missing} 条） 总帧数={total_frames}')
    print(f"{'':14}" + ''.join(f"{k:>{W}}" for k in _METRICS))
    print(f"{'序列均值(%)':<14}" + ''.join(_fmt(seq_means[k]) for k in _METRICS))
    print(f"{'帧加权(%)' :<14}" + ''.join(_fmt(frm_means[k]) for k in _METRICS))
    print('=' * 78)
    print(f'[RESULT] ' + ' '.join(f"{k}={seq_means[k]:.4f}" for k in _METRICS), flush=True)
    if args.dataset in ('gtot', 'rgbt234', 'vtuav_st', 'vtuav_lt'):
        print('[注] 官方口径为序列等权（seq_*）；SR==MSR、PS==MPR 对齐官方 VTUAV/rgbttoolkit。')

    # ── per_seq CSV ──
    csv_path = os.path.join(results_dir, f'per_seq_metrics_{args.out_prefix}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['seq_name'] + _METRICS + ['n_valid'])
        w.writeheader()
        for r in recs:
            w.writerow({**{'seq_name': r['seq_name'], 'n_valid': r['n_valid']},
                        **{k: f'{r[k]:.6f}' for k in _METRICS}})
    print(f'[INFO] per_seq_metrics_{args.out_prefix}.csv -> {csv_path}')

    # ── summary JSON ──
    summary = dict(
        results_dir=results_dir, dataset=args.dataset,
        n_sequences=len(recs), n_missing=n_missing, total_frames=total_frames,
        seq_means=seq_means, frm_means=frm_means,
    )
    summary_path = os.path.join(results_dir, f'summary_{args.out_prefix}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[INFO] summary_{args.out_prefix}.json       -> {summary_path}')


if __name__ == '__main__':
    main()
