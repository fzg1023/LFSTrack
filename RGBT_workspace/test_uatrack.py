#!/usr/bin/env python3
"""
LFSTrack — LasHeR / VTUAV / GTOT / RGBT210 / RGBT234 多进程并行测试
====================================================================
特性（参考 CADTrack 测试逻辑）：
  • N 个独立 spawn 子进程，每个加载自己的模型副本，串行处理分配到的序列子集
  • 序列按帧数升序后 round-robin 分配，各 worker 负载均衡
  • 子进程通过 multiprocessing.Queue 即时返回结果
  • 主进程实时打印带累计指标的进度表格
  • 保存 per_seq_metrics.csv（本次每条序列详情）
  • 追加写入 eval_history.csv（多 epoch 汇总对比）
  • 保存 summary.json

用法：
  python RGBT_workspace/test_uatrack.py \
      --script single_stream \
      --config rgbt_lfn_multilayer_gate_scratch_lasher \
      --checkpoint output/checkpoints/train/single_stream/<config>/BATrack_ep0050.pth.tar \
      --dataset lasher \
      --workers 2

支持数据集: lasher / vtuav_st / vtuav_lt / gtot / rgbt210 / rgbt234
服务器数据根目录由环境变量 RGBT_DATA_ROOT 指定（默认 /root/RGBTData）。
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List

import cv2
import numpy as np

# ── 项目根目录 ────────────────────────────────────────────────────────────────
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# 数据集配置
# ══════════════════════════════════════════════════════════════════════════════

# ── 服务器 RGBT 数据根目录（组织形式: <root>/<Dataset>） ──
# 可通过环境变量 RGBT_DATA_ROOT 覆盖（兼容旧名 SGTEST_DATA_ROOT），默认 /root/RGBTData
_SERVER_DATA_ROOT = os.environ.get('RGBT_DATA_ROOT') or os.environ.get('SGTEST_DATA_ROOT') or '/root/RGBTData'

DATASET_CFG = {
    'lasher': {
        'root':   '/home/fzg/data/lasher/testingset',
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'init.txt',
        'gt_all': 'visible.txt',
    },
    'rgbt234': {
        'root':   os.path.join(_SERVER_DATA_ROOT, 'RGBT234'),
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'visible.txt',
        'gt_all': 'visible.txt',
        'gt_i':   'infrared.txt',
    },
    'rgbt210': {
        'root':   os.path.join(_SERVER_DATA_ROOT, 'RGBT210'),
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'init.txt',
        'gt_all': 'init.txt',
    },
    'gtot': {
        'root':   os.path.join(_SERVER_DATA_ROOT, 'GTOT'),
        'rgb':    'v',
        'tir':    'i',
        'gt':     'groundTruth_v.txt',
        'gt_all': 'groundTruth_v.txt',
        'gt_i':   'groundTruth_i.txt',
    },
    'vtuav_st': {
        'root':   os.path.join(_SERVER_DATA_ROOT, 'VTUAV', 'test_ST'),
        'rgb':    'rgb',
        'tir':    'ir',
        'gt':     'rgb.txt',
        'gt_all': 'rgb.txt',
        'gt_i':   'ir.txt',
    },
    'vtuav_lt': {
        'root':   os.path.join(_SERVER_DATA_ROOT, 'VTUAV', 'test_LT'),
        'rgb':    'rgb',
        'tir':    'ir',
        'gt':     'rgb.txt',
        'gt_all': 'rgb.txt',
        'gt_i':   'ir.txt',
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# IO 工具
# ══════════════════════════════════════════════════════════════════════════════

def _list_frames(seq_dir: str, modal: str) -> List[str]:
    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []
    files = sorted(
        f for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    return [os.path.join(d, f) for f in files]


def _read_frame(path: str, is_tir: bool = False) -> np.ndarray:
    """读取一帧，TIR 支持 uint16，统一返回 3 通道 RGB uint8。
    模型输入为 6 通道: RGB(3ch) + TIR(3ch)。
    """
    if is_tir:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f'cv2.imread failed: {path}')
        if img.dtype == np.uint16:
            mn, mx = float(img.min()), float(img.max())
            img = ((img.astype(np.float32) - mn) / (mx - mn + 1e-6) * 255
                   ).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img)  # 恒为 HxWx3
    else:
        img = cv2.imread(path)
        if img is None:
            raise IOError(f'cv2.imread failed: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _match_size(tir: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """将 TIR 图像 resize 到与 RGB 相同尺寸（GTOT/RGBT210 等 RGB/TIR 分辨率可能不同）。"""
    if tir.shape[:2] != rgb.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if tir.ndim == 2:
        tir = tir[:, :, None]
    return tir


def _read_gt(path: str, dataset: str = '') -> List[List[float]]:
    """读取 GT 文件，支持逗号/空格分隔。
    LasHeR/RGBT234/RGBT210: (x, y, w, h)
    GTOT: (x1, y1, x2, y2) → 自动转换为 (x, y, w, h)
    """
    is_gtot = (dataset == 'gtot')
    if not os.path.isfile(path):
        return []
    bboxes = []
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
            vals = [float(v) for v in parts[:4]]
            if is_gtot:
                vals = [vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1]]
            bboxes.append(vals)
        except ValueError:
            continue
    return bboxes


# 全局变量：VTUAV init_idx 映射（延迟加载）
_VTUAV_INIT_IDX = None


def _load_vtuav_init_idx() -> Dict:
    """加载 VTUAV init_frame.npy 映射（只加载一次）"""
    global _VTUAV_INIT_IDX
    if _VTUAV_INIT_IDX is None:
        init_path = os.path.join(_PRJ_ROOT, 'lib/train/dataset/init_frame.npy')
        if os.path.exists(init_path):
            _VTUAV_INIT_IDX = np.load(init_path, allow_pickle=True).item()
            print(f"[INFO] 加载了 {len(_VTUAV_INIT_IDX)} 个 VTUAV 序列的 init_idx 映射")
        else:
            _VTUAV_INIT_IDX = {}
            print(f"[WARN] init_frame.npy 不存在: {init_path}")
    return _VTUAV_INIT_IDX


def _list_frames_vtuav(seq_dir: str, modal: str) -> List[str]:
    """VTUAV 特殊帧加载：使用 frame_id*10+init_idx 命名映射（与训练一致）。"""
    seq_name = os.path.basename(seq_dir)
    init_idx_dict = _load_vtuav_init_idx()
    init_idx = init_idx_dict.get(seq_name, 0)

    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []

    all_files = {}
    for f in os.listdir(d):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            all_files[f] = os.path.join(d, f)

    frame_paths = []
    frame_id = 0
    while True:
        frame_name = f"{(frame_id * 10 + init_idx):06d}.jpg"
        if frame_name in all_files:
            frame_paths.append(all_files[frame_name])
            frame_id += 1
            continue
        found = False
        for ext in ['.png', '.jpeg', '.bmp', '.JPG', '.PNG']:
            alt_name = f"{(frame_id * 10 + init_idx):06d}{ext}"
            if alt_name in all_files:
                frame_paths.append(all_files[alt_name])
                frame_id += 1
                found = True
                break
        if not found:
            break
    return frame_paths


def _count_frames(seq_dir: str, modal: str = 'visible') -> int:
    return len(_list_frames(seq_dir, modal))


def _get_seq_dirs(dataset: str) -> List[str]:
    dc = DATASET_CFG.get(dataset)
    if dc is None:
        raise ValueError(f'未知数据集: {dataset}，支持: {list(DATASET_CFG)}')
    root = dc['root']
    if not os.path.isdir(root):
        raise FileNotFoundError(f'数据集目录不存在: {root}')
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


# ══════════════════════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════════════════════

_SUCCESS_THRESHOLDS   = np.linspace(0, 1, 21)
_METRICS = ['AO', 'SR', 'SR50', 'SR75', 'PS', 'NPS', 'MSR', 'MPR']
# 口径对齐官方 VTUAV (calcPlotErr_TPR.m / plot_ST.m) 及 rgbttoolkit:
#   逐帧对两模态 GT (visible / infrared) 各算一次取更优:
#     Success/重叠: err    = max(IoU_v, IoU_i)  -> SR/MSR 曲线 (阈值比较用 > t)
#     Precision/中心: errC = min(CLE_v, CLE_i)  -> PS/MPR (<= 20px)
#   官方 plot_ST.m 中 PR@20 == MPR、SR AUC == MSR（同一数值）。
#   无第二模态 GT 的数据集 (LasHeR/RGBT210) 退化为 RGB 单模态口径。
#   GTOT: MPR = min CLE < 5px (rgbttoolkit 口径); PS 仍统一用 <= 20px。
_MPRMSR_DATASETS = {'gtot', 'rgbt234', 'vtuav_st', 'vtuav_lt'}


def _iou(b1, b2) -> float:
    # [x,y,w,h] 半开区间口径; 与官方 calcRectInt 的闭区间 (x+w-1) 数学等价
    # (intersection 宽度中 -1 与 +1 相互抵消, union 相同)。
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.


def _compute_metrics(preds, gts, gts_i=None, dataset: str = '') -> dict:
    """返回单序列指标。
    口径对齐官方 VTUAV (calcPlotErr_TPR.m / plot_ST.m) 与 rgbttoolkit:
      逐帧取两模态 GT 更优: err = max(IoU_v, IoU_i), errC = min(CLE_v, CLE_i)
      SR  = max IoU 曲线 AUC (官方: 21 阈值简单平均, > t)
      PS  = min CLE <= 20px 帧占比 (官方口径下 == MPR)
    无第二模态 GT 时退化为 RGB 单模态口径。
    """
    if gts_i:
        gts_i = list(gts_i)
    valid = [(fi, p, g) for fi, (p, g) in enumerate(zip(preds, gts))
             if len(p) >= 4 and len(g) >= 4 and g[2] > 0 and g[3] > 0]
    if not valid:
        return dict(AO=-1., SR=-1., SR50=-1., SR75=-1.,
                    PS=-1., NPS=-1., MSR=-1., MPR=-1., n_valid=0)

    use_mpr_msr = dataset in _MPRMSR_DATASETS and gts_i is not None
    mpr_thr = 5.0 if dataset == 'gtot' else 20.0

    ious, nd = [], []
    mpr_dists = []  # PS/MPR: min(CLE_v, CLE_i)
    msr_ious  = []  # SR/MSR: max(IoU_v, IoU_i)

    for fi, p, g in valid:
        iou = _iou(p, g)
        ious.append(iou)
        cx_p = p[0] + p[2] / 2;  cy_p = p[1] + p[3] / 2
        cx_g = g[0] + g[2] / 2;  cy_g = g[1] + g[3] / 2
        d = float(np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2))
        nd.append(d / float(np.sqrt(g[2] * g[3])) if g[2] * g[3] > 0 else 0.)

        if use_mpr_msr:
            # 两模态 GT 取最优（rgbttoolkit: min CLE / max IoU）
            best_iou, best_dist = iou, d
            g2 = gts_i[fi] if fi < len(gts_i) else None
            if g2 is not None and len(g2) >= 4 and g2[2] > 0 and g2[3] > 0:
                iou_i = _iou(p, g2)
                if iou_i > best_iou:
                    best_iou = iou_i
                cx2 = g2[0] + g2[2] / 2;  cy2 = g2[1] + g2[3] / 2
                d_i = float(np.sqrt((cx_p - cx2) ** 2 + (cy_p - cy2) ** 2))
                if d_i < best_dist:
                    best_dist = d_i
            msr_ious.append(best_iou)
            mpr_dists.append(best_dist)
        else:
            msr_ious.append(iou)
            mpr_dists.append(d)

    ia  = np.array(ious,       dtype=np.float64)
    na  = np.array(nd,         dtype=np.float64)
    mia = np.array(msr_ious,   dtype=np.float64)
    mda = np.array(mpr_dists,  dtype=np.float64)

    # Success/SR: max(IoU_v, IoU_i) 曲线 (官方 VTUAV: err = max(errV, errI))，
    # 无第二模态 GT 时退化为 RGB IoU 曲线 (LasHeR/RGBT210)。
    # AUC 口径对齐官方 plot_ST.m: 21 个阈值上的简单平均 (temp/size(successX,2))。
    sr_curve = np.array([(mia > t).mean() for t in _SUCCESS_THRESHOLDS])
    sr_auc   = float(sr_curve.mean())

    return dict(
        AO   = float(ia.mean()),
        SR   = sr_auc,
        SR50 = float((ia >= 0.50).mean()),
        SR75 = float((ia >= 0.75).mean()),
        # Precision/PR: min(CLE_v, CLE_i) <= 20px (官方 VTUAV: errCenter = min(errCenterV, errCenterI);
        # 官方口径下 PR@20 == MPR)
        PS   = float((mda <= 20.).mean()),
        NPS  = float((na <= 0.5).mean()),
        MSR  = sr_auc,
        MPR  = float((mda < mpr_thr).mean()) if dataset == 'gtot' else float((mda <= mpr_thr).mean()),
        n_valid = len(valid),
    )


def _save_pred(preds: list, seq_name: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{seq_name}.txt'), 'w') as f:
        for b in preds:
            f.write(','.join(f'{v:.4f}' for v in b) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# 子进程 worker — LFSTrack 版本
# ══════════════════════════════════════════════════════════════════════════════

def _worker(worker_id: int,
            seq_dirs: List[str],
            result_dir: str,
            checkpoint: str,
            script_name: str,
            yaml_name: str,
            epoch: int,
            dataset: str,
            out_q: mp.Queue):
    """
    spawn 子进程入口。
    独立加载 LFSTrack 模型，串行处理分配到的序列子集。
    """
    try:
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id % torch.cuda.device_count())
        torch.set_num_threads(1)

        # ── 加载参数与 tracker — single-stream only ──
        if script_name == 'single_stream_gated':
            from lib.test.parameter.single_stream_gated import parameters as tracker_params
        elif script_name == 'single_stream':
            from lib.test.parameter.single_stream import parameters as tracker_params
        else:
            raise ValueError(f"Unsupported script: {script_name}")
        params = tracker_params(yaml_name, epoch)
        params.checkpoint = checkpoint  # 覆盖为指定路径

        from lib.test.tracker.bat import BATTrack
        tracker = BATTrack(params)
        out_q.put(('ready', worker_id))
    except Exception:
        out_q.put(('init_error', worker_id, traceback.format_exc()))
        return

    dc = DATASET_CFG[dataset]

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        t0 = time.perf_counter()
        try:
            rgb_paths = _list_frames(seq_dir, dc['rgb'])
            tir_paths = _list_frames(seq_dir, dc['tir'])
            if dataset in ('vtuav_st', 'vtuav_lt'):
                rgb_paths = _list_frames_vtuav(seq_dir, dc['rgb'])
                tir_paths = _list_frames_vtuav(seq_dir, dc['tir'])
            gt_all = _read_gt(os.path.join(seq_dir, dc.get('gt_all', dc['gt'])), dataset)
            if not gt_all:
                gt_all = _read_gt(os.path.join(seq_dir, dc['gt']), dataset)
            # 第二模态 GT（rgbttoolkit MPR/MSR 用）: GTOT groundTruth_i.txt / RGBT234 infrared.txt
            gt_i_all = []
            if dc.get('gt_i'):
                gt_i_all = _read_gt(os.path.join(seq_dir, dc['gt_i']), dataset)

            n = min(len(rgb_paths), len(tir_paths), len(gt_all))
            if gt_i_all:
                n = min(n, len(gt_i_all))
            if n < 1:
                raise ValueError('no valid frames/gt')

            # 第 0 帧：初始化 — RGB(3ch) + TIR(3ch) 拼接为 HxWx6
            rgb0 = _read_frame(rgb_paths[0], is_tir=False)
            tir0 = _match_size(_read_frame(tir_paths[0], is_tir=True), rgb0)
            image0 = np.concatenate([rgb0, tir0], axis=2)  # HxWx6
            init_info = {'init_bbox': gt_all[0]}
            tracker.initialize(image0, init_info)
            preds = [list(gt_all[0])]

            t_track = time.perf_counter()
            for fi in range(1, n):
                rgb_f = _read_frame(rgb_paths[fi], is_tir=False)
                tir_f = _match_size(_read_frame(tir_paths[fi], is_tir=True), rgb_f)
                image_f = np.concatenate([rgb_f, tir_f], axis=2)  # HxWx6
                out = tracker.track(image_f)
                preds.append(list(out['target_bbox']))

            elapsed = time.perf_counter() - t0
            fps     = (n - 1) / max(time.perf_counter() - t_track, 1e-6)

            _save_pred(preds, seq_name, result_dir)
            m = _compute_metrics(preds, gt_all[:n], gt_i_all[:n] if gt_i_all else None, dataset)
            m.update(seq_name=seq_name, fps=float(fps), elapsed=float(elapsed))
            out_q.put(('result', worker_id, m))

        except Exception:
            elapsed = time.perf_counter() - t0
            out_q.put(('seq_error', worker_id, seq_name,
                       traceback.format_exc(), float(elapsed)))


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser('LFSTrack 多进程测试')
    p.add_argument('--script',     default='single_stream',
                   choices=['bat', 'single_stream', 'single_stream_mamba', 'single_stream_mamba_hybrid', 'single_stream_gated', 'single_stream_distill', 'single_stream_gatedffn'],
                   help='训练脚本名')
    p.add_argument('--config',     default='rgbt',
                   help='yaml 配置文件名 (experiments/<script>/<config>.yaml)')
    p.add_argument('--checkpoint', required=True,
                   help='模型权重文件路径 (如 ./BATrack_ep0050.pth.tar)')
    p.add_argument('--dataset',    default='lasher',
                   choices=list(DATASET_CFG))
    p.add_argument('--save_dir',
                   default=os.path.join(_PRJ_ROOT, 'RGBT_workspace', 'results'))
    p.add_argument('--workers',    type=int, default=2,
                   help='并行 worker 进程数（单 GPU 建议 1~2）')
    p.add_argument('--epoch',      type=int, default=0,
                   help='权重 epoch 号（用于结果目录命名，使用自定义 checkpoint 时设 0）')
    p.add_argument('--sequence',   default='',
                   help='只跑单条序列（调试用）')
    args = p.parse_args()

    # ── 将 checkpoint 转为绝对路径（spawn 子进程可能 cwd 不同）──
    args.checkpoint = os.path.abspath(args.checkpoint)

    # ── config 路径: experiments/<script>/<config>.yaml ──
    yaml_name = args.config

    # ── 结果目录: save_dir/<dataset>/<script>_<config>_<epoch_tag>/ ─────────
    ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_name = f'{args.script}_{args.config}_{ckpt_tag}'
    result_dir = os.path.join(args.save_dir, args.dataset, save_name)
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 80)
    print(f'  Script     : {args.script}')
    print(f'  Config     : {args.config}')
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  Dataset    : {args.dataset}')
    print(f'  Result dir : {result_dir}')
    print(f'  Workers    : {args.workers}')
    print('=' * 80)

    # ── 收集并过滤序列 ────────────────────────────────────────────────────────
    seq_dirs = _get_seq_dirs(args.dataset)
    if args.sequence:
        seq_dirs = [d for d in seq_dirs
                    if os.path.basename(d) == args.sequence]
    n_seqs = len(seq_dirs)
    if n_seqs == 0:
        print('[ERROR] 没有找到任何序列，请检查数据集路径和 --sequence 参数。')
        sys.exit(1)

    # ── 按帧数升序后 round-robin 分配 ─────────────────────────────────────────
    dc = DATASET_CFG[args.dataset]
    sorted_dirs = sorted(seq_dirs,
                         key=lambda d: _count_frames(d, dc['rgb']))
    n_workers   = min(args.workers, n_seqs)
    chunks: List[List[str]] = [[] for _ in range(n_workers)]
    for i, d in enumerate(sorted_dirs):
        chunks[i % n_workers].append(d)
    chunks    = [c for c in chunks if c]
    n_workers = len(chunks)

    print(f'[INFO] 共 {n_seqs} 条序列，启动 {n_workers} 个 worker')
    print(f'       每 worker 约 {max(len(c) for c in chunks)} 条序列')

    # ── 表头 ─────────────────────────────────────────────────────────────────
    HDR = (f"\n{'#':<7} {'序列名':<32} "
           f"{'AO':>6} {'SR':>6} {'SR50':>6} {'SR75':>6} "
           f"{'PS':>6} {'NPS':>6} {'MSR':>6} {'MPR':>6} "
           f"{'FPS':>6} {'耗时s':>7} "
           f"{'cPS':>6} {'cSR':>6} {'cFPS':>7} {'cMSR':>7} {'cMPR':>7}")
    SEP = '-' * (len(HDR) - 1)
    print(HDR)
    print(SEP)

    # ── 启动子进程 ────────────────────────────────────────────────────────────
    out_q: mp.Queue = mp.Queue()
    procs = []
    for wid, chunk in enumerate(chunks):
        proc = mp.Process(
            target=_worker,
            args=(wid, chunk, result_dir,
                  args.checkpoint, args.script, yaml_name, args.epoch,
                  args.dataset, out_q),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    # ── 等待所有 worker 完成模型加载 ─────────────────────────────────────────
    print(f'\n[INFO] 等待 {n_workers} 个 worker 加载模型...', flush=True)
    ready = 0
    while ready < n_workers:
        msg = out_q.get()
        if msg[0] == 'ready':
            ready += 1
            print(f'[INFO]   worker {msg[1]} 就绪 ({ready}/{n_workers})', flush=True)
        elif msg[0] == 'init_error':
            print(f'[ERROR]  worker {msg[1]} 加载失败:\n{msg[2]}', flush=True)
            ready += 1
    print(f'[INFO] 所有 worker 就绪，开始追踪...\n', flush=True)

    # ── 收集结果 ──────────────────────────────────────────────────────────────
    all_recs: Dict[str, dict] = {}
    done  = 0
    t_all = time.perf_counter()
    cum_ps_vals: list = []  # 累计 PS 值
    cum_sr_vals: list = []  # 累计 SR 值
    cum_fps_vals: list = []  # 累计 FPS 值
    cum_msr_vals: list = []  # 累计 MSR 值（序列均值）
    cum_mpr_vals: list = []  # 累计 MPR 值（序列均值）

    while done < n_seqs:
        msg = out_q.get()
        if msg[0] == 'result':
            _, wid, m = msg
            done += 1
            all_recs[m['seq_name']] = m

            # ── 累计统计 ──
            cum_ps_str = '   ---'
            cum_sr_str = '   ---'
            cum_fps_str = '   ---'
            cum_msr_str = '    ---'
            cum_mpr_str = '    ---'
            if m['fps'] > 0:
                cum_fps_vals.append(m['fps'])
                cum_fps = float(np.mean(cum_fps_vals))
                cum_fps_str = f'{cum_fps:6.1f}'
            if m['AO'] >= 0 and m['PS'] >= 0:
                cum_ps_vals.append(m['PS'])
                cum_sr_vals.append(m['SR'])
                cum_ps = float(np.mean(cum_ps_vals))
                cum_sr = float(np.mean(cum_sr_vals))
                cum_ps_str = f'{cum_ps:6.3f}'
                cum_sr_str = f'{cum_sr:6.3f}'
            if m['AO'] >= 0 and m['MSR'] >= 0:
                cum_msr_vals.append(m['MSR'])
                cum_mpr_vals.append(m['MPR'])
                cum_msr = float(np.mean(cum_msr_vals))
                cum_mpr = float(np.mean(cum_mpr_vals))
                cum_msr_str = f'{cum_msr:7.3f}'
                cum_mpr_str = f'{cum_mpr:7.3f}'

            if m['AO'] >= 0:
                print(
                    f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<32s} "
                    f"{m['AO']:6.3f} {m['SR']:6.3f} "
                    f"{m['SR50']:6.3f} {m['SR75']:6.3f} "
                    f"{m['PS']:6.3f} {m['NPS']:6.3f} "
                    f"{m['MSR']:6.3f} {m['MPR']:6.3f} "
                    f"{m['fps']:6.1f} {m['elapsed']:7.1f} "
                    f"{cum_ps_str} {cum_sr_str} {cum_fps_str} {cum_msr_str} {cum_mpr_str}",
                    flush=True)
            else:
                print(
                    f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<32s} "
                    f"{'---':>6} {'---':>6} {'---':>6} {'---':>6} "
                    f"{'---':>6} {'---':>6} {'---':>6} {'---':>6} "
                    f"{m.get('fps', 0.):6.1f} {m.get('elapsed', 0.):7.1f} "
                    f"{cum_ps_str} {cum_sr_str} {cum_fps_str} {cum_msr_str} {cum_mpr_str}",
                    flush=True)

        elif msg[0] == 'seq_error':
            _, wid, seq_name, tb, elapsed = msg
            done += 1
            err_line = tb.strip().splitlines()[-1][:80]
            print(
                f"[{done:03d}/{n_seqs:03d}] {seq_name:<32s} "
                f"[ERROR] {err_line}  ({elapsed:.1f}s) w={wid}",
                flush=True)
            print(f"  Full traceback:\n{tb}", flush=True)
            all_recs[seq_name] = dict(
                seq_name=seq_name, AO=-1., SR=-1., SR50=-1., SR75=-1.,
                PS=-1., NPS=-1., MSR=-1., MPR=-1.,
                n_valid=0, fps=0., elapsed=elapsed)

    for proc in procs:
        proc.join(timeout=30)

    t_total = time.perf_counter() - t_all
    print(SEP)
    print(f'\n[INFO] 追踪完成，总耗时 {t_total / 60:.1f} 分钟', flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 汇总统计
    # ══════════════════════════════════════════════════════════════════════════
    recs  = list(all_recs.values())
    valid = [r for r in recs if r['AO'] >= 0]

    if valid:
        total_frames = sum(r['n_valid'] for r in valid)
        seq_means: Dict[str, float] = {}
        frm_means: Dict[str, float] = {}
        for k in _METRICS:
            seq_means[k] = float(np.mean([r[k] for r in valid]))
            frm_means[k] = (
                float(sum(r[k] * r['n_valid'] for r in valid) / total_frames)
                if total_frames > 0 else -1.)
        mfps = float(np.mean([r['fps'] for r in valid if r['fps'] > 0]))
    else:
        seq_means = {k: -1. for k in _METRICS}
        frm_means = {k: -1. for k in _METRICS}
        mfps = 0.
        total_frames = 0

    W = 9
    def _fmt_pct(v):
        return '---' if v is None or v < 0 else f'{v * 100:>{W}.2f}'
    print(f'\n{"=" * 78}')
    print(f"[汇总] {len(valid)}/{len(recs)} 条有效序列  "
          f"总帧数={total_frames}  平均FPS={mfps:.1f}")
    print(f"{'':12}" + ''.join(f"{k:>{W}}" for k in _METRICS))
    print(f"{'序列均值(%)':<12}" + ''.join(_fmt_pct(seq_means[k]) for k in _METRICS))
    print(f"{'帧加权(%)' :<12}" + ''.join(_fmt_pct(frm_means[k]) for k in _METRICS))
    print(f'{"=" * 78}')
    print(f"[RESULT] " +
          " ".join(f"{k}={seq_means[k]:.4f}" for k in _METRICS), flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # per_seq_metrics.csv
    # ══════════════════════════════════════════════════════════════════════════
    per_seq_fields = ['seq_name'] + _METRICS + ['n_valid', 'fps', 'elapsed', 'cMSR', 'cMPR']
    csv_path = os.path.join(result_dir, 'per_seq_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=per_seq_fields)
        w.writeheader()
        # 按完成顺序计算累计 MSR/MPR 序列均值
        csv_cum_msr, csv_cum_mpr, csv_cum_n = 0.0, 0.0, 0
        for r in recs:
            if r.get('MSR', -1) >= 0:
                csv_cum_msr += r['MSR']
                csv_cum_mpr += r['MPR']
                csv_cum_n += 1
            row = {k: r.get(k, '') for k in per_seq_fields if k not in ('cMSR', 'cMPR')}
            row['cMSR'] = f'{csv_cum_msr / csv_cum_n:.4f}' if csv_cum_n > 0 else '-1'
            row['cMPR'] = f'{csv_cum_mpr / csv_cum_n:.4f}' if csv_cum_n > 0 else '-1'
            w.writerow(row)
    print(f'[INFO] per_seq_metrics.csv  → {csv_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # eval_history.csv（追加）
    # ══════════════════════════════════════════════════════════════════════════
    history_dir = os.path.join(args.save_dir, args.dataset)
    history_csv = os.path.join(history_dir, 'eval_history.csv')
    os.makedirs(history_dir, exist_ok=True)

    hist_fields = (
        ['ckpt_tag', 'checkpoint', 'dataset', 'n_valid', 'n_total', 'mean_fps'] +
        [f'seq_{k}' for k in _METRICS] +
        [f'frm_{k}' for k in _METRICS]
    )
    hist_row: dict = {
        'ckpt_tag':   save_name,
        'checkpoint': os.path.basename(args.checkpoint),
        'dataset':    args.dataset,
        'n_valid':    len(valid),
        'n_total':    len(recs),
        'mean_fps':   f'{mfps:.2f}',
    }
    for k in _METRICS:
        hist_row[f'seq_{k}'] = f'{seq_means[k]:.4f}'
        hist_row[f'frm_{k}'] = f'{frm_means[k]:.4f}'

    write_header = not os.path.isfile(history_csv)

    # ── 兼容旧版 header（列名/列数不一致时重建整个 CSV，避免列错位）──
    old_header, old_rows = None, []
    if os.path.isfile(history_csv):
        with open(history_csv, 'r', newline='', encoding='utf-8') as f:
            rd = csv.reader(f)
            old_header = next(rd, None)
            old_rows = list(rd)
    if old_header is not None and old_header != hist_fields:
        # 旧列名 → 新列名映射（SS 改名为 SR），缺失的新列填空
        col_map = {'seq_SS': 'seq_SR', 'frm_SS': 'frm_SR'}
        with open(history_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=hist_fields)
            w.writeheader()
            for old_row in old_rows:
                if len(old_row) == len(hist_fields):
                    # 行本就是新字段顺序写入的（仅 header 旧），直接对齐
                    d = dict(zip(hist_fields, old_row))
                else:
                    d = {}
                    for i, c in enumerate(old_header):
                        if i < len(old_row):
                            nc = col_map.get(c, c)
                            if nc in hist_fields:
                                d[nc] = old_row[i]
                for hf in hist_fields:
                    d.setdefault(hf, '')
                w.writerow(d)
        print(f'[INFO] eval_history.csv 已迁移至新列结构 ({len(old_rows)} 行保留)')

    with open(history_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=hist_fields)
        if write_header:
            w.writeheader()
        w.writerow(hist_row)
    print(f'[INFO] eval_history.csv     → {history_csv}')

    # ══════════════════════════════════════════════════════════════════════════
    # summary.json
    # ══════════════════════════════════════════════════════════════════════════
    import json
    summary = dict(
        checkpoint=args.checkpoint, dataset=args.dataset,
        script=args.script, config=args.config, epoch=args.epoch,
        n_sequences=len(recs), n_valid=len(valid), mean_fps=mfps,
        seq_means={k: seq_means[k] for k in _METRICS},
        frm_means={k: frm_means[k] for k in _METRICS},
        total_time_min=f'{t_total / 60:.1f}',
    )
    summary_path = os.path.join(result_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[INFO] summary.json         → {summary_path}')


if __name__ == '__main__':
    main()

