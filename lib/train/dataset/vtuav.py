import os
import os.path
import numpy as np
import torch
import csv
import pandas
from collections import OrderedDict

from .base_video_dataset import BaseVideoDataset
from lib.train.admin.environment import env_settings
from lib.train.dataset.depth_utils import get_x_frame
"""
VTUAV dataset loader (ported from OmniAdapt, based on HMFT).
Frame naming uses frame_id*10+init_idx (see init_frame.npy).
GT file rgb.txt/ir.txt: 8 columns per row, first 4 = (x, y, w, h).
"""


class VTUAV(BaseVideoDataset):

    def __init__(self, root=None, split=None, modality="RGBT"):
        if modality is None:
            raise ValueError('Unknown modality mode.')
        else:
            self.modality = modality

        root = env_settings().vtuav_dir if root is None else root
        super().__init__("VTUAV", root)

        self.dtype = 'rgbrgb'

        if split is not None:
            if split == 'train':
                file_path = os.path.join(self.root, 'train')
            elif split == 'val_st':
                file_path = os.path.join(self.root, 'test_ST')
            elif split == 'val_lt':
                file_path = os.path.join(self.root, 'test_LT')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = os.listdir(file_path)
            self.root = file_path

        self.init_idx = np.load(os.path.join(os.path.dirname(__file__), "init_frame.npy"), allow_pickle=True).item()
        self.sequence_list = sequence_list

    def get_name(self):
        return 'vtuav'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self):
        return os.listdir(self.root)

    def _read_bb_anno(self, seq_path):
        if self.modality in ['RGB', 'RGBT']:
            bb_anno_file = os.path.join(seq_path, "rgb.txt")
            gt = pandas.read_csv(bb_anno_file, delimiter=' ', header=None, dtype=np.float32,
                                 na_filter=False, low_memory=False).values
        elif self.modality in ['T']:
            bb_anno_file = os.path.join(seq_path, "ir.txt")
            gt = np.loadtxt(bb_anno_file).astype(np.float32)
        else:
            raise ValueError('Unknown modality mode.')

        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover > 0)

        return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        seq_name = seq_path.split('/')[-1]
        if seq_name in self.init_idx:
            init_idx = self.init_idx[seq_name]
        else:
            init_idx = 0
        nz = 6
        return (os.path.join(seq_path, 'rgb', str(frame_id * 10 + init_idx).zfill(nz) + '.jpg'),
                os.path.join(seq_path, 'ir', str(frame_id * 10 + init_idx).zfill(nz) + '.jpg'))

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # (h,w,6)

    def get_class_name(self, seq_id):
        return None

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta
