import torch
import numpy as np

class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm

class PreprocessorMM(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,6,H,W)
        return img_tensor_norm


class PreprocessorRGBT(object):
    """(已弃用) 4 通道 RGB+TIR 预处理，仅为兼容旧 4ch 实验保留。
    当前 6ch 输入统一使用 PreprocessorMM。
    兼容 4/6 通道自动选择对应统计量。
    """
    def __init__(self):
        self.mean4 = torch.tensor([0.485, 0.456, 0.406, 0.485]).view((1, 4, 1, 1)).cuda()
        self.std4  = torch.tensor([0.229, 0.224, 0.225, 0.229]).view((1, 4, 1, 1)).cuda()
        self.mean6 = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()
        self.std6  = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        if img_tensor.shape[1] == 4:
            img_tensor_norm = ((img_tensor / 255.0) - self.mean4) / self.std4  # (1,4,H,W)
        else:
            img_tensor_norm = ((img_tensor / 255.0) - self.mean6) / self.std6  # (1,6,H,W)
        return img_tensor_norm


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
