import torch
from torch import nn

class STEFunction(torch.autograd.Function):
    """
    直通估计器 (Straight-Through Estimator)
    前向传播进行硬二值化，反向传播跳过二值化直接传梯度
    """
    @staticmethod
    def forward(ctx, input_mask, threshold=0.5):
        return (input_mask > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def get_ste_mask(mask_logit, threshold=0.1):
    """
    计算高斯剪枝掩码
    """
    soft_mask = torch.sigmoid(mask_logit)
    hard_mask = STEFunction.apply(soft_mask, threshold)
    return soft_mask + (hard_mask - soft_mask).detach()

def get_sh_masks(sh_logits, threshold=0.1):
    """
    计算 SH 自适应剪枝掩码
    """
    soft_sh = torch.sigmoid(sh_logits)                      # [N, 3]
    hard_sh = STEFunction.apply(soft_sh, threshold)         # [N, 3]

    return soft_sh + (hard_sh - soft_sh).detach()

def apply_sh_masks(shs, masks):
    """
    独立掩码应用到SH:
    DC 不动
    l=1 -> 1:4
    l=2 -> 4:9
    l=3 -> 9:16

    如果后面还有额外通道，保留 tail，不要截断。
    """
    assert shs.shape[1] >= 16, f"Expected >=16 SH channels, got {shs.shape[1]}"

    sh0 = shs[:, 0:1, :]
    sh1 = shs[:, 1:4, :] * masks[:, 0:1].unsqueeze(-1)
    sh2 = shs[:, 4:9, :] * masks[:, 1:2].unsqueeze(-1)
    sh3 = shs[:, 9:16, :] * masks[:, 2:3].unsqueeze(-1)

    tail = shs[:, 16:, :] if shs.shape[1] > 16 else None

    if tail is not None:
        return torch.cat([sh0, sh1, sh2, sh3, tail], dim=1)
    else:
        return torch.cat([sh0, sh1, sh2, sh3], dim=1)