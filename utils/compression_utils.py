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

def get_sh_masks(sh_logits, threshold=0.5):
    """
    计算自适应 SH 级联掩码
    """
    soft_sh = torch.sigmoid(sh_logits)
    m1 = STEFunction.apply(soft_sh[:, 0:1], threshold)
    m2 = STEFunction.apply(soft_sh[:, 1:2], threshold)
    m3 = STEFunction.apply(soft_sh[:, 2:3], threshold)
    
    # 级联依赖：1阶关了，2、3阶必须关
    eff_m1 = m1
    eff_m2 = m1 * m2
    eff_m3 = m1 * m2 * m3
    
    hard_masks = torch.cat([eff_m1, eff_m2, eff_m3], dim=-1)
    return soft_sh + (hard_masks - soft_sh).detach()

def apply_sh_masks(shs, masks):
    """
    将掩码应用到球谐函数系数上
    shs 形状: [N, 16, 3] (1个DC + 15个额外SH系数)
    masks 形状: [N, 3] (对应 1, 2, 3 阶的级联硬掩码)
    """
    # 这里的 shs 是特征向量，通常索引 0 是 DC，1-15 是 SH
    # 保持 DC 不变，只对高阶项做掩码
    
    # 第1阶: 包含 3 个系数 (索引 1, 2, 3)
    shs[:, 1:4, :] = shs[:, 1:4, :] * masks[:, 0:1].unsqueeze(-1)
    
    # 第2阶: 包含 5 个系数 (索引 4, 5, 6, 7, 8)
    shs[:, 4:9, :] = shs[:, 4:9, :] * masks[:, 1:2].unsqueeze(-1)
    
    # 第3阶: 包含 7 个系数 (索引 9, 10, 11, 12, 13, 14, 15)
    shs[:, 9:16, :] = shs[:, 9:16, :] * masks[:, 2:3].unsqueeze(-1)
    
    return shs