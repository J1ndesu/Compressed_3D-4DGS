"""Straight-through gates for Gaussian and spatial spherical harmonic pruning."""

import torch
from torch import nn

class STEFunction(torch.autograd.Function):
    """Binarize with a strict threshold and pass input gradients through unchanged."""
    @staticmethod
    def forward(ctx, input_mask, threshold=0.5):
        return (input_mask > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def get_ste_mask(mask_logit, threshold=0.1):
    """Return binary Gaussian gates with sigmoid surrogate gradients.

    Args:
        mask_logit: Floating-point logits, normally shaped (N, 1).
        threshold: Keep entries whose sigmoid value is strictly greater.

    Returns:
        A tensor with the input shape: binary in the forward pass, with
        gradients through sigmoid(mask_logit) in the backward pass.
    """
    soft_mask = torch.sigmoid(mask_logit)
    hard_mask = STEFunction.apply(soft_mask, threshold)
    return soft_mask + (hard_mask - soft_mask).detach()

def get_sh_masks(sh_logits, threshold=0.1):
    """Return independent straight-through gates for spatial SH bands 1, 2, 3.

    Args:
        sh_logits: Per-Gaussian band logits of shape (N, 3).
        threshold: Keep bands whose sigmoid value is strictly greater.

    Returns:
        Binary forward gates of shape (N, 3), with sigmoid surrogate gradients.
        Keeping a higher band does not require keeping any lower band.
    """
    soft_sh = torch.sigmoid(sh_logits)                      # [N, 3]
    hard_sh = STEFunction.apply(soft_sh, threshold)         # [N, 3]

    return soft_sh + (hard_sh - soft_sh).detach()

def apply_sh_masks(shs, masks):
    """Gate spatial SH bands while preserving DC, extra channels, and tensor shape.

    Args:
        shs: Coefficients of shape (N, C, 3), with C >= 16 and DC at index 0.
        masks: Gates of shape (N, 3), one for each spatial band.

    Returns:
        Coefficients of shape (N, C, 3). Bands 1, 2, and 3 occupy slices
        1:4, 4:9, and 9:16. Channels from index 16 onward pass through.
        Zeroed bands remain allocated; this function does not pack a bitstream.
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
