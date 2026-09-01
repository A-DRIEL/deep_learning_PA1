import torch


def iou_score(pred_binary, target_binary, eps=1e-7):
    """IoU entre duas máscaras binárias (B, H, W) ou (H, W)."""
    intersection = (pred_binary & target_binary).float().sum()
    union = (pred_binary | target_binary).float().sum()
    return ((intersection + eps) / (union + eps)).item()


def dice_score(pred_binary, target_binary, eps=1e-7):
    """Dice entre duas máscaras binárias (B, H, W) ou (H, W)."""
    intersection = (pred_binary & target_binary).float().sum()
    denom = pred_binary.float().sum() + target_binary.float().sum()
    return ((2 * intersection + eps) / (denom + eps)).item()