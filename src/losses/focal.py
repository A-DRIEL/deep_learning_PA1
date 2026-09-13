import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss multiclasse (Lin et al., 2017), para o rótulo de 3 classes
    (fundo / interior / borda) da Trilha A.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Ajustando os parâmetros ela cobre as 4 variantes pedidas no enunciado:
    - gamma=0, alpha=None      -> CE comum
    - gamma=0, alpha=pesos     -> CE balanceada (equivalente ao que já
                                   existe em train_dsb2018_3class.py)
    - gamma>0, alpha=None      -> focal "pura"
    - gamma>0, alpha=pesos     -> focal balanceada
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor (num_classes,) ou None
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C, H, W) ; target: (B, H, W) long
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_unsq = target.unsqueeze(1)
        log_pt = log_probs.gather(1, target_unsq).squeeze(1)
        pt = probs.gather(1, target_unsq).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss