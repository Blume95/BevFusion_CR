import torch
import torch.nn as nn


class computeLoss(nn.Module):
    def __init__(self):
        super(computeLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, target, x, x_backward, pred_forward):
        loss_cycle = self.l1(x, x_backward)
        loss_bce = self.bce(pred, target)
        loss_bce_forward = self.bce(pred_forward, target)
        loss = 0.1 * loss_cycle + loss_bce + loss_bce_forward

        return loss, loss_cycle, loss_bce, loss_bce_forward
