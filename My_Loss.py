import torch
import torch.nn as nn

class my_loss(nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, pred, target):
        # pred and target: shape (..., 3) with (x, y, theta)
        dx = torch.abs(pred[:, 0] - target[:, 0])
        dy = torch.abs(pred[:, 1] - target[:, 1])

        # Conversione in radianti
        target_theta_rad = torch.deg2rad(target[:, 2])
        pred_theta_rad = torch.deg2rad(pred[:, 2])

        # Differenza angolare
        dtheta = target_theta_rad - pred_theta_rad
        dtheta = torch.abs(torch.sin(dtheta))

        loss = 50 (dx + dy) + 10 * dtheta
        
        return loss.mean()
