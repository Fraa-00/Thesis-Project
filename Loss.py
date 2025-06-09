import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred, target):
        # pred and target: shape (..., 3) with (x, y, theta)
        dx = pred[:, 0] - target[:, 0]
        dy = pred[:, 1] - target[:, 1]

        # Conversione in radianti
        target_theta_rad = torch.deg2rad(target[:, 2])
        pred_theta_rad = torch.deg2rad(pred[:, 2])

        # Differenza angolare
        dtheta = target_theta_rad - pred_theta_rad
        dtheta = torch.abs(torch.sin(dtheta))

        loss = dx**2 + dy**2 + dtheta**2
        
        return loss.mean()
