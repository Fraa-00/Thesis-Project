import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from Network import Marepo_Regressor, DinoV2, MegaLoc, MLP
from My_DataLoader import rgb_transforms
import torchvision.transforms.functional as TF
from My_Loss import my_loss

def evaluation_loop(model, mlp, second_encoder, val_loader, loss_fn, device):
    model.eval()
    if second_encoder:
        second_encoder.eval()
    mlp.eval()

    val_running = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            feat1 = model.get_features(TF.rgb_to_grayscale(imgs))
            feat1_flat = feat1.flatten(2).permute(0, 2, 1)
            if second_encoder:
                feat2 = second_encoder(imgs)
                feat2_flat = feat2.flatten(2).permute(0, 2, 1)
                feats = torch.cat([feat1_flat, feat2_flat], dim=-1)
            else:
                feats = feat1_flat

            pooled = torch.max(feats, dim=1)[0]
            preds = mlp(pooled)
            loss_val_batch = loss_fn(preds, targets)
            batch_loss = loss_val_batch.mean().item() if loss_val_batch.dim() > 0 else loss_val_batch.item()
            val_running += batch_loss

    return val_running / len(val_loader)