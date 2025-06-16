import torch

def evaluation_loop(model, val_loader, loss_fn, device):
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss_val_batch = loss_fn(preds, targets)
            batch_loss = loss_val_batch.mean().item() if loss_val_batch.dim() > 0 else loss_val_batch.item()
            val_running += batch_loss
    return val_running / len(val_loader)